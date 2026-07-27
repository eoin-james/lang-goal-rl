"""Stage 6 proof gate: end-to-end demo across ad-hoc live phrasings -- task success + time-to-redirect.

Per model seed (reusing stage 3's already-trained `GoalEmbeddingExtractor`-
based SAC checkpoints zero-shot, no new RL training -- these are the same
checkpoints stage 4's `eval_held_out.py`/`eval_nn_lookup_held_out.py` loaded,
since stage 6 needs a policy that consumes a 16-dim goal embedding, unlike
stage 5's literal-xyz stage-1 checkpoints), this script runs:

1. A literal-goal sanity check (`train.py`'s `evaluate_literal`, identical to
   every prior stage) -- confirms the reused checkpoint still performs the
   base task before trusting anything downstream.
2. A no-switch control for Set A (stage 4's 14
   `held_out_paraphrases.HELD_OUT_PARAPHRASES`) using the exact same
   eval protocol stage 4's attempt 4 used (`evaluate_language_goal`, ground
   truth = `compute_region_centroid`, 50 episodes/instruction) but with
   `LiveGoalController.instruction_to_goal_embedding` supplying the
   embedding instead of stage 4's script directly calling
   `nearest_neighbor_projection`. Since `LiveGoalController` wraps that exact
   mechanism at k=1 over the identical 84-sentence combined reference, this
   control doubles as this experiment's required sanity cross-check: it
   should approximately reproduce stage 4's already-measured 0.571
   mean/1.000 median result. If it doesn't, something in this script's
   wiring is wrong and Set B shouldn't be trusted until that's fixed (see
   ROADMAP.md-referenced brief).
3. The same no-switch control for Set B (`set_b_vocabulary.SET_B_INSTRUCTIONS`,
   7 brand-new phrasings never used anywhere in this project before -- see
   that module's docstring for the disjointness guarantee).
4. The actual mid-episode live-language-switch test, for both sets: each
   instruction is paired with one other, different-region instruction from
   the same set (`build_pairs`); a single episode runs
   `rollout_with_goal_switch_timed` (this script's own instrumented copy of
   `midepisode_regoal.rollout_with_goal_switch` -- see that function's
   docstring for why it's a local duplicate, not an import) targeting the
   first instruction's live-encoded goal embedding + region centroid for
   `SWITCH_STEP` steps, then switching to the second instruction's, all the
   way to `MAX_STEPS`. Task success is judged against the *second*
   instruction's `compute_region_centroid` -- the region-vs-point lesson
   applies here exactly as it has since stage 3. Time-to-redirect is the
   number of steps from `SWITCH_STEP` to the first post-switch step whose
   `info["is_success"]` is true, for episodes that ever reach it; episodes
   that never do are recorded with `time_to_redirect=None` ("did not
   redirect"), never averaged in as an arbitrary large number and never
   dropped silently.

Results are dumped to `runs/seed_<k>/results.json` for
`aggregate_and_report.py` to assemble into `report.md`; per-instruction and
per-pair summary lines are also printed so `runs/seed_<k>/stdout.log`
(written by `launch_seeds.sh`'s redirect) is independently readable.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import numpy.typing as npt
import torch
from stable_baselines3 import SAC

EXPERIMENT_DIR = Path(__file__).parent
STAGE3_DIR = EXPERIMENT_DIR.parent / "03_language_goal_projection"
sys.path.insert(0, str(STAGE3_DIR))

from train import (  # noqa: E402 -- STAGE3_DIR must be on sys.path first, see module docstring
    DEFAULT_ENCODER_PATH,
    ENV_ID,
    compute_region_centroid,
    evaluate_language_goal,
    evaluate_literal,
    load_frozen_encoder,
)

from set_b_vocabulary import (  # noqa: E402 -- same-directory script-relative import, matches project convention
    set_b_region_names,
    set_b_texts,
    verify_disjoint,
)

from lang_goal_rl.episode_recording import _pin_desired_goal_embedding
from lang_goal_rl.goal_embedding_extractor import GoalEmbeddingExtractor
from lang_goal_rl.held_out_paraphrases import held_out_region_names, held_out_texts
from lang_goal_rl.live_goal_controller import LiveGoalController

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_CHECKPOINT_DIR = STAGE3_DIR / "checkpoints"

MAX_STEPS = 50
"""FetchReach-v4's registered `max_episode_steps` -- same total budget every
prior stage's episodes used."""

SWITCH_STEP = 20
"""Fixed mid-episode switch point (step 20 of 50). Stage 5 already measured
the switch-point sweep (10/20/30/40) and found no meaningful difference for
the literal-goal mechanism itself; this stage is testing the *language*
pipeline layered on top of that already-answered mechanism question, so one
representative mid-episode point is enough."""

DEFAULT_CONTROL_EPISODES = 50
"""Episodes per instruction for the no-switch control -- matches stage 4's
`eval_nn_lookup_held_out.py` exactly, so the Set-A control is an
apples-to-apples reproduction of stage 4's already-measured 0.571 mean/1.000
median result, not a differently-powered new measurement."""

SET_A_CONTROL_BASE_SEED = 30_000
SET_B_CONTROL_BASE_SEED = 40_000
SET_A_SWITCH_BASE_SEED = 50_000
SET_B_SWITCH_BASE_SEED = 60_000
"""Fresh seed ranges, offset from every earlier stage's (1000, 5000, 9000,
13000, ...) so no reset seed is silently reused across stages for a
different purpose. Each is independent of `--seed` (the *model* checkpoint
under test) -- all 3 model seeds are evaluated against identical episode
conditions, the same convention stage 5's `run_regoal_eval.py` used, so
comparing across model seeds isn't confounded by different episode draws."""

SET_A_PAIRING_SEED = 1
SET_B_PAIRING_SEED = 2
"""Fixed seeds for `build_pairs` -- the instruction pairing is computed once
and reused identically across all 3 model seeds, not re-randomized per run."""


@dataclass(frozen=True)
class TimedGoalSwitchResult:
    """Outcome of one instrumented mid-episode goal-switch rollout.

    Attributes:
        success: Whether `info["is_success"]` was truthy on the *final*
            post-switch step -- same definition `midepisode_regoal.
            rollout_with_goal_switch`'s `success` field uses, kept
            consistent with every stage 3-5 success metric.
        first_success_step: Absolute step index (counting from episode
            start) of the first post-switch step whose `info["is_success"]`
            was truthy, or `None` if the episode never reached the new goal
            before ending. `time_to_redirect = first_success_step -
            switch_step` when not `None`.
        switch_step: The `switch_step` argument, echoed back.
        n_steps: Total env steps run (pre- and post-switch combined).

    """

    success: bool
    first_success_step: int | None
    switch_step: int
    n_steps: int


def _goal_input_context(model: SAC, embedding: torch.Tensor | None) -> AbstractContextManager[None]:
    """Pin `model.actor.features_extractor`'s desired-goal output to `embedding`, or do nothing.

    Same helper `midepisode_regoal._goal_input_context` implements; kept as
    a local copy here rather than imported since that name is private to its
    own module and `rollout_with_goal_switch_timed` (this file's own
    instrumented rollout, see its docstring) is the only caller.
    """
    if embedding is None:
        return nullcontext()
    features_extractor = cast("GoalEmbeddingExtractor", model.actor.features_extractor)
    return _pin_desired_goal_embedding(features_extractor, embedding)


def rollout_with_goal_switch_timed(
    model: SAC,
    env: gym.Env,
    *,
    initial_goal_xyz: npt.ArrayLike,
    switch_step: int,
    new_goal_xyz: npt.ArrayLike,
    max_steps: int,
    base_seed: int,
    initial_goal_embedding: torch.Tensor,
    new_goal_embedding: torch.Tensor,
) -> TimedGoalSwitchResult:
    """Roll out a mid-episode goal switch, tracking the first post-switch success step.

    Identical control flow to `midepisode_regoal.rollout_with_goal_switch`'s
    embedding-substitution mode (reset, pin `initial_goal_embedding` for
    `switch_step` steps against `initial_goal_xyz`'s env ground truth,
    overwrite the env's goal, pin `new_goal_embedding` for the remaining
    steps against `new_goal_xyz`) -- duplicated locally, not imported,
    because "first step whose `is_success` flips true" (needed for stage 6's
    time-to-redirect metric) is stage-6-specific instrumentation that
    `rollout_with_goal_switch` doesn't expose (it only tracks the *final*
    step's success). See this experiment's report.md Anomalies section for
    a note flagging this as a candidate for promotion into
    `midepisode_regoal.py` if a future stage needs the same timing data.

    Args:
        model: A trained SAC model with a `GoalEmbeddingExtractor`-based
            `model.actor.features_extractor` (stage 2/3/4-style checkpoint).
        env: The FetchReach-v4 env instance to roll out on.
        initial_goal_xyz: Env ground-truth goal for the first `switch_step`
            steps, shape `(3,)`.
        switch_step: Step at which the switch happens. Must be `>= 1` and
            `< max_steps`.
        new_goal_xyz: Env ground-truth goal from `switch_step` onward, shape
            `(3,)`. `success`/`first_success_step` are judged against this.
        max_steps: Total episode budget (pre- + post-switch combined).
        base_seed: Seed passed to `env.reset(seed=base_seed)`.
        initial_goal_embedding: Policy's desired-goal input for the
            pre-switch phase, shape `(embed_dim,)`.
        new_goal_embedding: Policy's desired-goal input for the post-switch
            phase, shape `(embed_dim,)`.

    Returns:
        A `TimedGoalSwitchResult`.

    Raises:
        ValueError: If `switch_step < 1`, `switch_step >= max_steps`, or
            `max_steps` exceeds the env's registered `max_episode_steps`.

    """
    spec = getattr(env, "spec", None)
    registered_limit = getattr(spec, "max_episode_steps", None)
    if registered_limit is not None and max_steps > registered_limit:
        msg = (
            f"max_steps ({max_steps}) exceeds the env's registered max_episode_steps "
            f"({registered_limit})"
        )
        raise ValueError(msg)
    if switch_step < 1:
        msg = f"switch_step must be >= 1 (got {switch_step}) -- a switch at step 0 isn't mid-episode"
        raise ValueError(msg)
    if switch_step >= max_steps:
        msg = f"switch_step ({switch_step}) must be < max_steps ({max_steps}) to leave a post-switch step"
        raise ValueError(msg)

    initial_goal = np.asarray(initial_goal_xyz, dtype=np.float64)
    new_goal = np.asarray(new_goal_xyz, dtype=np.float64)

    obs, _info = env.reset(seed=base_seed)
    env.unwrapped.goal = initial_goal.copy()
    obs["desired_goal"] = initial_goal.copy()

    n_steps = 0
    terminated = truncated = False
    with _goal_input_context(model, initial_goal_embedding):
        while n_steps < switch_step and not (terminated or truncated):
            action, _state = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(action)
            n_steps += 1

    env.unwrapped.goal = new_goal.copy()
    obs["desired_goal"] = new_goal.copy()
    is_success = False
    first_success_step: int | None = None

    with _goal_input_context(model, new_goal_embedding):
        while not (terminated or truncated) and n_steps < max_steps:
            action, _state = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            n_steps += 1
            is_success = bool(info.get("is_success", is_success))
            if is_success and first_success_step is None:
                first_success_step = n_steps

    return TimedGoalSwitchResult(
        success=is_success,
        first_success_step=first_success_step,
        switch_step=switch_step,
        n_steps=n_steps,
    )


def build_pairs(regions: Sequence[str], *, seed: int) -> list[tuple[int, int]]:
    """Pair every instruction index with one different-region instruction, deterministically.

    Args:
        regions: Row-aligned region name per instruction.
        seed: Seed for the partner draw; the same `(regions, seed)` always
            produces the same pairing, so it can be computed once and reused
            identically across every model seed.

    Returns:
        A list of `(index, partner_index)` pairs, one per input instruction,
        `regions[partner_index] != regions[index]` guaranteed for every
        pair.

    """
    rng = np.random.default_rng(seed)
    n = len(regions)
    pairs = []
    for index in range(n):
        candidate_indices = [j for j in range(n) if regions[j] != regions[index]]
        partner_index = int(rng.choice(candidate_indices))
        pairs.append((index, partner_index))
    return pairs


def run_no_switch_control(
    model: SAC,
    env: gym.Env,
    controller: LiveGoalController,
    texts: Sequence[str],
    regions: Sequence[str],
    *,
    base_seed: int,
    n_episodes: int,
    label: str,
) -> list[dict]:
    """Run the no-switch, live-embedding control for every instruction in one vocabulary set.

    Args:
        model: The trained SAC checkpoint under test.
        env: The FetchReach-v4 env instance to roll out on.
        controller: `LiveGoalController` supplying each instruction's live
            goal embedding.
        texts: Instructions to evaluate.
        regions: Row-aligned ground-truth region name per instruction.
        base_seed: First instruction's base eval seed; instruction `i` uses
            `base_seed + i * n_episodes`.
        n_episodes: Episodes per instruction.
        label: Log-line prefix (`"set_a"` or `"set_b"`).

    Returns:
        Per-instruction result dicts: `instruction`, `region`,
        `success_rate`, `n_episodes`.

    """
    results = []
    for index, (instruction, region_name) in enumerate(zip(texts, regions, strict=True)):
        embedding = controller.instruction_to_goal_embedding(instruction)
        instruction_base_seed = base_seed + index * n_episodes
        success_rate = evaluate_language_goal(
            model,
            env,
            region_name=region_name,
            projected_embedding=embedding,
            n_episodes=n_episodes,
            base_seed=instruction_base_seed,
        )
        results.append(
            {"instruction": instruction, "region": region_name, "success_rate": success_rate, "n_episodes": n_episodes},
        )
        print(
            f'{label}_no_switch_success_rate={success_rate:.3f} instruction="{instruction}" '
            f'region="{region_name}" over {n_episodes} episodes',
        )
    return results


def run_switch_suite(
    model: SAC,
    env: gym.Env,
    controller: LiveGoalController,
    texts: Sequence[str],
    regions: Sequence[str],
    pairs: list[tuple[int, int]],
    *,
    base_seed: int,
    label: str,
) -> list[dict]:
    """Run the live mid-episode instruction-switch test for every pair in one vocabulary set.

    Args:
        model: The trained SAC checkpoint under test.
        env: The FetchReach-v4 env instance to roll out on.
        controller: `LiveGoalController` supplying each instruction's live
            goal embedding.
        texts: Instructions in this vocabulary set.
        regions: Row-aligned ground-truth region name per instruction.
        pairs: `(index, partner_index)` pairs from `build_pairs`.
        base_seed: First pair's episode seed; pair `i` uses `base_seed + i`.
        label: Log-line prefix (`"set_a"` or `"set_b"`).

    Returns:
        One result dict per pair: `instr1`/`region1`/`instr2`/`region2`,
        `success`, `first_success_step`, `time_to_redirect` (`None` if the
        episode never redirected), `switch_step`, `n_steps`.

    """
    results = []
    for pair_index, (first_index, second_index) in enumerate(pairs):
        instr1, region1 = texts[first_index], regions[first_index]
        instr2, region2 = texts[second_index], regions[second_index]

        embedding1 = controller.instruction_to_goal_embedding(instr1)
        xyz1 = compute_region_centroid(region1)
        embedding2 = controller.instruction_to_goal_embedding(instr2)
        xyz2 = compute_region_centroid(region2)

        episode_seed = base_seed + pair_index
        result = rollout_with_goal_switch_timed(
            model,
            env,
            initial_goal_xyz=xyz1,
            switch_step=SWITCH_STEP,
            new_goal_xyz=xyz2,
            max_steps=MAX_STEPS,
            base_seed=episode_seed,
            initial_goal_embedding=embedding1,
            new_goal_embedding=embedding2,
        )
        time_to_redirect = (
            None if result.first_success_step is None else result.first_success_step - result.switch_step
        )
        results.append(
            {
                "instr1": instr1,
                "region1": region1,
                "instr2": instr2,
                "region2": region2,
                "success": result.success,
                "first_success_step": result.first_success_step,
                "time_to_redirect": time_to_redirect,
                "switch_step": result.switch_step,
                "n_steps": result.n_steps,
            },
        )
        redirect_text = "did not redirect" if time_to_redirect is None else f"time_to_redirect={time_to_redirect}"
        print(
            f'{label}_switch task_success={result.success} {redirect_text} '
            f'instr1="{instr1}" region1="{region1}" -> instr2="{instr2}" region2="{region2}"',
        )
    return results


def print_switch_summary(label: str, episodes: list[dict]) -> None:
    """Print the proof-gate metric (task success + time-to-redirect) for one vocabulary set.

    Args:
        label: Log-line prefix (`"set_a"` or `"set_b"`).
        episodes: `run_switch_suite`'s return value.

    """
    successes = [episode["success"] for episode in episodes]
    redirect_times = [episode["time_to_redirect"] for episode in episodes if episode["time_to_redirect"] is not None]
    task_success_rate = float(np.mean(successes))
    redirect_success_rate = len(redirect_times) / len(episodes)
    mean_ttr = float(np.mean(redirect_times)) if redirect_times else None
    median_ttr = float(np.median(redirect_times)) if redirect_times else None
    print(
        f"{label}_switch_task_success_rate={task_success_rate:.3f} "
        f"redirect_success_rate={redirect_success_rate:.3f} "
        f"mean_time_to_redirect={mean_ttr} median_time_to_redirect={median_ttr} "
        f"n_episodes={len(episodes)} n_redirected={len(redirect_times)}",
    )


def main() -> None:
    """Load one seed's saved SAC checkpoint zero-shot and run the full stage-6 live eval suite."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Stage-3 SAC checkpoint seed (0/1/2)")
    parser.add_argument("--sanity-episodes", type=int, default=50)
    parser.add_argument("--control-episodes", type=int, default=DEFAULT_CONTROL_EPISODES)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--encoder-path", type=Path, default=DEFAULT_ENCODER_PATH)
    args = parser.parse_args()

    disjointness_problems = verify_disjoint()
    if disjointness_problems:
        msg = "Set B overlaps prior vocabulary:\n" + "\n".join(disjointness_problems)
        raise AssertionError(msg)

    gym.register_envs(gymnasium_robotics)
    env = gym.make(ENV_ID)

    checkpoint_path = args.checkpoint_dir / f"seed_{args.seed}.zip"
    model = SAC.load(checkpoint_path, env=env)
    print(f"loaded checkpoint from {checkpoint_path} (no new training, seed={args.seed})")

    literal_success_rate = evaluate_literal(model, env, args.sanity_episodes)
    print(f"literal_sanity_success_rate={literal_success_rate:.3f} over {args.sanity_episodes} episodes")

    encoder = load_frozen_encoder(args.encoder_path)
    controller = LiveGoalController(encoder)

    set_a_texts, set_a_regions = held_out_texts(), held_out_region_names()
    set_b_texts_, set_b_regions_ = set_b_texts(), set_b_region_names()

    set_a_control = run_no_switch_control(
        model, env, controller, set_a_texts, set_a_regions,
        base_seed=SET_A_CONTROL_BASE_SEED, n_episodes=args.control_episodes, label="set_a",
    )
    set_b_control = run_no_switch_control(
        model, env, controller, set_b_texts_, set_b_regions_,
        base_seed=SET_B_CONTROL_BASE_SEED, n_episodes=args.control_episodes, label="set_b",
    )

    set_a_pairs = build_pairs(set_a_regions, seed=SET_A_PAIRING_SEED)
    set_b_pairs = build_pairs(set_b_regions_, seed=SET_B_PAIRING_SEED)

    set_a_switch = run_switch_suite(
        model, env, controller, set_a_texts, set_a_regions, set_a_pairs,
        base_seed=SET_A_SWITCH_BASE_SEED, label="set_a",
    )
    set_b_switch = run_switch_suite(
        model, env, controller, set_b_texts_, set_b_regions_, set_b_pairs,
        base_seed=SET_B_SWITCH_BASE_SEED, label="set_b",
    )

    print_switch_summary("set_a", set_a_switch)
    print_switch_summary("set_b", set_b_switch)

    output = {
        "model_seed": args.seed,
        "literal_sanity_success_rate": literal_success_rate,
        "sanity_episodes": args.sanity_episodes,
        "switch_step": SWITCH_STEP,
        "max_steps": MAX_STEPS,
        "control_episodes": args.control_episodes,
        "set_a_no_switch_control": set_a_control,
        "set_b_no_switch_control": set_b_control,
        "set_a_switch_episodes": set_a_switch,
        "set_b_switch_episodes": set_b_switch,
    }
    results_path = EXPERIMENT_DIR / "runs" / f"seed_{args.seed}" / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(output, indent=2))
    print(f"results_saved={results_path}")

    env.close()


if __name__ == "__main__":
    main()
