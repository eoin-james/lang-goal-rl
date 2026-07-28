"""Stage 8 proof gate: relative-move success vs. budget-matched fresh baseline.

Per model seed (a stage-1 literal-xyz checkpoint reused zero-shot, no new RL
training -- same reuse rationale `midepisode_regoal.py` already applies to
stage 5), this script runs:

1. A literal-goal sanity check, copied from stage 5's own local copy of
   `train.py`'s `evaluate()` (same held-out seed convention, same
   deterministic rollout loop) -- confirms the reused checkpoint still
   performs the plain literal-goal task, and flags the known SAC
   deterministic-eval collapse signature (seeds 2 and 7, per ROADMAP.md's
   Known risks) before trusting any relative-move result from that seed.
2. For every `(switch_step, direction, magnitude)` combination and
   `EPISODES_PER_COMBO` episodes each:
   - a fresh `initial_goal_xyz` is drawn from a randomly-chosen
     `goal_region_vocabulary` region (`sample_region_goals`) so the achieved
     position at `switch_step` is a genuinely varied "wherever a prior
     command actually left the robot," not a fixed or near-reset point.
   - `relative_move.rollout_with_relative_move` -- the mechanism under test.
   - `midepisode_regoal.rollout_fresh_with_budget`, judged against the
     *same* `resolved_target_xyz` the relative-move rollout resolved to
     (the already-clipped target, per the plan's explicit choice -- clipping
     may have moved the target off the raw requested point, and both
     conditions must be judged against the same point to be a fair
     comparison), with the same post-switch step budget
     (`max_steps - switch_step`).

`MAGNITUDES` includes one small distance (5cm, matching the "move left 5cm"
example from `PHASES.md`), one larger in-bounds-stressing distance (15cm,
`MEASURED_GOAL_BOX`'s per-axis half-range), and one deliberately larger than
the box's full per-axis range (~30cm) so it clips regardless of the
starting position on the move axis -- confirmed empirically via the
`was_clipped` fraction printed per magnitude bucket below, not assumed.

Results are dumped to `runs/seed_<k>/results.json` for
`aggregate_and_report.py` to assemble into `report.md`/`evidence.md`;
per-combo summary lines are also printed so `runs/seed_<k>/stdout.log`
(written by `launch_seeds.sh`'s redirect) is independently readable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3 import SAC

from lang_goal_rl.goal_region_vocabulary import region_names, sample_region_goals
from lang_goal_rl.midepisode_regoal import rollout_fresh_with_budget
from lang_goal_rl.relative_move import DIRECTION_UNIT_VECTORS, rollout_with_relative_move

EXPERIMENT_DIR = Path(__file__).parent
CHECKPOINT_DIR = EXPERIMENT_DIR.parent / "01_uvfa_her_baseline" / "checkpoints"

ENV_ID = "FetchReach-v4"

MAX_STEPS = 50
"""FetchReach-v4's registered `max_episode_steps` -- the total budget every
relative-move rollout uses (pre- + post-switch combined)."""

SWITCH_STEPS: tuple[int, ...] = (10, 25, 40)
"""Early/mid/late switch points spanning the 50-step episode, per the plan's
explicit requirement that "current position" vary across when a prior
command actually left the robot -- not just a fresh-reset position."""

DIRECTIONS: tuple[str, ...] = tuple(DIRECTION_UNIT_VECTORS.keys())
"""All 6 production direction labels. Provisional pending stage 7's human
sign-off (see `relative_move.py`'s module docstring) -- irrelevant to this
experiment, which tests the relative-move *mechanism*, not whether "forward"
is the correct camera-frame label."""

MAGNITUDES: dict[str, float] = {
    "small_5cm": 0.05,
    "medium_15cm": 0.15,
    "clip_forcing_35cm": 0.35,
}
"""Three magnitude buckets: `small_5cm` matches PHASES.md's own "move left
5cm" example; `medium_15cm` equals `MEASURED_GOAL_BOX`'s per-axis half-range
(~0.1499m) -- large enough to stress the mechanism while normally landing
in-bounds from a typical achieved position; `clip_forcing_35cm` exceeds the
box's full per-axis range (~0.2998m on every axis), which guarantees
`was_clipped=True` on the move axis regardless of the starting position
within the box (current + 0.35 always overshoots axis_max when current is
already within [axis_min, axis_max] and the move is positive, and
symmetrically for a negative move) -- confirmed empirically per-seed below,
not just asserted algebraically."""

DEFAULT_EPISODES_PER_COMBO = 20
DEFAULT_SANITY_EPISODES = 50

RELATIVE_MOVE_BASE_SEED = 20_000
"""Base seed for relative-move eval episodes, offset clear of stage 5's
range (9000 + 500_000 offset) and stage 6's ranges so no reset seed is
silently reused across stages for a different purpose."""

INITIAL_GOAL_SEED_OFFSET = 700_000
"""Offset applied to an episode's seed before sampling `initial_goal_xyz`'s
region and point, so that draw never reuses the same seed as the episode's
own `env.reset(seed=...)` call -- mirrors stage 5's
`REGION_B_SEED_OFFSET` convention."""


def literal_goal_sanity_check(model: SAC, env: gym.Env, n_episodes: int) -> float:
    """Roll out the policy on the env's own randomly-sampled goal, no override.

    A same-directory local copy of stage 5's `run_regoal_eval.py` function of
    the same name (itself mirroring `train.py`'s `evaluate()`) -- this
    project's established convention is intra-directory reuse only, not
    reaching across experiment directories.

    Args:
        model: The trained SAC checkpoint under test.
        env: The FetchReach-v4 env instance to roll out on.
        n_episodes: Number of held-out episodes to evaluate.

    Returns:
        Mean success rate over `n_episodes`.
    """
    successes = []
    for episode in range(n_episodes):
        obs, _info = env.reset(seed=1000 + episode)
        terminated = truncated = False
        is_success = False
        while not (terminated or truncated):
            action, _state = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            is_success = bool(info.get("is_success", is_success))
        successes.append(is_success)
    return float(np.mean(successes))


def sample_initial_goal(seed: int) -> np.ndarray:
    """Sample a varied `initial_goal_xyz` from a randomly-chosen goal region.

    Args:
        seed: Deterministic seed for both the region choice and the
            within-region point draw.

    Returns:
        An xyz point, shape `(3,)`, so the achieved position at
        `switch_step` genuinely varies trial to trial rather than always
        starting near the same spot.
    """
    rng = np.random.default_rng(seed)
    region = rng.choice(region_names())
    return sample_region_goals(str(region), 1, seed=seed)[0]


def run_combo_eval(
    model: SAC,
    env: gym.Env,
    *,
    switch_step: int,
    direction: str,
    magnitude_label: str,
    distance_m: float,
    n_episodes: int,
    base_seed: int,
) -> dict[str, float | int | str | list[bool]]:
    """Run `n_episodes` relative-move + budget-matched-baseline pairs for one combo.

    Args:
        model: The trained SAC checkpoint under test (literal-xyz mode --
            `model.actor` is never touched by either rollout function).
        env: The FetchReach-v4 env instance to roll out on.
        switch_step: Step at which the relative move triggers.
        direction: A key into `DIRECTION_UNIT_VECTORS`.
        magnitude_label: Key into `MAGNITUDES`, echoed back for aggregation.
        distance_m: The magnitude in meters (`MAGNITUDES[magnitude_label]`).
        n_episodes: Number of episodes to sample and run for this combo.
        base_seed: First episode's seed; episode `i` uses `base_seed + i`.

    Returns:
        A dict with the combo's identifying fields, per-episode success
        lists for both conditions, the clip flag list, and the aggregated
        success/clip rates.
    """
    relative_move_successes: list[bool] = []
    baseline_successes: list[bool] = []
    was_clipped_flags: list[bool] = []
    for episode_index in range(n_episodes):
        episode_seed = base_seed + episode_index
        initial_goal_xyz = sample_initial_goal(episode_seed + INITIAL_GOAL_SEED_OFFSET)

        result = rollout_with_relative_move(
            model,
            env,
            initial_goal_xyz=initial_goal_xyz,
            switch_step=switch_step,
            direction=direction,
            distance_m=distance_m,
            max_steps=MAX_STEPS,
            base_seed=episode_seed,
        )
        relative_move_successes.append(result.success)
        was_clipped_flags.append(result.was_clipped)

        # Judged against `resolved_target_xyz` (the already-clipped target),
        # never the raw requested direction/distance -- clipping may have
        # moved the target, and the baseline must chase the same point the
        # relative-move rollout was actually judged against to be a fair
        # comparison. Same post-switch budget as the swap's remaining steps.
        baseline_success = rollout_fresh_with_budget(
            model,
            env,
            goal_xyz=result.resolved_target_xyz,
            max_steps=MAX_STEPS - switch_step,
            base_seed=episode_seed,
        )
        baseline_successes.append(baseline_success)

    return {
        "switch_step": switch_step,
        "direction": direction,
        "magnitude_label": magnitude_label,
        "distance_m": distance_m,
        "n_episodes": n_episodes,
        "relative_move_successes": relative_move_successes,
        "baseline_successes": baseline_successes,
        "was_clipped_flags": was_clipped_flags,
        "relative_move_success_rate": float(np.mean(relative_move_successes)),
        "baseline_success_rate": float(np.mean(baseline_successes)),
        "clip_rate": float(np.mean(was_clipped_flags)),
    }


def main() -> None:
    """Load one seed's checkpoint zero-shot and run the full stage-8 eval suite."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Model checkpoint seed (0-9)")
    parser.add_argument("--episodes-per-combo", type=int, default=DEFAULT_EPISODES_PER_COMBO)
    parser.add_argument("--sanity-episodes", type=int, default=DEFAULT_SANITY_EPISODES)
    args = parser.parse_args()

    gym.register_envs(gymnasium_robotics)
    env = gym.make(ENV_ID)

    checkpoint_path = CHECKPOINT_DIR / f"seed_{args.seed}.zip"
    model = SAC.load(checkpoint_path, env=env)
    print(f"loaded checkpoint from {checkpoint_path} (zero-shot, no new training, seed={args.seed})")

    sanity_success_rate = literal_goal_sanity_check(model, env, args.sanity_episodes)
    print(
        f"sanity_check_success_rate={sanity_success_rate:.3f} over {args.sanity_episodes} episodes "
        "(literal control, full 50-step episode, no relative move)"
    )

    combo_results = []
    combo_index = 0
    for switch_step in SWITCH_STEPS:
        for direction in DIRECTIONS:
            for magnitude_label, distance_m in MAGNITUDES.items():
                base_seed = RELATIVE_MOVE_BASE_SEED + combo_index * 1000
                result = run_combo_eval(
                    model,
                    env,
                    switch_step=switch_step,
                    direction=direction,
                    magnitude_label=magnitude_label,
                    distance_m=distance_m,
                    n_episodes=args.episodes_per_combo,
                    base_seed=base_seed,
                )
                combo_results.append(result)
                print(
                    f"switch_step={switch_step} direction={direction!r} "
                    f"magnitude={magnitude_label} ({distance_m}m) "
                    f"relative_move_success_rate={result['relative_move_success_rate']:.3f} "
                    f"baseline_success_rate={result['baseline_success_rate']:.3f} "
                    f"clip_rate={result['clip_rate']:.3f} over {args.episodes_per_combo} episodes"
                )
                combo_index += 1

    output = {
        "model_seed": args.seed,
        "sanity_check_success_rate": sanity_success_rate,
        "sanity_check_episodes": args.sanity_episodes,
        "combo_results": combo_results,
    }
    results_path = EXPERIMENT_DIR / "runs" / f"seed_{args.seed}" / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(output, indent=2))
    print(f"results_saved={results_path}")

    env.close()


if __name__ == "__main__":
    main()
