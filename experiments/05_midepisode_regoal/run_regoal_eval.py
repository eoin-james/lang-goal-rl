"""Stage 5 proof gate: zero-shot mid-episode goal-swap vs. budget-matched fresh baseline.

Per model seed (a stage-1 checkpoint reused zero-shot, no new RL training --
see `experiments/01_uvfa_her_baseline/provision_checkpoints.py` for how those
checkpoints were provisioned), this script runs:

1. A literal-goal sanity check: `train.py`'s own `evaluate()` (full
   50-step episode, no swap, targeting the env's own randomly-sampled goal)
   reused verbatim -- confirms the retrained checkpoint actually learned the
   task before trusting anything else, exactly like every prior stage's
   report did.
2. For each `switch_step` in `SWITCH_STEPS` and `EPISODES_PER_COMBO`
   episodes: `goal_a`/`goal_b` sampled from two distinct
   `goal_region_vocabulary` regions (never the same region twice, so every
   swap is a genuinely different retarget), then three rollouts per episode:
   - swap: `midepisode_regoal.rollout_with_goal_switch` (the mechanism under
     test)
   - budget-matched baseline: `rollout_fresh_with_budget(goal_xyz=goal_b,
     max_steps=MAX_STEPS - switch_step)` -- the fair comparison, same final
     goal and same remaining budget as the swap's post-switch phase
   - full-budget reference: `rollout_fresh_with_budget(goal_xyz=goal_b,
     max_steps=MAX_STEPS)` -- not the primary comparison, just the ceiling
     this policy could reach on goal_b given the whole episode

Results are dumped to `runs/seed_<k>/results.json` for
`aggregate_and_report.py` to assemble into `report.md`; per-switch_step
summary lines are also printed so `runs/seed_<k>/stdout.log` (written by
`launch_seeds.sh`'s redirect) is independently readable.
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
from lang_goal_rl.midepisode_regoal import (
    rollout_fresh_with_budget,
    rollout_with_goal_switch,
)

EXPERIMENT_DIR = Path(__file__).parent
CHECKPOINT_DIR = EXPERIMENT_DIR.parent / "01_uvfa_her_baseline" / "checkpoints"

ENV_ID = "FetchReach-v4"

MAX_STEPS = 50
"""FetchReach-v4's registered `max_episode_steps` -- the total budget every
swap rollout and full-budget reference uses."""

SWITCH_STEPS: tuple[int, ...] = (10, 20, 30, 40)
"""Early/mid/late switch points spanning a meaningful range of the 50-step episode."""

DEFAULT_EPISODES_PER_COMBO = 40
DEFAULT_SANITY_EPISODES = 50

REGOAL_BASE_SEED = 9000
"""Base seed for the swap-eval episodes, offset well clear of stage 1's
training seeds (0-9), literal eval seeds (1000+), and stage 3's language
eval seeds (5000+) so no reset seed is silently reused across stages for a
different purpose."""

REGION_B_SEED_OFFSET = 500_000
"""Offset applied to region_b's sample_region_goals seed so it never draws
from the same rejection-sampling seed as region_a's draw for the same episode."""


def literal_goal_sanity_check(model: SAC, env: gym.Env, n_episodes: int) -> float:
    """Roll out the policy on the env's own randomly-sampled goal, no override.

    Mirrors `experiments/01_uvfa_her_baseline/train.py`'s `evaluate()` exactly
    (same held-out seed convention `1000 + episode`, same deterministic
    rollout loop, same `is_success` bookkeeping) -- confirms the reused
    checkpoint still performs the plain literal-goal task before trusting any
    of the swap results below. Not imported directly from `train.py` because
    that module lives in a different experiment directory; this repo's
    existing cross-file reuse (e.g. `experiments/03_language_goal_projection/
    eval_fixed_projection.py` importing from `train.py`) is intra-directory
    only, so a same-directory local copy matches the established convention
    rather than reaching across experiment directories.

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


def sample_goal_pair(seed: int) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Sample two goals from two distinct, seed-determined regions.

    Args:
        seed: Deterministic seed for both the region choice and the
            within-region point draw.

    Returns:
        `(goal_a, goal_b, region_a_name, region_b_name)` -- `goal_a` and
        `goal_b` are guaranteed to come from different regions, so the swap
        is never a same-region no-op.
    """
    names = region_names()
    rng = np.random.default_rng(seed)
    region_a, region_b = rng.choice(names, size=2, replace=False)
    goal_a = sample_region_goals(str(region_a), 1, seed=seed)[0]
    goal_b = sample_region_goals(str(region_b), 1, seed=seed + REGION_B_SEED_OFFSET)[0]
    return goal_a, goal_b, str(region_a), str(region_b)


def run_switch_step_eval(
    model: SAC,
    env: gym.Env,
    switch_step: int,
    n_episodes: int,
    base_seed: int,
) -> dict[str, float | int]:
    """Run the swap / budget-matched-baseline / full-budget-reference triad for one switch_step.

    Args:
        model: The trained SAC checkpoint under test (literal-xyz mode).
        env: The FetchReach-v4 env instance to roll out on.
        switch_step: Step at which the swap condition retargets.
        n_episodes: Number of (goal_a, goal_b) episodes to sample and run.
        base_seed: First episode's seed; episode `i` uses `base_seed + i`.

    Returns:
        A dict with `switch_step`, `n_episodes`, and the three success rates.
    """
    swap_successes = []
    baseline_successes = []
    fullbudget_successes = []
    for episode_index in range(n_episodes):
        episode_seed = base_seed + episode_index
        goal_a, goal_b, _region_a, _region_b = sample_goal_pair(episode_seed)

        swap_result = rollout_with_goal_switch(
            model,
            env,
            initial_goal_xyz=goal_a,
            switch_step=switch_step,
            new_goal_xyz=goal_b,
            max_steps=MAX_STEPS,
            base_seed=episode_seed,
        )
        swap_successes.append(swap_result.success)

        baseline_success = rollout_fresh_with_budget(
            model,
            env,
            goal_xyz=goal_b,
            max_steps=MAX_STEPS - switch_step,
            base_seed=episode_seed,
        )
        baseline_successes.append(baseline_success)

        fullbudget_success = rollout_fresh_with_budget(
            model,
            env,
            goal_xyz=goal_b,
            max_steps=MAX_STEPS,
            base_seed=episode_seed,
        )
        fullbudget_successes.append(fullbudget_success)

    return {
        "switch_step": switch_step,
        "n_episodes": n_episodes,
        "swap_success_rate": float(np.mean(swap_successes)),
        "budget_matched_baseline_success_rate": float(np.mean(baseline_successes)),
        "full_budget_reference_success_rate": float(np.mean(fullbudget_successes)),
    }


def main() -> None:
    """Load one seed's checkpoint zero-shot and run the full stage-5 eval suite."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, required=True, help="Model checkpoint seed (0/1/2)"
    )
    parser.add_argument(
        "--episodes-per-combo", type=int, default=DEFAULT_EPISODES_PER_COMBO
    )
    parser.add_argument("--sanity-episodes", type=int, default=DEFAULT_SANITY_EPISODES)
    args = parser.parse_args()

    gym.register_envs(gymnasium_robotics)
    env = gym.make(ENV_ID)

    checkpoint_path = CHECKPOINT_DIR / f"seed_{args.seed}.zip"
    model = SAC.load(checkpoint_path, env=env)
    print(
        f"loaded checkpoint from {checkpoint_path} (zero-shot, no new training, seed={args.seed})"
    )

    sanity_success_rate = literal_goal_sanity_check(model, env, args.sanity_episodes)
    print(
        f"sanity_check_success_rate={sanity_success_rate:.3f} over {args.sanity_episodes} episodes "
        f"(literal control, full {MAX_STEPS}-step episode, no swap)"
    )

    switch_step_results = []
    for switch_step_index, switch_step in enumerate(SWITCH_STEPS):
        base_seed = REGOAL_BASE_SEED + switch_step_index * 1000
        result = run_switch_step_eval(
            model, env, switch_step, args.episodes_per_combo, base_seed
        )
        switch_step_results.append(result)
        print(
            f"switch_step={switch_step} "
            f"swap_success_rate={result['swap_success_rate']:.3f} "
            f"budget_matched_baseline_success_rate={result['budget_matched_baseline_success_rate']:.3f} "
            f"full_budget_reference_success_rate={result['full_budget_reference_success_rate']:.3f} "
            f"over {args.episodes_per_combo} episodes"
        )

    output = {
        "model_seed": args.seed,
        "sanity_check_success_rate": sanity_success_rate,
        "sanity_check_episodes": args.sanity_episodes,
        "switch_step_results": switch_step_results,
    }
    results_path = EXPERIMENT_DIR / "runs" / f"seed_{args.seed}" / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(output, indent=2))
    print(f"results_saved={results_path}")

    env.close()


if __name__ == "__main__":
    main()
