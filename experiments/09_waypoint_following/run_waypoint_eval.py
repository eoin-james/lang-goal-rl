"""Stage 9 proof gate: does the N=2 goal-switch mechanism compound error over longer chains?

Uses `waypoint_following.rollout_with_waypoints` -- already proven equivalent
to `midepisode_regoal.rollout_with_goal_switch` at N=2 by this project's own
regression test suite (`tests/lang_goal_rl/test_waypoint_following.py`'s
`TestEquivalenceWithMidepisodeRegoal`, which passes; see
`report.md`/`evidence.md` for the run that confirms it). That equivalence is
settled. What this script measures is new: whether the mechanism still holds
up at N=3 and N=5, and whether it's robust to a tight per-leg step budget or
only works when the budget is generous.

Parameterized by `--seed` to select which of `experiments/
01_uvfa_her_baseline/checkpoints/seed_<k>.zip` to load, zero-shot. The
original run used only `seed_0` and applied "tiered for speed" along the
episode-count axis instead of across model seeds -- reviewer feedback
(see `evidence.md`'s Reviewer verdict) confirmed that departure from
CONTRACTS.md's multi-seed convention left the "no compounding
degradation" claim unable to distinguish "this mechanism is robust" from
"this one already-oracle-solvable checkpoint has no room to fail." This
script now runs identically across every healthy checkpoint (seeds
0,1,3,4,5,6,8,9 -- excluding 2,7, the documented SAC deterministic-eval
collapse seeds, ROADMAP.md Known risks) at the already-validated
50-episodes/condition count; the original seed_0 tier1/final runs stay
under `runs/` as-is, new seeds land under `runs/seed_<k>/`.

Two waypoint-sequence kinds, both precomputed as absolute xyz *before* the
episode starts (the stage-9 v1 scope limit -- see the plan's "Open items
already decided"; true live-relative chaining, where a leg's target is
computed from the robot's actual position at the moment that leg starts
rather than from the previous leg's precomputed target, is explicit future
work, not attempted here):

- "literal": each leg sampled from a distinct
  `goal_region_vocabulary` region, so every leg in the chain is a
  meaningfully different target from every other leg.
- "relative": leg 0 is a literal xyz point (sampled from the "center"
  region); every subsequent leg is `relative_move.compute_relative_goal`
  applied to the *previous leg's own target* (never the robot's live
  achieved position at that point in the episode -- that distinction is
  exactly the documented v1/v2 scope split above).

For every (sequence kind, chain length, per-leg budget) condition, each
episode runs the chain once (`rollout_with_waypoints`) and, for every leg in
that same chain, a budget-matched fresh baseline
(`midepisode_regoal.rollout_fresh_with_budget`) targeting that leg's exact
goal with that leg's exact step budget, from a fresh reset using the same
episode seed -- the same "same start, same budget, only difference is
whether a different goal was pursued first" comparison stage 5 already
established, generalized to every leg position instead of just the one
post-switch phase. Per-leg success rates (not just whether the final
waypoint was reached) are what reveal compounding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import numpy.typing as npt
from stable_baselines3 import SAC

from lang_goal_rl.goal_region_vocabulary import region_names, sample_region_goals
from lang_goal_rl.midepisode_regoal import rollout_fresh_with_budget
from lang_goal_rl.relative_move import DIRECTION_UNIT_VECTORS, compute_relative_goal
from lang_goal_rl.waypoint_following import rollout_with_waypoints

EXPERIMENT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = EXPERIMENT_DIR.parent / "01_uvfa_her_baseline" / "checkpoints"
"""Directory holding this project's stage-1 literal-xyz SAC+HER checkpoints,
one per model seed -- this whole experiment reuses them zero-shot, per the
task brief ("same checkpoint stage 8 uses, for consistency across Phase
2a"). `main()`'s `--seed` argument selects `seed_<k>.zip` from here; never
pass 2 or 7, the documented SAC deterministic-eval collapse seeds
(ROADMAP.md Known risks)."""

ENV_ID = "FetchReach-v4"

TIGHT_BUDGET = 9
GENEROUS_BUDGET = 18
BUDGETS: dict[str, int] = {"tight": TIGHT_BUDGET, "generous": GENEROUS_BUDGET}
"""Per-leg step budgets. "tight" sits in the task brief's 8-10 range,
"generous" in its 15-20 range."""

CHAIN_LENGTHS: tuple[int, ...] = (2, 3, 5)

SEQUENCE_KINDS: tuple[str, ...] = ("literal", "relative")

MAX_TOTAL_BUDGET = max(CHAIN_LENGTHS) * max(BUDGETS.values())
"""5 legs x 18 steps/leg = 90 -- the longest any single chain rollout in this
experiment needs. FetchReach-v4's default `max_episode_steps` is 50, well
short of this, so the env below is created with an explicit
`max_episode_steps` override big enough to cover every condition without
`rollout_with_waypoints`/`rollout_fresh_with_budget`'s own
`_ensure_within_env_step_limit` guard rejecting the run. This only changes
when the env's `TimeLimit` wrapper would truncate -- it does not change
step-by-step MuJoCo dynamics, and the policy's observation has no
episode-step feature, so behavior on steps 1-50 is identical to the
default-length env; the checkpoint was never trained to run this long, but
nothing about its architecture assumes a horizon, either -- this is
measured, not assumed, via the literal-goal sanity check below."""
ENV_MAX_EPISODE_STEPS = 100
"""Round number comfortably above `MAX_TOTAL_BUDGET`."""

LITERAL_REGION_SEED_OFFSET = 700_000
LITERAL_POINT_SEED_OFFSET = 800_000
RELATIVE_LEG0_SEED_OFFSET = 900_000
RELATIVE_DIRECTION_SEED_OFFSET = 950_000
RELATIVE_MOVE_DISTANCE_M = 0.15
"""Comparable to `MEASURED_GOAL_BOX`'s per-axis half-range (~0.15m) -- large
enough that each relative leg is a meaningfully different target from the
one before it, small enough that `compute_relative_goal`'s clip-to-box
doesn't collapse most legs onto the box's boundary."""

CONDITION_BASE_SEED_STRIDE = 10_000
"""Each (sequence_kind, chain_length, budget_name) condition gets its own
block of episode seeds, `CONDITION_BASE_SEED_STRIDE` apart, so no two
conditions' episodes ever draw the same (region choice / relative-move
direction / env reset) randomness by accident."""

SANITY_COLLAPSE_THRESHOLD = 0.8
"""Below this, the checkpoint's literal-goal sanity check is flagged as a
possible SAC deterministic-eval collapse (ROADMAP.md Known risks) rather
than trusted at face value."""


def literal_goal_sanity_check(model: SAC, env: gym.Env, n_episodes: int) -> float:
    """Roll out the policy on the env's own randomly-sampled goal, no override.

    Local copy of stage 5/6's own sanity-check helper (same `1000 + episode`
    held-out seed convention, same deterministic rollout loop) -- this
    project's convention is same-directory local copies rather than
    cross-experiment-directory imports (see `run_regoal_eval.py`'s docstring
    for the same rationale). Confirms the reused checkpoint still performs
    the plain literal-goal task, on the *default*-length env, before
    trusting anything measured on the extended-length env below.

    Args:
        model: The trained SAC checkpoint under test.
        env: A FetchReach-v4 env instance at its default 50-step episode
            length.
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


def generate_literal_waypoints(
    n_legs: int, seed: int
) -> tuple[list[npt.NDArray[np.floating]], list[str]]:
    """Sample `n_legs` xyz waypoints from `n_legs` distinct goal regions.

    Args:
        n_legs: Number of waypoints/legs to generate. Must be `<= 7` (the
            number of regions `goal_region_vocabulary` defines).
        seed: Deterministic seed for both the region choice and the
            within-region point draws.

    Returns:
        `(waypoints, region_names_used)`, both length `n_legs`, in leg
        order. Every region name is distinct, so every leg targets a
        meaningfully different part of the goal box from every other leg.
    """
    names = region_names()
    rng = np.random.default_rng(seed + LITERAL_REGION_SEED_OFFSET)
    chosen_regions = rng.choice(np.array(names), size=n_legs, replace=False)
    waypoints = [
        sample_region_goals(str(region), 1, seed=seed + LITERAL_POINT_SEED_OFFSET + leg_index)[0]
        for leg_index, region in enumerate(chosen_regions)
    ]
    return waypoints, [str(region) for region in chosen_regions]


def generate_relative_waypoints(
    n_legs: int, seed: int, distance_m: float = RELATIVE_MOVE_DISTANCE_M
) -> tuple[list[npt.NDArray[np.floating]], list[str]]:
    """Chain `relative_move.compute_relative_goal` off each leg's own precomputed target.

    Leg 0 is a literal xyz point sampled from the "center" region. Every
    subsequent leg is computed as `compute_relative_goal(waypoints[-1],
    direction, distance_m)` -- relative to the *previous leg's own target*,
    never the robot's actual live position when that leg starts. This is
    the stage-9 v1 scope limit stated in the plan: legs are precomputed as
    absolute xyz before the episode begins; true live-relative chaining is
    explicit future work.

    Args:
        n_legs: Number of waypoints/legs to generate.
        seed: Deterministic seed for leg 0's draw and every direction choice.
        distance_m: Signed distance moved per relative leg.

    Returns:
        `(waypoints, directions_used)` -- `waypoints` has length `n_legs`,
        `directions_used` has length `n_legs - 1` (one per relative leg;
        leg 0 has no direction, it's the literal starting point).
    """
    leg0 = sample_region_goals("center", 1, seed=seed + RELATIVE_LEG0_SEED_OFFSET)[0]
    waypoints = [leg0]
    direction_names = list(DIRECTION_UNIT_VECTORS.keys())
    rng = np.random.default_rng(seed + RELATIVE_DIRECTION_SEED_OFFSET)
    directions_used = []
    for _ in range(n_legs - 1):
        direction = str(rng.choice(np.array(direction_names)))
        waypoints.append(compute_relative_goal(waypoints[-1], direction, distance_m))
        directions_used.append(direction)
    return waypoints, directions_used


def run_condition(
    model: SAC,
    env: gym.Env,
    *,
    sequence_kind: str,
    chain_len: int,
    budget_name: str,
    n_episodes: int,
    base_seed: int,
) -> dict:
    """Run one (sequence_kind, chain_len, budget) condition for `n_episodes` episodes.

    Every episode runs the chain once and, for every leg in that chain, a
    budget-matched fresh baseline targeting that exact leg's goal with that
    leg's exact per-leg budget, seeded identically to the chain so both
    conditions share the same random start position -- the only difference
    is whether a different goal was pursued first.

    Args:
        model: The trained SAC checkpoint under test.
        env: A FetchReach-v4 env instance created with `max_episode_steps`
            covering this condition's full `chain_len * budget`.
        sequence_kind: "literal" or "relative" -- see
            `generate_literal_waypoints`/`generate_relative_waypoints`.
        chain_len: Number of waypoints/legs.
        budget_name: "tight" or "generous" -- key into `BUDGETS`.
        n_episodes: Number of episodes to run for this condition.
        base_seed: First episode's seed; episode `i` uses `base_seed + i`,
            for both the chain rollout and every one of its legs' baselines.

    Returns:
        A dict with per-leg chain/baseline success-rate lists (length
        `chain_len`, leg-index order), the whole-chain `all_succeeded` rate,
        the raw per-episode per-leg chain-success bits (for correlation
        analysis -- do failures at different leg positions land on the same
        episode, or independent ones?), and enough metadata to identify the
        condition.
    """
    budget = BUDGETS[budget_name]
    per_leg_chain_successes: list[list[bool]] = [[] for _ in range(chain_len)]
    per_leg_baseline_successes: list[list[bool]] = [[] for _ in range(chain_len)]
    all_succeeded_flags: list[bool] = []
    per_episode_chain_bits: list[list[bool]] = []

    for episode_index in range(n_episodes):
        episode_seed = base_seed + episode_index
        if sequence_kind == "literal":
            waypoints, _meta = generate_literal_waypoints(chain_len, seed=episode_seed)
        else:
            waypoints, _meta = generate_relative_waypoints(chain_len, seed=episode_seed)

        chain_result = rollout_with_waypoints(
            model, env, waypoints=waypoints, steps_per_leg=budget, base_seed=episode_seed
        )
        all_succeeded_flags.append(chain_result.all_succeeded)
        per_episode_chain_bits.append(list(chain_result.per_waypoint_success))

        for leg_index in range(chain_len):
            per_leg_chain_successes[leg_index].append(chain_result.per_waypoint_success[leg_index])
            baseline_success = rollout_fresh_with_budget(
                model,
                env,
                goal_xyz=waypoints[leg_index],
                max_steps=budget,
                base_seed=episode_seed,
            )
            per_leg_baseline_successes[leg_index].append(baseline_success)

    return {
        "sequence_kind": sequence_kind,
        "chain_len": chain_len,
        "budget_name": budget_name,
        "budget": budget,
        "n_episodes": n_episodes,
        "per_leg_chain_success_rate": [float(np.mean(x)) for x in per_leg_chain_successes],
        "per_leg_baseline_success_rate": [float(np.mean(x)) for x in per_leg_baseline_successes],
        "all_succeeded_rate": float(np.mean(all_succeeded_flags)),
        "per_episode_chain_bits": per_episode_chain_bits,
    }


def main() -> None:
    """Run every (sequence_kind, chain_len, budget) condition and dump results to JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Model checkpoint seed (seed_<k>.zip); never 2 or 7")
    parser.add_argument("--episodes", type=int, required=True, help="Episodes per condition")
    parser.add_argument("--sanity-episodes", type=int, default=50)
    parser.add_argument("--tag", type=str, required=True, help="Tier tag, e.g. 'tier1' or 'final'")
    args = parser.parse_args()

    checkpoint_path = CHECKPOINT_DIR / f"seed_{args.seed}.zip"

    gym.register_envs(gymnasium_robotics)

    sanity_env = gym.make(ENV_ID)
    model = SAC.load(checkpoint_path, env=sanity_env)
    print(f"loaded checkpoint from {checkpoint_path} (zero-shot, no new training, seed={args.seed})")

    sanity_rate = literal_goal_sanity_check(model, sanity_env, args.sanity_episodes)
    print(
        f"sanity_check_success_rate={sanity_rate:.3f} over {args.sanity_episodes} episodes "
        "(literal control, default 50-step episode, no waypoint chain)"
    )
    sanity_env.close()
    if sanity_rate < SANITY_COLLAPSE_THRESHOLD:
        print(
            f"WARNING: sanity_check_success_rate={sanity_rate:.3f} is below "
            f"{SANITY_COLLAPSE_THRESHOLD} -- resembles the known SAC "
            "deterministic-eval collapse signature (ROADMAP.md Known risks)."
        )

    env = gym.make(ENV_ID, max_episode_steps=ENV_MAX_EPISODE_STEPS)
    model = SAC.load(checkpoint_path, env=env)

    condition_results = []
    condition_index = 0
    for sequence_kind in SEQUENCE_KINDS:
        for chain_len in CHAIN_LENGTHS:
            for budget_name in BUDGETS:
                base_seed = 20_000 + condition_index * CONDITION_BASE_SEED_STRIDE
                result = run_condition(
                    model,
                    env,
                    sequence_kind=sequence_kind,
                    chain_len=chain_len,
                    budget_name=budget_name,
                    n_episodes=args.episodes,
                    base_seed=base_seed,
                )
                condition_results.append(result)
                per_leg_chain = ", ".join(f"{rate:.3f}" for rate in result["per_leg_chain_success_rate"])
                per_leg_baseline = ", ".join(
                    f"{rate:.3f}" for rate in result["per_leg_baseline_success_rate"]
                )
                print(
                    f"kind={sequence_kind} chain_len={chain_len} budget={budget_name}({result['budget']}) "
                    f"chain_per_leg=[{per_leg_chain}] baseline_per_leg=[{per_leg_baseline}] "
                    f"all_succeeded_rate={result['all_succeeded_rate']:.3f} over {args.episodes} episodes"
                )
                condition_index += 1

    env.close()

    output = {
        "tag": args.tag,
        "model_seed": args.seed,
        "checkpoint": str(checkpoint_path),
        "sanity_check_success_rate": sanity_rate,
        "sanity_check_episodes": args.sanity_episodes,
        "episodes_per_condition": args.episodes,
        "conditions": condition_results,
    }
    results_path = EXPERIMENT_DIR / "runs" / f"seed_{args.seed}" / f"{args.tag}_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(output, indent=2))
    print(f"results_saved={results_path}")


if __name__ == "__main__":
    main()
