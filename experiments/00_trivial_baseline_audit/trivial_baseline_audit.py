"""Audit whether FetchReach-v4's reset geometry inflates this project's reported success rates.

Not a ROADMAP stage — a standalone, reproducible credibility check. A demo GIF
review noticed one episode that started unusually close to its goal, raising the
question of whether the whole task might be trivially easy (i.e. whether the
project's 0.548-1.000 success rates across stages 1-6 could be a geometry
artifact rather than learned behavior). This script answers that directly by
measuring two things, using the exact same `info["is_success"]` criterion and
50-step episode length every other stage in this project uses:

1. How far a freshly-reset episode's `achieved_goal` starts from `desired_goal`,
   before any action is taken (the reset-to-goal distance distribution).
2. How often policies with zero learned behavior — no-op, random action,
   and a straight-line oracle — succeed under that same criterion.

See `experiments/00_trivial_baseline_audit/report.md` for the write-up this
script's output backs.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np

ENV_ID = "FetchReach-v4"
MAX_EPISODE_STEPS = 50
"""FetchReach-v4's episode length, matching every other stage in this project."""

# Seed blocks kept far apart so the three policy runs and the reset-distance
# sample never draw from overlapping reset seeds.
RESET_DISTANCE_SEED_OFFSET = 0
NOOP_SEED_OFFSET = 100_000
RANDOM_SEED_OFFSET = 200_000
ORACLE_SEED_OFFSET = 300_000

ORACLE_GAIN = 10.0
"""Scales the (desired - achieved) position delta before clipping to the
action space. Chosen empirically: FetchReach's action space is a normalized
position-control delta, and this gain reaches the goal in a handful of steps
without saturating early (see the plain straight-line-to-goal check this
script runs)."""


def build_env() -> gym.Env:
    """Register gymnasium-robotics and construct the FetchReach-v4 env."""
    gym.register_envs(gymnasium_robotics)
    return gym.make(ENV_ID)


def measure_reset_distances(env: gym.Env, n_samples: int, seed: int) -> np.ndarray:
    """Reset the env n_samples times and record the pre-action goal distance each time.

    Args:
        env: A FetchReach-v4 environment instance.
        n_samples: Number of resets to sample.
        seed: Base seed; reset `i` uses `seed + i`, so the sample is
            deterministic and reproducible.

    Returns:
        Array of shape `(n_samples,)`: the Euclidean distance between
        `desired_goal` and `achieved_goal` immediately after each reset,
        before any action is taken.
    """
    distances = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        obs, _info = env.reset(seed=seed + i)
        distances[i] = np.linalg.norm(obs["desired_goal"] - obs["achieved_goal"])
    return distances


def run_episodes(
    env: gym.Env,
    policy_fn: Callable[[dict[str, np.ndarray]], np.ndarray],
    n_episodes: int,
    seed: int,
) -> tuple[list[bool], list[int | None]]:
    """Roll out a policy for n_episodes and record success + steps-to-first-success.

    Uses `info["is_success"]` as the success criterion and lets the env's own
    `TimeLimit` wrapper enforce the 50-step episode length — the same
    mechanism every other stage in this project relies on.

    Args:
        env: A FetchReach-v4 environment instance.
        policy_fn: Maps an observation dict to an action array.
        n_episodes: Number of episodes to run.
        seed: Base seed; episode `i` resets with `seed + i`.

    Returns:
        Tuple of `(successes, steps_to_success)`: parallel lists, one entry
        per episode. `steps_to_success[i]` is `None` if episode `i` never
        succeeded.
    """
    successes: list[bool] = []
    steps_to_success: list[int | None] = []
    for i in range(n_episodes):
        obs, _info = env.reset(seed=seed + i)
        terminated = truncated = False
        is_success = False
        first_success_step: int | None = None
        step = 0
        while not (terminated or truncated):
            step += 1
            action = policy_fn(obs)
            obs, _reward, terminated, truncated, info = env.step(action)
            is_success = bool(info.get("is_success", is_success))
            if is_success and first_success_step is None:
                first_success_step = step
        successes.append(is_success)
        steps_to_success.append(first_success_step)
    return successes, steps_to_success


def noop_policy(obs: dict[str, np.ndarray], action_dim: int) -> np.ndarray:
    """Return the all-zero action, regardless of observation."""
    return np.zeros(action_dim, dtype=np.float32)


def oracle_policy(obs: dict[str, np.ndarray], gain: float = ORACLE_GAIN) -> np.ndarray:
    """Move the gripper directly toward the goal, ignoring the gripper-open/close dim.

    Args:
        obs: FetchReach observation dict with `achieved_goal` and `desired_goal`.
        gain: Scale applied to the position delta before clipping to [-1, 1].

    Returns:
        A 4-dim action: the gain-scaled, clipped straight-line direction to the
        goal in the first 3 dims, zero for the (irrelevant, no-gripper-task) 4th.
    """
    delta = obs["desired_goal"] - obs["achieved_goal"]
    action = np.zeros(4, dtype=np.float32)
    action[:3] = np.clip(delta * gain, -1.0, 1.0)
    return action


def summarize_distances(
    distances: np.ndarray, success_threshold: float
) -> dict[str, float]:
    """Compute the percentile summary of a reset-distance sample.

    Args:
        distances: Array of reset-to-goal distances.
        success_threshold: FetchReach's success radius, for the
            fraction-already-within-threshold statistic.

    Returns:
        Dict of summary statistics in meters (except the fraction, in [0, 1]).
    """
    return {
        "n_samples": int(distances.size),
        "min": float(np.min(distances)),
        "p10": float(np.percentile(distances, 10)),
        "p25": float(np.percentile(distances, 25)),
        "median": float(np.median(distances)),
        "p75": float(np.percentile(distances, 75)),
        "p90": float(np.percentile(distances, 90)),
        "max": float(np.max(distances)),
        "fraction_within_success_threshold": float(
            np.mean(distances <= success_threshold)
        ),
        "success_threshold": float(success_threshold),
    }


def summarize_policy_run(
    successes: list[bool], steps_to_success: list[int | None]
) -> dict[str, float | None]:
    """Compute the success rate and median steps-to-success for one policy's run.

    Args:
        successes: Per-episode success booleans.
        steps_to_success: Per-episode steps-to-first-success (`None` if never).

    Returns:
        Dict with `success_rate`, `n_episodes`, `n_successes`, and
        `median_steps_to_success` (`None` if no episode ever succeeded).
    """
    reached = [s for s in steps_to_success if s is not None]
    return {
        "n_episodes": len(successes),
        "n_successes": int(sum(successes)),
        "success_rate": float(np.mean(successes)),
        "median_steps_to_success": float(np.median(reached)) if reached else None,
    }


def plot_reset_distance_histogram(
    distances: np.ndarray, success_threshold: float, out_path: Path
) -> Path:
    """Histogram the reset-to-goal distances with the success threshold marked.

    A one-off chart (not routed through `lang_goal_rl.reporting`) — none of
    that module's existing plot functions fit a single-sample histogram with
    a threshold line, and this chart isn't reused by any other stage.

    Args:
        distances: Array of reset-to-goal distances, in meters.
        success_threshold: FetchReach's success radius, in meters.
        out_path: Destination PNG path; parent directories are created if
            missing.

    Returns:
        The path the PNG was written to (same as `out_path`).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(distances, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.axvline(
        success_threshold,
        color="#C44E52",
        linestyle="--",
        linewidth=2,
        label=f"success threshold ({success_threshold:.2f} m)",
    )
    frac_within = float(np.mean(distances <= success_threshold))
    ax.set_xlabel("reset-to-goal distance (m)")
    ax.set_ylabel(f"count (of {distances.size} resets)")
    ax.set_title(
        "FetchReach-v4 reset-to-goal distance distribution\n"
        f"only {frac_within:.1%} of resets start inside the success threshold"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def print_summary(
    distance_summary: dict[str, float],
    policy_summaries: dict[str, dict[str, float | None]],
) -> None:
    """Print a human-readable summary of every measurement to stdout."""
    print("=== Reset-to-goal distance distribution ===")
    print(f"n_samples={distance_summary['n_samples']}")
    print(f"success_threshold={distance_summary['success_threshold']:.4f} m")
    for key in ("min", "p10", "p25", "median", "p75", "p90", "max"):
        print(f"{key}={distance_summary[key]:.4f} m")
    print(
        "fraction_within_success_threshold="
        f"{distance_summary['fraction_within_success_threshold']:.4f}"
    )

    print()
    print("=== Trivial-policy success rates ===")
    for name, summary in policy_summaries.items():
        median_steps = summary["median_steps_to_success"]
        median_steps_text = (
            f"{median_steps:.1f}"
            if median_steps is not None
            else "n/a (never succeeded)"
        )
        print(
            f"{name}: success_rate={summary['success_rate']:.4f} "
            f"({summary['n_successes']}/{summary['n_episodes']} episodes), "
            f"median_steps_to_success={median_steps_text}"
        )


def main() -> None:
    """Run the full audit: reset-distance distribution + no-op/random/oracle success rates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed; all seed blocks derive from this.",
    )
    parser.add_argument("--n-reset-samples", type=int, default=500)
    parser.add_argument("--n-episodes", type=int, default=500)
    args = parser.parse_args()

    experiment_dir = Path(__file__).parent
    runs_dir = experiment_dir / "runs"
    charts_dir = experiment_dir / "charts"
    runs_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    env = build_env()
    assert env.spec is not None
    assert env.spec.max_episode_steps == MAX_EPISODE_STEPS, (
        f"expected FetchReach-v4 episode length {MAX_EPISODE_STEPS}, got {env.spec.max_episode_steps}"
    )
    success_threshold = float(env.unwrapped.distance_threshold)
    assert env.action_space.shape is not None
    action_dim = int(env.action_space.shape[0])

    distances = measure_reset_distances(
        env, args.n_reset_samples, seed=args.seed + RESET_DISTANCE_SEED_OFFSET
    )
    distance_summary = summarize_distances(distances, success_threshold)

    env.action_space.seed(args.seed + RANDOM_SEED_OFFSET)
    policies: dict[str, Callable[[dict[str, np.ndarray]], np.ndarray]] = {
        "no-op": lambda obs: noop_policy(obs, action_dim),
        "random": lambda obs: env.action_space.sample(),
        "oracle": oracle_policy,
    }
    seed_offsets = {
        "no-op": NOOP_SEED_OFFSET,
        "random": RANDOM_SEED_OFFSET,
        "oracle": ORACLE_SEED_OFFSET,
    }

    policy_summaries: dict[str, dict[str, float | None]] = {}
    for name, policy_fn in policies.items():
        successes, steps_to_success = run_episodes(
            env, policy_fn, args.n_episodes, seed=args.seed + seed_offsets[name]
        )
        policy_summaries[name] = summarize_policy_run(successes, steps_to_success)

    env.close()

    print_summary(distance_summary, policy_summaries)

    results = {
        "env_id": ENV_ID,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "seed": args.seed,
        "reset_distance_distribution": distance_summary,
        "trivial_policies": policy_summaries,
    }
    (runs_dir / "results.json").write_text(json.dumps(results, indent=2))
    np.save(runs_dir / "reset_distances.npy", distances)

    chart_path = plot_reset_distance_histogram(
        distances, success_threshold, charts_dir / "reset_distance_histogram.png"
    )
    print(f"\nchart written to {chart_path}")
    print(f"results written to {runs_dir / 'results.json'}")


if __name__ == "__main__":
    main()
