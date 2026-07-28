"""Generate 6 before/after GIFs testing whether `AXIS_DIRECTIONS`' labeling looks right on camera.

Stage 7's whole point: `goal_region_vocabulary.py`'s `AXIS_DIRECTIONS` (which
xyz sign is "forward" vs "back", "left" vs "right", "up" vs "down") is this
project's own labeling convention, not a measured fact -- its docstring says
so explicitly. Nobody has watched the robot move and confirmed a human would
actually call that motion "left" or "forward". This script produces the
visual evidence a human needs to make that call; it does not make the call
itself (see `report.md` / `evidence.md`).

For each of the 6 directional regions in `AXIS_DIRECTIONS` (all regions
except "center", which has no direction to check), records one episode via
`lang_goal_rl.episode_recording.record_episode_with_goal_switch` against
`experiments/01_uvfa_her_baseline/checkpoints/seed_0.zip` -- a healthy seed
(seeds 2 and 7 are excluded project-wide: both stage 1's and stage 5's
reports document them as showing the SAC deterministic-eval-collapse
signature, unrelated to goal-conditioning). Literal-xyz mode throughout (no
`goal_embedding_override`) -- this is a coordinate-convention check, not a
language-layer check, so the policy always sees the env's real
`desired_goal`.

Phase 1 (steps 0..`SWITCH_STEP`) targets `MEASURED_GOAL_BOX.centroid` --
"start near centroid" per the task brief. A quick check
(`uv run python -c "..."` against `FetchReach-v4`, logged in this stage's
`runs/` dir) found FetchReach-v4's fixed initial gripper pose is already
`8.7e-5` m from `MEASURED_GOAL_BOX.centroid` -- effectively the same point,
well inside the 5cm success tolerance. `SWITCH_STEP=5` is therefore just
enough steps to show a few clearly-at-centroid "before" frames, not a
meaningful travel requirement. Phase 2 (the remaining steps up to
`MAX_STEPS=50`, matching every other stage's FetchReach-v4 episode length)
targets one real in-region point per direction, drawn via
`sample_region_goals(region_name, 1, seed=...)` -- "your call on the
simplest correct way to get one real in-region xyz target" per the task
brief; `sample_region_goals` is the existing, already-tested function for
exactly that.

Tries-cap convention (matching every other stage's `make_demo.py`, e.g.
`05_midepisode_regoal/make_demo.py`'s `ATTEMPT_SEEDS`): up to
`ATTEMPTS_PER_DIRECTION=3` seeds are tried per direction, stopping at the
first real success (`EpisodeRecording.success`, grounded in the env's own
`info["is_success"]`). If none of the 3 succeed, the last attempt's
recording is kept and printed as an honest failure -- never forced, never
silently dropped. Each direction's seed range is `BASE_SEED + direction_idx
* 10 + attempt_idx`, disjoint from every other stage's seed ranges (1000s,
2000s, 5000s, 6000s, 9000s) and from `REGION_TARGET_SEED_OFFSET`'s range,
per the project's dedup convention.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC

from lang_goal_rl.episode_recording import record_episode_with_goal_switch
from lang_goal_rl.goal_region_vocabulary import (
    AXIS_DIRECTIONS,
    MEASURED_GOAL_BOX,
    region_names,
    sample_region_goals,
)

EXPERIMENT_DIR = Path(__file__).parent
CHARTS_DIR = EXPERIMENT_DIR / "charts"

ENV_ID = "FetchReach-v4"
CHECKPOINT_PATH = (
    EXPERIMENT_DIR.parent / "01_uvfa_her_baseline" / "checkpoints" / "seed_0.zip"
)
"""Stage 1's seed_0 -- a healthy checkpoint (seeds 2 and 7 excluded project-wide
for the documented SAC eval-collapse signature). Reused unchanged: this stage
tests the coordinate convention, not the policy or the language layer."""

DIRECTIONS: tuple[str, ...] = tuple(name for name in region_names() if name != "center")
"""All 6 directional regions from `REGIONS`, in that tuple's own order
(`reach forward, reach back, reach left, reach right, reach up high, reach
down low`) -- `center` excluded, it has no direction to check."""

CENTROID = MEASURED_GOAL_BOX.centroid

SWITCH_STEP = 5
"""Steps spent targeting `CENTROID` before switching to the region target.
Small deliberately: FetchReach-v4's fixed initial gripper pose is already
~8.7e-5 m from `CENTROID` (see module docstring), so this is "a few frames
sitting at centroid for the GIF's before-shot", not a real travel
requirement."""

MAX_STEPS = 50
"""Matches every other stage's FetchReach-v4 episode length."""

BASE_SEED = 7000
"""Disjoint from every other stage's seed ranges (1000s: stage 1/5's
EVAL_SEEDS/ATTEMPT_SEEDS; 2000s: stage 2; 5000s/6000s/9000s: stage 3/5's
LANGUAGE_EVAL_SEEDS/BASELINE_EVAL_SEEDS/REGOAL_BASE_SEED), per the project's
per-stage seed-dedup convention."""

ATTEMPTS_PER_DIRECTION = 3
"""Tries-cap matching every other stage's `make_demo.py` retry discipline
(e.g. stage 5's `ATTEMPT_SEEDS`, 3 attempts) -- stop at the first real
success, keep and honestly label the last attempt if none succeed."""

REGION_TARGET_SEED_OFFSET = 500_000
"""Same convention as stage 5's `REGION_B_SEED_OFFSET` -- keeps the region-
target draw's seed disjoint from the env-reset seed even though they're
different RNG streams (`sample_region_goals` uses its own
`np.random.default_rng`, never the gym env's), so a human diffing seed
values across this project's scripts never has to wonder whether two
different-looking seeds secretly collide."""

AXIS_NAMES = ("x", "y", "z")


def axis_and_sign(region_name: str) -> tuple[str, str]:
    """Look up which axis and sign `AXIS_DIRECTIONS` assigns a directional region to.

    Args:
        region_name: One of `DIRECTIONS` (a non-"center" region name).

    Returns:
        `(axis_name, sign)`, e.g. `("x", "positive")` for "reach forward".

    Raises:
        ValueError: If `region_name` isn't one of `AXIS_DIRECTIONS`' 6 entries.

    """
    for axis_idx, (negative_name, positive_name) in enumerate(AXIS_DIRECTIONS):
        if region_name == negative_name:
            return AXIS_NAMES[axis_idx], "negative"
        if region_name == positive_name:
            return AXIS_NAMES[axis_idx], "positive"
    msg = f"{region_name!r} is not one of AXIS_DIRECTIONS' 6 directional regions"
    raise ValueError(msg)


def slug_for(region_name: str) -> str:
    """Turn a region name like "reach up high" into a filename stem like "reach_up_high"."""
    return region_name.replace(" ", "_")


def record_direction(env: gym.Env, model: SAC, region_name: str, direction_idx: int) -> dict[str, Any]:
    """Record one direction's before/after GIF, trying up to `ATTEMPTS_PER_DIRECTION` seeds.

    Writes the selected attempt's GIF to `charts/<slug>.gif` -- the first
    real success found, or the last attempt if none succeeded (labeled
    honestly in the returned dict and in stdout, never silently dropped).

    Args:
        env: A `render_mode="rgb_array"` `FetchReach-v4` instance.
        model: The loaded SAC checkpoint.
        region_name: One of `DIRECTIONS`.
        direction_idx: This region's index in `DIRECTIONS`, used only to
            derive a disjoint per-direction seed range from `BASE_SEED`.

    Returns:
        A dict with this direction's region name, axis/sign, success,
        attempts used, the seed and target that produced the kept GIF, step
        count, and total travel -- everything `evidence.md` needs, measured
        rather than transcribed by hand.

    """
    out_path = CHARTS_DIR / f"{slug_for(region_name)}.gif"
    axis, sign = axis_and_sign(region_name)
    attempt_seeds = [
        BASE_SEED + direction_idx * 10 + attempt_idx
        for attempt_idx in range(ATTEMPTS_PER_DIRECTION)
    ]

    last_attempt: tuple[int, int, list[float], Any, Path] | None = None
    with tempfile.TemporaryDirectory(prefix=f"stage7-{slug_for(region_name)}-") as scratch_dir:
        scratch_root = Path(scratch_dir)
        for attempt_number, seed in enumerate(attempt_seeds, start=1):
            target = sample_region_goals(
                region_name, 1, seed=seed + REGION_TARGET_SEED_OFFSET
            )[0]
            candidate_path = scratch_root / f"attempt_{attempt_number}.gif"
            env.reset(seed=seed)
            result = record_episode_with_goal_switch(
                env,
                model,
                out_path=candidate_path,
                initial_goal_xyz=CENTROID,
                switch_step=SWITCH_STEP,
                new_goal_xyz=target,
                max_steps=MAX_STEPS,
            )
            print(
                f"[{region_name}] axis={axis} sign={sign} attempt={attempt_number}/"
                f"{ATTEMPTS_PER_DIRECTION} seed={seed} target={target.tolist()} "
                f"success={result.success} n_steps={result.n_steps} "
                f"total_travel={result.total_travel:.4f}"
            )
            last_attempt = (attempt_number, seed, target.tolist(), result, candidate_path)
            if result.success:
                shutil.copyfile(candidate_path, out_path)
                return {
                    "region": region_name,
                    "axis": axis,
                    "sign": sign,
                    "success": True,
                    "attempts_used": attempt_number,
                    "attempts_cap": ATTEMPTS_PER_DIRECTION,
                    "seed": seed,
                    "target": target.tolist(),
                    "n_steps": result.n_steps,
                    "total_travel": result.total_travel,
                    "gif": out_path,
                }

        assert last_attempt is not None  # ATTEMPTS_PER_DIRECTION >= 1
        attempt_number, seed, target_list, result, candidate_path = last_attempt
        print(
            f"[{region_name}] WARNING: no success across {ATTEMPTS_PER_DIRECTION} "
            "attempts -- keeping the last recording, labeled honestly as a failure"
        )
        shutil.copyfile(candidate_path, out_path)
        return {
            "region": region_name,
            "axis": axis,
            "sign": sign,
            "success": False,
            "attempts_used": ATTEMPTS_PER_DIRECTION,
            "attempts_cap": ATTEMPTS_PER_DIRECTION,
            "seed": seed,
            "target": target_list,
            "n_steps": result.n_steps,
            "total_travel": result.total_travel,
            "gif": out_path,
        }


def main() -> None:
    """Generate all 6 directional demo GIFs into `charts/` and print the real, measured outcome for each."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    gym.register_envs(gymnasium_robotics)
    env = gym.make(ENV_ID, render_mode="rgb_array")
    model = SAC.load(CHECKPOINT_PATH, env=env)
    print(f"loaded checkpoint from {CHECKPOINT_PATH} (no training, eval-only)")
    print(f"centroid (phase-1 target, all directions): {CENTROID.tolist()}")

    results = [
        record_direction(env, model, region_name, direction_idx)
        for direction_idx, region_name in enumerate(DIRECTIONS)
    ]
    env.close()

    print("\n=== summary ===")
    for r in results:
        print(
            f"{r['region']} (axis={r['axis']}, sign={r['sign']}): success={r['success']} "
            f"attempts_used={r['attempts_used']}/{r['attempts_cap']} seed={r['seed']} "
            f"n_steps={r['n_steps']} total_travel={r['total_travel']:.4f} gif={r['gif']}"
        )


if __name__ == "__main__":
    main()
