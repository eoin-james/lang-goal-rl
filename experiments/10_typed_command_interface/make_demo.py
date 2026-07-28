"""Generate the stage-10 demo GIF: goto, then move, then waypoints, in ONE continuous episode.

Neither `episode_recording.record_episode` (one phase) nor
`record_episode_with_goal_switch` (exactly two phases) fits a `goto` -> `move` -> `waypoints`
sequence (four goal-holding segments total once the 2-leg waypoint chain is counted) --
per the task's explicit permission, this is a small one-off recording script rather than a
forced fit onto either existing helper. It reuses the same rendering primitive those helpers
use (`env.render()` on a `render_mode="rgb_array"` env, encoded via `imageio.mimsave`) and the
same real pipeline every other script in this stage drives: `parse_command` ->
`CommandExecutor.apply_command`/`advance` -> `clip_to_box` -> `env.step`. Nothing about this
script is a special demo-only code path -- it is `interactive_demo.run_commands`'s exact step
loop, minus the stdin thread and terminal printing, plus frame capture.

Segment lengths sum to exactly 50 steps -- FetchReach-v4's own registered `max_episode_steps`
-- so this is one real, complete episode under the env's own limit, not an extended or
truncated one:

- `goto` a point in the "reach left" region: 15 steps.
- `move reach up high 0.15` from wherever the robot actually ended up after the goto phase
  (the live achieved position, resolved the same way `run_command_eval.py`'s move eval
  resolves it): 15 steps.
- `waypoints`: two legs, "reach down low" then "reach forward", 10 steps/leg (steps_per_leg=10,
  matching `CommandExecutor`'s constructor): 20 steps.

Regions are hand-picked (not randomly drawn) for visual separation on camera, following every
earlier stage's demo-script precedent (e.g. stage 5/6's `make_demo.py` choosing "reach left"
then "reach up high" by hand). Checkpoint is `seed_0`, the same healthy seed every other
stage's `make_demo.py` defaults to.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import SAC

from lang_goal_rl.command_executor import CommandExecutor
from lang_goal_rl.command_grammar import parse_command
from lang_goal_rl.goal_region_vocabulary import MEASURED_GOAL_BOX, sample_region_goals
from lang_goal_rl.relative_move import clip_to_box

EXPERIMENT_DIR = Path(__file__).parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
DEMOS_DIR = REPO_ROOT / "demos"

ENV_ID = "FetchReach-v4"
CHECKPOINT_PATH = EXPERIMENT_DIR.parent / "01_uvfa_her_baseline" / "checkpoints" / "seed_0.zip"
OUT_PATH = DEMOS_DIR / "09_stage10_typed_command_capstone.gif"

SEED = 12000
"""A fresh base seed, disjoint from every other stage's demo-script seed ranges (clips 1-8
each use their own documented range -- see `demos/README.md`)."""

GOTO_REGION = "reach left"
MOVE_DIRECTION = "reach up high"
MOVE_DISTANCE_M = 0.15
WAYPOINT_REGIONS = ("reach down low", "reach forward")
"""Four hand-picked regions/directions spanning most of the workspace (left, up, down,
forward) so all three command types are visually unambiguous in one continuous clip."""

GOTO_STEPS = 15
MOVE_STEPS = 15
STEPS_PER_LEG = 10
N_WAYPOINT_LEGS = len(WAYPOINT_REGIONS)
WAYPOINTS_STEPS = STEPS_PER_LEG * N_WAYPOINT_LEGS
TOTAL_STEPS = GOTO_STEPS + MOVE_STEPS + WAYPOINTS_STEPS
"""15 + 15 + 20 = 50 -- FetchReach-v4's own registered `max_episode_steps`, so this is one
real, complete episode under the env's own limit, not an extended or truncated one."""

FPS = 10
"""Matches every `record_episode*`/live-eval script's default elsewhere in this project."""


def _run_phase(model: SAC, env: gym.Env, obs: dict, target: np.ndarray, *, n_steps: int, frames: list, positions: list) -> tuple[dict, bool]:
    """Write `target` into the env/obs, then step `n_steps` times, capturing a frame each step.

    Args:
        model: The trained SAC checkpoint under test.
        env: A `render_mode="rgb_array"` FetchReach-v4 env instance.
        obs: The current observation.
        target: The xyz goal to hold for this phase, already clipped.
        n_steps: Number of env steps to run.
        frames: Accumulator list; this phase's rendered frames are appended in place.
        positions: Accumulator list; this phase's achieved xyz positions are appended in place.

    Returns:
        `(obs, phase_success)` -- the updated observation and whether `info["is_success"]`
        was truthy on this phase's final step.
    """
    env.unwrapped.goal = target.copy()
    obs["desired_goal"] = target.copy()
    is_success = False
    for _ in range(n_steps):
        action, _state = model.predict(obs, deterministic=True)
        obs, _reward, _terminated, _truncated, info = env.step(action)
        is_success = bool(info.get("is_success", is_success))
        frames.append(env.render())
        positions.append(np.asarray(obs["achieved_goal"], dtype=np.float64))
    return obs, is_success


def main() -> None:
    """Record the goto -> move -> waypoints continuous episode and print the real, measured outcome per phase."""
    DEMOS_DIR.mkdir(parents=True, exist_ok=True)
    gym.register_envs(gymnasium_robotics)
    env = gym.make(ENV_ID, render_mode="rgb_array")
    model = SAC.load(CHECKPOINT_PATH, env=env)
    print(f"loaded checkpoint from {CHECKPOINT_PATH} (no training, eval-only, seed={SEED})")

    executor = CommandExecutor(box=MEASURED_GOAL_BOX, steps_per_leg=STEPS_PER_LEG)
    obs, _info = env.reset(seed=SEED)
    achieved = np.asarray(obs["achieved_goal"], dtype=np.float64)

    frames = [env.render()]
    positions = [achieved.copy()]

    # --- phase 1: goto ---
    goto_point = sample_region_goals(GOTO_REGION, 1, seed=SEED)[0]
    goto_command = parse_command(f"goto {goto_point[0]} {goto_point[1]} {goto_point[2]}")
    executor.apply_command(goto_command, current_achieved_xyz=achieved)
    goto_target = clip_to_box(executor.target_for_step(), box=MEASURED_GOAL_BOX)
    obs, goto_success = _run_phase(model, env, obs, goto_target, n_steps=GOTO_STEPS, frames=frames, positions=positions)
    achieved = np.asarray(obs["achieved_goal"], dtype=np.float64)
    print(f"[phase 1: goto {GOTO_REGION} {goto_target.tolist()}] success={goto_success} steps={GOTO_STEPS}")

    # --- phase 2: move (resolved live, from wherever the goto phase actually left the robot) ---
    move_command = parse_command(f"move {MOVE_DIRECTION} {MOVE_DISTANCE_M}")
    executor.apply_command(move_command, current_achieved_xyz=achieved)
    move_target = clip_to_box(executor.target_for_step(), box=MEASURED_GOAL_BOX)
    obs, move_success = _run_phase(model, env, obs, move_target, n_steps=MOVE_STEPS, frames=frames, positions=positions)
    achieved = np.asarray(obs["achieved_goal"], dtype=np.float64)
    print(f"[phase 2: move {MOVE_DIRECTION} {MOVE_DISTANCE_M}m -> {move_target.tolist()}] success={move_success} steps={MOVE_STEPS}")

    # --- phase 3: waypoints (2 legs, no reset since phase 1/2) ---
    waypoint_goals = [sample_region_goals(region, 1, seed=SEED + i)[0] for i, region in enumerate(WAYPOINT_REGIONS)]
    leg_text = ", ".join(f"{w[0]} {w[1]} {w[2]}" for w in waypoint_goals)
    waypoints_command = parse_command(f"waypoints {leg_text}")
    executor.apply_command(waypoints_command, current_achieved_xyz=achieved)

    leg_successes = []
    for leg_index, region in enumerate(WAYPOINT_REGIONS):
        leg_target = clip_to_box(executor.target_for_step(), box=MEASURED_GOAL_BOX)
        leg_success = False
        for _ in range(STEPS_PER_LEG):
            env.unwrapped.goal = leg_target.copy()
            obs["desired_goal"] = leg_target.copy()
            action, _state = model.predict(obs, deterministic=True)
            obs, _reward, _terminated, _truncated, info = env.step(action)
            leg_success = bool(info.get("is_success", leg_success))
            executor.advance(achieved_xyz=obs["achieved_goal"], is_success=leg_success)
            frames.append(env.render())
            positions.append(np.asarray(obs["achieved_goal"], dtype=np.float64))
        leg_successes.append(leg_success)
        print(f"[phase 3: waypoints leg {leg_index + 1}/{N_WAYPOINT_LEGS} ({region}) -> {leg_target.tolist()}] success={leg_success} steps={STEPS_PER_LEG}")

    env.close()

    total_travel = float(np.linalg.norm(np.diff(np.stack(positions), axis=0), axis=1).sum())
    imageio.mimsave(OUT_PATH, frames, duration=1000 / FPS)
    print(
        f"saved {OUT_PATH} ({len(frames)} frames, {TOTAL_STEPS} steps, total_travel={total_travel:.4f}) -- "
        f"real outcome: goto={goto_success} move={move_success} waypoints_legs={leg_successes}"
    )


if __name__ == "__main__":
    main()
