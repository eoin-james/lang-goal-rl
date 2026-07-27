"""Generate the stage-5 demo GIF: the robot's target changes mid-episode, no reset, and it still gets there.

Uses `lang_goal_rl.episode_recording.record_episode_with_goal_switch` against
`experiments/01_uvfa_her_baseline/checkpoints/seed_0.zip` -- the same healthy
seed stage 1's own `make_demo.py` uses (seeds 2 and 7 are excluded: this
stage's own `report.md` documents both as showing the SAC deterministic-
eval-collapse signature carried over from stage 1, unrelated to re-goaling
itself).

Goal pair and switch_step follow this stage's own `run_regoal_eval.py`
protocol as closely as a single illustrative episode can:

- Two goals drawn from two distinct `goal_region_vocabulary` regions via
  `sample_region_goals`, exactly like `run_regoal_eval.py`'s
  `sample_goal_pair` -- but with the two regions picked deliberately here
  ("reach left" then "reach up high") rather than randomly chosen, so the
  swap is visually unambiguous in the GIF rather than a coin-flip pair that
  might look like a small nudge.
- `switch_step=20`, one of the four switch points `run_regoal_eval.py`
  actually measured (`SWITCH_STEPS = (10, 20, 30, 40)`), roughly the middle
  of the 50-step episode.
- `max_steps=50`, matching `run_regoal_eval.py`'s `MAX_STEPS` (FetchReach-
  v4's registered episode length).

Literal-xyz mode (no embedding overrides): the policy sees the env's real
`desired_goal` in both phases, matching this stage's own scope -- testing the
re-goaling mechanism itself, not the embedding layer stages 2-4 cover.
Ground truth is judged against the *new* goal only, via
`record_episode_with_goal_switch`'s own success bookkeeping.

Seeds tried follow `run_regoal_eval.py`'s own seed convention:
`REGOAL_BASE_SEED + switch_step_index * 1000 + episode_index`, with
`switch_step_index=1` for `switch_step=20` (its position in `SWITCH_STEPS`)
-- so the first seed tried here is the exact `base_seed` that stage 5's own
switch_step=20 aggregate run used for its first episode. Up to 3 attempts are
tried (varying only the seed, not the region choice) before giving up
honestly, matching this project's other demo scripts' retry discipline.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC

from lang_goal_rl.episode_recording import record_episode_with_goal_switch
from lang_goal_rl.goal_region_vocabulary import sample_region_goals

EXPERIMENT_DIR = Path(__file__).parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
DEMOS_DIR = REPO_ROOT / "demos"

ENV_ID = "FetchReach-v4"
CHECKPOINT_PATH = (
    EXPERIMENT_DIR.parent / "01_uvfa_her_baseline" / "checkpoints" / "seed_0.zip"
)
OUT_PATH = DEMOS_DIR / "07_stage5_midepisode_switch.gif"

INITIAL_REGION = "reach left"
NEW_REGION = "reach up high"
"""Two regions chosen for visual clarity in the GIF -- `run_regoal_eval.py`'s
own `sample_goal_pair` picks two *random* distinct regions per episode;
picking these two by hand here just makes the swap obviously legible on
video rather than leaving it to a random draw."""

SWITCH_STEP = 20
MAX_STEPS = 50
"""Matches `run_regoal_eval.py`'s `SWITCH_STEPS` (one of the four measured
points) and `MAX_STEPS` (FetchReach-v4's registered episode length)."""

SWITCH_STEP_INDEX = 1
"""`SWITCH_STEP`'s position in `run_regoal_eval.py`'s `SWITCH_STEPS = (10,
20, 30, 40)` tuple -- used to reconstruct that script's own base_seed
convention below."""

REGOAL_BASE_SEED = 9000
REGION_B_SEED_OFFSET = 500_000
"""Copied from `run_regoal_eval.py` -- same base seed and region-B offset
convention, so the first seed tried here lines up with that script's own
switch_step=20, episode_index=0 draw."""

ATTEMPT_SEEDS = [
    REGOAL_BASE_SEED + SWITCH_STEP_INDEX * 1000 + episode_index
    for episode_index in range(3)
]
"""At most 3 (seed, goal-pair) attempts, per the task's honesty-over-luck
retry cap -- each seed draws its own goal pair via `sample_region_goals`, so
varying the seed also varies the exact within-region points, not just the
env's initial robot pose."""


def main() -> None:
    """Record the mid-episode goal-switch demo and print the real, measured outcome."""
    DEMOS_DIR.mkdir(parents=True, exist_ok=True)
    gym.register_envs(gymnasium_robotics)
    env = gym.make(ENV_ID, render_mode="rgb_array")
    model = SAC.load(CHECKPOINT_PATH, env=env)
    print(f"loaded checkpoint from {CHECKPOINT_PATH} (no training, eval-only)")

    for attempt, seed in enumerate(ATTEMPT_SEEDS, start=1):
        initial_goal = sample_region_goals(INITIAL_REGION, 1, seed=seed)[0]
        new_goal = sample_region_goals(NEW_REGION, 1, seed=seed + REGION_B_SEED_OFFSET)[
            0
        ]

        env.reset(seed=seed)
        result = record_episode_with_goal_switch(
            env,
            model,
            out_path=OUT_PATH,
            initial_goal_xyz=initial_goal,
            switch_step=SWITCH_STEP,
            new_goal_xyz=new_goal,
            max_steps=MAX_STEPS,
        )
        print(
            f"[stage5] attempt={attempt} seed={seed} "
            f"initial_goal={INITIAL_REGION} {initial_goal.tolist()} "
            f"new_goal={NEW_REGION} {new_goal.tolist()} "
            f"switch_step={SWITCH_STEP} success={result.success} n_steps={result.n_steps}"
        )
        if result.success:
            break
    else:
        print(
            "[stage5] WARNING: no success found across tried seeds -- keeping the last recording, labeled honestly"
        )

    env.close()


if __name__ == "__main__":
    main()
