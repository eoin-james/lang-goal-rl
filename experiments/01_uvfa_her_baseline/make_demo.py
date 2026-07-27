"""Generate the stage-1 demo GIF: literal xyz goal, no language involved at all.

Uses `lang_goal_rl.episode_recording.record_episode` against
`checkpoints/seed_0.zip` -- one of this stage's 8 healthy seeds (seeds 2 and
7 are excluded: `report.md`'s Pass 2 reviewer verdict documents both as
showing the SAC deterministic-eval-collapse signature, a known algorithm-
level fragility unrelated to goal-conditioning itself).

Literal-goal mode means `record_episode` is called with no
`goal_embedding_override` -- the policy sees exactly what `env.reset()`
samples as `desired_goal`, unmodified, matching this stage's own `evaluate()`
protocol in `train.py` (`obs, _info = env.reset(seed=1000 + episode)`,
deterministic actions). Ground truth is whatever `info["is_success"]`
reports; nothing here touches the env's goal.

Tries the same held-out eval seed range `evaluate()` uses (1000+) until a
real success is found, rather than assuming the first seed works -- even
though seed 0's own eval scored a clean 1.000 over 50 episodes (`report.md`),
so the first seed tried is expected, not guaranteed, to succeed.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC

from lang_goal_rl.episode_recording import record_episode

EXPERIMENT_DIR = Path(__file__).parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
DEMOS_DIR = REPO_ROOT / "demos"

ENV_ID = "FetchReach-v4"
CHECKPOINT_PATH = EXPERIMENT_DIR / "checkpoints" / "seed_0.zip"
OUT_PATH = DEMOS_DIR / "04_stage1_literal_baseline.gif"

EVAL_SEEDS = range(1000, 1010)
"""Matches `train.py`'s `evaluate()` held-out eval seed range (`1000 +
episode`) -- distinct from the 0-9 training-seed range."""


def main() -> None:
    """Record the literal-goal baseline and print the real, measured outcome."""
    DEMOS_DIR.mkdir(parents=True, exist_ok=True)
    gym.register_envs(gymnasium_robotics)
    env = gym.make(ENV_ID, render_mode="rgb_array")
    model = SAC.load(CHECKPOINT_PATH, env=env)
    print(f"loaded checkpoint from {CHECKPOINT_PATH} (no training, eval-only)")

    for seed in EVAL_SEEDS:
        env.reset(seed=seed)
        result = record_episode(env, model, out_path=OUT_PATH, max_steps=50)
        print(f"[stage1] seed={seed} success={result.success} n_steps={result.n_steps}")
        if result.success:
            break
    else:
        print(
            "[stage1] WARNING: no success found across tried seeds -- keeping the last recording, labeled honestly"
        )

    env.close()


if __name__ == "__main__":
    main()
