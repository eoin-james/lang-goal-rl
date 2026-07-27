"""Side-task for stage 5: retrain stage 1's SAC+HER baseline and persist checkpoints.

The original `train.py` never called `model.save(...)` despite stage 1 being
marked Done in ROADMAP.md -- no checkpoint exists on disk. Stage 5 (mid-episode
re-goaling) needs an existing trained literal-goal policy to test zero-shot, so
this script retrains 3 seeds with the *exact same* `build_model`/`evaluate`
helpers and hyperparameters as `train.py` (imported directly, not copied, so
there is no risk of hyperparameter drift from the original run) and adds the
one missing step: `model.save(...)`.

This is purely a checkpoint-provisioning step in service of stage 5 (and any
future stage that needs a literal-goal policy). It does not re-derive, judge,
or claim credit for stage 1's own proof gate -- that result and its report.md
are untouched. See `experiments/05_midepisode_regoal/report.md`'s "Checkpoint
provisioning" section for why this script exists.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
from train import ENV_ID, build_model, evaluate

EXPERIMENT_DIR = Path(__file__).parent


def main() -> None:
    """Train one seed of stage 1's SAC+HER model, evaluate it, and save its checkpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    gym.register_envs(gymnasium_robotics)
    env = gym.make(ENV_ID)

    model = build_model(env, args.seed)
    model.learn(total_timesteps=args.total_timesteps)

    success_rate = evaluate(model, env, args.eval_episodes)
    print(f"success_rate={success_rate:.3f} over {args.eval_episodes} episodes")

    checkpoint_path = EXPERIMENT_DIR / "checkpoints" / f"seed_{args.seed}.zip"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint_path)
    print(f"checkpoint_saved={checkpoint_path}")

    env.close()


if __name__ == "__main__":
    main()
