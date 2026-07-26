"""Stage 2 proof gate: SAC + HER + frozen contrastively-pretrained goal embedding on FetchReach-v4.

Same protocol as stage 1 (`experiments/01_uvfa_her_baseline/train.py`):
SAC+HER, identical hyperparameters, identical total-timesteps/eval-episodes
budget, identical held-out eval seeds. The only change is swapping the
literal xyz goal the policy/critic see for a learned, frozen embedding via
`GoalEmbeddingExtractor`. The encoder itself is pretrained exactly once, by
`pretrain_encoder.py`, and loaded here unchanged for every RL seed — RL seed
variance is never confounded with encoder-pretraining variance.
"""

import argparse
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from lang_goal_rl.goal_embedding_extractor import GoalEmbeddingExtractor
from lang_goal_rl.goal_encoder import GoalEncoder

ENV_ID = "FetchReach-v4"
DEFAULT_ENCODER_PATH = Path(__file__).parent / "artifacts" / "goal_encoder.pt"


def load_frozen_encoder(path: Path) -> GoalEncoder:
    """Load the once-pretrained GoalEncoder checkpoint shared across all RL seeds.

    Args:
        path: Path to the state dict saved by `pretrain_encoder.py`.

    Returns:
        The loaded `GoalEncoder`, in eval mode.
    """
    encoder = GoalEncoder(goal_dim=3)
    encoder.load_state_dict(torch.load(path, map_location="cpu"))
    encoder.eval()
    return encoder


def build_model(env: gym.Env, seed: int, encoder: GoalEncoder) -> SAC:
    """Construct the SAC+HER model — identical to stage 1 except for the features extractor.

    `GoalEmbeddingExtractor` deep-copies `encoder` internally (once per
    actor/critic/critic_target network) and freezes the copy, so this
    function's caller can safely reuse the same `encoder` object across all
    10 RL seeds without cross-seed weight sharing or drift.
    """
    return SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": 4,
            "goal_selection_strategy": "future",
        },
        learning_rate=1e-3,
        buffer_size=int(1e6),
        gamma=0.95,
        batch_size=256,
        policy_kwargs={
            "features_extractor_class": GoalEmbeddingExtractor,
            "features_extractor_kwargs": {"goal_encoder": encoder, "freeze_encoder": True},
            "net_arch": [256, 256, 256],
        },
        seed=seed,
        verbose=1,
    )


def evaluate(model: SAC, env: gym.Env, n_episodes: int) -> float:
    """Roll out the trained policy and return the success rate over n_episodes.

    Identical eval protocol to stage 1: deterministic actions, held-out eval
    seeds (1000+) distinct from training.
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


def main() -> None:
    """Train SAC+HER with a frozen goal-embedding extractor on FetchReach-v4 and report eval success rate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encoder-path", type=Path, default=DEFAULT_ENCODER_PATH)
    args = parser.parse_args()

    gym.register_envs(gymnasium_robotics)
    env = gym.make(ENV_ID)

    encoder = load_frozen_encoder(args.encoder_path)
    model = build_model(env, args.seed, encoder)
    model.learn(total_timesteps=args.total_timesteps)

    success_rate = evaluate(model, env, args.eval_episodes)
    print(f"success_rate={success_rate:.3f} over {args.eval_episodes} episodes")
    env.close()


if __name__ == "__main__":
    main()
