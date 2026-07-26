"""Stage 1 proof gate: SAC + HER on FetchReach-v4, literal xyz goal.

Success criterion: near-100% success rate over held-out eval episodes,
matching the known-easy difficulty of FetchReach in the SB3/HER literature.
"""

import argparse

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

ENV_ID = "FetchReach-v4"


def build_model(env: gym.Env, seed: int) -> SAC:
    """Construct the SAC+HER model with FetchReach-appropriate hyperparameters."""
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
        policy_kwargs={"net_arch": [256, 256, 256]},
        seed=seed,
        verbose=1,
    )


def evaluate(model: SAC, env: gym.Env, n_episodes: int) -> float:
    """Roll out the trained policy and return the success rate over n_episodes."""
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
    """Train SAC+HER on FetchReach-v4 and report the eval success rate."""
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
    env.close()


if __name__ == "__main__":
    main()
