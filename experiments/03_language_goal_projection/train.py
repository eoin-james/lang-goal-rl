"""Stage 3: train the stage-2-style SAC+HER policy, then run the language-goal substitution eval.

Two phases per seed, run back to back in one process:

1. Standard protocol (identical to `experiments/02_contrastive_goal_embedding/
   train.py`): SAC+HER with `GoalEmbeddingExtractor` wrapping stage 2's
   frozen `GoalEncoder`, same hyperparameters, same total-timesteps budget.
   Unlike stage 2, the trained model is saved to `checkpoints/seed_<k>.zip`
   (stage 2 didn't persist a checkpoint, which cost a retrain — see
   `.claude/agents/CONTRACTS.md`'s reuse-checkpoints rule). Evaluated with
   the exact stage-2 literal-goal-embedding protocol first, to confirm this
   run reproduces stage 2's result before testing anything language-related.

2. The actual stage-3 substitution test, once per fixed instruction
   (`goal_region_vocabulary.ALL_INSTRUCTIONS`): the env's `desired_goal` is
   overridden after every reset to a literal xyz point sampled from that
   instruction's real region (`sample_region_goals`) — this is what
   determines ground-truth success/failure, exactly like HER's literal
   goals always have. But the *policy* never sees that literal point run
   through the goal encoder; its features extractor is monkeypatched for
   the duration of this eval to substitute a fixed
   `projection(sentence_embedding(instruction))` vector wherever it would
   have computed `goal_encoder(desired_goal)`. `achieved_goal` still goes
   through the frozen `GoalEncoder` normally, from the real current state,
   every step — only the desired-goal input is substituted. See
   `evaluate_language_goal`'s docstring for why a monkeypatch (rather than
   modifying `GoalEmbeddingExtractor` itself) is the right tool here.
"""

from __future__ import annotations

import argparse
import types
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from lang_goal_rl.goal_embedding_extractor import GoalEmbeddingExtractor
from lang_goal_rl.goal_encoder import GoalEncoder
from lang_goal_rl.goal_region_vocabulary import (
    ALL_INSTRUCTIONS,
    MEASURED_GOAL_BOX,
    instruction_to_region,
    sample_region_goals,
)
from lang_goal_rl.language_embedding import encode_instructions
from lang_goal_rl.language_goal_projection import LanguageGoalProjection

ENV_ID = "FetchReach-v4"
EXPERIMENT_DIR = Path(__file__).parent
DEFAULT_ENCODER_PATH = (
    EXPERIMENT_DIR.parent / "02_contrastive_goal_embedding" / "artifacts" / "goal_encoder.pt"
)
DEFAULT_PROJECTION_PATH = EXPERIMENT_DIR / "artifacts" / "language_goal_projection.pt"

LITERAL_EVAL_BASE_SEED = 1000
"""Held-out eval seeds for the standard literal-goal protocol, identical to
stage 1/2's `evaluate()` — distinct from the 0-9 range used for training
seeds."""

LANGUAGE_EVAL_BASE_SEED = 5000
"""Held-out eval seeds for the language-goal substitution protocol. A
different offset from `LITERAL_EVAL_BASE_SEED` so the two protocols never
reuse the same env-reset seed for a different purpose, and different
instructions/regions use `LANGUAGE_EVAL_BASE_SEED + region_index * n_episodes`
so no two instructions share reset seeds either (see `main`)."""


def load_frozen_encoder(path: Path) -> GoalEncoder:
    """Load stage 2's pretrained `GoalEncoder` checkpoint, unchanged.

    Args:
        path: Path to the state dict saved by stage 2's `pretrain_encoder.py`.

    Returns:
        The loaded `GoalEncoder`, in eval mode.
    """
    encoder = GoalEncoder(goal_dim=3)
    encoder.load_state_dict(torch.load(path, map_location="cpu"))
    encoder.eval()
    return encoder


def load_projection(path: Path) -> LanguageGoalProjection:
    """Load a `LanguageGoalProjection` checkpoint saved by `train_projection.py`.

    Args:
        path: Path to the checkpoint dict (`input_dim`/`embed_dim`/`state_dict`).

    Returns:
        The loaded `LanguageGoalProjection`, in eval mode.
    """
    checkpoint = torch.load(path, map_location="cpu")
    projection = LanguageGoalProjection(input_dim=checkpoint["input_dim"], embed_dim=checkpoint["embed_dim"])
    projection.load_state_dict(checkpoint["state_dict"])
    projection.eval()
    return projection


def build_model(env: gym.Env, seed: int, encoder: GoalEncoder) -> SAC:
    """Construct the SAC+HER model — identical hyperparameters to stage 2's `build_model`."""
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


def evaluate_literal(model: SAC, env: gym.Env, n_episodes: int) -> float:
    """Roll out the trained policy under the standard literal-goal-embedding protocol.

    Identical to stage 1/2's `evaluate()`: deterministic actions, held-out
    eval seeds distinct from training, `desired_goal` untouched (whatever
    the env samples on reset).
    """
    successes = []
    for episode in range(n_episodes):
        obs, _info = env.reset(seed=LITERAL_EVAL_BASE_SEED + episode)
        terminated = truncated = False
        is_success = False
        while not (terminated or truncated):
            action, _state = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            is_success = bool(info.get("is_success", is_success))
        successes.append(is_success)
    return float(np.mean(successes))


def evaluate_language_goal(
    model: SAC,
    env: gym.Env,
    *,
    region_name: str,
    projected_embedding: torch.Tensor,
    n_episodes: int,
    base_seed: int,
) -> float:
    """Roll out the policy with its desired-goal input substituted by a language projection.

    Ground truth (what decides success/failure) is a literal xyz point
    sampled from `region_name` and written into the env's actual `goal`
    state after every reset (via `env.unwrapped.goal`) — every downstream
    `info["is_success"]`/`compute_reward` call in `MujocoRobotEnv.step`
    reads `self.goal` directly, so this is a real ground-truth override, not
    a cosmetic one (checked in `gymnasium_robotics.envs.robot_env`: `step`
    computes `is_success`/reward/termination against `self.goal`, and
    `MujocoFetchEnv._get_obs` reports `desired_goal` as `self.goal.copy()`).

    What the *policy* sees for its desired-goal input is substituted
    instead: `model.actor.features_extractor.forward` is monkeypatched for
    the duration of this function to return `projected_embedding` (broadcast
    over the batch) wherever it would otherwise have computed
    `goal_encoder(desired_goal)`. `achieved_goal` is untouched and still runs
    through the real frozen `GoalEncoder` every step. A monkeypatch (rather
    than a new `GoalEmbeddingExtractor` subclass built into `policy_kwargs`
    at construction time) is used because this substitution only needs to
    exist for eval rollouts on an *already-trained* model — rebuilding the
    model would discard the trained weights.

    Only `model.actor`'s extractor is patched: `SAC.predict` (used in the
    eval loop below, same as `evaluate_literal`) calls `self.actor(...)`
    exclusively (confirmed in `stable_baselines3/sac/policies.py`), and this
    stage's `build_model` uses `share_features_extractor=False` (SAC's
    default), so the critic's separate extractor instance is never touched
    and never needs to be for prediction.

    Args:
        model: A trained SAC model (same architecture as `build_model`).
        env: The FetchReach-v4 env instance `model` was trained/evaluated on.
        region_name: The instruction's region (`goal_region_vocabulary`),
            used to sample ground-truth targets.
        projected_embedding: The instruction's `projection(sentence_embedding)`
            output, shape `(embed_dim,)` — fixed for the whole eval.
        n_episodes: Number of eval episodes to run.
        base_seed: Base env-reset seed; episode `i` resets with `base_seed + i`
            and also samples its ground-truth target with the same seed
            (via `sample_region_goals`), giving this call a self-contained,
            reproducible seed range distinct from any other call's.

    Returns:
        Success rate over `n_episodes`, judged against the literal
        region-sampled ground truth.
    """
    targets = sample_region_goals(region_name, n_episodes, seed=base_seed, box=MEASURED_GOAL_BOX)

    extractor = model.actor.features_extractor
    original_forward = extractor.forward
    override = projected_embedding.detach().to(torch.float32)

    def patched_forward(self: GoalEmbeddingExtractor, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        flat_observation = self._flatten(observations["observation"])
        achieved_embedding = self.goal_encoder(observations["achieved_goal"])
        batch_size = achieved_embedding.shape[0]
        desired_embedding = override.expand(batch_size, -1)
        return torch.cat([flat_observation, achieved_embedding, desired_embedding], dim=1)

    extractor.forward = types.MethodType(patched_forward, extractor)
    try:
        successes = []
        for episode in range(n_episodes):
            obs, _info = env.reset(seed=base_seed + episode)
            target = targets[episode].astype(np.float64)
            env.unwrapped.goal = target.copy()
            obs["desired_goal"] = target.copy()
            terminated = truncated = False
            is_success = False
            while not (terminated or truncated):
                action, _state = model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, info = env.step(action)
                is_success = bool(info.get("is_success", is_success))
            successes.append(is_success)
    finally:
        extractor.forward = original_forward
    return float(np.mean(successes))


def main() -> None:
    """Train one seed's SAC+HER policy, save its checkpoint, and run both eval protocols."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--language-eval-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encoder-path", type=Path, default=DEFAULT_ENCODER_PATH)
    parser.add_argument("--projection-path", type=Path, default=DEFAULT_PROJECTION_PATH)
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=None,
        help="Defaults to checkpoints/seed_<seed>.zip under this experiment dir.",
    )
    args = parser.parse_args()
    checkpoint_out = args.checkpoint_out or (EXPERIMENT_DIR / "checkpoints" / f"seed_{args.seed}.zip")

    gym.register_envs(gymnasium_robotics)
    env = gym.make(ENV_ID)

    encoder = load_frozen_encoder(args.encoder_path)
    model = build_model(env, args.seed, encoder)
    model.learn(total_timesteps=args.total_timesteps)

    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint_out)
    print(f"saved checkpoint to {checkpoint_out}")

    literal_success_rate = evaluate_literal(model, env, args.eval_episodes)
    print(f"success_rate={literal_success_rate:.3f} over {args.eval_episodes} episodes")

    projection = load_projection(args.projection_path)
    for region_index, instruction in enumerate(ALL_INSTRUCTIONS):
        region_name = instruction_to_region(instruction)
        sentence_embedding = torch.from_numpy(encode_instructions([instruction]))
        with torch.no_grad():
            projected_embedding = projection(sentence_embedding).squeeze(0)

        base_seed = LANGUAGE_EVAL_BASE_SEED + region_index * args.language_eval_episodes
        language_success_rate = evaluate_language_goal(
            model,
            env,
            region_name=region_name,
            projected_embedding=projected_embedding,
            n_episodes=args.language_eval_episodes,
            base_seed=base_seed,
        )
        print(
            f'language_success_rate={language_success_rate:.3f} instruction="{instruction}" '
            f'region="{region_name}" over {args.language_eval_episodes} episodes',
        )

    env.close()


if __name__ == "__main__":
    main()
