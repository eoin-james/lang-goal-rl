"""Diagnostic: isolate whether the near-zero language-eval result is a substitution-mechanism
bug or an off-distribution projected-embedding problem.

Two checks against the already-trained seed_0 checkpoint:

1. Feed the policy `goal_encoder(literal_target)` (the *correct*, un-projected
   embedding for a region-sampled literal target) through the exact same
   monkeypatch machinery `evaluate_language_goal` uses. If this reproduces a
   near-stage-2 success rate, the substitution mechanism itself is sound and
   the failure is specific to the projected embedding's value, not to how it
   gets fed to the policy.
2. Compare the norm/scale of projected instruction embeddings against the
   norm/scale of `goal_encoder(desired_goal)` for goals actually sampled
   during training (uniform over the whole box) — the two need to be in a
   comparable numeric regime for the policy's learned features to make sense
   of either.
"""

from __future__ import annotations

import types
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from stable_baselines3 import SAC

from lang_goal_rl.goal_embedding_extractor import GoalEmbeddingExtractor
from lang_goal_rl.goal_encoder import GoalEncoder
from lang_goal_rl.goal_region_vocabulary import MEASURED_GOAL_BOX, sample_region_goals
from lang_goal_rl.language_embedding import encode_instructions
from lang_goal_rl.language_goal_projection import LanguageGoalProjection

EXPERIMENT_DIR = Path(__file__).parent
ENCODER_PATH = EXPERIMENT_DIR.parent / "02_contrastive_goal_embedding" / "artifacts" / "goal_encoder.pt"
PROJECTION_PATH = EXPERIMENT_DIR / "artifacts" / "language_goal_projection.pt"
CHECKPOINT_PATH = EXPERIMENT_DIR / "checkpoints" / "seed_0.zip"


def load_frozen_encoder(path: Path) -> GoalEncoder:
    """Load stage 2's pretrained `GoalEncoder` checkpoint, unchanged."""
    encoder = GoalEncoder(goal_dim=3)
    encoder.load_state_dict(torch.load(path, map_location="cpu"))
    encoder.eval()
    return encoder


def load_projection(path: Path) -> LanguageGoalProjection:
    """Load a `LanguageGoalProjection` checkpoint saved by `train_projection.py`."""
    checkpoint = torch.load(path, map_location="cpu")
    projection = LanguageGoalProjection(input_dim=checkpoint["input_dim"], embed_dim=checkpoint["embed_dim"])
    projection.load_state_dict(checkpoint["state_dict"])
    projection.eval()
    return projection


def run_with_override(model: SAC, env: gym.Env, target: np.ndarray, override: torch.Tensor, seed: int) -> bool:
    """Run one episode with the actor's desired-embedding input pinned to `override`."""
    extractor = model.actor.features_extractor
    original_forward = extractor.forward
    fixed = override.detach().to(torch.float32)

    def patched_forward(self: GoalEmbeddingExtractor, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        flat_observation = self._flatten(observations["observation"])
        achieved_embedding = self.goal_encoder(observations["achieved_goal"])
        batch_size = achieved_embedding.shape[0]
        desired_embedding = fixed.expand(batch_size, -1)
        return torch.cat([flat_observation, achieved_embedding, desired_embedding], dim=1)

    extractor.forward = types.MethodType(patched_forward, extractor)
    try:
        obs, _info = env.reset(seed=seed)
        env.unwrapped.goal = target.copy()
        obs["desired_goal"] = target.copy()
        terminated = truncated = False
        is_success = False
        while not (terminated or truncated):
            action, _state = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            is_success = bool(info.get("is_success", is_success))
        return is_success
    finally:
        extractor.forward = original_forward


def main() -> None:
    """Run both diagnostic checks and print the results."""
    gym.register_envs(gymnasium_robotics)
    env = gym.make("FetchReach-v4")

    encoder = load_frozen_encoder(ENCODER_PATH)
    projection = load_projection(PROJECTION_PATH)
    model = SAC.load(CHECKPOINT_PATH, env=env)

    # Check 1: correct (un-projected) embedding through the same monkeypatch machinery.
    n_episodes = 20
    targets = sample_region_goals("reach up high", n_episodes, seed=9000, box=MEASURED_GOAL_BOX)
    successes_correct_embedding = []
    for i in range(n_episodes):
        target = targets[i]
        with torch.no_grad():
            correct_embedding = encoder(torch.from_numpy(target).float().unsqueeze(0)).squeeze(0)
        successes_correct_embedding.append(
            run_with_override(model, env, target.astype(np.float64), correct_embedding, seed=9000 + i),
        )
    print(
        f"Check 1 (correct goal_encoder(literal_target) via monkeypatch): "
        f"success_rate={float(np.mean(successes_correct_embedding)):.3f} over {n_episodes} episodes",
    )

    # Check 2: scale comparison -- projected instruction embeddings vs. real training-time
    # goal_encoder(desired_goal) outputs (goals uniform over the whole measured box, matching
    # what FetchReach actually samples on reset during RL training).
    rng = np.random.default_rng(0)
    training_like_goals = rng.uniform(MEASURED_GOAL_BOX.axis_min, MEASURED_GOAL_BOX.axis_max, size=(500, 3))
    with torch.no_grad():
        training_like_embeddings = encoder(torch.from_numpy(training_like_goals).float())
    training_norms = training_like_embeddings.norm(dim=1)

    instructions = ["reach up high", "move your hand upward", "move your hand to the center"]
    sentence_embeddings = torch.from_numpy(encode_instructions(instructions))
    with torch.no_grad():
        projected = projection(sentence_embeddings)
    projected_norms = projected.norm(dim=1)

    print(
        f"Check 2 (embedding-space scale): training-time goal_encoder(desired_goal) norms: "
        f"mean={training_norms.mean():.3f} std={training_norms.std():.3f} "
        f"min={training_norms.min():.3f} max={training_norms.max():.3f}",
    )
    for instruction, norm in zip(instructions, projected_norms, strict=True):
        print(f'  projected embedding norm for "{instruction}" = {norm.item():.3f}')

    env.close()


if __name__ == "__main__":
    main()
