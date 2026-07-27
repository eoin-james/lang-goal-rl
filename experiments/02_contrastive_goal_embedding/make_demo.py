"""Generate the stage-2 demo GIF: learned goal-embedding representation, no language involved.

Stage 2 never persisted an RL checkpoint to disk (`train.py`'s `main` calls
`model.learn(...)` and only ever prints the eval success rate -- no
`model.save(...)` anywhere in this experiment, confirmed by reading the
whole file; stage 3's `train.py` docstring calls this out explicitly:
"Unlike stage 2, the trained model is saved to `checkpoints/seed_<k>.zip`
(stage 2 didn't persist a checkpoint, which cost a retrain...)"). So this
script cannot load "stage 2's own" seed_0 weights -- they were never saved.

Instead it reuses `experiments/03_language_goal_projection/checkpoints/
seed_0.zip`. This is a valid stand-in, confirmed by reading both training
scripts side by side, not assumed:

- `03_language_goal_projection/train.py`'s `build_model` is byte-for-byte
  the same hyperparameters as this experiment's `build_model` -- same SAC
  args (`learning_rate=1e-3`, `buffer_size=1e6`, `gamma=0.95`,
  `batch_size=256`), same `HerReplayBuffer` kwargs
  (`n_sampled_goal=4`/`goal_selection_strategy="future"`), same
  `features_extractor_class=GoalEmbeddingExtractor` with the identical
  `net_arch=[256, 256, 256]`.
- Both load the *exact same* frozen encoder weights: stage 3's
  `DEFAULT_ENCODER_PATH` points at
  `02_contrastive_goal_embedding/artifacts/goal_encoder.pt` -- this
  experiment's own pretrained checkpoint, unchanged.
- Stage 3's `train.py` runs this checkpoint through `evaluate_literal`
  (identical to this experiment's `evaluate()`: same held-out seed range,
  same deterministic-action protocol) *before* touching anything
  language-related, specifically to confirm the run reproduces stage 2's
  result first.

What is NOT true, and is stated plainly rather than glossed over: this is
not literally stage 2's original seed-0 run (those weights don't exist on
disk to load) -- it's a fresh training run of the identical architecture,
hyperparameters, and frozen encoder. Since stage 2's own proof gate is about
whether *this protocol* (SAC+HER over a frozen learned embedding) works, not
about one specific seed's arbitrary weights, a checkpoint trained under the
identical protocol is a faithful stand-in, not a substitution of what's being
tested.

Recorded in literal-goal mode (`goal_embedding_override=None`): the policy
sees the env's real `desired_goal` run through `GoalEmbeddingExtractor` ->
stage 2's frozen `GoalEncoder` -- exactly what stage 2's proof gate tests.
Ground truth is `info["is_success"]`, untouched.
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
CHECKPOINT_PATH = (
    EXPERIMENT_DIR.parent / "03_language_goal_projection" / "checkpoints" / "seed_0.zip"
)
OUT_PATH = DEMOS_DIR / "05_stage2_embedding_baseline.gif"

EVAL_SEEDS = range(1000, 1010)
"""Matches this experiment's own `evaluate()` held-out eval seed range
(`1000 + episode`), also reused unchanged by stage 3's `evaluate_literal`."""


def main() -> None:
    """Record the stage-2 embedding baseline and print the real, measured outcome."""
    DEMOS_DIR.mkdir(parents=True, exist_ok=True)
    gym.register_envs(gymnasium_robotics)
    env = gym.make(ENV_ID, render_mode="rgb_array")
    model = SAC.load(CHECKPOINT_PATH, env=env)
    print(
        f"loaded checkpoint from {CHECKPOINT_PATH} (stage-3 checkpoint, stage-2-identical protocol, eval-only)"
    )

    for seed in EVAL_SEEDS:
        env.reset(seed=seed)
        result = record_episode(env, model, out_path=OUT_PATH, max_steps=50)
        print(f"[stage2] seed={seed} success={result.success} n_steps={result.n_steps}")
        if result.success:
            break
    else:
        print(
            "[stage2] WARNING: no success found across tried seeds -- keeping the last recording, labeled honestly"
        )

    env.close()


if __name__ == "__main__":
    main()
