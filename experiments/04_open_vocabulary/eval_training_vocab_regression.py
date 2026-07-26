"""Stage 4, attempt 2: regression check -- does the retrained projection still ace the ORIGINAL stage-3 vocabulary?

Attempt 2's fix retrains `LanguageGoalProjection` on the 70-sentence
`augmented_training_vocabulary`, not `goal_region_vocabulary.ALL_INSTRUCTIONS`
(the original 14-sentence vocabulary stage 3 was built and passed against).
Before trusting the held-out numbers from `eval_held_out.py` (run against the
new projection), this script checks the augmentation fix didn't regress what
was already working: does the new projection still reproduce stage 3's
attempt-4 ~1.000 success rate on the *original* 14 training instructions?

Same eval protocol as `eval_held_out.py` (imported from stage 3's `train.py`:
`evaluate_literal`, `evaluate_language_goal`, `load_projection`), just over
`goal_region_vocabulary.ALL_INSTRUCTIONS` instead of
`held_out_paraphrases.held_out_texts()`, and using the new
`language_goal_projection_v5_augmented.pt` checkpoint instead of v3. No new
RL training -- reuses stage 3's already-trained SAC checkpoints, same as
`eval_held_out.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import torch
from stable_baselines3 import SAC

EXPERIMENT_DIR = Path(__file__).parent
STAGE3_DIR = EXPERIMENT_DIR.parent / "03_language_goal_projection"
sys.path.insert(0, str(STAGE3_DIR))

from train import (  # noqa: E402 -- STAGE3_DIR must be on sys.path first, see module docstring
    ENV_ID,
    evaluate_language_goal,
    evaluate_literal,
    load_projection,
)

from lang_goal_rl.goal_region_vocabulary import ALL_INSTRUCTIONS, instruction_to_region
from lang_goal_rl.language_embedding import encode_instructions

DEFAULT_CHECKPOINT_DIR = STAGE3_DIR / "checkpoints"
DEFAULT_PROJECTION_PATH = EXPERIMENT_DIR / "artifacts" / "language_goal_projection_v5_augmented.pt"

REGRESSION_EVAL_BASE_SEED = 7000
"""Distinct from stage 3's `LITERAL_EVAL_BASE_SEED` (1000) and
`LANGUAGE_EVAL_BASE_SEED` (5000), and from `eval_held_out.py`'s
`HELD_OUT_EVAL_BASE_SEED` (9000) -- so this script's env-reset seeds never
collide with any other stage-3/4 eval script's seed range."""


def main() -> None:
    """Load one seed's saved SAC checkpoint and evaluate the new projection on the ORIGINAL 14 training instructions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--language-eval-episodes", type=int, default=50)
    parser.add_argument("--projection-path", type=Path, default=DEFAULT_PROJECTION_PATH)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint or (DEFAULT_CHECKPOINT_DIR / f"seed_{args.seed}.zip")

    gym.register_envs(gymnasium_robotics)
    env = gym.make(ENV_ID)

    model = SAC.load(checkpoint_path, env=env)
    print(f"loaded checkpoint from {checkpoint_path} (no new training, seed={args.seed})")

    literal_success_rate = evaluate_literal(model, env, args.eval_episodes)
    print(f"success_rate={literal_success_rate:.3f} over {args.eval_episodes} episodes")

    projection = load_projection(args.projection_path)
    for region_index, instruction in enumerate(ALL_INSTRUCTIONS):
        region_name = instruction_to_region(instruction)
        sentence_embedding = torch.from_numpy(encode_instructions([instruction]))
        with torch.no_grad():
            projected_embedding = projection(sentence_embedding).squeeze(0)

        base_seed = REGRESSION_EVAL_BASE_SEED + region_index * args.language_eval_episodes
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
