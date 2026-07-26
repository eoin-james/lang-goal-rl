"""Stage 3 scale-fix retest: re-run the language-goal substitution eval against the
scale-fixed `LanguageGoalProjection`, reusing the 3 already-trained SAC checkpoints
from `checkpoints/seed_<k>.zip` -- **no new RL training happens in this script**.

The stage-3 FAIL (see `report.md`) was root-caused to the *projection's* output
norm sitting 5-10x outside the frozen `GoalEncoder`'s real operating range, not
to anything wrong with the trained SAC policies (their literal-goal eval was a
clean 1.000 on all 3 seeds). Per `.claude/agents/CONTRACTS.md`'s reuse-checkpoints
rule, retraining those policies again to re-test a fix that only touches the
projection layer would be pure waste -- this script loads each seed's saved
policy and evaluates it, exactly like `train.py`'s eval phase, but skips
`model.learn(...)` entirely and swaps in the new projection checkpoint.

Reuses `train.py`'s eval helpers (`evaluate_literal`, `evaluate_language_goal`,
`load_frozen_encoder`, `load_projection`) rather than duplicating them, since
this is the identical protocol against a different projection checkpoint --
any drift between the two would make the before/after comparison meaningless.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import torch
from stable_baselines3 import SAC

from train import (
    ALL_INSTRUCTIONS,
    DEFAULT_ENCODER_PATH,
    ENV_ID,
    LANGUAGE_EVAL_BASE_SEED,
    encode_instructions,
    evaluate_language_goal,
    evaluate_literal,
    instruction_to_region,
    load_frozen_encoder,
    load_projection,
)

EXPERIMENT_DIR = Path(__file__).parent
DEFAULT_CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"
DEFAULT_PROJECTION_PATH = EXPERIMENT_DIR / "artifacts" / "language_goal_projection_v2_fixed.pt"


def main() -> None:
    """Load one seed's saved SAC checkpoint and re-run both eval protocols against the fixed projection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--language-eval-episodes", type=int, default=50)
    parser.add_argument("--encoder-path", type=Path, default=DEFAULT_ENCODER_PATH)
    parser.add_argument("--projection-path", type=Path, default=DEFAULT_PROJECTION_PATH)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint or (DEFAULT_CHECKPOINT_DIR / f"seed_{args.seed}.zip")

    gym.register_envs(gymnasium_robotics)
    env = gym.make(ENV_ID)

    encoder = load_frozen_encoder(args.encoder_path)
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
