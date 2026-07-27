"""Stage 4, attempt 4, RL half: zero-training k-NN lookup as the projection, measuring actual held-out RL success.

Attempt 3's reviewer verdict: rather than running a cleaner tolerance
diagnostic, swap `LanguageGoalProjection`'s trained MLP out entirely and
measure held-out RL success using the already-built, zero-training
`nearest_neighbor_projection` (`nearest_neighbor_projection.py`) as the
inference-time mapping -- using the combined 84-sentence reference set
(`combined_vocabulary.py`: 14 original + 70 augmented) so every known
sentence is available as a reference point at once, fixing attempt 2's
accidental replace-not-extend bug in the same step. k=1 returns an *exact*
known training target, never a learned, potentially direction-distorted
approximation; if it substantially beats the MLP's attempt-2 RL success
(0.095 mean), that proves the MLP's own directional distortion -- not policy
tolerance -- was the real bottleneck. k=3 is also run (checked, not assumed
to be worse, per the task brief) since the earlier 14-sentence ceiling test
found k=1 best for *classification*, which does not automatically mean k=1
is also best for *RL success* on the larger 84-sentence reference.

No retraining anywhere in this script: reuses the same 3 already-trained
stage-3 SAC checkpoints (`03_language_goal_projection/checkpoints/
seed_{0,1,2}.zip`) attempts 1-3 used, and the same eval protocol (`train.py`'s
`evaluate_literal`/`evaluate_language_goal`, ground truth judged against each
instruction's true region centroid, per `ROADMAP.md`'s region-vs-point
lesson) -- this run is one CLI invocation per seed, matching `eval_held_out.
py`'s convention, so `--seed` selects which stage-3 checkpoint to load.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from stable_baselines3 import SAC

EXPERIMENT_DIR = Path(__file__).parent
STAGE3_DIR = EXPERIMENT_DIR.parent / "03_language_goal_projection"
sys.path.insert(0, str(STAGE3_DIR))

from train import (  # noqa: E402 -- STAGE3_DIR must be on sys.path first, see module docstring
    ENV_ID,
    evaluate_language_goal,
    evaluate_literal,
)

from combined_vocabulary import build_combined_reference, combined_instructions_and_regions, load_frozen_encoder
from lang_goal_rl.held_out_paraphrases import held_out_region_names, held_out_texts
from lang_goal_rl.language_embedding import encode_instructions
from lang_goal_rl.nearest_neighbor_projection import nearest_neighbor_projection

DEFAULT_CHECKPOINT_DIR = STAGE3_DIR / "checkpoints"

K_VALUES: tuple[int, ...] = (1, 3)
"""k=1 is the reviewer-specified setup; k=3 is checked here rather than
assumed worse, per the task brief's "don't assume" instruction -- see module
docstring."""

ATTEMPT4_EVAL_BASE_SEED = 13_000
"""Held-out eval seeds for this attempt, offset from attempt-1/2's
`HELD_OUT_EVAL_BASE_SEED` (9000, `eval_held_out.py`) and stage 3's
`LANGUAGE_EVAL_BASE_SEED`/`LITERAL_EVAL_BASE_SEED` (5000/1000) so this
script's env-reset seeds never collide with an earlier attempt's. Instruction
`i` under k-value index `k_index` (in `K_VALUES` order) uses
`ATTEMPT4_EVAL_BASE_SEED + k_index * n_instructions * n_episodes + i *
n_episodes`, so no two (k, instruction) pairs share reset seeds either."""


def main() -> None:
    """Load one seed's saved SAC checkpoint and evaluate it on all 14 held-out paraphrases, for every k in K_VALUES."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--language-eval-episodes", type=int, default=50)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint or (DEFAULT_CHECKPOINT_DIR / f"seed_{args.seed}.zip")

    instructions, _regions = combined_instructions_and_regions()
    n_unique = len(set(instructions))
    if n_unique != len(instructions):
        msg = (
            f"combined reference set has {len(instructions) - n_unique} duplicate instruction(s) -- "
            "expected 14 original + 70 augmented to be string-disjoint (verified in attempt 2)"
        )
        raise AssertionError(msg)
    print(f"combined reference set: {len(instructions)} instructions ({n_unique} unique, 14 original + 70 augmented)")

    gym.register_envs(gymnasium_robotics)
    env = gym.make(ENV_ID)

    model = SAC.load(checkpoint_path, env=env)
    print(f"loaded checkpoint from {checkpoint_path} (no new training, seed={args.seed})")

    literal_success_rate = evaluate_literal(model, env, args.eval_episodes)
    print(f"success_rate={literal_success_rate:.3f} over {args.eval_episodes} episodes")

    encoder = load_frozen_encoder()
    reference_raw_embeddings, reference_targets = build_combined_reference(encoder)

    texts = held_out_texts()
    regions = held_out_region_names()
    held_out_raw_embeddings = encode_instructions(texts)
    n_instructions = len(texts)

    for k_index, k in enumerate(K_VALUES):
        predicted = np.stack(
            [
                nearest_neighbor_projection(query, reference_raw_embeddings, reference_targets, k=k)
                for query in held_out_raw_embeddings
            ],
        )
        for index, (instruction, region_name) in enumerate(zip(texts, regions, strict=True)):
            projected_embedding = torch.from_numpy(predicted[index]).float()
            base_seed = (
                ATTEMPT4_EVAL_BASE_SEED
                + k_index * n_instructions * args.language_eval_episodes
                + index * args.language_eval_episodes
            )
            language_success_rate = evaluate_language_goal(
                model,
                env,
                region_name=region_name,
                projected_embedding=projected_embedding,
                n_episodes=args.language_eval_episodes,
                base_seed=base_seed,
            )
            print(
                f'k={k} language_success_rate={language_success_rate:.3f} instruction="{instruction}" '
                f'region="{region_name}" over {args.language_eval_episodes} episodes',
            )

    env.close()


if __name__ == "__main__":
    main()
