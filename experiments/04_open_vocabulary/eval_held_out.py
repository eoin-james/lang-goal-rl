"""Stage 4, part 2: RL success rate on held-out paraphrases -- the actual generalization test.

Stage 4's proof gate is "graceful degradation on unseen phrasing", and this
script is the half of it that needs the trained policy in the loop (part 1,
`diagnose_open_vocab.py`, covers the embedding-geometry half without any RL).

Per `.claude/agents/CONTRACTS.md`'s reuse-checkpoints rule and this stage's
explicit brief: **no retraining happens here**. This loads one of stage 3's
already-trained SAC checkpoints (`experiments/03_language_goal_projection/
checkpoints/seed_<k>.zip`) and stage 3's final, fixed-centroid-regression
projection checkpoint (`.../artifacts/language_goal_projection_v3.pt`,
attempt 3's checkpoint -- unchanged by attempt 4's eval-protocol fix, see
`report.md`), then evaluates 14 held-out paraphrases
(`held_out_paraphrases.HELD_OUT_PARAPHRASES`) that were never fed to
`train_projection`.

The eval protocol is identical to stage 3's attempt 4 (the fix for the
region-vs-point lesson in `ROADMAP.md`'s Known risks): ground truth is
`compute_region_centroid(region_name)` -- a fixed xyz point, precomputed
once per region and reused for every episode of a given instruction -- not
a freshly resampled point. Applying that lesson from the start here (rather
than rediscovering it) is why this script imports stage 3's `train.py`
helpers (`compute_region_centroid`, `evaluate_language_goal`,
`evaluate_literal`, `load_projection`) instead of reimplementing the eval
loop. Stage 3's directory is added to `sys.path` at import time since this
script lives in a different experiment directory
(`experiments/04_open_vocabulary/`) than `train.py`
(`experiments/03_language_goal_projection/`) -- unlike
`eval_fixed_projection.py`, which could `from train import ...` directly
because it sits next to `train.py`.
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

from lang_goal_rl.held_out_paraphrases import held_out_region_names, held_out_texts
from lang_goal_rl.language_embedding import encode_instructions

DEFAULT_CHECKPOINT_DIR = STAGE3_DIR / "checkpoints"
DEFAULT_PROJECTION_PATH = STAGE3_DIR / "artifacts" / "language_goal_projection_v3.pt"

HELD_OUT_EVAL_BASE_SEED = 9000
"""Held-out eval seeds, offset from stage 3's `LANGUAGE_EVAL_BASE_SEED` (5000)
and `LITERAL_EVAL_BASE_SEED` (1000) so this script's env-reset seeds never
collide with a stage-3 seed used for a different instruction/protocol.
Instruction `i` (in `held_out_paraphrases.held_out_texts()` order) uses
`HELD_OUT_EVAL_BASE_SEED + i * n_episodes`, so no two held-out instructions
share reset seeds either."""


def main() -> None:
    """Load one seed's saved SAC checkpoint and evaluate it on all 14 held-out paraphrases."""
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
    texts = held_out_texts()
    regions = held_out_region_names()
    for index, (instruction, region_name) in enumerate(zip(texts, regions, strict=True)):
        sentence_embedding = torch.from_numpy(encode_instructions([instruction]))
        with torch.no_grad():
            projected_embedding = projection(sentence_embedding).squeeze(0)

        base_seed = HELD_OUT_EVAL_BASE_SEED + index * args.language_eval_episodes
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
