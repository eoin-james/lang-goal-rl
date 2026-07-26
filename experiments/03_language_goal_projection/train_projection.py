"""Stage 3 projection pretraining: train LanguageGoalProjection once, shared across all RL seeds.

Mirrors stage 2's `pretrain_encoder.py` pattern: this is run exactly once,
before any RL seed, and every RL training/eval seed in this stage loads the
resulting checkpoint unchanged — RL seed variance must never be confounded
with projection-training variance (same reasoning as stage 2's frozen
`GoalEncoder`).

Depends on stage 2's frozen `GoalEncoder` checkpoint
(`experiments/02_contrastive_goal_embedding/artifacts/goal_encoder.pt`),
loaded read-only here and never modified — `train_projection` freezes it
internally too, but loading it fresh in this process makes that explicit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lang_goal_rl.goal_encoder import GoalEncoder
from lang_goal_rl.goal_region_vocabulary import ALL_INSTRUCTIONS, instruction_to_region
from lang_goal_rl.language_embedding import encode_instructions
from lang_goal_rl.language_goal_projection import LanguageGoalProjection, train_projection

STAGE2_ENCODER_PATH = (
    Path(__file__).parent.parent / "02_contrastive_goal_embedding" / "artifacts" / "goal_encoder.pt"
)
DEFAULT_OUT_PATH = Path(__file__).parent / "artifacts" / "language_goal_projection.pt"

PROJECTION_SEED = 0
"""Seed for the projection's weight init and region-sampling randomness during
training. Fixed and documented for the same reason as stage 2's
`PRETRAIN_SEED` — this projection is trained exactly once and then reused,
unchanged, across every RL seed."""

N_STEPS = 2_000
"""Optimizer steps. Matches stage 2 encoder pretraining's `N_PRETRAIN_STEPS`
order of magnitude — both are small MLPs regressed against a sampled target,
and 2000 steps was enough there to converge well past the point needed for
this stage's proof gate (see `check_no_collapse` in
`instruction_collapse_diagnostic.py` for the actual pass/fail measurement)."""


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


def main() -> None:
    """Train the language->goal-embedding projection once and save its checkpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-path", type=Path, default=STAGE2_ENCODER_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    args = parser.parse_args()

    torch.manual_seed(PROJECTION_SEED)

    encoder = load_frozen_encoder(args.encoder_path)
    print(f"loaded frozen stage-2 goal encoder from {args.encoder_path}")

    instructions = list(ALL_INSTRUCTIONS)
    region_names = [instruction_to_region(instruction) for instruction in instructions]
    sentence_embeddings = torch.from_numpy(encode_instructions(instructions))
    print(f"encoded {len(instructions)} fixed instructions -> shape {tuple(sentence_embeddings.shape)}")

    projection, loss_history = train_projection(
        encoder,
        sentence_embeddings,
        region_names,
        n_steps=args.n_steps,
        seed=PROJECTION_SEED,
    )

    early_mean = sum(loss_history[:20]) / 20
    late_mean = sum(loss_history[-20:]) / 20
    print(f"regression_loss early_mean(first 20 steps)={early_mean:.4f} late_mean(last 20 steps)={late_mean:.4f}")
    print(f"regression_loss final={loss_history[-1]:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_dim": projection.input_dim,
            "embed_dim": projection.embed_dim,
            "state_dict": projection.state_dict(),
        },
        args.out,
    )
    print(f"saved trained projection to {args.out}")


if __name__ == "__main__":
    main()
