"""Stage 4, attempt 2 fix: retrain `LanguageGoalProjection` on the 70-sentence augmented vocabulary.

Attempt 1 (see `report.md`'s top-level section, preserved verbatim) diagnosed
`LanguageGoalProjection` (384->64->16, ~25,600 parameters) as overfit to
`goal_region_vocabulary.ALL_INSTRUCTIONS` -- exactly 14 fixed sentences, 2 per
region -- with a zero-training nearest-neighbor ceiling test independently
confirming the raw sentence-embedding space already carries region-clustering
signal the MLP was discarding (0.714 NN-ceiling vs. 0.286 trained-MLP accuracy
on the same 14 held-out phrases). The reviewer's recommended fix, in order,
was: NN-ceiling test (done, attempt 1) -> data augmentation (this script) ->
smoothness regularization only if augmentation alone isn't enough.

This script is the data-augmentation half: it is `experiments/
03_language_goal_projection/train_projection.py`'s script with exactly one
substantive change -- the training vocabulary is
`augmented_training_vocabulary.AUGMENTED_INSTRUCTIONS` (70 sentences, 10 per
region) instead of `goal_region_vocabulary.ALL_INSTRUCTIONS` (14 sentences,
2 per region). Every hyperparameter (`n_steps=2000`, `learning_rate=1e-3`,
`n_target_samples=1000`, `box=MEASURED_GOAL_BOX`, `seed=0`) is unchanged from
the training run that produced `language_goal_projection_v3.pt` (the
checkpoint stage 3's passing attempt 4 and stage 4's attempt 1 both used --
see `report.md`'s attempt-3 section: "Loss ... dropped ... over 2000 steps",
confirming `train_projection.py`'s `N_STEPS=2000` default is what actually
trained v3). Reusing v3's hyperparameters isolates the retest to the one
variable stage 4's diagnosis actually implicates (vocabulary size/diversity),
not a confounded simultaneous change to how the projection is optimized.

Saved to a new filename (`language_goal_projection_v5_augmented.pt`) rather
than overwriting v3 -- v3 stays available for provenance/comparison, per
`.claude/agents/experiment-runner.md`'s reuse-checkpoints rule.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lang_goal_rl.augmented_training_vocabulary import AUGMENTED_INSTRUCTIONS, augmented_instruction_to_region
from lang_goal_rl.goal_encoder import GoalEncoder
from lang_goal_rl.language_embedding import encode_instructions
from lang_goal_rl.language_goal_projection import train_projection

EXPERIMENT_DIR = Path(__file__).parent
STAGE2_ENCODER_PATH = (
    EXPERIMENT_DIR.parent / "02_contrastive_goal_embedding" / "artifacts" / "goal_encoder.pt"
)
DEFAULT_OUT_PATH = EXPERIMENT_DIR / "artifacts" / "language_goal_projection_v5_augmented.pt"

PROJECTION_SEED = 0
"""Matches `train_projection.py`'s `PROJECTION_SEED` -- the checkpoint this
script's output is compared against (`language_goal_projection_v3.pt`) used
the same seed. Unchanged here since nothing about this retest's hypothesis
(vocabulary size/diversity) implicates weight-init or sampling seed."""

N_STEPS = 2_000
"""Matches `train_projection.py`'s `N_STEPS` -- confirmed (via
`report.md`'s attempt-3 section) to be the actual step count that trained
`language_goal_projection_v3.pt`. Unchanged so this retest isolates the one
variable being tested (vocabulary), not step count."""


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
    """Train the projection on the 70-sentence augmented vocabulary and save its checkpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-path", type=Path, default=STAGE2_ENCODER_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    args = parser.parse_args()

    torch.manual_seed(PROJECTION_SEED)

    encoder = load_frozen_encoder(args.encoder_path)
    print(f"loaded frozen stage-2 goal encoder from {args.encoder_path}")

    instructions = list(AUGMENTED_INSTRUCTIONS)
    mapping = augmented_instruction_to_region()
    region_names_row_aligned = [mapping[instruction] for instruction in instructions]
    sentence_embeddings = torch.from_numpy(encode_instructions(instructions))
    print(f"encoded {len(instructions)} augmented instructions -> shape {tuple(sentence_embeddings.shape)}")

    projection, loss_history = train_projection(
        encoder,
        sentence_embeddings,
        region_names_row_aligned,
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
