"""Fail-fast check (stage-3 scale-fix retest): does a trained `LanguageGoalProjection`'s
output land within the frozen `GoalEncoder`'s real operating-norm range?

This is the check the stage-3 FAIL reviewer asked for (see
`experiments/03_language_goal_projection/report.md`'s recommendation #2):
run `measure_reference_norms` + `check_projection_norm_range` immediately
after `train_projection` returns, before spending any RL evaluation budget
on a projection that's already known to be off-distribution. Unlike the
original diagnostic (`debug_language_eval.py`), this script's whole purpose
is to have its stdout redirected to a log file — the previous diagnostic's
output was never captured, which the reviewer flagged as an evidence gap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lang_goal_rl.goal_encoder import GoalEncoder
from lang_goal_rl.goal_region_vocabulary import ALL_INSTRUCTIONS
from lang_goal_rl.language_embedding import encode_instructions
from lang_goal_rl.language_goal_projection import (
    LanguageGoalProjection,
    check_projection_norm_range,
    measure_reference_norms,
)

STAGE2_ENCODER_PATH = (
    Path(__file__).parent.parent / "02_contrastive_goal_embedding" / "artifacts" / "goal_encoder.pt"
)
DEFAULT_PROJECTION_PATH = Path(__file__).parent / "artifacts" / "language_goal_projection_v2_fixed.pt"


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


def main() -> None:
    """Run the fail-fast norm-range check and print its `.summary()`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-path", type=Path, default=STAGE2_ENCODER_PATH)
    parser.add_argument("--projection-path", type=Path, default=DEFAULT_PROJECTION_PATH)
    args = parser.parse_args()

    encoder = load_frozen_encoder(args.encoder_path)
    projection = load_projection(args.projection_path)

    instructions = list(ALL_INSTRUCTIONS)
    sentence_embeddings = torch.from_numpy(encode_instructions(instructions))
    with torch.no_grad():
        projected = projection(sentence_embeddings)

    reference_norms = measure_reference_norms(encoder)
    result = check_projection_norm_range(projected, reference_norms, instructions)
    print(result.summary())
    print(f"PASSED={result.passed}")


if __name__ == "__main__":
    main()
