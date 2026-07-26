"""Re-run stage 3's collapse diagnostic against the trained projection checkpoint.

Independent re-verification of the builder's reported result — loads the
saved `LanguageGoalProjection` checkpoint fresh and calls
`instruction_collapse_diagnostic.check_no_collapse` directly, rather than
trusting the number the builder cited during the build.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lang_goal_rl.goal_encoder import GoalEncoder
from lang_goal_rl.goal_region_vocabulary import ALL_INSTRUCTIONS, instruction_to_region
from lang_goal_rl.instruction_collapse_diagnostic import check_no_collapse
from lang_goal_rl.language_embedding import encode_instructions
from lang_goal_rl.language_goal_projection import LanguageGoalProjection

STAGE2_ENCODER_PATH = (
    Path(__file__).parent.parent / "02_contrastive_goal_embedding" / "artifacts" / "goal_encoder.pt"
)
DEFAULT_PROJECTION_PATH = Path(__file__).parent / "artifacts" / "language_goal_projection.pt"


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
    """Load the trained projection + frozen encoder and print the collapse-diagnostic result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-path", type=Path, default=STAGE2_ENCODER_PATH)
    parser.add_argument("--projection-path", type=Path, default=DEFAULT_PROJECTION_PATH)
    args = parser.parse_args()

    encoder = load_frozen_encoder(args.encoder_path)
    projection = load_projection(args.projection_path)

    instructions = list(ALL_INSTRUCTIONS)
    region_names = [instruction_to_region(instruction) for instruction in instructions]
    sentence_embeddings = torch.from_numpy(encode_instructions(instructions))

    report = check_no_collapse(projection, encoder, sentence_embeddings, instructions, region_names)

    print(f"n_instructions={len(instructions)} n_distinct_regions={len(set(region_names))}")
    print(f"min_pairwise_distance={report.min_pairwise_distance:.6f}")
    print(f"mean_pairwise_distance={report.mean_pairwise_distance:.6f}")
    print(f"min_cross_region_pairwise_distance={report.min_cross_region_pairwise_distance:.6f}")
    print(f"collapse_epsilon={report.collapse_epsilon:.6f}")
    ratio = report.min_cross_region_pairwise_distance / report.collapse_epsilon
    print(f"min_cross_region_distance / collapse_epsilon = {ratio:.2f}x")
    print(f"is_collapsed={report.is_collapsed}")


if __name__ == "__main__":
    main()
