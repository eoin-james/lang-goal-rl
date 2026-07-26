"""Stage 4 addendum: zero-training nearest-neighbor-interpolation ceiling test (reviewer-requested).

`diagnose_open_vocab.py`'s Part 1 found the trained `LanguageGoalProjection`
MLP (384->64->16, ~25,600 params, fit to only 14 training points) classifies
just 4/14 (28.6%) held-out paraphrases to their correct region. The reviewer
diagnosed this as memorization, not an information-theoretic ceiling on what
the raw `all-MiniLM-L6-v2` embedding space can support -- and recommended a
cheap check *before* committing to the real fix (data augmentation): bypass
the learned MLP entirely and see whether a plain distance-weighted
interpolation over the raw 384-dim sentence embeddings (`nearest_neighbor_
projection.py`, shipped by the rl-builder for exactly this test) beats 28.6%.

This script builds the same 14-instruction reference set the MLP was trained
on, blends each held-out paraphrase's raw embedding against its `k` nearest
training instructions, and classifies the resulting 16-dim goal-space point
against the 7 region centroids -- using the exact `(n_samples=1000, seed=0)`
pair stage 3/4 used throughout, so the centroids and regression targets here
are bit-identical to the ones the MLP was actually trained/evaluated
against, not a separately invented approximation.

No MLP, no training, no RL -- pure embedding-space geometry, matching
`diagnose_open_vocab.py`'s scope. Tries k=1, 3, 5 and reports all three (not
just whichever looks best) per the experiment-runner's no-cherry-picking
rule.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lang_goal_rl.goal_encoder import GoalEncoder
from lang_goal_rl.goal_region_vocabulary import (
    ALL_INSTRUCTIONS,
    compute_region_target_embeddings,
    instruction_to_region,
    region_names,
)
from lang_goal_rl.held_out_paraphrases import held_out_region_names, held_out_texts
from lang_goal_rl.language_embedding import encode_instructions
from lang_goal_rl.language_goal_projection import precompute_instruction_targets
from lang_goal_rl.nearest_neighbor_projection import nearest_neighbor_projection
from lang_goal_rl.semantic_neighbor_diagnostic import (
    SemanticNeighborReport,
    diagnose_semantic_neighbors,
)

EXPERIMENT_DIR = Path(__file__).parent
STAGE2_ENCODER_PATH = (
    EXPERIMENT_DIR.parent
    / "02_contrastive_goal_embedding"
    / "artifacts"
    / "goal_encoder.pt"
)

CENTROID_N_SAMPLES = 1000
"""Matches `language_goal_projection.DEFAULT_N_TARGET_SAMPLES` and
`diagnose_open_vocab.py`'s `CENTROID_N_SAMPLES` -- the sample count stage
3/4 actually used to build the embedding-space regression targets and region
centroids `language_goal_projection_v3.pt` was trained/evaluated against."""

CENTROID_SEED = 0
"""Matches `train_projection.PROJECTION_SEED` / `diagnose_open_vocab.py`'s
`CENTROID_SEED` -- reproduces the identical centroids stage 3/4 used."""

K_VALUES: tuple[int, ...] = (1, 3, 5)
"""Every k tried, reported in full -- not cherry-picked to whichever looks best."""

MLP_PART1_CORRECT: dict[str, bool] = {
    "settle into the middle of the workspace": False,
    "return your hand to a neutral position": False,
    "push your arm out in front of you": False,
    "extend forward away from your body": False,
    "draw your hand back toward yourself": False,
    "retreat away from the front of the workspace": False,
    "swing your arm over to the left": False,
    "shift your gripper toward the left edge": False,
    "swing your arm over to the right": False,
    "shift your gripper toward the right edge": True,
    "raise your arm as high as it will go": True,
    "extend upward toward the ceiling": True,
    "lower your arm toward the floor": False,
    "drop your gripper down low": True,
}
"""Verbatim per-instruction verdicts from `report.md`'s existing Part 1 table
(the trained MLP's classification of the same 14 held-out paraphrases), kept
here as a fixed comparison point so this script can report which
instructions flip between the MLP and the NN-interpolation baseline."""


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


def classify_via_nearest_neighbor(
    k: int,
    held_out_raw_embeddings: np.ndarray,
    reference_raw_embeddings: np.ndarray,
    reference_targets: np.ndarray,
    centroid_embeddings: torch.Tensor,
    region_name_list: tuple[str, ...],
) -> SemanticNeighborReport:
    """Blend each held-out paraphrase via `nearest_neighbor_projection`, then classify by nearest region centroid.

    Args:
        k: Number of nearest training instructions to blend.
        held_out_raw_embeddings: Held-out paraphrases' raw 384-dim sentence
            embeddings, shape (14, 384).
        reference_raw_embeddings: Training instructions' raw 384-dim sentence
            embeddings, shape (14, 384).
        reference_targets: Training instructions' fixed 16-dim regression
            targets (own region's centroid), row-aligned with
            `reference_raw_embeddings`, shape (14, 16).
        centroid_embeddings: One 16-dim centroid per region, shape (7, 16).
        region_name_list: Region label for each row of `centroid_embeddings`.

    Returns:
        A `SemanticNeighborReport` over all 14 held-out paraphrases.

    """
    predicted = np.stack(
        [
            nearest_neighbor_projection(
                query, reference_raw_embeddings, reference_targets, k=k
            )
            for query in held_out_raw_embeddings
        ],
    )
    predicted_embeddings = torch.from_numpy(predicted).float()

    return diagnose_semantic_neighbors(
        query_embeddings=predicted_embeddings,
        query_instructions=held_out_texts(),
        query_true_region_names=held_out_region_names(),
        reference_embeddings=centroid_embeddings,
        reference_region_names=region_name_list,
    )


def _format_flips(report: SemanticNeighborReport) -> list[str]:
    """List instructions whose correct/incorrect verdict differs from the MLP's Part-1 result."""
    lines = []
    for result in report.results:
        mlp_correct = MLP_PART1_CORRECT[result.instruction]
        nn_correct = bool(result.is_correct)
        if mlp_correct == nn_correct:
            continue
        direction = (
            "WRONG (MLP) -> correct (NN)"
            if nn_correct
            else "correct (MLP) -> WRONG (NN)"
        )
        lines.append(f"    FLIP: {result.instruction!r} [{direction}]")
    return lines


def main() -> None:
    """Run the ceiling test for every k in `K_VALUES` and print a log-friendly readout."""
    encoder = load_frozen_encoder(STAGE2_ENCODER_PATH)

    train_region_names = [
        instruction_to_region(instruction) for instruction in ALL_INSTRUCTIONS
    ]
    reference_raw_embeddings = encode_instructions(ALL_INSTRUCTIONS)
    reference_targets = precompute_instruction_targets(
        encoder,
        train_region_names,
        n_samples=CENTROID_N_SAMPLES,
        seed=CENTROID_SEED,
    ).numpy()

    names = region_names()
    centroid_embeddings = compute_region_target_embeddings(
        encoder,
        names,
        n_samples=CENTROID_N_SAMPLES,
        seed=CENTROID_SEED,
    )

    held_out_raw_embeddings = encode_instructions(held_out_texts())

    print(
        "Nearest-neighbor-interpolation ceiling test (zero-training, bypasses LanguageGoalProjection MLP)"
    )
    print(
        "Reference: 14 training instructions' raw sentence embeddings + their region-centroid targets"
    )
    print("MLP Part-1 baseline (report.md): accuracy=0.286 (4/14)")
    print()

    for k in K_VALUES:
        report = classify_via_nearest_neighbor(
            k,
            held_out_raw_embeddings,
            reference_raw_embeddings,
            reference_targets,
            centroid_embeddings,
            names,
        )
        print(f"=== k={k} ===")
        print(report.summary())
        flips = _format_flips(report)
        if flips:
            print("  Flips vs. MLP Part-1:")
            for line in flips:
                print(line)
        else:
            print("  No flips vs. MLP Part-1.")
        print()


if __name__ == "__main__":
    main()
