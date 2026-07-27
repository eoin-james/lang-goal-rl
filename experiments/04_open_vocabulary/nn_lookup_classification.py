"""Stage 4, attempt 4, classification half: k-NN lookup over the combined 84-sentence reference set.

Attempt 3's reviewer verdict recommended the decisive next test: swap
`LanguageGoalProjection`'s trained MLP for the zero-training
`nearest_neighbor_projection` (k=1), drawing on the combined 84-sentence
reference set (14 original + 70 augmented, fixing attempt 2's accidental
replace-not-extend bug in the same step) instead of either vocabulary alone.
This script is the classification half (no RL, no SAC policy): for each of
`held_out_paraphrases.HELD_OUT_PARAPHRASES`' 14 held-out phrases, blend the
`k` nearest of the 84 reference sentences' targets in raw 384-dim
sentence-embedding space, then classify the blended 16-dim point by nearest
region centroid -- the same geometry check `nn_ceiling_test.py` ran over the
original 14-sentence reference set, generalized to the combined 84.

Both k=1 (the reviewer's specified setup) and k=3 (checked, not assumed, per
the task brief -- k=1 was the best classifier over the 14-sentence set but
that does not automatically transfer to an 84-sentence reference) are run
and reported in full.
"""

from __future__ import annotations

import numpy as np
import torch

from combined_vocabulary import (
    CENTROID_N_SAMPLES,
    CENTROID_SEED,
    build_combined_reference,
    combined_instructions_and_regions,
    load_frozen_encoder,
)
from lang_goal_rl.goal_region_vocabulary import compute_region_target_embeddings, region_names
from lang_goal_rl.held_out_paraphrases import held_out_region_names, held_out_texts
from lang_goal_rl.language_embedding import encode_instructions
from lang_goal_rl.nearest_neighbor_projection import nearest_neighbor_projection
from lang_goal_rl.semantic_neighbor_diagnostic import SemanticNeighborReport, diagnose_semantic_neighbors

K_VALUES: tuple[int, ...] = (1, 3)
"""k=1 is the reviewer-specified setup (best classifier over the 14-sentence
reference in the earlier ceiling test); k=3 is checked here rather than
assumed to still be worse over the larger 84-sentence reference, per the
task brief's "don't assume" instruction."""

PRIOR_BASELINES = (
    "attempt-1 MLP (14-sentence vocab) = 0.286 (4/14)",
    "attempt-2 MLP (70-sentence vocab) = 0.643 (9/14)",
    "14-sentence NN-ceiling, k=1 = 0.714 (10/14)",
)


def classify_via_nn_lookup(
    k: int,
    held_out_raw_embeddings: np.ndarray,
    reference_raw_embeddings: np.ndarray,
    reference_targets: np.ndarray,
    centroid_embeddings: torch.Tensor,
    region_name_list: tuple[str, ...],
) -> SemanticNeighborReport:
    """Blend each held-out paraphrase via `nearest_neighbor_projection` over the 84-sentence reference, then classify.

    Args:
        k: Number of nearest reference sentences to blend.
        held_out_raw_embeddings: Held-out paraphrases' raw 384-dim sentence
            embeddings, shape (14, 384).
        reference_raw_embeddings: Combined 84-sentence reference set's raw
            384-dim sentence embeddings.
        reference_targets: Combined reference set's fixed 16-dim region-
            centroid targets, row-aligned with `reference_raw_embeddings`.
        centroid_embeddings: One 16-dim centroid per region, shape (7, 16).
        region_name_list: Region label for each row of `centroid_embeddings`.

    Returns:
        A `SemanticNeighborReport` over all 14 held-out paraphrases.

    """
    predicted = np.stack(
        [
            nearest_neighbor_projection(query, reference_raw_embeddings, reference_targets, k=k)
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


def main() -> None:
    """Run the combined-84-sentence NN classification check for every k in `K_VALUES`."""
    instructions, _regions = combined_instructions_and_regions()
    n_unique = len(set(instructions))
    if n_unique != len(instructions):
        msg = (
            f"combined reference set has {len(instructions) - n_unique} duplicate instruction(s) -- "
            "expected 14 original + 70 augmented to be string-disjoint (verified in attempt 2)"
        )
        raise AssertionError(msg)
    print(f"combined reference set: {len(instructions)} instructions ({n_unique} unique, 14 original + 70 augmented)")

    encoder = load_frozen_encoder()
    reference_raw_embeddings, reference_targets = build_combined_reference(encoder)

    names = region_names()
    centroid_embeddings = compute_region_target_embeddings(
        encoder, names, n_samples=CENTROID_N_SAMPLES, seed=CENTROID_SEED,
    )

    held_out_raw_embeddings = encode_instructions(held_out_texts())

    print("Nearest-neighbor classification over the COMBINED 84-sentence reference set (attempt 4, zero-training)")
    print("Prior baselines (for comparison, not re-derived here):")
    for baseline in PRIOR_BASELINES:
        print(f"  {baseline}")
    print()

    for k in K_VALUES:
        report = classify_via_nn_lookup(
            k, held_out_raw_embeddings, reference_raw_embeddings, reference_targets, centroid_embeddings, names,
        )
        print(f"=== k={k} ===")
        print(report.summary())
        print()


if __name__ == "__main__":
    main()
