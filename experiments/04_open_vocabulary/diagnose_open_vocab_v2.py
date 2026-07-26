"""Stage 4, attempt 2: rerun Parts 1 and 3 against the augmented-vocabulary projection.

Same two diagnostics as `diagnose_open_vocab.py`, with exactly one change:
the projection checkpoint loaded is `artifacts/language_goal_projection_v5_augmented.pt`
(trained on `augmented_training_vocabulary.AUGMENTED_INSTRUCTIONS`, 70
sentences) instead of `language_goal_projection_v3.pt` (trained on
`goal_region_vocabulary.ALL_INSTRUCTIONS`, 14 sentences). Nothing about the
evaluation logic changes -- only which projection is loaded and, for Part 1,
which reference set its "own projected embeddings" means.

**Part 1 (`run_semantic_neighbor_diagnostic_v2`)**: mirrors attempt 1's
`run_semantic_neighbor_diagnostic` reference-set choice exactly (the training
instructions' own projected embeddings, not region centroids -- see that
function's docstring for why), but "the training instructions" now means the
70 `AUGMENTED_INSTRUCTIONS` this projection was actually trained on, not the
original 14. The query set (14 `HELD_OUT_PARAPHRASES`) is unchanged --
disjoint from both vocabularies, so this stays the same apples-to-apples test
attempt 1 ran, just against a projection trained on more data.

**Part 3 (`run_compositional_diagnostic`)**: reused unchanged from
`diagnose_open_vocab.py` -- it takes the projection and encoder as arguments
and has no dependency on which vocabulary trained the projection. Centroids
are still computed via `goal_region_vocabulary.compute_region_target_embeddings`
against the canonical 7 `region_names()`, at the same `(n_samples=1000,
seed=0)` pair used throughout stages 3/4, so compositional placement is
judged against the same fixed centroids attempt 1 used.
"""

from __future__ import annotations

from pathlib import Path

import torch

from diagnose_open_vocab import (
    CENTROID_N_SAMPLES,
    CENTROID_SEED,
    STAGE2_ENCODER_PATH,
    load_frozen_encoder,
    load_projection,
    run_compositional_diagnostic,
)
from lang_goal_rl.augmented_training_vocabulary import AUGMENTED_INSTRUCTIONS, augmented_instruction_to_region
from lang_goal_rl.held_out_paraphrases import held_out_region_names, held_out_texts
from lang_goal_rl.language_embedding import encode_instructions
from lang_goal_rl.language_goal_projection import LanguageGoalProjection
from lang_goal_rl.semantic_neighbor_diagnostic import (
    CompositionalPlacement,
    SemanticNeighborReport,
    diagnose_semantic_neighbors,
)

EXPERIMENT_DIR = Path(__file__).parent
AUGMENTED_PROJECTION_PATH = EXPERIMENT_DIR / "artifacts" / "language_goal_projection_v5_augmented.pt"

__all__ = ["CENTROID_N_SAMPLES", "CENTROID_SEED", "run_compositional_diagnostic"]
"""Re-exported for `generate_report_attempt2.py`, which reuses attempt 1's
centroid constants and the unchanged compositional-diagnostic function
rather than redefining them."""


def run_semantic_neighbor_diagnostic_v2(projection: LanguageGoalProjection) -> SemanticNeighborReport:
    """Classify each held-out paraphrase's projected embedding by nearest *augmented*-training instruction.

    Args:
        projection: The augmented-vocabulary-trained `LanguageGoalProjection`
            (see `load_projection`).

    Returns:
        A `SemanticNeighborReport` covering all 14 held-out paraphrases.

    """
    train_sentence_embeddings = torch.from_numpy(encode_instructions(AUGMENTED_INSTRUCTIONS))
    with torch.no_grad():
        train_projected = projection(train_sentence_embeddings)
    mapping = augmented_instruction_to_region()
    train_regions = [mapping[instruction] for instruction in AUGMENTED_INSTRUCTIONS]

    held_out_sentence_embeddings = torch.from_numpy(encode_instructions(held_out_texts()))
    with torch.no_grad():
        held_out_projected = projection(held_out_sentence_embeddings)

    return diagnose_semantic_neighbors(
        query_embeddings=held_out_projected,
        query_instructions=held_out_texts(),
        query_true_region_names=held_out_region_names(),
        reference_embeddings=train_projected,
        reference_region_names=train_regions,
    )


def _format_compositional(placement: CompositionalPlacement) -> str:
    """Render one `CompositionalPlacement` as a log-friendly line, mirroring `diagnose_open_vocab.py`'s helper."""
    component_flag = "component" if placement.nearest_is_component else "NEITHER component"
    return (
        f'  {placement.instruction!r} -> nearest={placement.nearest_region_name!r} ({component_flag}) '
        f"components={placement.component_region_names} balance={placement.component_distance_balance:.3f} "
        f"distances={placement.distances_by_region}"
    )


def main() -> None:
    """Run both diagnostics against the augmented-vocabulary projection and print a log-friendly readout."""
    projection = load_projection(AUGMENTED_PROJECTION_PATH)
    encoder = load_frozen_encoder(STAGE2_ENCODER_PATH)

    neighbor_report = run_semantic_neighbor_diagnostic_v2(projection)
    print(neighbor_report.summary())

    print()
    print("CompositionalPlacementReport:")
    for placement in run_compositional_diagnostic(projection, encoder):
        print(_format_compositional(placement))


if __name__ == "__main__":
    main()
