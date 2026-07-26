"""Stage 4, parts 1 and 3: semantic-neighbor diagnostic + compositional placement.

Neither part needs any RL: both only need the frozen sentence-transformer
and stage 3's trained `LanguageGoalProjection` (loaded unchanged, no
retraining -- `checkpoint["state_dict"]` from `language_goal_projection_v3.pt`
is only ever read here). Part 3 additionally needs stage 2's frozen
`GoalEncoder` to build region-centroid embeddings.

**Part 1 (`run_semantic_neighbor_diagnostic`)**: projects the 14 held-out
paraphrases (`held_out_paraphrases.HELD_OUT_PARAPHRASES`) through the
projection and classifies each by nearest region against a reference set.
Reference choice: the 14 *training* instructions'
(`goal_region_vocabulary.ALL_INSTRUCTIONS`) own projected embeddings, run
through the identical projection -- not the region centroids used to train
it. `semantic_neighbor_diagnostic.diagnose_semantic_neighbors`'s docstring
states either is a valid choice, since the classification itself is pure
geometry independent of which reference is used. Training instructions' own
projected embeddings are chosen here because stage 4's proof gate
("semantic neighbors land near each other in goal space") is a claim about
the projection *network's actual output geometry* for real encoded
sentences: whether an unseen phrasing's projected point sits closest to the
real points other real instructions' text produces, not to a separately
computed, idealized encoder-space average that no instruction's projection
ever exactly reproduces (attempt 3's report shows the projection converges
close to, but not exactly onto, its regression target).

**Part 3 (`run_compositional_diagnostic`)**: reports where each
compositional instruction's (`held_out_paraphrases.COMPOSITIONAL_INSTRUCTIONS`)
projected embedding lands relative to its two named component regions.
Unlike part 1, `diagnose_compositional_placement`'s interface specifically
requires region *centroid* embeddings -- there is no reference-choice
flexibility here. Centroids are computed via
`goal_region_vocabulary.compute_region_target_embeddings` against stage 2's
frozen `GoalEncoder`, using the exact `(n_samples, seed)` pair stage 3's
`train_projection.py` used to build the projection's own regression targets
(`n_samples=1000` matching `language_goal_projection.DEFAULT_N_TARGET_SAMPLES`,
`seed=0` matching `train_projection.PROJECTION_SEED`) -- so these are the
same points the projection was actually trained to match, not a separately
invented notion of "the region's center".

Per the stage-3 attempt-1 reviewer's documented lesson (save diagnostic
output to a log, don't just print and discard), `main()`'s output should be
redirected to `artifacts/semantic_neighbor_diagnostic_stdout.log` by the
caller. `generate_report.py` re-runs both diagnostics in-process (cheap,
deterministic, no RL) rather than parsing this log, so the saved log is
evidence, not the report's data source.
"""

from __future__ import annotations

from pathlib import Path

import torch

from lang_goal_rl.goal_encoder import GoalEncoder
from lang_goal_rl.goal_region_vocabulary import (
    ALL_INSTRUCTIONS,
    compute_region_target_embeddings,
    instruction_to_region,
    region_names,
)
from lang_goal_rl.held_out_paraphrases import (
    COMPOSITIONAL_INSTRUCTIONS,
    held_out_region_names,
    held_out_texts,
)
from lang_goal_rl.language_embedding import encode_instructions
from lang_goal_rl.language_goal_projection import LanguageGoalProjection
from lang_goal_rl.semantic_neighbor_diagnostic import (
    CompositionalPlacement,
    SemanticNeighborReport,
    diagnose_compositional_placement,
    diagnose_semantic_neighbors,
)

EXPERIMENT_DIR = Path(__file__).parent
STAGE2_ENCODER_PATH = (
    EXPERIMENT_DIR.parent / "02_contrastive_goal_embedding" / "artifacts" / "goal_encoder.pt"
)
STAGE3_PROJECTION_PATH = (
    EXPERIMENT_DIR.parent / "03_language_goal_projection" / "artifacts" / "language_goal_projection_v3.pt"
)

CENTROID_N_SAMPLES = 1000
"""Matches `language_goal_projection.DEFAULT_N_TARGET_SAMPLES` -- the sample
count `train_projection.py` actually used to precompute each region's
embedding-space regression target for `language_goal_projection_v3.pt`."""

CENTROID_SEED = 0
"""Matches `train_projection.PROJECTION_SEED` -- the seed
`language_goal_projection_v3.pt`'s training run actually used, so
`compute_region_target_embeddings` reproduces the identical centroids the
projection was regressed toward."""


def load_projection(path: Path) -> LanguageGoalProjection:
    """Load stage 3's trained `LanguageGoalProjection` checkpoint, unchanged.

    Args:
        path: Path to the checkpoint dict (`input_dim`/`embed_dim`/`state_dict`).

    Returns:
        The loaded `LanguageGoalProjection`, in eval mode.

    """
    checkpoint = torch.load(path, map_location="cpu")
    projection = LanguageGoalProjection(input_dim=checkpoint["input_dim"], embed_dim=checkpoint["embed_dim"])
    projection.load_state_dict(checkpoint["state_dict"])
    projection.eval()
    return projection


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


def run_semantic_neighbor_diagnostic(projection: LanguageGoalProjection) -> SemanticNeighborReport:
    """Classify each held-out paraphrase's projected embedding by nearest training instruction.

    Args:
        projection: Stage 3's trained `LanguageGoalProjection` (see `load_projection`).

    Returns:
        A `SemanticNeighborReport` covering all 14 held-out paraphrases.

    """
    train_sentence_embeddings = torch.from_numpy(encode_instructions(ALL_INSTRUCTIONS))
    with torch.no_grad():
        train_projected = projection(train_sentence_embeddings)
    train_regions = [instruction_to_region(instruction) for instruction in ALL_INSTRUCTIONS]

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


def run_compositional_diagnostic(
    projection: LanguageGoalProjection, encoder: GoalEncoder,
) -> tuple[CompositionalPlacement, ...]:
    """Report where each compositional instruction's projected embedding lands among region centroids.

    Args:
        projection: Stage 3's trained `LanguageGoalProjection` (see `load_projection`).
        encoder: Stage 2's frozen `GoalEncoder` (see `load_frozen_encoder`), used
            to build the region-centroid embeddings compositional placement is
            judged against.

    Returns:
        One `CompositionalPlacement` per `held_out_paraphrases.COMPOSITIONAL_INSTRUCTIONS`,
        in that tuple's order.

    """
    names = region_names()
    centroid_embeddings = compute_region_target_embeddings(
        encoder, names, n_samples=CENTROID_N_SAMPLES, seed=CENTROID_SEED,
    )

    placements = []
    for instruction in COMPOSITIONAL_INSTRUCTIONS:
        sentence_embedding = torch.from_numpy(encode_instructions([instruction.text]))
        with torch.no_grad():
            projected = projection(sentence_embedding).squeeze(0)
        placements.append(
            diagnose_compositional_placement(
                projected,
                instruction.text,
                instruction.component_region_names,
                centroid_embeddings,
                names,
            ),
        )
    return tuple(placements)


def _format_compositional(placement: CompositionalPlacement) -> str:
    """Render one `CompositionalPlacement` as a log-friendly line, mirroring `SemanticNeighborReport.summary()`."""
    component_flag = "component" if placement.nearest_is_component else "NEITHER component"
    return (
        f'  {placement.instruction!r} -> nearest={placement.nearest_region_name!r} ({component_flag}) '
        f"components={placement.component_region_names} balance={placement.component_distance_balance:.3f} "
        f"distances={placement.distances_by_region}"
    )


def main() -> None:
    """Run both diagnostics and print a log-friendly readout (redirect stdout to save it)."""
    projection = load_projection(STAGE3_PROJECTION_PATH)
    encoder = load_frozen_encoder(STAGE2_ENCODER_PATH)

    neighbor_report = run_semantic_neighbor_diagnostic(projection)
    print(neighbor_report.summary())

    print()
    print("CompositionalPlacementReport:")
    for placement in run_compositional_diagnostic(projection, encoder):
        print(_format_compositional(placement))


if __name__ == "__main__":
    main()
