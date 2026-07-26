"""Diagnostic: does an unseen phrasing's projected embedding land near its own region?

Stage 4's proof gate: "semantic neighbors land near each other in goal
space" for unseen phrasing. This module answers that with a 1-NN-style
geometric check rather than a raw distance number, following stage 3's
region-vs-point lesson (`ROADMAP.md`'s "Region-vs-point ground truth" Known
risks entry): a held-out paraphrase's ground truth is the region it's a new
phrasing of, and "near" is judged relative to *other regions' distances*,
not an arbitrary absolute threshold.

Two entry points, matching `held_out_paraphrases.py`'s two kinds of unseen
instruction:

1. `diagnose_semantic_neighbors` -- for held-out paraphrases with a known
   ground-truth region (`held_out_paraphrases.HELD_OUT_PARAPHRASES`):
   per-instruction nearest-region classification against a reference set
   (either the training instructions' projected embeddings, or region
   centroids -- the caller's choice, since correctness here is pure
   geometry and doesn't depend on which reference is used), plus an
   aggregate accuracy over the labeled subset.

2. `diagnose_compositional_placement` -- for compositional instructions
   with no single ground-truth region
   (`held_out_paraphrases.COMPOSITIONAL_INSTRUCTIONS`): reports where the
   projected embedding actually lands relative to its two named component
   regions (closer to one, balanced between both, or nearest some other
   region entirely) without asserting a pass/fail verdict -- "graceful
   degradation" is the thing being measured, not a binary correctness
   check that this instruction class was never designed to satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class NearestRegionMatch:
    """Result of a 1-NN classification of one query embedding against a labeled reference set.

    Attributes:
        nearest_region_name: The region of the single closest reference row.
        nearest_distance: That closest row's Euclidean distance to the query.
        distances_by_region: Minimum distance from the query to any
            reference row of each region present in the reference set.

    """

    nearest_region_name: str
    nearest_distance: float
    distances_by_region: dict[str, float]


def classify_nearest_region(
    query_embedding: torch.Tensor,
    reference_embeddings: torch.Tensor,
    reference_region_names: Sequence[str],
) -> NearestRegionMatch:
    """Classify `query_embedding` by its nearest row in `reference_embeddings`.

    Args:
        query_embedding: A single embedding, shape (embed_dim,).
        reference_embeddings: Labeled reference points, shape
            (n_reference, embed_dim). May be individual training-instruction
            embeddings or per-region centroids -- this function only does
            the distance geometry, the caller decides what "reference"
            means.
        reference_region_names: Region label for each row of
            `reference_embeddings`, same length and order.

    Returns:
        A `NearestRegionMatch`.

    Raises:
        ValueError: If `reference_embeddings` and `reference_region_names`
            have different row counts.

    """
    if reference_embeddings.shape[0] != len(reference_region_names):
        msg = (
            f"row count mismatch: reference_embeddings has {reference_embeddings.shape[0]} rows, "
            f"reference_region_names has {len(reference_region_names)}"
        )
        raise ValueError(msg)

    with torch.no_grad():
        distances = torch.linalg.norm(reference_embeddings - query_embedding, dim=1)

    distances_by_region: dict[str, float] = {}
    for region, distance in zip(
        reference_region_names, distances.tolist(), strict=True
    ):
        if region not in distances_by_region or distance < distances_by_region[region]:
            distances_by_region[region] = distance

    nearest_region_name = min(
        distances_by_region, key=lambda region: distances_by_region[region]
    )
    return NearestRegionMatch(
        nearest_region_name=nearest_region_name,
        nearest_distance=distances_by_region[nearest_region_name],
        distances_by_region=distances_by_region,
    )


@dataclass(frozen=True)
class NeighborResult:
    """One held-out instruction's nearest-region classification, plus its ground truth if known.

    Attributes:
        instruction: The instruction text.
        true_region_name: The instruction's known ground-truth region, or
            `None` if it has no single correct region (e.g. a compositional
            instruction routed through this diagnostic for information
            only).
        nearest_region_name: The region `classify_nearest_region` assigned.
        is_correct: `nearest_region_name == true_region_name`, or `None`
            when `true_region_name` is `None` (no verdict to give).
        distances_by_region: Same as `NearestRegionMatch.distances_by_region`
            -- kept per-result so a caller can inspect near-misses (e.g. an
            incorrect classification whose margin to the true region was
            tiny) rather than only seeing the boolean verdict.

    """

    instruction: str
    true_region_name: str | None
    nearest_region_name: str
    is_correct: bool | None
    distances_by_region: dict[str, float]


@dataclass(frozen=True)
class SemanticNeighborReport:
    """Aggregate result of running `diagnose_semantic_neighbors` over a batch of instructions.

    Attributes:
        results: One `NeighborResult` per query instruction, in input order.
        accuracy: Fraction of results with a known `true_region_name` that
            were classified correctly. `None` if no result has a known
            ground truth (nothing to average).

    """

    results: tuple[NeighborResult, ...]
    accuracy: float | None

    def summary(self) -> str:
        """Render a human-readable, log-friendly per-instruction breakdown."""
        n_labeled = sum(1 for r in self.results if r.true_region_name is not None)
        accuracy_text = "n/a" if self.accuracy is None else f"{self.accuracy:.3f}"
        lines = [
            f"SemanticNeighborReport: accuracy={accuracy_text} over {n_labeled} labeled instruction(s)"
        ]
        for result in self.results:
            verdict = (
                "?"
                if result.is_correct is None
                else ("correct" if result.is_correct else "WRONG")
            )
            lines.append(
                f"  [{verdict}] {result.instruction!r} -> nearest={result.nearest_region_name!r} "
                f"true={result.true_region_name!r} distances={result.distances_by_region}",
            )
        return "\n".join(lines)


def diagnose_semantic_neighbors(
    query_embeddings: torch.Tensor,
    query_instructions: Sequence[str],
    query_true_region_names: Sequence[str | None],
    reference_embeddings: torch.Tensor,
    reference_region_names: Sequence[str],
) -> SemanticNeighborReport:
    """Classify each held-out instruction's projected embedding by nearest reference region.

    Args:
        query_embeddings: Held-out instructions' projected embeddings,
            shape (n_query, embed_dim).
        query_instructions: Instruction text for each row, same length and
            order.
        query_true_region_names: Ground-truth region for each row (or
            `None` if unknown/inapplicable -- see `NeighborResult`'s
            docstring), same length and order.
        reference_embeddings: Labeled reference points to classify against
            (e.g. the training vocabulary's projected embeddings, or region
            centroids), shape (n_reference, embed_dim).
        reference_region_names: Region label for each row of
            `reference_embeddings`, same length and order.

    Returns:
        A `SemanticNeighborReport`.

    Raises:
        ValueError: If `query_instructions` and `query_true_region_names`
            don't match `query_embeddings`' row count.

    """
    n = query_embeddings.shape[0]
    if len(query_instructions) != n or len(query_true_region_names) != n:
        msg = (
            f"row count mismatch: query_embeddings has {n} rows, query_instructions has "
            f"{len(query_instructions)}, query_true_region_names has {len(query_true_region_names)}"
        )
        raise ValueError(msg)

    results: list[NeighborResult] = []
    for embedding, instruction, true_region in zip(
        query_embeddings,
        query_instructions,
        query_true_region_names,
        strict=True,
    ):
        match = classify_nearest_region(
            embedding, reference_embeddings, reference_region_names
        )
        is_correct = (
            None if true_region is None else match.nearest_region_name == true_region
        )
        results.append(
            NeighborResult(
                instruction=instruction,
                true_region_name=true_region,
                nearest_region_name=match.nearest_region_name,
                is_correct=is_correct,
                distances_by_region=match.distances_by_region,
            ),
        )

    labeled = [result.is_correct for result in results if result.is_correct is not None]
    accuracy = (sum(labeled) / len(labeled)) if labeled else None
    return SemanticNeighborReport(results=tuple(results), accuracy=accuracy)


@dataclass(frozen=True)
class CompositionalPlacement:
    """Where a compositional instruction's projected embedding lands relative to its two components.

    No pass/fail field by design -- the current 7-region vocabulary has no
    single correct answer for a compositional phrase, so this reports the
    geometry (closer to one component, balanced between both, or nearest
    something else entirely) and leaves interpretation to the caller.

    Attributes:
        instruction: The compositional instruction text.
        component_region_names: The two regions this instruction combines.
        distances_by_region: Distance from the projected embedding to every
            region present in the reference set (see
            `classify_nearest_region`'s `distances_by_region`), including
            but not limited to the two components.
        nearest_region_name: The single closest region overall.
        nearest_is_component: Whether `nearest_region_name` is one of
            `component_region_names`.
        component_distance_balance: `min(d0, d1) / max(d0, d1)` where `d0`,
            `d1` are the distances to the two components specifically
            (regardless of which region is nearest overall). `1.0` means
            equidistant from both components (sits on their "midline");
            `0.0` means exactly at one component's reference point and
            arbitrarily far from the other, relative to that component's
            own distance.

    """

    instruction: str
    component_region_names: tuple[str, str]
    distances_by_region: dict[str, float]
    nearest_region_name: str
    nearest_is_component: bool
    component_distance_balance: float


def diagnose_compositional_placement(
    query_embedding: torch.Tensor,
    instruction: str,
    component_region_names: tuple[str, str],
    region_centroid_embeddings: torch.Tensor,
    region_names: Sequence[str],
) -> CompositionalPlacement:
    """Report where a compositional instruction's projected embedding lands among region centroids.

    Args:
        query_embedding: The compositional instruction's projected
            embedding, shape (embed_dim,).
        instruction: The compositional instruction text.
        component_region_names: The two regions this instruction combines
            (e.g. `("reach up high", "reach left")` for "reach up and to
            the left" -- see `held_out_paraphrases.CompositionalInstruction`).
        region_centroid_embeddings: One centroid per region to compare
            against, shape (n_regions, embed_dim) (e.g. via
            `goal_region_vocabulary.compute_region_target_embeddings`).
        region_names: Region label for each row of
            `region_centroid_embeddings`, same length and order. Must
            include both of `component_region_names`.

    Returns:
        A `CompositionalPlacement`.

    Raises:
        ValueError: If either name in `component_region_names` is not
            present in `region_names`.

    """
    missing = [name for name in component_region_names if name not in region_names]
    if missing:
        msg = f"component_region_names {missing!r} not found in region_names {list(region_names)!r}"
        raise ValueError(msg)

    match = classify_nearest_region(
        query_embedding, region_centroid_embeddings, region_names
    )

    first_distance = match.distances_by_region[component_region_names[0]]
    second_distance = match.distances_by_region[component_region_names[1]]
    larger = max(first_distance, second_distance)
    balance = 1.0 if larger == 0.0 else min(first_distance, second_distance) / larger

    return CompositionalPlacement(
        instruction=instruction,
        component_region_names=component_region_names,
        distances_by_region=match.distances_by_region,
        nearest_region_name=match.nearest_region_name,
        nearest_is_component=match.nearest_region_name in component_region_names,
        component_distance_balance=balance,
    )
