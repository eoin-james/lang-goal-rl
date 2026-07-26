"""Diagnostic: whether the language projection collapses distinct instructions to one point.

Answers the second half of stage 3's proof gate. Two ideas combine:

1. `measure_instruction_separation` computes pairwise distances between the
   fixed vocabulary's *projected* embeddings, split into the global minimum
   (every instruction pair) and the minimum *across different regions*.
   Same-region synonyms (e.g. "reach up high" / "move your hand upward")
   are *expected* to sit close together — that's correct behavior, not
   collapse — so the pass/fail check (`is_collapsed`) uses the cross-region
   minimum, not the global one. The global minimum is still reported for
   transparency.

2. `collapse_epsilon_from_goal_encoder` grounds the "how close is too
   close" threshold in stage 2's own embedding space: a fraction of the
   smallest distance between two regions' *true* mean embeddings under the
   same frozen `GoalEncoder`. An arbitrary absolute number (e.g. "0.01")
   would be meaningless without knowing that space's scale; this measures
   the scale first, matching the project's practice elsewhere (e.g.
   `goal_region_vocabulary.py`'s box grounded in a real reset distribution)
   of deriving thresholds from data rather than picking them by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch

from lang_goal_rl.goal_region_vocabulary import (
    MEASURED_GOAL_BOX,
    compute_region_target_embeddings,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lang_goal_rl.goal_encoder import GoalEncoder
    from lang_goal_rl.goal_region_vocabulary import GoalBox
    from lang_goal_rl.language_goal_projection import LanguageGoalProjection


MIN_DISTINCT_REGIONS_FOR_EPSILON = 2
"""A cross-region distance is only meaningful with at least 2 distinct regions present."""


def _pairwise_distances(points: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Return the upper-triangular (excluding diagonal) pairwise Euclidean distances."""
    diff = points[:, None, :] - points[None, :, :]
    distance_matrix = np.linalg.norm(diff, axis=-1)
    rows, cols = np.triu_indices(points.shape[0], k=1)
    return distance_matrix[rows, cols]


@dataclass(frozen=True)
class SeparationReport:
    """Result of measuring how far apart the fixed vocabulary's projected embeddings land.

    Attributes:
        instructions: The instructions measured, in input order.
        region_names: Each instruction's region, same order as `instructions`.
        min_pairwise_distance: Smallest distance between any two instructions'
            projected embeddings, including same-region pairs.
        mean_pairwise_distance: Mean distance across all instruction pairs.
        min_cross_region_pairwise_distance: Smallest distance between two
            instructions from *different* regions. This is what
            `is_collapsed` is judged against — same-region synonyms are
            allowed (expected) to be close.
        collapse_epsilon: The threshold `min_cross_region_pairwise_distance`
            was compared against.
        is_collapsed: `True` if `min_cross_region_pairwise_distance <
            collapse_epsilon`.

    """

    instructions: tuple[str, ...]
    region_names: tuple[str, ...]
    min_pairwise_distance: float
    mean_pairwise_distance: float
    min_cross_region_pairwise_distance: float
    collapse_epsilon: float
    is_collapsed: bool


def measure_instruction_separation(
    embeddings: npt.NDArray[np.floating],
    instructions: Sequence[str],
    region_names: Sequence[str],
    collapse_epsilon: float,
) -> SeparationReport:
    """Measure pairwise separation between a batch of projected instruction embeddings.

    Args:
        embeddings: Array of shape (n_instructions, embed_dim) — one
            projected embedding per instruction.
        instructions: Instruction text, one per row of `embeddings`.
        region_names: Each instruction's region name, same order.
        collapse_epsilon: Threshold `min_cross_region_pairwise_distance` is
            compared against to set `is_collapsed`.

    Returns:
        A `SeparationReport`.

    Raises:
        ValueError: If `instructions`/`region_names` don't match
            `embeddings`' row count, or if `region_names` contains fewer
            than 2 distinct regions (no cross-region pair exists to judge).

    """
    embeddings = np.asarray(embeddings)
    n = embeddings.shape[0]
    if len(instructions) != n or len(region_names) != n:
        msg = (
            f"row count mismatch: embeddings has {n} rows, instructions has "
            f"{len(instructions)}, region_names has {len(region_names)}"
        )
        raise ValueError(msg)

    rows, cols = np.triu_indices(n, k=1)
    all_pair_distances = _pairwise_distances(embeddings)
    cross_region_mask = np.array(
        [region_names[int(r)] != region_names[int(c)] for r, c in zip(rows, cols, strict=True)],
    )
    if not np.any(cross_region_mask):
        msg = "no cross-region instruction pair found; region_names needs at least 2 distinct regions"
        raise ValueError(msg)

    min_cross_region = float(np.min(all_pair_distances[cross_region_mask]))
    return SeparationReport(
        instructions=tuple(instructions),
        region_names=tuple(region_names),
        min_pairwise_distance=float(np.min(all_pair_distances)),
        mean_pairwise_distance=float(np.mean(all_pair_distances)),
        min_cross_region_pairwise_distance=min_cross_region,
        collapse_epsilon=collapse_epsilon,
        is_collapsed=min_cross_region < collapse_epsilon,
    )


def collapse_epsilon_from_goal_encoder(
    goal_encoder: GoalEncoder,
    region_names: Sequence[str],
    *,
    box: GoalBox = MEASURED_GOAL_BOX,
    n_samples: int = 200,
    seed: int = 0,
    fraction: float = 0.1,
) -> float:
    """Ground the collapse epsilon in the real target space's scale.

    Computes each distinct region's true mean embedding under
    `goal_encoder` (via `compute_region_target_embeddings`) and returns
    `fraction` times the smallest pairwise distance between those region
    embeddings — i.e. "a small fraction of how far apart the *true* regions
    already are in this embedding space".

    Args:
        goal_encoder: Stage 2's frozen encoder.
        region_names: Region names to consider (duplicates collapsed to
            their unique set; order-preserving).
        box: Goal box to sample regions within.
        n_samples: xyz samples per region used to estimate its mean embedding.
        seed: Base seed for region sampling.
        fraction: Fraction of the smallest true inter-region distance to use
            as the threshold. 0.1 (10%) is small enough that a projection
            has to badly under-separate two regions to trip it, while still
            being well above floating-point noise.

    Returns:
        The collapse epsilon, a positive float.

    Raises:
        ValueError: If fewer than 2 distinct region names are given.

    """
    unique_names = list(dict.fromkeys(region_names))
    if len(unique_names) < MIN_DISTINCT_REGIONS_FOR_EPSILON:
        msg = f"need at least 2 distinct region names to ground an epsilon, got {unique_names!r}"
        raise ValueError(msg)

    target_embeddings = compute_region_target_embeddings(
        goal_encoder, unique_names, box=box, n_samples=n_samples, seed=seed,
    ).numpy()
    min_true_distance = float(np.min(_pairwise_distances(target_embeddings)))
    return fraction * min_true_distance


def check_no_collapse(
    projection: LanguageGoalProjection,
    goal_encoder: GoalEncoder,
    sentence_embeddings: torch.Tensor,
    instructions: Sequence[str],
    region_names: Sequence[str],
    *,
    box: GoalBox = MEASURED_GOAL_BOX,
    epsilon_fraction: float = 0.1,
    epsilon_n_samples: int = 200,
    epsilon_seed: int = 0,
) -> SeparationReport:
    """Run the full collapse check: project the vocabulary, ground epsilon, measure separation.

    Args:
        projection: The (typically trained) `LanguageGoalProjection` to check.
        goal_encoder: Stage 2's frozen encoder, used both to ground epsilon
            and as the space `projection` targets.
        sentence_embeddings: Frozen sentence embeddings, shape
            (n_instructions, input_dim), row-aligned with `instructions`.
        instructions: Instruction text, one per row of `sentence_embeddings`.
        region_names: Each instruction's region name, same order.
        box: Goal box to sample regions within.
        epsilon_fraction: Passed through to `collapse_epsilon_from_goal_encoder`.
        epsilon_n_samples: Passed through to `collapse_epsilon_from_goal_encoder`.
        epsilon_seed: Passed through to `collapse_epsilon_from_goal_encoder`.

    Returns:
        A `SeparationReport`.

    """
    with torch.no_grad():
        projected_embeddings = projection(sentence_embeddings).numpy()

    epsilon = collapse_epsilon_from_goal_encoder(
        goal_encoder,
        region_names,
        box=box,
        n_samples=epsilon_n_samples,
        seed=epsilon_seed,
        fraction=epsilon_fraction,
    )

    return measure_instruction_separation(
        projected_embeddings, instructions, region_names, collapse_epsilon=epsilon,
    )
