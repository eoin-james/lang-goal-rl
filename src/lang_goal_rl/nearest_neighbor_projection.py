"""Zero-training ceiling test for stage 4's projection-layer overfitting diagnosis.

Stage 4 (open vocabulary) failed: `LanguageGoalProjection` (the trained
384->16 MLP in `language_goal_projection.py`) memorized its 14-sentence
training vocabulary instead of generalizing -- held-out paraphrases hit only
28.6% nearest-correct-region accuracy (vs. 14.3% random chance) and RL
success collapsed to ~2%. Before committing to the real fix (more training
data), the reviewer's recommended next step is cheap: bypass the learned MLP
entirely and see whether a plain distance-weighted interpolation over the raw
384-dim sentence-transformer embeddings beats 28.6%. A pass confirms the raw
embedding space itself carries the region-clustering signal and the MLP is
the thing throwing it away; a fail means the raw space doesn't cluster by
region as well as assumed, and the diagnosis needs revisiting before doing
anything else.
"""

from __future__ import annotations

import numpy as np

DEFAULT_EPSILON = 1e-8
"""Added to every distance before inverting, so an exact embedding match
(distance 0) gets a large but finite weight instead of a division-by-zero
error."""


def nearest_neighbor_projection(
    query_embedding: np.ndarray,
    reference_embeddings: np.ndarray,
    reference_targets: np.ndarray,
    *,
    k: int = 3,
) -> np.ndarray:
    """Distance-weighted blend of the `k` nearest reference targets in raw embedding space.

    No model, no training, no state -- a pure function meant to be called
    once per held-out query as a ceiling test against a trained projection
    layer (see the module docstring).

    Args:
        query_embedding: The query's raw sentence-transformer embedding,
            shape (input_dim,).
        reference_embeddings: The known vocabulary's raw sentence-transformer
            embeddings, shape (n_references, input_dim).
        reference_targets: Each reference sentence's known target embedding
            in goal space (e.g. its region centroid from
            `language_goal_projection.precompute_instruction_targets`), shape
            (n_references, embed_dim), row-aligned with
            `reference_embeddings`.
        k: Number of nearest reference points to blend.

    Returns:
        The blended target embedding, shape (embed_dim,): a weighted average
        of the `k` nearest reference targets, weighted by inverse Euclidean
        distance in raw embedding space and normalized to sum to 1.

    Raises:
        ValueError: If `reference_embeddings` and `reference_targets` have
            different row counts, or if `k` exceeds the number of available
            reference points.

    """
    if reference_embeddings.shape[0] != reference_targets.shape[0]:
        msg = (
            f"row count mismatch: reference_embeddings has {reference_embeddings.shape[0]} rows, "
            f"reference_targets has {reference_targets.shape[0]}"
        )
        raise ValueError(msg)
    if k > reference_embeddings.shape[0]:
        msg = f"k={k} exceeds the {reference_embeddings.shape[0]} available reference points"
        raise ValueError(msg)

    distances = np.linalg.norm(reference_embeddings - query_embedding, axis=1)
    nearest_indices = np.argsort(distances)[:k]

    weights = 1.0 / (distances[nearest_indices] + DEFAULT_EPSILON)
    weights /= weights.sum()

    return weights @ reference_targets[nearest_indices]
