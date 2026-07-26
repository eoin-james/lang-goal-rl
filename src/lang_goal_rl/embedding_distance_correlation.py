"""Diagnostic: does distance in embedding space track true xyz distance?

Computes the Pearson correlation between all pairwise distances in a batch
of learned goal embeddings and the corresponding pairwise distances in the
literal xyz goal space. This is the numeric check behind the second half of
stage 2's proof gate ("distance-in-latent correlates with true task
distance").

Pearson (not Spearman) is used to stay dependency-light and consistent with
the rest of this module's numpy-only philosophy (see
`reporting.plot_embedding_projection`'s SVD-based PCA, which avoids
scikit-learn for the same reason) — scipy isn't in `uv.lock`. Pearson
measures linear correlation; it will catch gross failures (embedding
collapse, no relationship at all) and reward an approximately linear
distance-preserving mapping, which is the property a contrastively
pretrained encoder is expected to have at this stage's scope. It will
under-report a genuinely non-linear-but-monotonic relationship — if that
distinction matters later, a rank correlation should replace this.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _pairwise_distances(points: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Return the upper-triangular (excluding diagonal) pairwise Euclidean distances."""
    diff = points[:, None, :] - points[None, :, :]
    distance_matrix = np.linalg.norm(diff, axis=-1)
    rows, cols = np.triu_indices(points.shape[0], k=1)
    return distance_matrix[rows, cols]


def embedding_distance_correlation(
    embeddings: npt.NDArray[np.floating],
    true_coords: npt.NDArray[np.floating],
) -> float:
    """Correlate pairwise embedding distances with pairwise true-goal distances.

    Args:
        embeddings: Array of shape (n_samples, embed_dim) — learned goal
            embeddings.
        true_coords: Array of shape (n_samples, goal_dim) — the same
            samples' literal xyz goal coordinates.

    Returns:
        Pearson correlation coefficient in [-1, 1] between the two sets of
        pairwise distances. Returns 0.0 if either distance set has zero
        variance (e.g. all embeddings collapsed to one point), since
        correlation is undefined there and 0.0 reads as "no measurable
        distance-tracking relationship" for this diagnostic's purpose.

    Raises:
        ValueError: If `embeddings` and `true_coords` don't have the same
            number of samples.
    """
    if embeddings.shape[0] != true_coords.shape[0]:
        msg = (
            f"sample count mismatch: embeddings has {embeddings.shape[0]} rows, "
            f"true_coords has {true_coords.shape[0]}"
        )
        raise ValueError(msg)

    embedding_distances = _pairwise_distances(embeddings)
    true_distances = _pairwise_distances(true_coords)

    if np.std(embedding_distances) == 0.0 or np.std(true_distances) == 0.0:
        return 0.0

    correlation_matrix = np.corrcoef(embedding_distances, true_distances)
    return float(correlation_matrix[0, 1])
