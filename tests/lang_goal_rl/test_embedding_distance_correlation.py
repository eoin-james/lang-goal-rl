"""Tests for the embedding-vs-true-distance correlation diagnostic.

This is the numeric check behind stage 2's proof gate half that reads
"distance-in-latent correlates with true task distance" — given goal
embeddings and their true xyz coordinates, it reports whether pairwise
distances in one space track pairwise distances in the other.
"""

import numpy as np

from lang_goal_rl.embedding_distance_correlation import embedding_distance_correlation


class TestEmbeddingDistanceCorrelation:
    """Pearson correlation between pairwise embedding and true-goal distances."""

    def test_perfectly_matching_distances_give_correlation_near_one(self) -> None:
        rng = np.random.default_rng(0)
        true_coords = rng.normal(size=(20, 3))
        # Embeddings identical to true coords: distances match exactly.
        embeddings = true_coords.copy()

        correlation = embedding_distance_correlation(embeddings, true_coords)

        assert correlation > 0.99

    def test_scaled_embeddings_still_correlate_near_one(self) -> None:
        rng = np.random.default_rng(1)
        true_coords = rng.normal(size=(20, 3))
        # A uniform rescale preserves relative pairwise distances.
        embeddings = true_coords * 5.0

        correlation = embedding_distance_correlation(embeddings, true_coords)

        assert correlation > 0.99

    def test_random_unrelated_embeddings_give_low_correlation(self) -> None:
        rng = np.random.default_rng(2)
        true_coords = rng.normal(size=(50, 3))
        unrelated_embeddings = rng.normal(size=(50, 16))

        correlation = embedding_distance_correlation(unrelated_embeddings, true_coords)

        assert abs(correlation) < 0.4

    def test_collapsed_embeddings_give_low_or_undefined_correlation(self) -> None:
        rng = np.random.default_rng(3)
        true_coords = rng.normal(size=(10, 3))
        # All embeddings identical: zero variance in embedding distances.
        embeddings = np.zeros((10, 16))

        correlation = embedding_distance_correlation(embeddings, true_coords)

        assert correlation == 0.0

    def test_raises_on_mismatched_sample_counts(self) -> None:
        embeddings = np.zeros((5, 16))
        true_coords = np.zeros((4, 3))
        try:
            embedding_distance_correlation(embeddings, true_coords)
        except ValueError:
            return
        raise AssertionError("expected ValueError for mismatched sample counts")
