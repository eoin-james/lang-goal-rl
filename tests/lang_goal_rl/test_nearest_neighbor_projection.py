"""Tests for the nearest-neighbor projection ceiling test.

`nearest_neighbor_projection` is a zero-training baseline built to isolate a
diagnosis after stage 4 (open vocabulary) failed: `LanguageGoalProjection`
(the trained 384->16 MLP, see `language_goal_projection.py`) collapsed
held-out paraphrases to memorized training points (28.6% held-out
neighbor-region accuracy, ~2% RL success). Before spending a data-
augmentation budget fixing the MLP, this function checks whether the *raw*
384-dim sentence-transformer space -- bypassing the MLP entirely -- already
carries enough region-clustering signal to do better than 28.6% via plain
distance-weighted interpolation over the known training points. If it does,
the MLP is confirmed as the thing throwing signal away, not the encoder.
"""

from __future__ import annotations

import numpy as np
import pytest

from lang_goal_rl.nearest_neighbor_projection import nearest_neighbor_projection


class TestNearestNeighborProjection:
    """Distance-weighted blend of the k nearest reference targets in raw embedding space."""

    def test_blends_two_neighbors_by_inverse_distance_weight(self) -> None:
        """Hand-computed case: query sits at distance 2.0 from reference 0 and
        1.0 from reference 1 (the third reference is farther and excluded by
        k=2). Inverse-distance weights are 0.5 and 1.0, normalizing to 1/3
        and 2/3 -- so the blend should be pulled twice as hard toward
        reference 1's target as reference 0's.
        """
        reference_embeddings = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        reference_targets = np.array([[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]])
        query_embedding = np.array([2.0, 0.0])

        blended = nearest_neighbor_projection(
            query_embedding, reference_embeddings, reference_targets, k=2,
        )

        expected = np.array([10.0 / 3.0, 20.0 / 3.0])
        np.testing.assert_allclose(blended, expected, atol=1e-6)

    def test_k_equals_one_reduces_to_exact_nearest_neighbor_copy_through(self) -> None:
        """With k=1 only the single nearest reference is used, and its weight
        always normalizes to 1.0 regardless of distance -- so the output must
        equal that reference's target exactly, even without an exact
        embedding match.
        """
        reference_embeddings = np.array([[0.0, 0.0], [5.0, 5.0]])
        reference_targets = np.array([[1.0, 2.0], [9.0, 9.0]])
        query_embedding = np.array([0.5, 0.5])

        blended = nearest_neighbor_projection(
            query_embedding, reference_embeddings, reference_targets, k=1,
        )

        np.testing.assert_allclose(blended, reference_targets[0], atol=1e-6)

    def test_exact_match_dominates_the_blend_even_with_k_greater_than_one(self) -> None:
        """An exact embedding match has distance 0, so its inverse-distance
        weight (1 / epsilon) dwarfs every other neighbor's -- the blend
        should land almost exactly on that reference's target even when
        other, farther references are included via k=3.
        """
        reference_embeddings = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        reference_targets = np.array([[1.0, 1.0], [-5.0, -5.0], [8.0, -8.0]])
        query_embedding = np.array([0.0, 0.0])

        blended = nearest_neighbor_projection(
            query_embedding, reference_embeddings, reference_targets, k=3,
        )

        np.testing.assert_allclose(blended, reference_targets[0], atol=1e-4)

    def test_closer_neighbor_is_weighted_more_heavily_than_a_farther_one(self) -> None:
        """Monotonicity check independent of exact hand-computed values: pull
        the query toward reference 0 and verify the blend lands strictly
        closer to reference 0's target than the unweighted midpoint.
        """
        reference_embeddings = np.array([[0.0, 0.0], [1.0, 0.0]])
        reference_targets = np.array([[0.0, 0.0], [10.0, 0.0]])
        query_embedding = np.array([0.1, 0.0])

        blended = nearest_neighbor_projection(
            query_embedding, reference_embeddings, reference_targets, k=2,
        )

        midpoint = reference_targets.mean(axis=0)
        assert blended[0] < midpoint[0]

    def test_output_shape_matches_reference_target_dimensionality(self) -> None:
        rng = np.random.default_rng(0)
        reference_embeddings = rng.standard_normal((14, 384))
        reference_targets = rng.standard_normal((14, 16))
        query_embedding = rng.standard_normal(384)

        blended = nearest_neighbor_projection(
            query_embedding, reference_embeddings, reference_targets, k=3,
        )

        assert blended.shape == (16,)

    def test_raises_on_row_count_mismatch_between_references_and_targets(self) -> None:
        reference_embeddings = np.zeros((3, 4))
        reference_targets = np.zeros((2, 4))
        query_embedding = np.zeros(4)

        with pytest.raises(ValueError, match="row count mismatch"):
            nearest_neighbor_projection(query_embedding, reference_embeddings, reference_targets, k=2)

    def test_raises_when_k_exceeds_available_reference_points(self) -> None:
        reference_embeddings = np.zeros((2, 4))
        reference_targets = np.zeros((2, 4))
        query_embedding = np.zeros(4)

        with pytest.raises(ValueError, match="k=3 exceeds"):
            nearest_neighbor_projection(query_embedding, reference_embeddings, reference_targets, k=3)
