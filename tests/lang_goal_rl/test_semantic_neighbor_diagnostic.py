"""Tests for the semantic-neighbor diagnostic (stage 4's proof gate).

All tests use constructed synthetic embeddings with a known right answer
(clusters placed at fixed, well-separated coordinates) rather than the real
sentence-transformer/GoalEncoder pipeline -- this diagnostic is pure
geometry over whatever embeddings it's handed, so its correctness doesn't
depend on any trained model.
"""

from __future__ import annotations

import pytest
import torch

from lang_goal_rl.semantic_neighbor_diagnostic import (
    classify_nearest_region,
    diagnose_compositional_placement,
    diagnose_semantic_neighbors,
)


class TestClassifyNearestRegion:
    """classify_nearest_region: 1-NN over a reference set, grouped by region."""

    def test_query_near_a_cluster_is_classified_into_that_clusters_region(self) -> None:
        reference_embeddings = torch.tensor(
            [[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]]
        )
        reference_region_names = ["a", "a", "b", "b"]
        query = torch.tensor([0.05, 0.0])

        match = classify_nearest_region(
            query, reference_embeddings, reference_region_names
        )

        assert match.nearest_region_name == "a"

    def test_distances_by_region_reports_the_minimum_distance_per_region(self) -> None:
        reference_embeddings = torch.tensor([[0.0, 0.0], [5.0, 0.0], [10.0, 10.0]])
        reference_region_names = ["a", "a", "b"]
        query = torch.tensor([0.0, 0.0])

        match = classify_nearest_region(
            query, reference_embeddings, reference_region_names
        )

        assert match.distances_by_region["a"] == pytest.approx(0.0)
        assert match.distances_by_region["b"] == pytest.approx(
            float(torch.tensor([10.0, 10.0]).norm())
        )
        assert match.nearest_distance == pytest.approx(0.0)

    def test_raises_on_row_count_mismatch(self) -> None:
        reference_embeddings = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        with pytest.raises(ValueError, match="row count mismatch"):
            classify_nearest_region(
                torch.tensor([0.0, 0.0]), reference_embeddings, ["a"]
            )


class TestDiagnoseSemanticNeighbors:
    """diagnose_semantic_neighbors: per-instruction nearest-region check + aggregate accuracy."""

    def _reference(self) -> tuple[torch.Tensor, list[str]]:
        # Three well-separated training clusters: "up", "left", "forward".
        reference_embeddings = torch.tensor(
            [
                [0.0, 0.0, 10.0],
                [0.0, 0.0, 10.1],
                [0.0, 10.0, 0.0],
                [0.0, 10.1, 0.0],
                [10.0, 0.0, 0.0],
            ],
        )
        reference_region_names = ["up", "up", "left", "left", "forward"]
        return reference_embeddings, reference_region_names

    def test_correct_neighbor_is_flagged_is_correct_true(self) -> None:
        reference_embeddings, reference_region_names = self._reference()
        query_embeddings = torch.tensor([[0.0, 0.0, 9.9]])  # near the "up" cluster

        report = diagnose_semantic_neighbors(
            query_embeddings,
            query_instructions=["raise your arm"],
            query_true_region_names=["up"],
            reference_embeddings=reference_embeddings,
            reference_region_names=reference_region_names,
        )

        assert report.results[0].is_correct is True
        assert report.results[0].nearest_region_name == "up"
        assert report.accuracy == pytest.approx(1.0)

    def test_wrong_neighbor_is_flagged_is_correct_false(self) -> None:
        reference_embeddings, reference_region_names = self._reference()
        query_embeddings = torch.tensor([[0.0, 9.9, 0.0]])  # near "left" cluster

        report = diagnose_semantic_neighbors(
            query_embeddings,
            query_instructions=["raise your arm"],
            query_true_region_names=["up"],  # wrong on purpose
            reference_embeddings=reference_embeddings,
            reference_region_names=reference_region_names,
        )

        assert report.results[0].is_correct is False
        assert report.results[0].nearest_region_name == "left"
        assert report.accuracy == pytest.approx(0.0)

    def test_none_true_region_produces_none_is_correct_and_is_excluded_from_accuracy(
        self,
    ) -> None:
        reference_embeddings, reference_region_names = self._reference()
        query_embeddings = torch.tensor([[0.0, 0.0, 9.9], [0.0, 9.9, 0.0]])

        report = diagnose_semantic_neighbors(
            query_embeddings,
            query_instructions=["raise your arm", "reach up and to the left"],
            query_true_region_names=["up", None],
            reference_embeddings=reference_embeddings,
            reference_region_names=reference_region_names,
        )

        assert report.results[0].is_correct is True
        assert report.results[1].is_correct is None
        assert report.results[1].true_region_name is None
        # Accuracy is over the one labeled result only, not diluted by the unlabeled one.
        assert report.accuracy == pytest.approx(1.0)

    def test_accuracy_averages_across_multiple_labeled_results(self) -> None:
        reference_embeddings, reference_region_names = self._reference()
        query_embeddings = torch.tensor([[0.0, 0.0, 9.9], [0.0, 9.9, 0.0]])

        report = diagnose_semantic_neighbors(
            query_embeddings,
            query_instructions=["correct one", "wrong one"],
            query_true_region_names=["up", "up"],  # second is deliberately wrong
            reference_embeddings=reference_embeddings,
            reference_region_names=reference_region_names,
        )

        assert report.accuracy == pytest.approx(0.5)

    def test_raises_on_row_count_mismatch(self) -> None:
        reference_embeddings, reference_region_names = self._reference()
        query_embeddings = torch.tensor([[0.0, 0.0, 9.9]])

        with pytest.raises(ValueError, match="row count mismatch"):
            diagnose_semantic_neighbors(
                query_embeddings,
                query_instructions=["a", "b"],  # mismatched length
                query_true_region_names=["up"],
                reference_embeddings=reference_embeddings,
                reference_region_names=reference_region_names,
            )

    def test_summary_mentions_every_instruction(self) -> None:
        reference_embeddings, reference_region_names = self._reference()
        query_embeddings = torch.tensor([[0.0, 0.0, 9.9]])

        report = diagnose_semantic_neighbors(
            query_embeddings,
            query_instructions=["raise your arm"],
            query_true_region_names=["up"],
            reference_embeddings=reference_embeddings,
            reference_region_names=reference_region_names,
        )

        assert "raise your arm" in report.summary()


class TestDiagnoseCompositionalPlacement:
    """diagnose_compositional_placement: where a compositional phrase lands, no pass/fail verdict."""

    def _region_centroids(self) -> tuple[torch.Tensor, list[str]]:
        region_centroid_embeddings = torch.tensor(
            [[0.0, 0.0, 10.0], [0.0, 10.0, 0.0], [10.0, 0.0, 0.0]],
        )
        region_names = ["up", "left", "forward"]
        return region_centroid_embeddings, region_names

    def test_embedding_exactly_between_components_has_balance_close_to_one(
        self,
    ) -> None:
        region_centroid_embeddings, region_names = self._region_centroids()
        midpoint = torch.tensor(
            [0.0, 5.0, 5.0]
        )  # exact midpoint of "up" and "left" centroids

        placement = diagnose_compositional_placement(
            midpoint,
            instruction="reach up and to the left",
            component_region_names=("up", "left"),
            region_centroid_embeddings=region_centroid_embeddings,
            region_names=region_names,
        )

        assert placement.component_distance_balance == pytest.approx(1.0, abs=1e-4)
        assert placement.nearest_is_component is True

    def test_embedding_at_one_components_centroid_has_zero_balance(self) -> None:
        region_centroid_embeddings, region_names = self._region_centroids()
        at_up = torch.tensor([0.0, 0.0, 10.0])

        placement = diagnose_compositional_placement(
            at_up,
            instruction="reach up and to the left",
            component_region_names=("up", "left"),
            region_centroid_embeddings=region_centroid_embeddings,
            region_names=region_names,
        )

        assert placement.component_distance_balance == pytest.approx(0.0, abs=1e-4)
        assert placement.nearest_region_name == "up"
        assert placement.nearest_is_component is True

    def test_embedding_nearest_a_non_component_region_is_flagged_ungrounded(
        self,
    ) -> None:
        region_centroid_embeddings, region_names = self._region_centroids()
        at_forward = torch.tensor([10.0, 0.0, 0.0])

        placement = diagnose_compositional_placement(
            at_forward,
            instruction="reach up and to the left",
            component_region_names=("up", "left"),
            region_centroid_embeddings=region_centroid_embeddings,
            region_names=region_names,
        )

        assert placement.nearest_region_name == "forward"
        assert placement.nearest_is_component is False

    def test_distances_by_region_includes_every_region(self) -> None:
        region_centroid_embeddings, region_names = self._region_centroids()
        midpoint = torch.tensor([0.0, 5.0, 5.0])

        placement = diagnose_compositional_placement(
            midpoint,
            instruction="reach up and to the left",
            component_region_names=("up", "left"),
            region_centroid_embeddings=region_centroid_embeddings,
            region_names=region_names,
        )

        assert set(placement.distances_by_region) == {"up", "left", "forward"}

    def test_raises_when_a_component_region_is_not_in_region_names(self) -> None:
        region_centroid_embeddings, region_names = self._region_centroids()
        midpoint = torch.tensor([0.0, 5.0, 5.0])

        with pytest.raises(ValueError, match="component_region_names"):
            diagnose_compositional_placement(
                midpoint,
                instruction="reach up and to the left",
                component_region_names=("up", "somewhere else"),
                region_centroid_embeddings=region_centroid_embeddings,
                region_names=region_names,
            )
