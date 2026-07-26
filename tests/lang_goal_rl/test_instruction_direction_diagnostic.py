"""Tests for the instruction-direction (cosine-similarity) diagnostic.

Attempt 2's reviewer asked a specific open question: is directional
accuracy against an instruction's TRUE region centroid actually what
predicts RL success, or is there a confound (e.g. FetchReach's fixed
success radius just happening to favor goals near the robot's reset
position, independent of projection quality)? This module builds the
measurement the experiment-runner needs to correlate against per-instruction
success rates already collected in attempt 2 -- it does not run that
correlation itself.
"""

from __future__ import annotations

import pytest
import torch

from lang_goal_rl.goal_encoder import GoalEncoder
from lang_goal_rl.goal_region_vocabulary import GoalBox
from lang_goal_rl.instruction_direction_diagnostic import (
    InstructionDirectionAlignment,
    cosine_similarity_to_true_centroid,
    measure_instruction_direction_alignment,
)
from lang_goal_rl.language_goal_projection import LanguageGoalProjection, precompute_instruction_targets

SYNTHETIC_BOX = GoalBox(axis_min=torch.zeros(3).numpy(), axis_max=torch.ones(3).numpy())


class TestCosineSimilarityToTrueCentroid:
    """Pure function: cosine similarity per row between precomputed projected outputs and true centroids."""

    def test_identical_vectors_give_similarity_one(self) -> None:
        projected = torch.tensor([[1.0, 2.0, 3.0]])
        true_centroids = torch.tensor([[1.0, 2.0, 3.0]])

        result = cosine_similarity_to_true_centroid(projected, true_centroids, ["a"], ["region-a"])

        assert result.cosine_similarities[0] == pytest.approx(1.0)

    def test_orthogonal_vectors_give_similarity_zero(self) -> None:
        projected = torch.tensor([[1.0, 0.0]])
        true_centroids = torch.tensor([[0.0, 1.0]])

        result = cosine_similarity_to_true_centroid(projected, true_centroids, ["a"], ["region-a"])

        assert result.cosine_similarities[0] == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors_give_similarity_negative_one(self) -> None:
        projected = torch.tensor([[1.0, 2.0, -3.0]])
        true_centroids = torch.tensor([[-1.0, -2.0, 3.0]])

        result = cosine_similarity_to_true_centroid(projected, true_centroids, ["a"], ["region-a"])

        assert result.cosine_similarities[0] == pytest.approx(-1.0)

    def test_returns_one_similarity_per_instruction_in_input_order(self) -> None:
        projected = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        true_centroids = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

        result = cosine_similarity_to_true_centroid(
            projected, true_centroids, ["aligned", "orthogonal", "opposite"], ["r1", "r2", "r3"],
        )

        assert result.instructions == ("aligned", "orthogonal", "opposite")
        assert result.cosine_similarities[0] == pytest.approx(1.0)
        assert result.cosine_similarities[1] == pytest.approx(0.0, abs=1e-6)
        assert result.cosine_similarities[2] == pytest.approx(-1.0)

    def test_scale_invariant_to_positive_rescaling(self) -> None:
        """Cosine similarity measures direction only -- this diagnostic is
        deliberately orthogonal to `check_projection_norm_range`'s
        scale-based check, not a duplicate of it.
        """
        projected = torch.tensor([[3.0, 4.0]])
        true_centroids = torch.tensor([[1.0, 0.0]])

        small = cosine_similarity_to_true_centroid(projected * 0.01, true_centroids, ["a"], ["r"])
        huge = cosine_similarity_to_true_centroid(projected * 500.0, true_centroids, ["a"], ["r"])

        assert small.cosine_similarities[0] == pytest.approx(huge.cosine_similarities[0])

    def test_raises_on_row_count_mismatch(self) -> None:
        projected = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        true_centroids = torch.tensor([[1.0, 0.0]])

        try:
            cosine_similarity_to_true_centroid(projected, true_centroids, ["only-one-label"], ["r1"])
        except ValueError:
            return
        raise AssertionError("expected ValueError for row count mismatch")

    def test_returns_an_instruction_direction_alignment_instance(self) -> None:
        projected = torch.tensor([[1.0, 0.0]])
        true_centroids = torch.tensor([[1.0, 0.0]])

        result = cosine_similarity_to_true_centroid(projected, true_centroids, ["a"], ["r"])

        assert isinstance(result, InstructionDirectionAlignment)

    def test_as_dict_maps_instruction_to_its_cosine_similarity(self) -> None:
        projected = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        true_centroids = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

        result = cosine_similarity_to_true_centroid(projected, true_centroids, ["aligned", "orthogonal"], ["r1", "r2"])
        mapping = result.as_dict()

        assert mapping["aligned"] == pytest.approx(1.0)
        assert mapping["orthogonal"] == pytest.approx(0.0, abs=1e-6)


class TestMeasureInstructionDirectionAlignment:
    """measure_instruction_direction_alignment composes projection + true-centroid computation into one report."""

    def test_perfectly_aligned_projection_gives_similarity_near_one(self) -> None:
        """A projection whose output IS its region's true centroid should
        score ~1.0 -- the diagnostic's sanity anchor before trusting it on a
        real (partially misaligned) trained projection.
        """
        torch.manual_seed(0)
        goal_encoder = GoalEncoder(goal_dim=3, embed_dim=4, hidden_dim=8)
        instructions = ["move left", "move right"]
        region_names = ["reach left", "reach right"]
        sentence_embeddings = torch.randn(2, 8)

        targets = precompute_instruction_targets(
            goal_encoder, region_names, box=SYNTHETIC_BOX, n_samples=200, seed=0,
        )

        class _FixedOutputProjection(LanguageGoalProjection):
            def forward(self, sentence_embeddings: torch.Tensor) -> torch.Tensor:
                del sentence_embeddings
                return targets

        projection = _FixedOutputProjection(input_dim=8, embed_dim=4, hidden_dim=6)

        result = measure_instruction_direction_alignment(
            projection,
            goal_encoder,
            sentence_embeddings,
            instructions,
            region_names,
            box=SYNTHETIC_BOX,
            n_samples=200,
            seed=0,
        )

        for similarity in result.cosine_similarities:
            assert similarity == pytest.approx(1.0, abs=1e-4)

    def test_returns_finite_similarities_in_valid_range_for_an_untrained_projection(self) -> None:
        torch.manual_seed(0)
        goal_encoder = GoalEncoder(goal_dim=3, embed_dim=4, hidden_dim=8)
        instructions = ["move left", "move right"]
        region_names = ["reach left", "reach right"]
        sentence_embeddings = torch.randn(2, 8)
        projection = LanguageGoalProjection(input_dim=8, embed_dim=4, hidden_dim=6)

        result = measure_instruction_direction_alignment(
            projection,
            goal_encoder,
            sentence_embeddings,
            instructions,
            region_names,
            box=SYNTHETIC_BOX,
            n_samples=50,
            seed=0,
        )

        assert result.instructions == tuple(instructions)
        assert result.region_names == tuple(region_names)
        for similarity in result.cosine_similarities:
            assert -1.0 - 1e-6 <= similarity <= 1.0 + 1e-6

    def test_leaves_goal_encoder_and_projection_parameters_unchanged(self) -> None:
        torch.manual_seed(0)
        goal_encoder = GoalEncoder(goal_dim=3, embed_dim=4, hidden_dim=8)
        projection = LanguageGoalProjection(input_dim=8, embed_dim=4, hidden_dim=6)
        encoder_before = [p.clone() for p in goal_encoder.parameters()]
        projection_before = [p.clone() for p in projection.parameters()]

        measure_instruction_direction_alignment(
            projection,
            goal_encoder,
            torch.randn(2, 8),
            ["a", "b"],
            ["reach left", "reach right"],
            box=SYNTHETIC_BOX,
            n_samples=30,
            seed=0,
        )

        for p_before, p_after in zip(encoder_before, goal_encoder.parameters(), strict=True):
            assert torch.equal(p_before, p_after)
        for p_before, p_after in zip(projection_before, projection.parameters(), strict=True):
            assert torch.equal(p_before, p_after)

    def test_deterministic_for_a_given_seed(self) -> None:
        torch.manual_seed(0)
        goal_encoder = GoalEncoder(goal_dim=3, embed_dim=4, hidden_dim=8)
        projection = LanguageGoalProjection(input_dim=8, embed_dim=4, hidden_dim=6)
        sentence_embeddings = torch.randn(2, 8)
        instructions = ["a", "b"]
        region_names = ["reach left", "reach right"]

        first = measure_instruction_direction_alignment(
            projection, goal_encoder, sentence_embeddings, instructions, region_names,
            box=SYNTHETIC_BOX, n_samples=30, seed=5,
        )
        second = measure_instruction_direction_alignment(
            projection, goal_encoder, sentence_embeddings, instructions, region_names,
            box=SYNTHETIC_BOX, n_samples=30, seed=5,
        )

        assert first.cosine_similarities == second.cosine_similarities
