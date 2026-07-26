"""Tests for the instruction-collapse diagnostic.

Answers stage 3's proof-gate question ("projection doesn't collapse distinct
instructions to one point") with a concrete number: the minimum pairwise
distance between *different-region* instructions' projected embeddings,
checked against an epsilon grounded in stage 2's own embedding-space scale
(a fraction of the smallest distance between two regions' true mean
embeddings under the same frozen GoalEncoder).
"""

from __future__ import annotations

import numpy as np
import torch

from lang_goal_rl.goal_encoder import GoalEncoder
from lang_goal_rl.goal_region_vocabulary import GoalBox
from lang_goal_rl.instruction_collapse_diagnostic import (
    check_no_collapse,
    collapse_epsilon_from_goal_encoder,
    measure_instruction_separation,
)
from lang_goal_rl.language_goal_projection import LanguageGoalProjection

SYNTHETIC_BOX = GoalBox(axis_min=np.zeros(3), axis_max=np.ones(3))


class ConstantProjection(LanguageGoalProjection):
    """A projection that ignores its input and always outputs the same vector.

    Used to construct a deliberately collapsed case for the diagnostic to
    catch, independent of whether real training happens to collapse.
    """

    def forward(self, sentence_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size = sentence_embeddings.shape[0]
        return torch.zeros(batch_size, self.embed_dim)


class TestMeasureInstructionSeparation:
    """measure_instruction_separation computes pairwise distances between projected instructions."""

    def test_min_pairwise_distance_is_zero_for_identical_embeddings(self) -> None:
        embeddings = np.zeros((3, 2))
        report = measure_instruction_separation(
            embeddings,
            instructions=["a", "b", "c"],
            region_names=["r1", "r2", "r3"],
            collapse_epsilon=0.1,
        )
        assert report.min_pairwise_distance == 0.0
        assert report.min_cross_region_pairwise_distance == 0.0

    def test_min_pairwise_distance_is_positive_for_distinct_embeddings(self) -> None:
        embeddings = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        report = measure_instruction_separation(
            embeddings,
            instructions=["a", "b", "c"],
            region_names=["r1", "r2", "r3"],
            collapse_epsilon=0.1,
        )
        assert report.min_pairwise_distance > 0.0

    def test_cross_region_distance_ignores_same_region_pairs(self) -> None:
        # Two instructions in the same region ("r1") are deliberately placed
        # close together (synonyms are expected to be close); a third, in a
        # different region, is far away. The global min should pick up the
        # close same-region pair; the cross-region min should not.
        embeddings = np.array([[0.0, 0.0], [0.01, 0.0], [10.0, 10.0]])
        report = measure_instruction_separation(
            embeddings,
            instructions=["a1", "a2", "b1"],
            region_names=["r1", "r1", "r2"],
            collapse_epsilon=0.1,
        )
        assert report.min_pairwise_distance < 0.1
        assert report.min_cross_region_pairwise_distance > 1.0

    def test_is_collapsed_true_when_cross_region_distance_below_epsilon(self) -> None:
        embeddings = np.array([[0.0, 0.0], [0.001, 0.0]])
        report = measure_instruction_separation(
            embeddings,
            instructions=["a", "b"],
            region_names=["r1", "r2"],
            collapse_epsilon=0.5,
        )
        assert report.is_collapsed is True

    def test_is_collapsed_false_when_cross_region_distance_above_epsilon(self) -> None:
        embeddings = np.array([[0.0, 0.0], [5.0, 0.0]])
        report = measure_instruction_separation(
            embeddings,
            instructions=["a", "b"],
            region_names=["r1", "r2"],
            collapse_epsilon=0.5,
        )
        assert report.is_collapsed is False

    def test_raises_when_only_one_region_is_present(self) -> None:
        embeddings = np.array([[0.0, 0.0], [1.0, 0.0]])
        try:
            measure_instruction_separation(
                embeddings,
                instructions=["a", "b"],
                region_names=["r1", "r1"],
                collapse_epsilon=0.1,
            )
        except ValueError:
            return
        raise AssertionError("expected ValueError when no cross-region pair exists")


class TestCollapseEpsilonFromGoalEncoder:
    """collapse_epsilon_from_goal_encoder grounds the threshold in the real target space's scale."""

    def test_returns_a_positive_float(self) -> None:
        goal_encoder = GoalEncoder(goal_dim=3, embed_dim=4, hidden_dim=8)
        epsilon = collapse_epsilon_from_goal_encoder(
            goal_encoder,
            region_names=["reach left", "reach right", "reach up high"],
            box=SYNTHETIC_BOX,
            n_samples=20,
            seed=0,
        )
        assert epsilon > 0.0

    def test_smaller_fraction_gives_a_smaller_epsilon(self) -> None:
        goal_encoder = GoalEncoder(goal_dim=3, embed_dim=4, hidden_dim=8)
        small = collapse_epsilon_from_goal_encoder(
            goal_encoder,
            region_names=["reach left", "reach right"],
            box=SYNTHETIC_BOX,
            n_samples=20,
            seed=0,
            fraction=0.05,
        )
        large = collapse_epsilon_from_goal_encoder(
            goal_encoder,
            region_names=["reach left", "reach right"],
            box=SYNTHETIC_BOX,
            n_samples=20,
            seed=0,
            fraction=0.5,
        )
        assert small < large


class TestCheckNoCollapse:
    """check_no_collapse composes projection + epsilon grounding into one pass/fail report."""

    def test_flags_collapse_for_a_constant_projection(self) -> None:
        torch.manual_seed(0)
        goal_encoder = GoalEncoder(goal_dim=3, embed_dim=4, hidden_dim=8)
        instructions = ["move left", "move right"]
        region_names = ["reach left", "reach right"]
        sentence_embeddings = torch.randn(2, 8)
        projection = ConstantProjection(input_dim=8, embed_dim=4, hidden_dim=6)

        report = check_no_collapse(
            projection,
            goal_encoder,
            sentence_embeddings,
            instructions,
            region_names,
            box=SYNTHETIC_BOX,
        )

        assert report.is_collapsed is True

    def test_does_not_flag_collapse_for_a_well_trained_projection(self) -> None:
        torch.manual_seed(0)
        from lang_goal_rl.language_goal_projection import train_projection

        goal_encoder = GoalEncoder(goal_dim=3, embed_dim=4, hidden_dim=8)
        instructions = ["move left", "move right"]
        region_names = ["reach left", "reach right"]
        sentence_embeddings = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        )

        projection, _history = train_projection(
            goal_encoder,
            sentence_embeddings,
            region_names,
            box=SYNTHETIC_BOX,
            n_steps=200,
            n_target_samples=30,
            learning_rate=5e-3,
            seed=0,
            projection=LanguageGoalProjection(input_dim=8, embed_dim=4, hidden_dim=6),
        )

        report = check_no_collapse(
            projection,
            goal_encoder,
            sentence_embeddings,
            instructions,
            region_names,
            box=SYNTHETIC_BOX,
        )

        assert report.is_collapsed is False
