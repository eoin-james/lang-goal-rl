"""Tests for the GoalEncoder MLP that maps literal xyz goals to embeddings.

Covers the three properties the stage-2 proof gate depends on: the encoder
produces the configured embedding dimensionality, is deterministic (no
stochastic layers), and handles batched input the way SB3's feature
extractors feed it.
"""

import torch

from lang_goal_rl.goal_encoder import GoalEncoder


class TestGoalEncoder:
    """GoalEncoder maps a (batch, goal_dim) tensor to (batch, embed_dim)."""

    def test_output_shape_matches_configured_embed_dim(self) -> None:
        encoder = GoalEncoder(goal_dim=3, embed_dim=16)
        goals = torch.zeros(5, 3)
        embeddings = encoder(goals)
        assert embeddings.shape == (5, 16)

    def test_output_shape_for_single_sample(self) -> None:
        encoder = GoalEncoder(goal_dim=3, embed_dim=16)
        goals = torch.rand(1, 3)
        embeddings = encoder(goals)
        assert embeddings.shape == (1, 16)

    def test_deterministic_given_fixed_weights(self) -> None:
        encoder = GoalEncoder(goal_dim=3, embed_dim=16)
        encoder.eval()
        goals = torch.rand(4, 3)
        first_pass = encoder(goals)
        second_pass = encoder(goals)
        assert torch.equal(first_pass, second_pass)

    def test_handles_large_batch(self) -> None:
        encoder = GoalEncoder(goal_dim=3, embed_dim=16)
        goals = torch.rand(256, 3)
        embeddings = encoder(goals)
        assert embeddings.shape == (256, 16)

    def test_supports_custom_embed_dim_and_goal_dim(self) -> None:
        encoder = GoalEncoder(goal_dim=7, embed_dim=4)
        goals = torch.rand(3, 7)
        embeddings = encoder(goals)
        assert embeddings.shape == (3, 4)
