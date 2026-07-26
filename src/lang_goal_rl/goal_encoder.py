"""GoalEncoder: a small MLP mapping a literal xyz goal to a learned embedding.

Stage 2 swaps the literal goal FetchReach hands to the policy/critic for a
learned continuous embedding. This is the encoder that produces it — see
`goal_embedding_extractor.py` for where it plugs into SB3, and
`contrastive.py` for the objective used to pretrain it so that distance in
its output space tracks true xyz distance.
"""

from __future__ import annotations

import torch
from torch import nn

DEFAULT_EMBED_DIM = 16
"""Embedding dimensionality used unless the caller overrides it.

FetchReach's goal is 3D xyz. 16 is picked as a modest expansion — enough
capacity for the encoder to learn a nonlinear reparameterization (and for
the contrastive objective to spread distinct goals apart) without making
the policy/critic input dominated by goal features, given FetchReach's
`observation` field is only 10D.
"""


class GoalEncoder(nn.Module):
    """Small MLP mapping a (batch, goal_dim) literal goal to (batch, embed_dim).

    Deterministic by construction (no dropout, no batch normalization) so
    that repeated calls with the same weights and input are reproducible —
    required for HER-style relabeling, which re-encodes the same literal
    goal many times across a replay buffer.
    """

    def __init__(
        self,
        goal_dim: int = 3,
        embed_dim: int = DEFAULT_EMBED_DIM,
        hidden_dim: int = 64,
    ) -> None:
        """Build the encoder's layers.

        Args:
            goal_dim: Dimensionality of the literal goal vector (3 for
                FetchReach's xyz `achieved_goal`/`desired_goal`).
            embed_dim: Dimensionality of the output embedding.
            hidden_dim: Width of the single hidden layer.
        """
        super().__init__()
        self.goal_dim = goal_dim
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        """Map a batch of literal goals to embeddings.

        Args:
            goal: Tensor of shape (batch, goal_dim).

        Returns:
            Tensor of shape (batch, embed_dim).
        """
        return self.net(goal.to(torch.float32))
