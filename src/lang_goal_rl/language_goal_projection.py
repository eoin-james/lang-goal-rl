"""LanguageGoalProjection: frozen sentence embedding -> stage 2's goal-embedding space.

Stage 3's proof gate needs a sentence to land at a point in the *same*
16-dim space stage 2's `GoalEncoder` already defines (see
`goal_encoder.py`), so stage 2's trained policies stay chainable. This
module owns two things:

1. `LanguageGoalProjection` — the small learnable layer doing the mapping.
2. `train_projection` — the procedure that fits it, given a frozen
   `GoalEncoder` and a fixed instruction vocabulary's region assignments
   (see `goal_region_vocabulary.py`).

Loss choice: for each instruction, a batch of true xyz goals is sampled from
its region (`sample_region_goals`) and averaged through the frozen
`GoalEncoder` to get that region's mean embedding — a stochastic estimate of
"where this region truly sits". Regressing the projection's output straight
to that mean with MSE (no separation term) was considered and rejected:
nothing prevents two regions whose *true* mean embeddings happen to sit
close together (adjacent regions of a small, contrastively-pretrained goal
space aren't guaranteed to be far apart) from converging to near-identical
projected points — that's exactly the failure stage 3's proof gate needs
ruled out. Reusing `contrastive.info_nce_loss` (already used to pretrain
`GoalEncoder` itself, see `contrastive.py`) with each instruction's
projected embedding as the anchor and its region's mean embedding as the
positive gets the MSE-like pull *and* an explicit push against every other
instruction's region embedding in the same training batch — directly
targeting "distinct instructions don't collapse".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from lang_goal_rl.contrastive import info_nce_loss
from lang_goal_rl.goal_encoder import DEFAULT_EMBED_DIM
from lang_goal_rl.goal_region_vocabulary import MEASURED_GOAL_BOX, sample_region_goals

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lang_goal_rl.goal_encoder import GoalEncoder
    from lang_goal_rl.goal_region_vocabulary import GoalBox

DEFAULT_INPUT_DIM = 384
"""Matches `language_embedding.LANGUAGE_EMBED_DIM` (`all-MiniLM-L6-v2`'s output size)."""


class LanguageGoalProjection(nn.Module):
    """Small MLP mapping a (batch, input_dim) sentence embedding to (batch, embed_dim).

    Same Linear-ReLU-Linear shape as `GoalEncoder` (see `goal_encoder.py`),
    for the same reason: a single hidden layer gives the mapping enough
    capacity to reach a nonlinear region of the 16-dim target space without
    over-parameterizing a projection that's only ever trained against a
    closed, ~14-instruction fixed vocabulary — a deeper network would mostly
    add overfitting risk here, not useful capacity.
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        embed_dim: int = DEFAULT_EMBED_DIM,
        hidden_dim: int = 64,
    ) -> None:
        """Build the projection's layers.

        Args:
            input_dim: Dimensionality of the frozen sentence embedding
                (384 for `all-MiniLM-L6-v2`).
            embed_dim: Dimensionality of stage 2's goal-embedding space (16
                for `GoalEncoder`'s default).
            hidden_dim: Width of the single hidden layer.

        """
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, sentence_embeddings: torch.Tensor) -> torch.Tensor:
        """Map a batch of sentence embeddings to stage 2's goal-embedding space.

        Args:
            sentence_embeddings: Tensor of shape (batch, input_dim).

        Returns:
            Tensor of shape (batch, embed_dim).

        """
        return self.net(sentence_embeddings.to(torch.float32))


def _region_mean_embeddings(
    goal_encoder: GoalEncoder,
    region_names: Sequence[str],
    box: GoalBox,
    n_samples: int,
    seed: int,
) -> torch.Tensor:
    """One row per name in `region_names`: the mean frozen-encoder embedding of a fresh region sample.

    Resampled (with a different `seed`) by the caller on every training
    step, so this is a noisy but unbiased per-step estimate of each region's
    true embedding centroid rather than a single fixed target — acting as a
    mild data-augmentation effect on the regression/contrastive target.
    """
    rows = []
    with torch.no_grad():
        for i, name in enumerate(region_names):
            goals = sample_region_goals(name, n_samples, seed=seed + i, box=box)
            embeddings = goal_encoder(torch.from_numpy(goals).float())
            rows.append(embeddings.mean(dim=0))
    return torch.stack(rows)


def train_projection(
    goal_encoder: GoalEncoder,
    sentence_embeddings: torch.Tensor,
    region_names: Sequence[str],
    *,
    box: GoalBox = MEASURED_GOAL_BOX,
    n_steps: int = 500,
    n_goal_samples_per_step: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 0,
    projection: LanguageGoalProjection | None = None,
) -> tuple[LanguageGoalProjection, list[float]]:
    """Train a projection so each instruction's embedding lands near its region and away from others.

    `goal_encoder` is frozen throughout (its parameters' `requires_grad` is
    left untouched, and the region-embedding computation runs under
    `torch.no_grad()`) — only `projection`'s weights are updated.

    Args:
        goal_encoder: Stage 2's pretrained `GoalEncoder`, used as a frozen
            source of ground-truth region embeddings.
        sentence_embeddings: Precomputed frozen sentence embeddings, shape
            (n_instructions, input_dim), one row per instruction. Computed
            once outside this function (e.g. via
            `language_embedding.encode_instructions`) since the encoder is
            frozen and the fixed vocabulary's text never changes.
        region_names: Region name for each row of `sentence_embeddings`, same
            length and order (e.g. via
            `goal_region_vocabulary.instruction_to_region`).
        box: Goal box to sample regions within.
        n_steps: Number of optimizer steps.
        n_goal_samples_per_step: xyz samples averaged per region, per step,
            to estimate that step's target embedding (see
            `_region_mean_embeddings`).
        learning_rate: Adam learning rate for `projection`'s parameters.
        seed: Seed for region-sampling randomness across training steps.
        projection: An existing `LanguageGoalProjection` to continue training.
            If `None`, a fresh one is constructed sized to
            `sentence_embeddings`' and `goal_encoder`'s dimensions.

    Returns:
        A tuple `(projection, loss_history)`: the trained module and the
        InfoNCE loss recorded at every step (useful for a caller checking
        that training actually reduced the loss).

    Raises:
        ValueError: If `sentence_embeddings` and `region_names` have
            different lengths.

    """
    if sentence_embeddings.shape[0] != len(region_names):
        msg = (
            f"row count mismatch: sentence_embeddings has {sentence_embeddings.shape[0]} rows, "
            f"region_names has {len(region_names)}"
        )
        raise ValueError(msg)

    resolved_projection = projection or LanguageGoalProjection(
        input_dim=sentence_embeddings.shape[1], embed_dim=goal_encoder.embed_dim,
    )

    goal_encoder.eval()
    for parameter in goal_encoder.parameters():
        parameter.requires_grad = False

    optimizer = torch.optim.Adam(resolved_projection.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)

    frozen_sentence_embeddings = sentence_embeddings.detach().to(torch.float32)

    loss_history: list[float] = []
    for _step in range(n_steps):
        step_seed = int(rng.integers(0, 2**31 - 1))
        positive = _region_mean_embeddings(
            goal_encoder, region_names, box, n_goal_samples_per_step, step_seed,
        )
        anchor = resolved_projection(frozen_sentence_embeddings)

        loss = info_nce_loss(anchor, positive)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    return resolved_projection, loss_history
