"""InfoNCE-style contrastive loss for pretraining `GoalEncoder`.

Scoped adaptation of Eysenbach et al.'s Contrastive RL, not a reproduction:
the paper replaces the TD critic loss itself with a contrastive objective
trained jointly with RL. Here the contrastive objective is used to
*pretrain* `GoalEncoder` on sampled (state's achieved-goal, desired-goal)
pairs before RL training — see `goal_embedding_extractor.py` for how the
resulting encoder is then frozen and dropped into SB3's feature extractor.
The RL loop and HER's replay buffer never see this loss directly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as f  # noqa: N812 -- `f` mirrors the common torch.nn.functional alias

DEFAULT_TEMPERATURE = 0.1
"""Softmax temperature. Lower sharpens the contrast between the matched
positive and the in-batch negatives; 0.1 is a standard InfoNCE default."""


def info_nce_loss(
    anchor_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
) -> torch.Tensor:
    """Compute an InfoNCE loss between matched anchor/positive embedding pairs.

    Each row `i` of `anchor_embeddings` is treated as matched with row `i`
    of `positive_embeddings` (e.g. a state's embedding and its true
    achieved-goal's embedding); every other row of `positive_embeddings` in
    the batch serves as an in-batch negative. Minimizing this loss pulls
    matched pairs together (higher cosine similarity) and pushes unmatched
    pairs apart, which is the mechanism intended to make embedding-space
    distance track true task-relevant distance.

    Args:
        anchor_embeddings: Tensor of shape (batch, embed_dim).
        positive_embeddings: Tensor of shape (batch, embed_dim), row-aligned
            with `anchor_embeddings`.
        temperature: Softmax temperature applied to the similarity logits.

    Returns:
        Scalar loss tensor (mean cross-entropy over the batch).

    Raises:
        ValueError: If the two inputs have different batch sizes.
    """
    if anchor_embeddings.shape[0] != positive_embeddings.shape[0]:
        msg = (
            f"batch size mismatch: anchor_embeddings has "
            f"{anchor_embeddings.shape[0]} rows, positive_embeddings has "
            f"{positive_embeddings.shape[0]}"
        )
        raise ValueError(msg)

    anchors = f.normalize(anchor_embeddings, dim=1)
    positives = f.normalize(positive_embeddings, dim=1)

    similarity_logits = anchors @ positives.T / temperature
    matched_indices = torch.arange(anchors.shape[0], device=anchors.device)
    return f.cross_entropy(similarity_logits, matched_indices)
