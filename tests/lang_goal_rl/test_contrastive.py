"""Tests for the InfoNCE-style contrastive loss used to pretrain GoalEncoder.

The loss should reward a batch where each anchor's embedding is closest to
its own matched positive (the state's true achieved-goal embedding) among
all positives in the batch, and penalize a batch where positives are
randomly shuffled relative to their anchors. This is the mechanism the
stage-2 proof gate leans on for "distance-in-latent correlates with true
task distance".
"""

import torch

from lang_goal_rl.contrastive import info_nce_loss


class TestInfoNceLoss:
    """info_nce_loss scores matched anchor/positive pairs against negatives."""

    def test_loss_is_lower_for_matched_pairs_than_shuffled_pairs(self) -> None:
        torch.manual_seed(0)
        batch_size, embed_dim = 8, 16
        anchors = torch.nn.functional.normalize(torch.randn(batch_size, embed_dim), dim=1)
        # Matched positives: near-identical to their anchor (small noise).
        matched_positives = anchors + 0.01 * torch.randn(batch_size, embed_dim)

        matched_loss = info_nce_loss(anchors, matched_positives)

        shuffle = torch.randperm(batch_size)
        shuffled_positives = matched_positives[shuffle]
        shuffled_loss = info_nce_loss(anchors, shuffled_positives)

        assert matched_loss.item() < shuffled_loss.item()

    def test_returns_scalar_tensor(self) -> None:
        anchors = torch.randn(4, 8)
        positives = torch.randn(4, 8)
        loss = info_nce_loss(anchors, positives)
        assert loss.dim() == 0

    def test_loss_is_finite_and_nonnegative(self) -> None:
        anchors = torch.randn(6, 10)
        positives = torch.randn(6, 10)
        loss = info_nce_loss(anchors, positives)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_raises_on_mismatched_batch_sizes(self) -> None:
        anchors = torch.randn(4, 8)
        positives = torch.randn(5, 8)
        try:
            info_nce_loss(anchors, positives)
        except ValueError:
            return
        raise AssertionError("expected ValueError for mismatched batch sizes")

    def test_temperature_controls_loss_sharpness(self) -> None:
        torch.manual_seed(1)
        anchors = torch.nn.functional.normalize(torch.randn(8, 16), dim=1)
        positives = anchors.clone()

        loss_low_temp = info_nce_loss(anchors, positives, temperature=0.05)
        loss_high_temp = info_nce_loss(anchors, positives, temperature=1.0)

        # With perfectly matched pairs, a sharper (lower) temperature drives
        # the loss closer to zero than a flatter (higher) temperature.
        assert loss_low_temp.item() < loss_high_temp.item()
