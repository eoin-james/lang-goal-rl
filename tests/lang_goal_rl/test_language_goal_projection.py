"""Tests for LanguageGoalProjection and its training procedure.

LanguageGoalProjection maps a frozen sentence-transformer embedding
(384-dim) into stage 2's `GoalEncoder` output space (16-dim) — the
mapping the roadmap's stage 3 proof gate depends on. `train_projection`
fits it against a fixed instruction vocabulary using region-grounded xyz
sampling (see `goal_region_vocabulary.py`) and stage 2's frozen encoder as
the regression/contrastive target — never against gym/MuJoCo directly, so
these tests stay fast and offline.
"""

from __future__ import annotations

import torch

from lang_goal_rl.goal_encoder import GoalEncoder
from lang_goal_rl.goal_region_vocabulary import GoalBox
from lang_goal_rl.language_goal_projection import LanguageGoalProjection, train_projection

SYNTHETIC_BOX = GoalBox(axis_min=torch.zeros(3).numpy(), axis_max=torch.ones(3).numpy())


class TestLanguageGoalProjection:
    """LanguageGoalProjection is a trainable nn.Module mapping input_dim -> embed_dim."""

    def test_forward_output_shape_matches_embed_dim(self) -> None:
        projection = LanguageGoalProjection(input_dim=384, embed_dim=16)
        batch = torch.randn(5, 384)
        output = projection(batch)
        assert output.shape == (5, 16)

    def test_default_dims_match_minilm_output_and_stage_two_embed_dim(self) -> None:
        projection = LanguageGoalProjection()
        batch = torch.randn(3, 384)
        assert projection(batch).shape == (3, 16)

    def test_is_a_real_trainable_module_with_parameters(self) -> None:
        projection = LanguageGoalProjection(input_dim=8, embed_dim=4, hidden_dim=6)
        parameters = list(projection.parameters())
        assert len(parameters) > 0
        assert all(p.requires_grad for p in parameters)

    def test_forward_pass_on_a_batch_produces_finite_output(self) -> None:
        projection = LanguageGoalProjection(input_dim=8, embed_dim=4, hidden_dim=6)
        batch = torch.randn(10, 8)
        output = projection(batch)
        assert torch.isfinite(output).all()

    def test_different_inputs_produce_different_outputs_before_training(self) -> None:
        torch.manual_seed(0)
        projection = LanguageGoalProjection(input_dim=8, embed_dim=4, hidden_dim=6)
        a = torch.randn(1, 8)
        b = torch.randn(1, 8)
        assert not torch.allclose(projection(a), projection(b))


class TestTrainProjection:
    """train_projection fits a projection to separate distinct instructions' regions."""

    def test_returns_a_language_goal_projection_and_a_loss_history(self) -> None:
        torch.manual_seed(0)
        goal_encoder = GoalEncoder(goal_dim=3, embed_dim=4, hidden_dim=8)
        sentence_embeddings = torch.randn(2, 8)
        region_names = ["reach left", "reach right"]

        projection, loss_history = train_projection(
            goal_encoder,
            sentence_embeddings,
            region_names,
            box=SYNTHETIC_BOX,
            n_steps=5,
            n_goal_samples_per_step=8,
            seed=0,
            projection=LanguageGoalProjection(input_dim=8, embed_dim=4, hidden_dim=6),
        )

        assert isinstance(projection, LanguageGoalProjection)
        assert len(loss_history) == 5

    def test_training_reduces_mean_loss(self) -> None:
        torch.manual_seed(0)
        goal_encoder = GoalEncoder(goal_dim=3, embed_dim=4, hidden_dim=8)
        # Two well-separated fake "instructions" pointing at two distinct
        # regions -- a projection with enough steps should learn to tell
        # them apart, i.e. the contrastive loss should trend down.
        sentence_embeddings = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        )
        region_names = ["reach left", "reach right"]

        _projection, loss_history = train_projection(
            goal_encoder,
            sentence_embeddings,
            region_names,
            box=SYNTHETIC_BOX,
            n_steps=150,
            n_goal_samples_per_step=16,
            learning_rate=5e-3,
            seed=0,
            projection=LanguageGoalProjection(input_dim=8, embed_dim=4, hidden_dim=6),
        )

        early_mean = sum(loss_history[:20]) / 20
        late_mean = sum(loss_history[-20:]) / 20
        assert late_mean < early_mean

    def test_leaves_goal_encoder_parameters_unchanged(self) -> None:
        torch.manual_seed(0)
        goal_encoder = GoalEncoder(goal_dim=3, embed_dim=4, hidden_dim=8)
        before = [p.clone() for p in goal_encoder.parameters()]
        sentence_embeddings = torch.randn(2, 8)
        region_names = ["reach left", "reach right"]

        train_projection(
            goal_encoder,
            sentence_embeddings,
            region_names,
            box=SYNTHETIC_BOX,
            n_steps=5,
            n_goal_samples_per_step=8,
            seed=0,
            projection=LanguageGoalProjection(input_dim=8, embed_dim=4, hidden_dim=6),
        )

        after = list(goal_encoder.parameters())
        for p_before, p_after in zip(before, after, strict=True):
            assert torch.equal(p_before, p_after)

    def test_constructs_a_default_projection_when_none_is_provided(self) -> None:
        torch.manual_seed(0)
        goal_encoder = GoalEncoder(goal_dim=3, embed_dim=16, hidden_dim=8)
        sentence_embeddings = torch.randn(2, 384)
        region_names = ["reach left", "reach right"]

        projection, _loss_history = train_projection(
            goal_encoder,
            sentence_embeddings,
            region_names,
            box=SYNTHETIC_BOX,
            n_steps=2,
            n_goal_samples_per_step=8,
            seed=0,
        )

        assert projection.input_dim == 384
        assert projection.embed_dim == 16
