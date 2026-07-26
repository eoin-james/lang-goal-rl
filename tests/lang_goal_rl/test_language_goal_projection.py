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

import numpy as np
import pytest
import torch

from lang_goal_rl.contrastive import info_nce_loss
from lang_goal_rl.goal_encoder import GoalEncoder
from lang_goal_rl.goal_region_vocabulary import GoalBox, compute_region_target_embeddings
from lang_goal_rl.language_goal_projection import (
    LanguageGoalProjection,
    ProjectionNormCheck,
    _region_mean_embeddings,
    check_projection_norm_range,
    combined_projection_loss,
    measure_reference_norms,
    train_projection,
)

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

    def test_trained_projection_output_norms_track_target_region_norms(self) -> None:
        """Regression test for stage 3's FAIL: with the norm-matching term enabled,
        the trained projection's output norms should land close to its regions'
        true target-embedding norms -- not merely well-separated from each other.
        """
        torch.manual_seed(0)
        goal_encoder = GoalEncoder(goal_dim=3, embed_dim=4, hidden_dim=8)
        sentence_embeddings = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        )
        region_names = ["reach left", "reach right"]

        projection, _loss_history = train_projection(
            goal_encoder,
            sentence_embeddings,
            region_names,
            box=SYNTHETIC_BOX,
            n_steps=150,
            n_goal_samples_per_step=16,
            learning_rate=5e-3,
            seed=0,
            projection=LanguageGoalProjection(input_dim=8, embed_dim=4, hidden_dim=6),
            norm_loss_weight=10.0,
        )

        target_embeddings = compute_region_target_embeddings(
            goal_encoder, region_names, box=SYNTHETIC_BOX, n_samples=500, seed=999,
        )
        with torch.no_grad():
            output_norms = projection(sentence_embeddings).norm(dim=1)
        target_norms = target_embeddings.norm(dim=1)

        # Not exact (both sides are stochastic estimates from a small net),
        # but should be within 25% of each other -- a world apart from stage
        # 3's actual 5-10x mismatch.
        assert torch.allclose(output_norms, target_norms, rtol=0.25, atol=0.02)

    def test_norm_loss_weight_zero_reproduces_the_old_scale_unconstrained_behavior(self) -> None:
        """`norm_loss_weight=0.0` should recover the pre-fix behavior: the loss
        used for the very first step is exactly `info_nce_loss` on that step's
        anchor/positive, with no scale term mixed in.

        Reuses the exact same `goal_encoder`/`projection` instances for both
        the manual reference computation and the `train_projection` call
        (rather than reconstructing them) so the two computations start from
        identical weights -- no need to replay torch's global RNG stream.
        """
        goal_encoder = GoalEncoder(goal_dim=3, embed_dim=4, hidden_dim=8)
        sentence_embeddings = torch.randn(2, 8)
        region_names = ["reach left", "reach right"]
        projection = LanguageGoalProjection(input_dim=8, embed_dim=4, hidden_dim=6)

        # Step 0's positive only depends on numpy's RNG (independent of
        # torch's), so this reproduces exactly what train_projection's first
        # iteration will draw for `seed=0`.
        step_seed = int(np.random.default_rng(0).integers(0, 2**31 - 1))
        expected_positive = _region_mean_embeddings(goal_encoder, region_names, SYNTHETIC_BOX, 8, step_seed)
        expected_anchor = projection(sentence_embeddings.detach().to(torch.float32))
        expected_first_loss = info_nce_loss(expected_anchor, expected_positive).item()

        _projection, loss_history_no_norm_term = train_projection(
            goal_encoder,
            sentence_embeddings,
            region_names,
            box=SYNTHETIC_BOX,
            n_steps=1,
            n_goal_samples_per_step=8,
            seed=0,
            projection=projection,
            norm_loss_weight=0.0,
        )

        assert loss_history_no_norm_term[0] == expected_first_loss


class TestCombinedProjectionLoss:
    """combined_projection_loss adds a norm-matching term that info_nce_loss alone cannot express."""

    def test_plain_info_nce_loss_is_scale_invariant_to_anchor_rescaling(self) -> None:
        """Root-cause demonstration: `info_nce_loss` normalizes both inputs
        internally, so rescaling the anchor by any positive constant has
        provably zero effect -- this is *why* stage 3's `train_projection`
        could not learn a scale-correct output using this loss alone.
        """
        torch.manual_seed(0)
        anchor = torch.randn(6, 5)
        positive = torch.randn(6, 5)

        baseline = info_nce_loss(anchor, positive)
        rescaled_small = info_nce_loss(anchor * 0.01, positive)
        rescaled_huge = info_nce_loss(anchor * 500.0, positive)

        assert torch.isclose(baseline, rescaled_small, atol=1e-5)
        assert torch.isclose(baseline, rescaled_huge, atol=1e-5)

    def test_combined_loss_is_not_scale_invariant_and_penalizes_norm_mismatch(self) -> None:
        """The fix: adding the norm-matching term breaks the scale invariance
        `info_nce_loss` alone has, and specifically penalizes an anchor whose
        norm has drifted away from its positive's norm.
        """
        torch.manual_seed(0)
        anchor = torch.randn(6, 5)
        positive = torch.randn(6, 5)

        baseline = combined_projection_loss(anchor, positive, norm_loss_weight=10.0)
        rescaled_huge = combined_projection_loss(anchor * 50.0, positive, norm_loss_weight=10.0)

        assert not torch.isclose(baseline, rescaled_huge, atol=1e-3)
        assert rescaled_huge.item() > baseline.item()

    def test_zero_norm_loss_weight_reduces_to_plain_info_nce_loss(self) -> None:
        torch.manual_seed(0)
        anchor = torch.randn(4, 3)
        positive = torch.randn(4, 3)

        combined = combined_projection_loss(anchor, positive, norm_loss_weight=0.0)
        plain = info_nce_loss(anchor, positive)

        assert torch.isclose(combined, plain)

    def test_returns_scalar_tensor(self) -> None:
        anchor = torch.randn(3, 4)
        positive = torch.randn(3, 4)
        loss = combined_projection_loss(anchor, positive)
        assert loss.dim() == 0


class TestMeasureReferenceNorms:
    """measure_reference_norms samples the frozen encoder's real operating-range norms."""

    def test_returns_one_norm_per_sample(self) -> None:
        goal_encoder = GoalEncoder(goal_dim=3, embed_dim=4, hidden_dim=8)
        norms = measure_reference_norms(goal_encoder, box=SYNTHETIC_BOX, n_samples=50, seed=0)
        assert norms.shape == (50,)
        assert torch.isfinite(norms).all()
        assert (norms >= 0).all()

    def test_deterministic_for_a_given_seed(self) -> None:
        goal_encoder = GoalEncoder(goal_dim=3, embed_dim=4, hidden_dim=8)
        first = measure_reference_norms(goal_encoder, box=SYNTHETIC_BOX, n_samples=20, seed=7)
        second = measure_reference_norms(goal_encoder, box=SYNTHETIC_BOX, n_samples=20, seed=7)
        assert torch.equal(first, second)

    def test_leaves_goal_encoder_parameters_unchanged(self) -> None:
        goal_encoder = GoalEncoder(goal_dim=3, embed_dim=4, hidden_dim=8)
        before = [p.clone() for p in goal_encoder.parameters()]
        measure_reference_norms(goal_encoder, box=SYNTHETIC_BOX, n_samples=20, seed=0)
        after = list(goal_encoder.parameters())
        for p_before, p_after in zip(before, after, strict=True):
            assert torch.equal(p_before, p_after)


class TestCheckProjectionNormRange:
    """check_projection_norm_range is the fail-fast gate against a repeat of stage 3's scale mismatch."""

    def test_flags_a_projection_whose_norms_are_far_outside_the_reference_range(self) -> None:
        # Reference norms cluster tightly around 0.04, matching the real
        # measured GoalEncoder range from experiments/03's report.
        reference_norms = torch.full((100,), 0.04)
        # Reproduces stage 3's actual failure shape: outputs ~5-10x the reference mean.
        projected = torch.tensor([[0.3, 0.0], [0.0, 0.35]])

        result = check_projection_norm_range(projected, reference_norms, ["reach up high", "reach down low"])

        assert result.passed is False
        assert result.out_of_range_instructions == ("reach up high", "reach down low")

    def test_passes_a_projection_whose_norms_sit_within_tolerance(self) -> None:
        reference_norms = torch.full((100,), 0.04)
        projected = torch.tensor([[0.03, 0.0], [0.0, 0.05]])

        result = check_projection_norm_range(projected, reference_norms, ["a", "b"])

        assert result.passed is True
        assert result.out_of_range_instructions == ()

    def test_flags_only_the_specific_instruction_that_is_out_of_range(self) -> None:
        reference_norms = torch.full((100,), 0.04)
        projected = torch.tensor([[0.04, 0.0], [0.0, 1.0]])

        result = check_projection_norm_range(projected, reference_norms, ["in range", "out of range"])

        assert result.passed is False
        assert result.out_of_range_instructions == ("out of range",)

    def test_bounds_are_derived_from_reference_mean_and_tolerance(self) -> None:
        reference_norms = torch.full((10,), 0.1)
        projected = torch.tensor([[0.1, 0.0]])

        result = check_projection_norm_range(projected, reference_norms, ["x"], tolerance=3.0)

        assert result.lower_bound == pytest.approx(0.1 / 3.0)
        assert result.upper_bound == pytest.approx(0.1 * 3.0)

    def test_raises_on_row_count_mismatch(self) -> None:
        reference_norms = torch.full((10,), 0.1)
        projected = torch.tensor([[0.1, 0.0], [0.2, 0.0]])

        try:
            check_projection_norm_range(projected, reference_norms, ["only-one-label"])
        except ValueError:
            return
        raise AssertionError("expected ValueError for row count mismatch")

    def test_returns_a_projection_norm_check_instance(self) -> None:
        reference_norms = torch.full((10,), 0.1)
        projected = torch.tensor([[0.1, 0.0]])
        result = check_projection_norm_range(projected, reference_norms, ["x"])
        assert isinstance(result, ProjectionNormCheck)

    def test_summary_is_a_printable_string_that_reports_pass_fail_and_bounds(self) -> None:
        reference_norms = torch.full((100,), 0.04)
        projected = torch.tensor([[0.3, 0.0]])
        result = check_projection_norm_range(projected, reference_norms, ["reach up high"])

        text = result.summary()

        assert isinstance(text, str)
        assert "FAILED" in text
        assert "reach up high" in text
        assert "OUT OF RANGE" in text

    def test_summary_reports_passed_when_in_range(self) -> None:
        reference_norms = torch.full((100,), 0.04)
        projected = torch.tensor([[0.04, 0.0]])
        result = check_projection_norm_range(projected, reference_norms, ["ok"])

        text = result.summary()

        assert "PASSED" in text
        assert "OUT OF RANGE" not in text
