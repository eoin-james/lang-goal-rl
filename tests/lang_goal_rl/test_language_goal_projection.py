"""Tests for LanguageGoalProjection and its training procedure.

LanguageGoalProjection maps a frozen sentence-transformer embedding
(384-dim) into stage 2's `GoalEncoder` output space (16-dim) — the
mapping the roadmap's stage 3 proof gate depends on. `train_projection`
fits it against a fixed instruction vocabulary using region-grounded xyz
sampling (see `goal_region_vocabulary.py`) and stage 2's frozen encoder as
the regression target — never against gym/MuJoCo directly, so these tests
stay fast and offline.

**Attempt 3 rewrite (see `experiments/03_language_goal_projection/report.md`,
attempt 2's reviewer verdict):** attempt 2's InfoNCE + norm-matching loss
resampled each region's target embedding every training step from a small
64-sample batch — a stochastic estimate that adds directional noise the
closed, 14-instruction vocabulary doesn't need to tolerate, since the
target regions' true centroids are already known to be well separated
(24.68x collapse margin, independent of anything the projection learns).
This file now tests direct regression to a *fixed*, precomputed-once
target per instruction instead.
"""

from __future__ import annotations

import pytest
import torch

from lang_goal_rl.goal_encoder import GoalEncoder
from lang_goal_rl.goal_region_vocabulary import GoalBox, compute_region_target_embeddings
from lang_goal_rl.language_goal_projection import (
    LanguageGoalProjection,
    ProjectionNormCheck,
    check_projection_norm_range,
    measure_reference_norms,
    precompute_instruction_targets,
    regression_loss,
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


class TestRegressionLoss:
    """regression_loss is a plain MSE pull toward a fixed target -- no scale invariance, no separation term."""

    def test_pulls_a_random_initial_output_toward_a_known_fixed_target(self) -> None:
        """The core claim this rewrite depends on: unlike InfoNCE's stochastic
        per-step targets, a *fixed* target is a simple, directly checkable
        optimization problem -- no noise to control for.
        """
        torch.manual_seed(0)
        anchor = torch.randn(3, 4, requires_grad=True)
        target = torch.tensor([[1.0, 2.0, -1.0, 0.5], [0.0, 0.0, 0.0, 0.0], [-2.0, 1.0, 1.0, -1.0]])

        optimizer = torch.optim.Adam([anchor], lr=0.1)
        for _ in range(300):
            optimizer.zero_grad()
            loss = regression_loss(anchor, target)
            loss.backward()
            optimizer.step()

        assert torch.allclose(anchor.detach(), target, atol=0.05)

    def test_zero_for_identical_tensors(self) -> None:
        x = torch.randn(4, 5)
        assert regression_loss(x, x).item() == 0.0

    def test_larger_for_a_bigger_mismatch(self) -> None:
        target = torch.zeros(3, 2)
        small_mismatch = torch.full((3, 2), 0.1)
        big_mismatch = torch.full((3, 2), 5.0)

        assert regression_loss(big_mismatch, target) > regression_loss(small_mismatch, target)

    def test_returns_scalar_tensor(self) -> None:
        anchor = torch.randn(3, 4)
        target = torch.randn(3, 4)
        loss = regression_loss(anchor, target)
        assert loss.dim() == 0

    def test_is_not_invariant_to_anchor_rescaling(self) -> None:
        """Direct contrast with attempt 2's diagnosed root cause: InfoNCE's
        `F.normalize()` made it provably blind to output scale. Plain MSE
        regression has no such blind spot -- rescaling the anchor away from
        the target strictly increases the loss.
        """
        torch.manual_seed(0)
        anchor = torch.randn(6, 5)
        target = torch.randn(6, 5)

        baseline = regression_loss(anchor, target)
        rescaled = regression_loss(anchor * 50.0, target)

        assert rescaled.item() > baseline.item()


class TestPrecomputeInstructionTargets:
    """precompute_instruction_targets computes each region's true centroid ONCE, not per training step."""

    def test_returns_one_row_per_region_name(self) -> None:
        goal_encoder = GoalEncoder(goal_dim=3, embed_dim=4, hidden_dim=8)
        region_names = ["reach left", "reach right", "reach left"]

        targets = precompute_instruction_targets(
            goal_encoder, region_names, box=SYNTHETIC_BOX, n_samples=50, seed=0,
        )

        assert targets.shape == (3, 4)

    def test_duplicate_region_names_get_an_identical_target(self) -> None:
        """Two instructions sharing a region (e.g. synonyms) must regress toward
        the *same* fixed point -- not two independently-sampled noisy estimates
        of the same region, which would reintroduce the exact per-target noise
        this rewrite removes.
        """
        goal_encoder = GoalEncoder(goal_dim=3, embed_dim=4, hidden_dim=8)
        region_names = ["reach up high", "reach down low", "reach up high"]

        targets = precompute_instruction_targets(
            goal_encoder, region_names, box=SYNTHETIC_BOX, n_samples=50, seed=0,
        )

        assert torch.equal(targets[0], targets[2])
        assert not torch.equal(targets[0], targets[1])

    def test_deterministic_for_a_given_seed(self) -> None:
        goal_encoder = GoalEncoder(goal_dim=3, embed_dim=4, hidden_dim=8)
        region_names = ["reach left", "reach right"]

        first = precompute_instruction_targets(goal_encoder, region_names, box=SYNTHETIC_BOX, n_samples=30, seed=3)
        second = precompute_instruction_targets(goal_encoder, region_names, box=SYNTHETIC_BOX, n_samples=30, seed=3)

        assert torch.equal(first, second)

    def test_leaves_goal_encoder_parameters_unchanged(self) -> None:
        goal_encoder = GoalEncoder(goal_dim=3, embed_dim=4, hidden_dim=8)
        before = [p.clone() for p in goal_encoder.parameters()]
        region_names = ["reach left", "reach right"]

        precompute_instruction_targets(goal_encoder, region_names, box=SYNTHETIC_BOX, n_samples=30, seed=0)

        after = list(goal_encoder.parameters())
        for p_before, p_after in zip(before, after, strict=True):
            assert torch.equal(p_before, p_after)

    def test_matches_a_direct_call_to_compute_region_target_embeddings_for_unique_names(self) -> None:
        """Sanity-checks this is genuinely reusing the same grounded-sampling
        machinery `goal_region_vocabulary` already provides for unique
        regions, not a parallel reimplementation.
        """
        goal_encoder = GoalEncoder(goal_dim=3, embed_dim=4, hidden_dim=8)
        unique_names = ["reach left", "reach right"]

        expected = compute_region_target_embeddings(
            goal_encoder, unique_names, box=SYNTHETIC_BOX, n_samples=30, seed=0,
        )
        actual = precompute_instruction_targets(
            goal_encoder, unique_names, box=SYNTHETIC_BOX, n_samples=30, seed=0,
        )

        assert torch.equal(expected, actual)


class TestTrainProjection:
    """train_projection fits a projection via direct regression to fixed, precomputed-once targets."""

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
            n_target_samples=30,
            seed=0,
            projection=LanguageGoalProjection(input_dim=8, embed_dim=4, hidden_dim=6),
        )

        assert isinstance(projection, LanguageGoalProjection)
        assert len(loss_history) == 5

    def test_training_reduces_mean_loss(self) -> None:
        torch.manual_seed(0)
        goal_encoder = GoalEncoder(goal_dim=3, embed_dim=4, hidden_dim=8)
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
            n_target_samples=30,
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
            n_target_samples=30,
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
            n_target_samples=30,
            seed=0,
        )

        assert projection.input_dim == 384
        assert projection.embed_dim == 16

    def test_target_is_precomputed_once_and_not_resampled_per_step(self, monkeypatch) -> None:  # noqa: ANN001
        """Regression test for attempt 2's exact diagnosed defect: the target
        must be computed once regardless of `n_steps`, not resampled every
        step. Spies on `precompute_instruction_targets` (the function
        `train_projection` must call) and asserts it runs exactly once even
        across many optimizer steps.
        """
        import lang_goal_rl.language_goal_projection as module

        call_count = 0
        original = module.precompute_instruction_targets

        def counting_wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            nonlocal call_count
            call_count += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(module, "precompute_instruction_targets", counting_wrapper)

        torch.manual_seed(0)
        goal_encoder = GoalEncoder(goal_dim=3, embed_dim=4, hidden_dim=8)
        sentence_embeddings = torch.randn(2, 8)
        region_names = ["reach left", "reach right"]

        module.train_projection(
            goal_encoder,
            sentence_embeddings,
            region_names,
            box=SYNTHETIC_BOX,
            n_steps=50,
            n_target_samples=30,
            seed=0,
            projection=LanguageGoalProjection(input_dim=8, embed_dim=4, hidden_dim=6),
        )

        assert call_count == 1

    def test_trained_projection_output_matches_fixed_target_closely(self) -> None:
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
            n_steps=300,
            n_target_samples=200,
            learning_rate=5e-3,
            seed=0,
            projection=LanguageGoalProjection(input_dim=8, embed_dim=4, hidden_dim=6),
        )

        target = precompute_instruction_targets(goal_encoder, region_names, box=SYNTHETIC_BOX, n_samples=200, seed=0)
        with torch.no_grad():
            output = projection(sentence_embeddings)

        assert torch.allclose(output, target, atol=0.05)

    def test_trained_projection_output_norms_track_target_norms_without_a_separate_norm_term(self) -> None:
        """Verifies (rather than assumes) attempt 3's simplification claim:
        matching the target vector exactly with plain MSE regression also
        matches its norm, with no separate norm-matching loss term needed --
        attempt 2 needed one because InfoNCE's normalization made it blind to
        scale; plain regression has no such blind spot (see
        `TestRegressionLoss.test_is_not_invariant_to_anchor_rescaling`).
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
            n_steps=300,
            n_target_samples=200,
            learning_rate=5e-3,
            seed=0,
            projection=LanguageGoalProjection(input_dim=8, embed_dim=4, hidden_dim=6),
        )

        target = precompute_instruction_targets(goal_encoder, region_names, box=SYNTHETIC_BOX, n_samples=200, seed=0)
        with torch.no_grad():
            output_norms = projection(sentence_embeddings).norm(dim=1)
        target_norms = target.norm(dim=1)

        assert torch.allclose(output_norms, target_norms, rtol=0.1, atol=0.02)

    def test_raises_on_row_count_mismatch(self) -> None:
        goal_encoder = GoalEncoder(goal_dim=3, embed_dim=4, hidden_dim=8)
        sentence_embeddings = torch.randn(2, 8)
        region_names = ["reach left"]

        try:
            train_projection(
                goal_encoder,
                sentence_embeddings,
                region_names,
                box=SYNTHETIC_BOX,
                n_steps=1,
                n_target_samples=10,
                seed=0,
            )
        except ValueError:
            return
        raise AssertionError("expected ValueError for row count mismatch")


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
    """check_projection_norm_range is a post-hoc sanity check on a trained projection's output scale."""

    def test_flags_a_projection_whose_norms_are_far_outside_the_reference_range(self) -> None:
        reference_norms = torch.full((100,), 0.04)
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
