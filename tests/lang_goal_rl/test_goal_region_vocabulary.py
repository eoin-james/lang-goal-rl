"""Tests for the fixed instruction vocabulary grounded in FetchReach-v4's measured goal box.

Two kinds of checks live here: (1) that the *measurement* mechanism
(`measure_goal_box`) is deterministic and produces a box consistent with the
frozen `MEASURED_GOAL_BOX` constant this module's regions are derived from,
and (2) that region classification / sampling / vocabulary lookups behave
correctly against small, synthetic boxes (fast, independent of MuJoCo).
"""

from __future__ import annotations

import numpy as np
import torch

from lang_goal_rl.goal_encoder import GoalEncoder
from lang_goal_rl.goal_region_vocabulary import (
    ALL_INSTRUCTIONS,
    MEASURED_GOAL_BOX,
    REGIONS,
    GoalBox,
    classify_region,
    compute_region_target_embeddings,
    instruction_to_region,
    measure_goal_box,
    region_names,
    sample_region_goals,
)

SYNTHETIC_BOX = GoalBox(axis_min=np.array([0.0, 0.0, 0.0]), axis_max=np.array([1.0, 1.0, 1.0]))


class TestMeasureGoalBox:
    """measure_goal_box grounds the vocabulary in FetchReach-v4's real reset distribution."""

    def test_is_deterministic_for_the_same_seed(self) -> None:
        first = measure_goal_box(n_samples=200, seed=0)
        second = measure_goal_box(n_samples=200, seed=0)
        assert np.array_equal(first.axis_min, second.axis_min)
        assert np.array_equal(first.axis_max, second.axis_max)

    def test_uses_at_least_1000_resets_and_stays_within_the_frozen_box(self) -> None:
        # Seeds 0..999 are a subset of the seeds used to derive
        # MEASURED_GOAL_BOX (seed=0, n_samples=2000), so the smaller box's
        # min/max must fall within the frozen box's range.
        measured = measure_goal_box(n_samples=1000, seed=0)
        assert np.all(measured.axis_min >= MEASURED_GOAL_BOX.axis_min)
        assert np.all(measured.axis_max <= MEASURED_GOAL_BOX.axis_max)
        # And each axis should span roughly FetchReach's known ~0.3-unit box
        # (sanity bound, not an exact reproduction requirement).
        span = measured.axis_max - measured.axis_min
        assert np.all(span > 0.2)
        assert np.all(span < 0.35)

    def test_frozen_measured_goal_box_has_positive_range_on_every_axis(self) -> None:
        span = MEASURED_GOAL_BOX.axis_max - MEASURED_GOAL_BOX.axis_min
        assert np.all(span > 0.0)


class TestGoalBox:
    """GoalBox derives centroid and half-range from its min/max bounds."""

    def test_centroid_is_the_midpoint(self) -> None:
        np.testing.assert_allclose(SYNTHETIC_BOX.centroid, [0.5, 0.5, 0.5])

    def test_half_range_is_half_the_span(self) -> None:
        np.testing.assert_allclose(SYNTHETIC_BOX.half_range, [0.5, 0.5, 0.5])


class TestClassifyRegion:
    """classify_region assigns a point to exactly one of the 7 fixed regions."""

    def test_point_at_centroid_is_center(self) -> None:
        assert classify_region(SYNTHETIC_BOX.centroid, SYNTHETIC_BOX) == "center"

    def test_point_near_centroid_within_threshold_is_center(self) -> None:
        near_centroid = SYNTHETIC_BOX.centroid + np.array([0.05, -0.05, 0.0])
        assert classify_region(near_centroid, SYNTHETIC_BOX) == "center"

    def test_point_extreme_on_positive_x_is_reach_forward(self) -> None:
        point = np.array([1.0, 0.5, 0.5])
        assert classify_region(point, SYNTHETIC_BOX) == "reach forward"

    def test_point_extreme_on_negative_x_is_reach_back(self) -> None:
        point = np.array([0.0, 0.5, 0.5])
        assert classify_region(point, SYNTHETIC_BOX) == "reach back"

    def test_point_extreme_on_positive_y_is_reach_left(self) -> None:
        point = np.array([0.5, 1.0, 0.5])
        assert classify_region(point, SYNTHETIC_BOX) == "reach left"

    def test_point_extreme_on_negative_y_is_reach_right(self) -> None:
        point = np.array([0.5, 0.0, 0.5])
        assert classify_region(point, SYNTHETIC_BOX) == "reach right"

    def test_point_extreme_on_positive_z_is_reach_up_high(self) -> None:
        point = np.array([0.5, 0.5, 1.0])
        assert classify_region(point, SYNTHETIC_BOX) == "reach up high"

    def test_point_extreme_on_negative_z_is_reach_down_low(self) -> None:
        point = np.array([0.5, 0.5, 0.0])
        assert classify_region(point, SYNTHETIC_BOX) == "reach down low"

    def test_every_region_name_returned_is_a_defined_region(self) -> None:
        defined_names = region_names()
        rng = np.random.default_rng(0)
        for _ in range(200):
            point = rng.uniform(SYNTHETIC_BOX.axis_min, SYNTHETIC_BOX.axis_max)
            assert classify_region(point, SYNTHETIC_BOX) in defined_names


class TestRegionsAndVocabulary:
    """REGIONS/ALL_INSTRUCTIONS define the fixed, non-empty instruction vocabulary."""

    def test_there_are_between_6_and_8_regions(self) -> None:
        assert 6 <= len(REGIONS) <= 8

    def test_every_region_has_at_least_one_instruction(self) -> None:
        assert all(len(region.instructions) >= 1 for region in REGIONS)

    def test_region_names_are_unique(self) -> None:
        names = region_names()
        assert len(names) == len(set(names))

    def test_all_instructions_are_unique(self) -> None:
        assert len(ALL_INSTRUCTIONS) == len(set(ALL_INSTRUCTIONS))

    def test_all_instructions_matches_the_union_of_region_instructions(self) -> None:
        expected = {instruction for region in REGIONS for instruction in region.instructions}
        assert set(ALL_INSTRUCTIONS) == expected

    def test_instruction_to_region_resolves_every_fixed_instruction(self) -> None:
        for instruction in ALL_INSTRUCTIONS:
            assert instruction_to_region(instruction) in region_names()

    def test_instruction_to_region_raises_on_unknown_instruction(self) -> None:
        try:
            instruction_to_region("this is not in the fixed vocabulary")
        except ValueError:
            return
        raise AssertionError("expected ValueError for an out-of-vocabulary instruction")


class TestSampleRegionGoals:
    """sample_region_goals rejection-samples xyz points that classify into a given region."""

    def test_returns_the_requested_number_of_points(self) -> None:
        goals = sample_region_goals("center", n_samples=25, seed=0, box=SYNTHETIC_BOX)
        assert goals.shape == (25, 3)

    def test_every_returned_point_classifies_into_the_requested_region(self) -> None:
        for region in REGIONS:
            goals = sample_region_goals(region.name, n_samples=15, seed=1, box=SYNTHETIC_BOX)
            for point in goals:
                assert classify_region(point, SYNTHETIC_BOX) == region.name

    def test_is_deterministic_for_the_same_seed(self) -> None:
        first = sample_region_goals("reach left", n_samples=10, seed=7, box=SYNTHETIC_BOX)
        second = sample_region_goals("reach left", n_samples=10, seed=7, box=SYNTHETIC_BOX)
        np.testing.assert_array_equal(first, second)


class TestComputeRegionTargetEmbeddings:
    """compute_region_target_embeddings returns one mean GoalEncoder embedding per region."""

    def test_returns_one_row_per_requested_region_name(self) -> None:
        goal_encoder = GoalEncoder(goal_dim=3, embed_dim=4, hidden_dim=8)
        names = ["reach left", "reach right", "center"]
        embeddings = compute_region_target_embeddings(
            goal_encoder, names, box=SYNTHETIC_BOX, n_samples=20, seed=0
        )
        assert embeddings.shape == (3, 4)

    def test_output_is_a_torch_tensor(self) -> None:
        goal_encoder = GoalEncoder(goal_dim=3, embed_dim=4, hidden_dim=8)
        embeddings = compute_region_target_embeddings(
            goal_encoder, ["center"], box=SYNTHETIC_BOX, n_samples=20, seed=0
        )
        assert isinstance(embeddings, torch.Tensor)
