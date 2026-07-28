"""Tests for interactive_demo's match-quality diagnostic helpers.

Both helpers are pure geometry/formatting over whatever embeddings or `GoalMatch` they're
handed, so these tests use small constructed arrays with a known right answer rather than the
real sentence-transformer/GoalEncoder pipeline -- matching this project's convention for testing
diagnostics (see `test_semantic_neighbor_diagnostic.py`), not the components the diagnostics sit
on top of.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lang_goal_rl.interactive_demo import (
    _describe_match_quality,
    _leave_one_out_baseline_distance,
)
from lang_goal_rl.live_goal_controller import GoalMatch


def _make_match(*, distance: float) -> GoalMatch:
    return GoalMatch(
        embedding=torch.zeros(16),
        reference_instruction="reach up high",
        region_name="reach up high",
        distance=distance,
    )


class TestLeaveOneOutBaselineDistance:
    """The empirical "typical distance between two known sentences" baseline."""

    def test_two_tight_clusters_give_a_baseline_matching_their_within_cluster_gap(self) -> None:
        reference_embeddings = np.array(
            [[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]]
        )

        baseline = _leave_one_out_baseline_distance(reference_embeddings)

        # Every point's nearest *other* point is its own cluster-mate, distance 0.1 in every
        # case -- so the mean is 0.1 and the std is 0, giving an exact baseline of 0.1.
        assert baseline == pytest.approx(0.1)

    def test_wider_spread_between_nearest_neighbors_gives_a_larger_baseline(self) -> None:
        tight = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]])
        wide = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]])

        assert _leave_one_out_baseline_distance(wide) > _leave_one_out_baseline_distance(tight)

    def test_returns_a_python_float(self) -> None:
        reference_embeddings = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

        baseline = _leave_one_out_baseline_distance(reference_embeddings)

        assert isinstance(baseline, float)


class TestDescribeMatchQuality:
    """One-line verdict: confident match vs. genuine extrapolation."""

    def test_distance_at_or_below_baseline_is_reported_as_a_confident_match(self) -> None:
        match = _make_match(distance=0.5)

        description = _describe_match_quality(match, baseline_distance=1.0)

        assert "confident match" in description
        assert "extrapolation" not in description

    def test_distance_exactly_at_the_baseline_counts_as_confident(self) -> None:
        match = _make_match(distance=1.0)

        description = _describe_match_quality(match, baseline_distance=1.0)

        assert "confident match" in description

    def test_distance_above_baseline_is_reported_as_an_extrapolation(self) -> None:
        match = _make_match(distance=2.5)

        description = _describe_match_quality(match, baseline_distance=1.0)

        assert description.startswith("extrapolation")

    def test_description_reports_both_the_matchs_distance_and_the_baseline(self) -> None:
        match = _make_match(distance=2.5)

        description = _describe_match_quality(match, baseline_distance=1.0)

        assert "2.500" in description
        assert "1.000" in description
