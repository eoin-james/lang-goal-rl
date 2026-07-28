"""Tests for interactive_demo's match-quality diagnostic helpers and stage 10's --interface wiring.

The match-quality helpers are pure geometry/formatting over whatever embeddings or `GoalMatch`
they're handed, so these tests use small constructed arrays with a known right answer rather than
the real sentence-transformer/GoalEncoder pipeline -- matching this project's convention for
testing diagnostics (see `test_semantic_neighbor_diagnostic.py`), not the components the
diagnostics sit on top of.

Stage 10's additions are tested at the same "pure function, no live env/model" level: `_clip_and_
write_goal` against a lightweight fake env (a plain object with a `.unwrapped.goal` attribute,
never a real gymnasium env -- no MuJoCo needed), and `main`'s `--interface`/`--checkpoint`
default-selection wiring by monkeypatching `run`/`run_commands` to record what they were called
with, never actually opening a window or loading a checkpoint. Nothing here can exercise the real
live-loop rendering/stdin-thread mechanism headlessly -- that gap is called out in the rl-builder
report rather than silently assumed covered.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lang_goal_rl import interactive_demo
from lang_goal_rl.goal_region_vocabulary import GoalBox
from lang_goal_rl.interactive_demo import (
    DEFAULT_CHECKPOINT,
    DEFAULT_COMMANDS_CHECKPOINT,
    _clip_and_write_goal,
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


def _make_fake_env() -> SimpleNamespace:
    """A minimal stand-in for a gymnasium env exposing only what `_clip_and_write_goal` touches."""
    return SimpleNamespace(unwrapped=SimpleNamespace(goal=None))


class TestClipAndWriteGoal:
    """The single clip-for-safety-and-log point every stage 10 command target funnels through."""

    def _box(self) -> GoalBox:
        return GoalBox(axis_min=np.array([0.0, 0.0, 0.0]), axis_max=np.array([1.0, 1.0, 1.0]))

    def test_a_goal_inside_the_box_is_written_unchanged_and_nothing_is_logged(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        env = _make_fake_env()
        observation: dict = {}
        target = np.array([0.5, 0.5, 0.5])

        result = _clip_and_write_goal(env, observation, target, box=self._box(), context="goto 0.5 0.5 0.5")

        assert np.allclose(result, target)
        assert np.allclose(env.unwrapped.goal, target)
        assert np.allclose(observation["desired_goal"], target)
        assert capsys.readouterr().out == ""

    def test_a_goal_outside_the_box_is_clipped_and_the_clip_is_logged(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        env = _make_fake_env()
        observation: dict = {}
        target = np.array([5.0, 0.5, 0.5])

        result = _clip_and_write_goal(env, observation, target, box=self._box(), context="goto 5 0.5 0.5")

        assert np.allclose(result, np.array([1.0, 0.5, 0.5]))
        assert np.allclose(env.unwrapped.goal, np.array([1.0, 0.5, 0.5]))
        assert np.allclose(observation["desired_goal"], np.array([1.0, 0.5, 0.5]))
        printed = capsys.readouterr().out
        assert "clipped" in printed
        assert "goto 5 0.5 0.5" in printed

    def test_env_and_observation_get_independent_copies_not_the_same_array(self) -> None:
        env = _make_fake_env()
        observation: dict = {}
        target = np.array([0.5, 0.5, 0.5])

        _clip_and_write_goal(env, observation, target, box=self._box(), context="goto")
        env.unwrapped.goal[0] = 999.0

        assert observation["desired_goal"][0] == pytest.approx(0.5)


class TestInterfaceFlagWiring:
    """`main()`'s --interface/--checkpoint default-selection, without opening a live env."""

    def _run_main_capturing_dispatch(self, monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> dict:
        captured: dict = {}
        monkeypatch.setattr(
            interactive_demo, "run", lambda **kwargs: captured.update(mode="language", kwargs=kwargs)
        )
        monkeypatch.setattr(
            interactive_demo,
            "run_commands",
            lambda **kwargs: captured.update(mode="commands", kwargs=kwargs),
        )
        monkeypatch.setattr(sys, "argv", ["interactive_demo.py", *argv])
        interactive_demo.main()
        return captured

    def test_default_interface_is_language_with_the_language_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = self._run_main_capturing_dispatch(monkeypatch, [])

        assert captured["mode"] == "language"
        assert captured["kwargs"]["checkpoint"] == DEFAULT_CHECKPOINT

    def test_interface_commands_dispatches_to_run_commands_with_the_literal_xyz_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = self._run_main_capturing_dispatch(monkeypatch, ["--interface", "commands"])

        assert captured["mode"] == "commands"
        assert captured["kwargs"]["checkpoint"] == DEFAULT_COMMANDS_CHECKPOINT

    def test_an_explicit_checkpoint_overrides_the_interface_default_in_commands_mode(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path,
    ) -> None:
        custom = tmp_path / "custom.zip"
        captured = self._run_main_capturing_dispatch(
            monkeypatch, ["--interface", "commands", "--checkpoint", str(custom)],
        )

        assert captured["kwargs"]["checkpoint"] == custom

    def test_steps_per_leg_is_forwarded_to_run_commands(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = self._run_main_capturing_dispatch(
            monkeypatch, ["--interface", "commands", "--steps-per-leg", "3"],
        )

        assert captured["kwargs"]["steps_per_leg"] == 3

    def test_language_mode_never_receives_a_steps_per_leg_kwarg(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = self._run_main_capturing_dispatch(monkeypatch, [])

        assert "steps_per_leg" not in captured["kwargs"]
