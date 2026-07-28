"""Tests for command_executor: stage 10's preempt/queue state machine.

Deliberately headless -- no gymnasium env, no trained model, no MuJoCo
startup cost. `CommandExecutor` never touches an env or a policy; every
test here drives it with plain numpy arrays standing in for
`current_achieved_xyz`/`achieved_xyz`, exactly the separation-of-concerns
the module is designed around (see `command_executor.py`'s module
docstring).
"""

from __future__ import annotations

import numpy as np
import pytest

from lang_goal_rl.command_executor import CommandExecutor
from lang_goal_rl.command_grammar import (
    GotoCommand,
    MoveCommand,
    ResetCommand,
    StopCommand,
    WaypointsCommand,
)
from lang_goal_rl.goal_region_vocabulary import GoalBox
from lang_goal_rl.relative_move import compute_relative_goal

# A wide synthetic box so every fixture point in this file sits comfortably inside it --
# these tests are about the state machine's sequencing logic, not clipping, which
# `TestClipping` below exercises deliberately with an intentionally tight box.
_WIDE_BOX = GoalBox(axis_min=np.array([-100.0, -100.0, -100.0]), axis_max=np.array([100.0, 100.0, 100.0]))

_SYNTHETIC_DIRECTIONS = {
    "reach forward": np.array([1.0, 0.0, 0.0]),
    "reach left": np.array([0.0, 1.0, 0.0]),
}


def _make_executor(**kwargs) -> CommandExecutor:
    kwargs.setdefault("box", _WIDE_BOX)
    return CommandExecutor(**kwargs)


class TestGotoPreemptsAWaypointQueue:
    def test_goto_replaces_an_in_progress_waypoint_queue_and_becomes_the_active_target(
        self,
    ) -> None:
        executor = _make_executor(steps_per_leg=2)
        waypoints = WaypointsCommand(
            goals=(np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0])),
        )
        executor.apply_command(waypoints, current_achieved_xyz=np.array([0.0, 0.0, 0.0]))
        executor.advance(achieved_xyz=np.array([0.0, 0.0, 0.0]), is_success=False)  # 1 of 2 steps into leg 0

        new_goal = np.array([9.0, 9.0, 9.0])
        executor.apply_command(GotoCommand(xyz=new_goal), current_achieved_xyz=np.array([0.0, 0.0, 0.0]))

        assert np.allclose(executor.target_for_step(), new_goal)

        # The queue is gone: stepping past leg 0's old budget must not advance anywhere --
        # the target stays pinned on the new Goto goal.
        for _ in range(10):
            executor.advance(achieved_xyz=np.array([0.0, 0.0, 0.0]), is_success=False)
        assert np.allclose(executor.target_for_step(), new_goal)


class TestMovePreemptsAWaypointQueue:
    def test_move_replaces_the_queue_and_resolves_a_relative_target(self) -> None:
        executor = _make_executor(steps_per_leg=2)
        waypoints = WaypointsCommand(goals=(np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])))
        executor.apply_command(waypoints, current_achieved_xyz=np.array([0.0, 0.0, 0.0]))

        current = np.array([5.0, 5.0, 5.0])
        move = MoveCommand(direction="reach forward", distance_m=0.5)
        executor.apply_command(move, current_achieved_xyz=current)

        expected = compute_relative_goal(
            current, "reach forward", 0.5, direction_vectors=_SYNTHETIC_DIRECTIONS, box=_WIDE_BOX,
        )
        assert np.allclose(executor.target_for_step(), expected)

        for _ in range(10):
            executor.advance(achieved_xyz=current, is_success=False)
        assert np.allclose(executor.target_for_step(), expected)


class TestStopPreemptsAWaypointQueue:
    def test_stop_sets_the_target_to_exactly_the_current_achieved_xyz(self) -> None:
        executor = _make_executor(steps_per_leg=2)
        waypoints = WaypointsCommand(goals=(np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])))
        executor.apply_command(waypoints, current_achieved_xyz=np.array([0.0, 0.0, 0.0]))

        achieved = np.array([1.23, 4.56, 7.89])
        executor.apply_command(StopCommand(), current_achieved_xyz=achieved)

        assert np.allclose(executor.target_for_step(), achieved)

        for _ in range(10):
            executor.advance(achieved_xyz=achieved, is_success=False)
        assert np.allclose(executor.target_for_step(), achieved)


class TestResetPreemptsAWaypointQueue:
    def test_reset_clears_the_queue_and_leaves_the_last_active_goal_in_place(self) -> None:
        executor = _make_executor(steps_per_leg=1)
        first_goal = np.array([3.0, 3.0, 3.0])
        executor.apply_command(GotoCommand(xyz=first_goal), current_achieved_xyz=np.array([0.0, 0.0, 0.0]))

        waypoints = WaypointsCommand(goals=(np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])))
        executor.apply_command(waypoints, current_achieved_xyz=np.array([0.0, 0.0, 0.0]))
        assert np.allclose(executor.target_for_step(), np.array([1.0, 0.0, 0.0]))

        executor.apply_command(ResetCommand(), current_achieved_xyz=np.array([0.0, 0.0, 0.0]))

        # Reset preempts the queue (no more leg-advancing) but doesn't invent a new goal --
        # the target stays whatever leg 0 had already set it to.
        assert np.allclose(executor.target_for_step(), np.array([1.0, 0.0, 0.0]))
        for _ in range(10):
            executor.advance(achieved_xyz=np.array([0.0, 0.0, 0.0]), is_success=False)
        assert np.allclose(executor.target_for_step(), np.array([1.0, 0.0, 0.0]))


class TestWaypointAdvanceMatchesWaypointFollowingsRule:
    """Advance after exactly `steps_per_leg` calls to `advance`, regardless of `is_success`."""

    def test_target_changes_at_exactly_the_step_count_boundary(self) -> None:
        executor = _make_executor(steps_per_leg=3)
        goals = (np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0]))
        executor.apply_command(WaypointsCommand(goals=goals), current_achieved_xyz=np.array([0.0, 0.0, 0.0]))

        assert np.allclose(executor.target_for_step(), goals[0])
        for _ in range(2):  # 2 of 3 steps into leg 0 -- must not have advanced yet
            executor.advance(achieved_xyz=np.array([0.0, 0.0, 0.0]), is_success=False)
            assert np.allclose(executor.target_for_step(), goals[0])

        executor.advance(achieved_xyz=np.array([0.0, 0.0, 0.0]), is_success=False)  # 3rd step -- boundary
        assert np.allclose(executor.target_for_step(), goals[1])

    def test_advances_regardless_of_is_success_value(self) -> None:
        executor = _make_executor(steps_per_leg=1)
        goals = (np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]))
        executor.apply_command(WaypointsCommand(goals=goals), current_achieved_xyz=np.array([0.0, 0.0, 0.0]))

        executor.advance(achieved_xyz=np.array([0.0, 0.0, 0.0]), is_success=False)
        assert np.allclose(executor.target_for_step(), goals[1])

    def test_a_three_leg_chain_advances_at_every_leg_boundary_in_order(self) -> None:
        executor = _make_executor(steps_per_leg=2)
        goals = (np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0]))
        executor.apply_command(WaypointsCommand(goals=goals), current_achieved_xyz=np.array([0.0, 0.0, 0.0]))

        seen_targets = []
        for _ in range(6):
            seen_targets.append(np.array(executor.target_for_step()))
            executor.advance(achieved_xyz=np.array([0.0, 0.0, 0.0]), is_success=True)

        expected = [goals[0], goals[0], goals[1], goals[1], goals[2], goals[2]]
        for seen, wanted in zip(seen_targets, expected, strict=True):
            assert np.allclose(seen, wanted)

    def test_queue_exhaustion_leaves_the_final_leg_as_the_target(self) -> None:
        executor = _make_executor(steps_per_leg=1)
        goals = (np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]))
        executor.apply_command(WaypointsCommand(goals=goals), current_achieved_xyz=np.array([0.0, 0.0, 0.0]))

        for _ in range(10):
            executor.advance(achieved_xyz=np.array([0.0, 0.0, 0.0]), is_success=False)

        assert np.allclose(executor.target_for_step(), goals[1])

    def test_single_leg_waypoints_command_never_advances_past_its_only_goal(self) -> None:
        executor = _make_executor(steps_per_leg=1)
        goal = np.array([1.0, 0.0, 0.0])
        executor.apply_command(WaypointsCommand(goals=(goal,)), current_achieved_xyz=np.array([0.0, 0.0, 0.0]))

        for _ in range(5):
            executor.advance(achieved_xyz=np.array([0.0, 0.0, 0.0]), is_success=False)
            assert np.allclose(executor.target_for_step(), goal)

    def test_advance_is_a_no_op_for_a_single_active_goto_goal(self) -> None:
        executor = _make_executor(steps_per_leg=1)
        goal = np.array([5.0, 5.0, 5.0])
        executor.apply_command(GotoCommand(xyz=goal), current_achieved_xyz=np.array([0.0, 0.0, 0.0]))

        for _ in range(10):
            executor.advance(achieved_xyz=np.array([0.0, 0.0, 0.0]), is_success=True)
            assert np.allclose(executor.target_for_step(), goal)


class TestMoveResolvesAtApplyTimeNotLater:
    """Move's target is frozen from `current_achieved_xyz` the moment `apply_command` runs."""

    def test_a_different_achieved_xyz_passed_to_a_later_advance_call_never_changes_the_target(
        self,
    ) -> None:
        executor = _make_executor()
        resolve_time_position = np.array([0.0, 0.0, 0.0])
        move = MoveCommand(direction="reach forward", distance_m=1.0)
        executor.apply_command(move, current_achieved_xyz=resolve_time_position)

        expected = compute_relative_goal(
            resolve_time_position, "reach forward", 1.0,
            direction_vectors=_SYNTHETIC_DIRECTIONS, box=_WIDE_BOX,
        )
        assert np.allclose(executor.target_for_step(), expected)

        # If the target were (incorrectly) resolved lazily inside target_for_step/advance
        # rather than eagerly inside apply_command, feeding a wildly different "current"
        # position here would change the resolved target. It must not.
        far_away_position = np.array([50.0, -50.0, 20.0])
        executor.advance(achieved_xyz=far_away_position, is_success=False)
        assert np.allclose(executor.target_for_step(), expected)
        assert not np.allclose(expected, far_away_position + np.array([1.0, 0.0, 0.0]))


class TestStopSetsTargetToExactAchievedXyz:
    def test_stop_target_equals_the_passed_achieved_xyz_exactly(self) -> None:
        executor = _make_executor()
        achieved = np.array([2.71, 3.14, 1.61])

        executor.apply_command(StopCommand(), current_achieved_xyz=achieved)

        assert np.array_equal(executor.target_for_step(), achieved)


class TestTargetForStepBeforeAnyCommand:
    def test_raises_a_clear_error_when_no_command_has_ever_been_applied(self) -> None:
        executor = _make_executor()

        with pytest.raises(ValueError, match="no goal"):
            executor.target_for_step()


class TestConstructorValidation:
    def test_steps_per_leg_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="steps_per_leg"):
            CommandExecutor(box=_WIDE_BOX, steps_per_leg=0)


class TestWaypointsCommandRequiresAtLeastOneGoal:
    def test_an_empty_goals_tuple_raises(self) -> None:
        executor = _make_executor()

        with pytest.raises(ValueError, match="at least one"):
            executor.apply_command(WaypointsCommand(goals=()), current_achieved_xyz=np.array([0.0, 0.0, 0.0]))
