"""Tests for command_grammar: stage 10's deterministic typed-command parser.

Pure text -> `Command` dataclass, no env/model/GoalBox dependency -- these
tests import nothing beyond `command_grammar` itself and numpy, matching
the module's own "fully TDD-able in isolation" design goal (no gymnasium,
no torch, no MuJoCo startup cost).
"""

from __future__ import annotations

import numpy as np
import pytest

from lang_goal_rl.command_grammar import (
    KNOWN_DIRECTIONS,
    CommandParseError,
    GotoCommand,
    MoveCommand,
    ResetCommand,
    StopCommand,
    WaypointsCommand,
    parse_command,
)


class TestGoto:
    """'goto X Y Z' -> GotoCommand(xyz=[X, Y, Z])."""

    def test_parses_three_floats_into_a_goto_command(self) -> None:
        command = parse_command("goto 1.3 0.7 0.5")

        assert isinstance(command, GotoCommand)
        assert np.allclose(command.xyz, np.array([1.3, 0.7, 0.5]))

    def test_verb_is_case_insensitive(self) -> None:
        command = parse_command("GoTo 1.3 0.7 0.5")

        assert isinstance(command, GotoCommand)
        assert np.allclose(command.xyz, np.array([1.3, 0.7, 0.5]))

    def test_accepts_negative_and_integer_coordinates(self) -> None:
        command = parse_command("goto -1 2 0")

        assert isinstance(command, GotoCommand)
        assert np.allclose(command.xyz, np.array([-1.0, 2.0, 0.0]))

    def test_tolerates_extra_whitespace_between_tokens(self) -> None:
        command = parse_command("goto   1.3   0.7   0.5  ")

        assert isinstance(command, GotoCommand)
        assert np.allclose(command.xyz, np.array([1.3, 0.7, 0.5]))

    def test_too_few_coordinates_raises_with_actionable_message(self) -> None:
        with pytest.raises(CommandParseError, match="3 numbers"):
            parse_command("goto 1.3 0.7")

    def test_too_many_coordinates_raises_with_actionable_message(self) -> None:
        with pytest.raises(CommandParseError, match="3 numbers"):
            parse_command("goto 1.3 0.7 0.5 0.1")

    def test_non_numeric_coordinate_raises_naming_the_bad_token(self) -> None:
        with pytest.raises(CommandParseError, match="banana"):
            parse_command("goto 1.3 banana 0.5")

    def test_no_coordinates_at_all_raises(self) -> None:
        with pytest.raises(CommandParseError, match="3 numbers"):
            parse_command("goto")


class TestMove:
    """'move DIRECTION DISTANCE_M' -> MoveCommand. DIRECTION is one of KNOWN_DIRECTIONS."""

    def test_parses_a_known_direction_and_distance(self) -> None:
        command = parse_command("move reach left 0.05")

        assert isinstance(command, MoveCommand)
        assert command.direction == "reach left"
        assert command.distance_m == pytest.approx(0.05)

    def test_verb_and_direction_are_both_case_insensitive(self) -> None:
        command = parse_command("MOVE Reach Left 0.05")

        assert isinstance(command, MoveCommand)
        assert command.direction == "reach left"

    def test_negative_distance_is_accepted(self) -> None:
        command = parse_command("move reach forward -0.05")

        assert isinstance(command, MoveCommand)
        assert command.distance_m == pytest.approx(-0.05)

    def test_all_six_known_directions_parse(self) -> None:
        for direction in KNOWN_DIRECTIONS:
            command = parse_command(f"move {direction} 0.1")
            assert isinstance(command, MoveCommand)
            assert command.direction == direction

    def test_unknown_direction_raises_naming_the_bad_token_and_known_options(self) -> None:
        with pytest.raises(CommandParseError, match="reach diagonally"):
            parse_command("move reach diagonally 0.05")

    def test_missing_distance_raises_with_actionable_message(self) -> None:
        with pytest.raises(CommandParseError, match="direction and a distance"):
            parse_command("move reach left")

    def test_non_numeric_distance_raises_naming_the_bad_token(self) -> None:
        with pytest.raises(CommandParseError, match="lots"):
            parse_command("move reach left lots")

    def test_bare_move_with_no_arguments_raises(self) -> None:
        with pytest.raises(CommandParseError, match="direction and a distance"):
            parse_command("move")


class TestWaypoints:
    """'waypoints X Y Z, X Y Z, ...' -> WaypointsCommand(goals=(...))."""

    def test_parses_two_comma_separated_legs(self) -> None:
        command = parse_command("waypoints 1.3 0.7 0.5, 1.4 0.8 0.6")

        assert isinstance(command, WaypointsCommand)
        assert len(command.goals) == 2
        assert np.allclose(command.goals[0], np.array([1.3, 0.7, 0.5]))
        assert np.allclose(command.goals[1], np.array([1.4, 0.8, 0.6]))

    def test_parses_a_single_leg(self) -> None:
        command = parse_command("waypoints 1.3 0.7 0.5")

        assert isinstance(command, WaypointsCommand)
        assert len(command.goals) == 1
        assert np.allclose(command.goals[0], np.array([1.3, 0.7, 0.5]))

    def test_parses_many_legs_and_tolerates_ragged_whitespace_around_commas(self) -> None:
        command = parse_command("waypoints 1.0 0.0 0.0,2.0 0.0 0.0 ,  3.0 0.0 0.0")

        assert isinstance(command, WaypointsCommand)
        assert len(command.goals) == 3
        assert np.allclose(command.goals[2], np.array([3.0, 0.0, 0.0]))

    def test_verb_is_case_insensitive(self) -> None:
        command = parse_command("WAYPOINTS 1.0 0.0 0.0")

        assert isinstance(command, WaypointsCommand)

    def test_no_legs_at_all_raises(self) -> None:
        with pytest.raises(CommandParseError, match="at least one"):
            parse_command("waypoints")

    def test_a_leg_with_wrong_arity_raises_naming_which_leg(self) -> None:
        with pytest.raises(CommandParseError, match="leg 2"):
            parse_command("waypoints 1.0 0.0 0.0, 2.0 0.0")

    def test_a_trailing_comma_leaving_an_empty_leg_raises(self) -> None:
        with pytest.raises(CommandParseError, match="leg 2"):
            parse_command("waypoints 1.0 0.0 0.0,")

    def test_a_leg_with_a_non_numeric_coordinate_raises(self) -> None:
        with pytest.raises(CommandParseError, match="nope"):
            parse_command("waypoints 1.0 0.0 0.0, 2.0 nope 0.0")


class TestStop:
    """'stop' -> StopCommand(), no arguments allowed."""

    def test_parses_stop(self) -> None:
        assert isinstance(parse_command("stop"), StopCommand)

    def test_verb_is_case_insensitive(self) -> None:
        assert isinstance(parse_command("STOP"), StopCommand)

    def test_stop_with_trailing_whitespace_still_parses(self) -> None:
        assert isinstance(parse_command("  stop  "), StopCommand)

    def test_stop_with_extra_arguments_raises(self) -> None:
        with pytest.raises(CommandParseError, match="no arguments"):
            parse_command("stop now")


class TestReset:
    """'reset' -> ResetCommand(), no arguments allowed."""

    def test_parses_reset(self) -> None:
        assert isinstance(parse_command("reset"), ResetCommand)

    def test_verb_is_case_insensitive(self) -> None:
        assert isinstance(parse_command("Reset"), ResetCommand)

    def test_reset_with_extra_arguments_raises(self) -> None:
        with pytest.raises(CommandParseError, match="no arguments"):
            parse_command("reset now")


class TestUnsupportedInput:
    """Anything not matching one of the five verbs is rejected loudly, never silently guessed."""

    def test_empty_string_raises(self) -> None:
        with pytest.raises(CommandParseError, match="empty"):
            parse_command("")

    def test_whitespace_only_string_raises(self) -> None:
        with pytest.raises(CommandParseError, match="empty"):
            parse_command("   ")

    def test_unknown_verb_raises_naming_the_bad_verb(self) -> None:
        with pytest.raises(CommandParseError, match="fly"):
            parse_command("fly to the moon")

    def test_a_plain_english_sentence_is_rejected_not_guessed(self) -> None:
        with pytest.raises(CommandParseError):
            parse_command("please move your hand to the left a little bit")

    def test_command_parse_error_is_a_value_error(self) -> None:
        assert issubclass(CommandParseError, ValueError)
