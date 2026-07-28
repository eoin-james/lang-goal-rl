"""Stage 10: deterministic typed-command grammar. Pure text -> `Command`, no env/model dependency.

Phase 2a's whole point is to ship a controller whose *own* correctness isn't
confounded with a language model's -- see `PHASES.md`'s "ship deterministic
typed commands before adding learned language grounding" line. This module
is the parser half of that: plain-text command strings in, one of five
typed `Command` dataclasses out, or a `CommandParseError` naming exactly
what was wrong. It never guesses at malformed or unrecognized input
(`PHASES.md`'s "detect ambiguous or unsupported instructions instead of
silently mapping them to the nearest known behavior") -- every rejection
path raises with a message that says what was wrong and, where there's a
fixed known set to fall back on, what was expected instead.

Deliberately dependency-free: this module imports nothing beyond `dataclasses`,
`numpy`, and the standard library, so its tests never pull in gymnasium,
gymnasium_robotics, or torch (all three come in transitively the moment
anything imports `relative_move.py` or `goal_region_vocabulary.py`, since
the latter loads a `GoalEncoder`). That keeps this module "fully TDD-able in
isolation" -- the design goal named in the stage 10 plan.

Grammar (whitespace-separated tokens; the verb is case-insensitive):

    goto X Y Z
        Absolute xyz goal. Exactly 3 numbers.

    move DIRECTION DISTANCE_M
        Relative move. DIRECTION must be one of `KNOWN_DIRECTIONS` below
        (case-insensitive, e.g. "reach left" or "REACH LEFT") --
        deliberately the *full* multi-word region-name phrase, matching
        `relative_move.DIRECTION_UNIT_VECTORS`'s keys and
        `goal_region_vocabulary.REGIONS`' names exactly, rather than
        inventing a second, shorter set of direction tokens (e.g. bare
        "left") that would need its own separate mapping back to those
        names. DISTANCE_M is a float in meters, parsed with a leading sign
        allowed (negative moves the robot the opposite way along the same
        unit vector).

    waypoints X Y Z, X Y Z, ...
        One or more comma-separated legs, each exactly 3 numbers. At least
        one leg is required.

    stop
        No arguments.

    reset
        No arguments.

KNOWN_DIRECTIONS is a hand-maintained literal, not imported from
`relative_move.DIRECTION_UNIT_VECTORS` -- importing it would pull in this
module's forbidden dependency chain (see above). If Stage 7's still-pending
human sign-off ever renames a direction (not just corrects a vector's
sign), this tuple needs the matching one-line edit alongside
`DIRECTION_UNIT_VECTORS`'s own docstring-flagged update.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

KNOWN_DIRECTIONS: tuple[str, ...] = (
    "reach forward",
    "reach back",
    "reach left",
    "reach right",
    "reach up high",
    "reach down low",
)
"""The six move directions `parse_command` accepts, lower-case. Mirrors
`relative_move.DIRECTION_UNIT_VECTORS`'s keys -- see the module docstring
for why this is a hand-duplicated literal rather than an import."""


@dataclass(frozen=True)
class GotoCommand:
    """Absolute xyz goal.

    Attributes:
        xyz: The target point, shape (3,).
    """

    xyz: npt.NDArray[np.floating]


@dataclass(frozen=True)
class MoveCommand:
    """Relative move from wherever the robot actually is when this command is applied.

    Attributes:
        direction: One of `KNOWN_DIRECTIONS`, always lower-case regardless
            of the case the user typed.
        distance_m: Signed distance in meters to move along `direction`'s
            unit vector.
    """

    direction: str
    distance_m: float


@dataclass(frozen=True)
class WaypointsCommand:
    """An ordered chain of one or more absolute-xyz legs to visit in sequence.

    Attributes:
        goals: The ordered legs, each shape (3,). Never empty --
            `parse_command` rejects a "waypoints" command with zero legs.
    """

    goals: tuple[npt.NDArray[np.floating], ...]


@dataclass(frozen=True)
class StopCommand:
    """Hold in place: the executor sets the active goal to the robot's current position."""


@dataclass(frozen=True)
class ResetCommand:
    """Restart the episode. Preempts any in-progress waypoint queue; see `CommandExecutor`."""


Command = GotoCommand | MoveCommand | WaypointsCommand | StopCommand | ResetCommand


class CommandParseError(ValueError):
    """Raised by `parse_command` on anything malformed, ambiguous, or unrecognized."""


def _parse_xyz(tokens: list[str], *, context: str) -> npt.NDArray[np.floating]:
    """Parse exactly 3 whitespace-split tokens into an xyz array, or raise with `context`.

    Args:
        tokens: The tokens expected to be 3 floats.
        context: Short label prefixed onto any raised message (e.g. "'goto'"
            or "'waypoints' leg 2") so the error names which part of the
            command was wrong.

    Returns:
        The parsed point, shape (3,).

    Raises:
        CommandParseError: If `tokens` doesn't have exactly 3 entries, or
            any entry isn't a valid float.
    """
    if len(tokens) != 3:
        msg = f"{context} needs exactly 3 numbers (x y z), got {len(tokens)} token(s): {tokens!r}"
        raise CommandParseError(msg)
    coords: list[float] = []
    for token in tokens:
        try:
            coords.append(float(token))
        except ValueError as error:
            msg = f"{context}: {token!r} is not a valid number"
            raise CommandParseError(msg) from error
    return np.array(coords, dtype=np.float64)


def _parse_move(remainder: str) -> MoveCommand:
    """Parse "DIRECTION... DISTANCE_M" (everything after the "move" verb).

    Args:
        remainder: The command text with the "move" verb already stripped.

    Returns:
        The parsed `MoveCommand`.

    Raises:
        CommandParseError: If there are fewer than 3 tokens (every known
            direction is at least 2 words, plus a distance -- 3 is the
            floor for any valid command), the direction (all tokens but
            the last, lower-cased) isn't in `KNOWN_DIRECTIONS`, or the last
            token isn't a valid float.
    """
    tokens = remainder.split()
    if len(tokens) < 3:
        msg = (
            f"'move' needs a direction and a distance in meters, e.g. 'move reach left 0.05'; "
            f"got {remainder!r}"
        )
        raise CommandParseError(msg)
    *direction_tokens, distance_token = tokens
    direction = " ".join(direction_tokens).lower()
    if direction not in KNOWN_DIRECTIONS:
        msg = (
            f"'move' direction {direction!r} is not recognized -- expected one of: "
            f"{', '.join(KNOWN_DIRECTIONS)}"
        )
        raise CommandParseError(msg)
    try:
        distance_m = float(distance_token)
    except ValueError as error:
        msg = f"'move' distance {distance_token!r} is not a valid number of meters"
        raise CommandParseError(msg) from error
    return MoveCommand(direction=direction, distance_m=distance_m)


def _parse_waypoints(remainder: str) -> WaypointsCommand:
    """Parse "X Y Z, X Y Z, ..." (everything after the "waypoints" verb) into legs.

    Args:
        remainder: The command text with the "waypoints" verb already
            stripped.

    Returns:
        The parsed `WaypointsCommand`, with at least one leg.

    Raises:
        CommandParseError: If `remainder` is empty (no legs at all), or any
            comma-separated leg doesn't parse as exactly 3 numbers.
    """
    if not remainder:
        msg = (
            "'waypoints' needs at least one comma-separated 'x y z' leg, "
            "e.g. 'waypoints 1.3 0.7 0.5, 1.4 0.8 0.6'"
        )
        raise CommandParseError(msg)
    legs = [leg.strip() for leg in remainder.split(",")]
    goals = tuple(
        _parse_xyz(leg.split(), context=f"'waypoints' leg {index + 1}")
        for index, leg in enumerate(legs)
    )
    return WaypointsCommand(goals=goals)


def parse_command(text: str) -> Command:
    """Parse one line of typed-command text into a `Command`.

    Args:
        text: The raw command text, e.g. "goto 1.3 0.7 0.5", "move reach
            left 0.05", "waypoints 1.3 0.7 0.5, 1.4 0.8 0.6", "stop", or
            "reset". The verb is matched case-insensitively; everything
            after it is parsed as described in the module docstring's
            grammar.

    Returns:
        One of `GotoCommand`, `MoveCommand`, `WaypointsCommand`,
        `StopCommand`, `ResetCommand`.

    Raises:
        CommandParseError: If `text` is empty/whitespace-only, names an
            unrecognized verb, or the verb's own arguments are malformed
            (see `_parse_xyz`, `_parse_move`, `_parse_waypoints`). Never
            silently guesses at an unsupported or ambiguous instruction --
            see the module docstring.
    """
    stripped = text.strip()
    if not stripped:
        msg = "empty command -- expected one of: goto, move, waypoints, stop, reset"
        raise CommandParseError(msg)

    verb, _, remainder = stripped.partition(" ")
    remainder = remainder.strip()
    verb_lower = verb.lower()

    if verb_lower == "goto":
        return GotoCommand(xyz=_parse_xyz(remainder.split(), context="'goto'"))
    if verb_lower == "move":
        return _parse_move(remainder)
    if verb_lower == "waypoints":
        return _parse_waypoints(remainder)
    if verb_lower == "stop":
        if remainder:
            msg = f"'stop' takes no arguments, got {remainder!r}"
            raise CommandParseError(msg)
        return StopCommand()
    if verb_lower == "reset":
        if remainder:
            msg = f"'reset' takes no arguments, got {remainder!r}"
            raise CommandParseError(msg)
        return ResetCommand()

    msg = f"unknown command verb {verb!r} -- expected one of: goto, move, waypoints, stop, reset"
    raise CommandParseError(msg)
