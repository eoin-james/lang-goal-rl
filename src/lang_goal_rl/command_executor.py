"""Stage 10: the preempt/queue state machine behind the typed-command live loop.

Deliberately separated from rendering/IO (no gymnasium `env`, no SB3
`model`, no stdin/stdout) so it's unit-testable headless -- every method
here takes plain xyz arrays and returns plain xyz arrays. `interactive_demo.
py`'s `run_commands` owns the actual env-stepping loop and is the only
caller that touches a live environment; this class only tracks *which xyz
goal is currently active* and *when a waypoint chain should move to its
next leg*.

Preempt rule (locked in by the stage 10 plan, matching stage 6's
already-validated finding that live goal-swapping costs nothing
measurable): a new `Goto`, `Move`, `Stop`, or `Reset` command always
replaces any in-progress waypoint queue immediately -- there is no
"finish the current leg first" grace period. Only `Waypoints` populates a
new queue; every other command clears it and sets (or, for `Reset`,
leaves alone) a single active goal instead.

Waypoint leg advancement reuses `waypoint_following.rollout_with_waypoints`'s
exact documented rule -- advance to the next leg after `steps_per_leg`
calls to `advance`, regardless of whether `is_success` was ever true during
that leg -- so a command typed live behaves identically to what stage 9
already validated, not a divergent reimplementation. This class doesn't
call `rollout_with_waypoints` directly (that function owns its own
env-stepping loop and expects to run a whole episode itself); instead it
reimplements just the leg-boundary bookkeeping, one `advance()` call per
env step, matching `rollout_with_waypoints`'s `while n_steps < leg_end_step`
condition step-for-step.

`Stop`'s hold-in-place semantics (target := the exact `current_achieved_xyz`
passed in) are UNTESTED beyond the pure state-machine assertion in
`test_command_executor.py` -- nothing here proves the trained policy
actually holds position without drifting once handed its own current spot
as a goal. That's explicitly deferred to stage 10's scripted-harness eval
(a position-hold-drift check), per the plan. Treat `Stop` as a plausible
design, not a validated capability, until that eval lands.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from lang_goal_rl.command_grammar import (
    Command,
    GotoCommand,
    MoveCommand,
    ResetCommand,
    StopCommand,
    WaypointsCommand,
)
from lang_goal_rl.goal_region_vocabulary import MEASURED_GOAL_BOX, GoalBox
from lang_goal_rl.relative_move import compute_relative_goal

DEFAULT_STEPS_PER_LEG = 9
"""Matches stage 9's validated "tight" per-leg budget (`experiments/
09_waypoint_following`'s `report.md`): 9 steps/leg scored 0.978-1.000
whole-chain success across all 8 healthy checkpoints, chain lengths up to
5. Chosen over stage 9's "generous" 18-step budget specifically for the
*live* interface -- a shorter per-leg wait makes waypoint transitions
visibly snappier in a real-time demo, and the tight budget was already
shown reliable enough that the extra margin the generous budget buys isn't
needed here. If the stage 10 scripted-harness eval finds this too tight
under live conditions specifically, bump to 18 (stage 9's "generous"
value), not some new untested number."""


class CommandExecutor:
    """Tracks the single active xyz goal (or in-progress waypoint queue) for the live loop.

    Call `apply_command` once per typed command and `advance` once per env
    step (regardless of whether a command arrived that step); read
    `target_for_step` whenever the caller needs to know what to write into
    `env.unwrapped.goal`.
    """

    def __init__(self, *, box: GoalBox = MEASURED_GOAL_BOX, steps_per_leg: int = DEFAULT_STEPS_PER_LEG) -> None:
        """Construct an idle executor with no active goal yet.

        Args:
            box: The `GoalBox` passed through to `relative_move.
                compute_relative_goal` when resolving a `Move` command.
                `Goto`/`Waypoints`/`Stop` targets are stored as given,
                unclipped -- clipping-for-safety happens once, at the
                single point `interactive_demo.py` writes a target into the
                live env, so it applies uniformly to every command type
                rather than being split across two modules.
            steps_per_leg: Number of `advance()` calls a waypoint leg holds
                before moving to the next leg. Must be `>= 1`.

        Raises:
            ValueError: If `steps_per_leg < 1`.
        """
        if steps_per_leg < 1:
            msg = f"steps_per_leg must be >= 1 (got {steps_per_leg})"
            raise ValueError(msg)
        self._box = box
        self._steps_per_leg = steps_per_leg
        self._active_goal: npt.NDArray[np.floating] | None = None
        self._waypoint_legs: list[npt.NDArray[np.floating]] | None = None
        self._steps_in_leg = 0

    def _clear_waypoint_queue(self) -> None:
        """Drop any in-progress waypoint chain, reverting to single-active-goal mode."""
        self._waypoint_legs = None
        self._steps_in_leg = 0

    def apply_command(self, command: Command, *, current_achieved_xyz: npt.ArrayLike) -> None:
        """Apply one parsed `Command`, updating the active goal and/or waypoint queue.

        Args:
            command: One of `command_grammar`'s five `Command` variants.
            current_achieved_xyz: The robot's real, live position at the
                moment this command is applied, shape (3,). Used by `Move`
                (resolved via `compute_relative_goal` right here, eagerly --
                never lazily re-resolved later against a different
                position) and by `Stop` (the new hold target, verbatim).
                Ignored by `Goto` and `Waypoints` (their targets are fully
                specified by the command itself) and by `Reset` (which sets
                no new goal at all).

        Raises:
            ValueError: If a `WaypointsCommand` with an empty `goals` tuple
                is applied.
        """
        if isinstance(command, GotoCommand):
            self._clear_waypoint_queue()
            self._active_goal = np.asarray(command.xyz, dtype=np.float64).copy()
        elif isinstance(command, MoveCommand):
            self._clear_waypoint_queue()
            self._active_goal = compute_relative_goal(
                current_achieved_xyz, command.direction, command.distance_m, box=self._box,
            )
        elif isinstance(command, WaypointsCommand):
            if len(command.goals) == 0:
                msg = "WaypointsCommand must contain at least one goal"
                raise ValueError(msg)
            self._waypoint_legs = [np.asarray(goal, dtype=np.float64).copy() for goal in command.goals]
            self._steps_in_leg = 0
            self._active_goal = self._waypoint_legs[0].copy()
        elif isinstance(command, StopCommand):
            self._clear_waypoint_queue()
            self._active_goal = np.asarray(current_achieved_xyz, dtype=np.float64).copy()
        elif isinstance(command, ResetCommand):
            self._clear_waypoint_queue()
            # No new goal: a bare "reset" restarts the episode, it doesn't imply a new
            # target -- interactive_demo.py re-applies whatever target_for_step() already
            # returns once the env itself has been reset, mirroring language mode's
            # existing "reset" handling (which re-applies the current match, not a new one).
        else:
            msg = f"unhandled command type: {type(command).__name__}"
            raise TypeError(msg)

    def target_for_step(self) -> npt.NDArray[np.floating]:
        """Return the currently active xyz goal.

        Returns:
            The active goal, shape (3,) -- either a single `Goto`/`Move`/
            `Stop` target, or the current leg of an in-progress waypoint
            queue.

        Raises:
            ValueError: If no command has ever been applied yet.
        """
        if self._active_goal is None:
            msg = "no goal has been set yet -- apply a Goto/Move/Waypoints command before requesting a target"
            raise ValueError(msg)
        return self._active_goal

    def advance(self, *, achieved_xyz: npt.ArrayLike, is_success: bool) -> None:
        """Register one env step against the current leg's budget; move on if the leg is done.

        A no-op when a single `Goto`/`Move`/`Stop` goal is active -- there's
        no queue to advance through. When a waypoint queue is active,
        advances to the next leg once `steps_per_leg` calls have been made
        against the current one, *regardless of `is_success`* -- matching
        `waypoint_following.rollout_with_waypoints`'s documented rule that a
        merely-unreached goal never cuts a leg short, only the step budget
        does. `achieved_xyz` and `is_success` are accepted for a stable
        "call once per env step" signature but neither one drives *when* to
        advance (only the call count does); they exist so a future caller
        could add per-leg diagnostics without changing this signature, not
        because this method uses them today.

        Args:
            achieved_xyz: The robot's real position after the step just
                taken. Not read.
            is_success: Whether `info["is_success"]` was truthy on the step
                just taken. Not read -- never gates advancement.
        """
        del achieved_xyz, is_success
        if self._waypoint_legs is None:
            return
        self._steps_in_leg += 1
        if self._steps_in_leg < self._steps_per_leg:
            return
        self._steps_in_leg = 0
        remaining_legs = self._waypoint_legs[1:]
        if not remaining_legs:
            self._waypoint_legs = None  # queue exhausted -- fall back to idle single-goal mode
            return
        self._waypoint_legs = remaining_legs
        self._active_goal = self._waypoint_legs[0].copy()
