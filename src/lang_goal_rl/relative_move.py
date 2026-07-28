"""Stage 8: move relative to wherever the robot actually is, not to a fixed point.

Stage 5's mid-episode re-goaling proved a policy can accept a brand-new
*literal* goal partway through an episode. Stage 8 is a distinct capability:
"move 10cm forward from here" -- where "here" is the robot's real, live
position at the moment the command lands, not a coordinate anyone chose in
advance. Two pieces:

- `compute_relative_goal` / `clip_to_box`: the pure math -- given a current
  xyz, a named direction, and a distance in meters, compute the target and
  clamp it into a `GoalBox` so an aggressive move can never request a point
  the env would never sample as a goal. Both take an explicit
  `direction_vectors` mapping as a parameter (defaulting to
  `DIRECTION_UNIT_VECTORS`) precisely so their correctness never depends on
  which real-world direction labels turn out to be right -- see the
  `DIRECTION_UNIT_VECTORS` docstring below.
- `rollout_with_relative_move`: runs the actual mid-episode switch. Reuses
  `midepisode_regoal._run_goal_phase` for both phases -- the pre-switch
  phase is byte-for-byte the same step loop `rollout_with_goal_switch`
  already runs, and factoring it out (stage 8's design decision, see
  ROADMAP.md/plan) means this module never grows a second, independently
  maintained copy of that loop. What's genuinely new here is what happens
  *between* the two phases: instead of a caller-supplied `new_goal_xyz`,
  the target is computed from `obs["achieved_goal"]` as observed at the
  end of the pre-switch phase -- the actual capability under test is
  "relative to an arbitrary real position", not "relative to a known
  point", so the target cannot be precomputed before the rollout starts.

DIRECTION_UNIT_VECTORS -- PROVISIONAL, pending Stage 7 sign-off
-----------------------------------------------------------------
`DIRECTION_UNIT_VECTORS` below encodes the same six directions and signs as
`goal_region_vocabulary.AXIS_DIRECTIONS` (x = depth back/forward, y =
lateral right/left, z = height down/up), because that is this project's
current best guess at FetchReach's camera-frame convention. Stage 7
generated visual-check clips to let a human confirm those labels actually
match what the camera shows, and that human sign-off has **not yet
happened** (deliberately deferred, per the approved plan, so Stage 8's math
isn't blocked on it). This module's tests never depend on
`DIRECTION_UNIT_VECTORS`'s specific values -- `compute_relative_goal`,
`clip_to_box`, and `rollout_with_relative_move` all accept an injectable
`direction_vectors` parameter and are tested against a synthetic dict. If
Stage 7's sign-off finds a label wrong (e.g. "left" and "right" swapped),
the fix is a one-line edit to the literal dict below -- nothing else in this
module, `midepisode_regoal.py`, or `goal_region_vocabulary.py` needs to
change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from lang_goal_rl.goal_region_vocabulary import MEASURED_GOAL_BOX, GoalBox
from lang_goal_rl.midepisode_regoal import _ensure_within_env_step_limit, _run_goal_phase

if TYPE_CHECKING:
    import gymnasium as gym
    from stable_baselines3 import SAC

DIRECTION_UNIT_VECTORS: dict[str, npt.NDArray[np.floating]] = {
    "reach forward": np.array([1.0, 0.0, 0.0]),  # x axis, positive -- AXIS_DIRECTIONS[0][1]
    "reach back": np.array([-1.0, 0.0, 0.0]),  # x axis, negative -- AXIS_DIRECTIONS[0][0]
    "reach left": np.array([0.0, 1.0, 0.0]),  # y axis, positive -- AXIS_DIRECTIONS[1][1]
    "reach right": np.array([0.0, -1.0, 0.0]),  # y axis, negative -- AXIS_DIRECTIONS[1][0]
    "reach up high": np.array([0.0, 0.0, 1.0]),  # z axis, positive -- AXIS_DIRECTIONS[2][1]
    "reach down low": np.array([0.0, 0.0, -1.0]),  # z axis, negative -- AXIS_DIRECTIONS[2][0]
}
"""Production direction-name -> unit-vector mapping. PROVISIONAL -- see the
module docstring. A hand-maintained literal (not derived at import time from
`goal_region_vocabulary.AXIS_DIRECTIONS`), so correcting a label after
Stage 7's sign-off is exactly one edit to exactly this dict."""


@dataclass(frozen=True)
class RelativeMoveResult:
    """Outcome of one `rollout_with_relative_move` episode.

    Attributes:
        success: Whether `info["is_success"]` was truthy on the *final*
            post-switch step, judged against `resolved_target_xyz` (the
            already-clipped target) -- never against the raw, pre-clip
            request.
        n_steps: Total env steps run, pre- and post-switch combined.
        switch_step: The `switch_step` argument, echoed back.
        resolved_target_xyz: The actual post-switch goal used, i.e.
            `compute_relative_goal` applied to the achieved position
            observed at `switch_step`, already clipped into `box`.
        was_clipped: Whether clipping actually changed the requested point
            (`True` iff the unclipped `current + distance_m * direction`
            fell outside `box` on at least one axis).
    """

    success: bool
    n_steps: int
    switch_step: int
    resolved_target_xyz: npt.NDArray[np.floating]
    was_clipped: bool


def clip_to_box(point: npt.ArrayLike, box: GoalBox = MEASURED_GOAL_BOX) -> npt.NDArray[np.floating]:
    """Clamp `point` into `box`'s axis-aligned bounds, per axis independently.

    Args:
        point: An xyz point, shape `(3,)`.
        box: The `GoalBox` to clamp into.

    Returns:
        `point` with each axis clamped to `[box.axis_min, box.axis_max]` on
        that axis. Axes are independent: an out-of-bounds x doesn't affect
        y or z.
    """
    return np.clip(np.asarray(point, dtype=np.float64), box.axis_min, box.axis_max)


def _relative_target_point(
    current_xyz: npt.ArrayLike,
    direction: str,
    distance_m: float,
    direction_vectors: dict[str, npt.NDArray[np.floating]],
) -> npt.NDArray[np.floating]:
    """`current_xyz + distance_m * direction_vectors[direction]`, unclipped.

    Factored out of `compute_relative_goal` so `rollout_with_relative_move`
    can compute the same raw point to detect whether clipping actually
    changed it (`RelativeMoveResult.was_clipped`), without duplicating this
    formula.

    Args:
        current_xyz: The position to move relative to, shape `(3,)`.
        direction: A key into `direction_vectors`.
        distance_m: Signed distance in meters to move along the direction's
            unit vector.
        direction_vectors: Direction-name -> unit-vector mapping.

    Returns:
        The unclipped target point, shape `(3,)`.

    Raises:
        ValueError: If `direction` is not a key in `direction_vectors`.
    """
    if direction not in direction_vectors:
        msg = f"{direction!r} is not in the given direction_vectors mapping (known: {sorted(direction_vectors)})"
        raise ValueError(msg)
    current = np.asarray(current_xyz, dtype=np.float64)
    unit_vector = np.asarray(direction_vectors[direction], dtype=np.float64)
    return current + distance_m * unit_vector


def compute_relative_goal(
    current_xyz: npt.ArrayLike,
    direction: str,
    distance_m: float,
    *,
    direction_vectors: dict[str, npt.NDArray[np.floating]] = DIRECTION_UNIT_VECTORS,
    box: GoalBox = MEASURED_GOAL_BOX,
) -> npt.NDArray[np.floating]:
    """`current_xyz + distance_m * direction_vectors[direction]`, clipped into `box`.

    `direction_vectors` is a parameter (defaulting to the production
    `DIRECTION_UNIT_VECTORS`) so callers -- including this module's own
    tests -- can inject a synthetic mapping and verify the pure math
    without depending on which real-world direction labels Stage 7's
    pending human sign-off eventually confirms.

    Args:
        current_xyz: The position to move relative to, shape `(3,)`.
        direction: A key into `direction_vectors`.
        distance_m: Signed distance in meters to move along the direction's
            unit vector.
        direction_vectors: Direction-name -> unit-vector mapping.
        box: The box to clip the result into.

    Returns:
        The target point, shape `(3,)`, clamped into `box`.

    Raises:
        ValueError: If `direction` is not a key in `direction_vectors`.
    """
    raw_target = _relative_target_point(current_xyz, direction, distance_m, direction_vectors)
    return clip_to_box(raw_target, box=box)


def rollout_with_relative_move(
    model: SAC,
    env: gym.Env,
    *,
    initial_goal_xyz: npt.ArrayLike,
    switch_step: int,
    direction: str,
    distance_m: float,
    max_steps: int,
    base_seed: int,
    box: GoalBox = MEASURED_GOAL_BOX,
    direction_vectors: dict[str, npt.NDArray[np.floating]] = DIRECTION_UNIT_VECTORS,
) -> RelativeMoveResult:
    """Roll out one episode: target `initial_goal_xyz`, then move relative to wherever that left the robot.

    Structurally identical to `midepisode_regoal.rollout_with_goal_switch`
    for the pre-switch phase (same `_run_goal_phase` helper, same
    reset-then-run-`switch_step`-steps shape). The difference is what
    happens at the switch: `rollout_with_goal_switch` takes its post-switch
    goal as a caller-supplied argument known before the rollout starts;
    this function instead reads `obs["achieved_goal"]` at the end of the
    pre-switch phase -- the robot's actual position at that instant -- and
    computes the post-switch target from *that*, via `compute_relative_goal`.
    That's the capability under test: "relative to an arbitrary real
    position reached by a prior command", not "relative to a known point",
    so the target genuinely cannot be precomputed.

    Success is judged against `resolved_target_xyz` (the already-clipped
    target), never against the raw, pre-clip request -- a command that
    would overshoot the box still gets judged fairly against the point the
    robot was actually asked to reach.

    Args:
        model: A trained SAC model. Literal-xyz only (no embedding-mode
            params, unlike `rollout_with_goal_switch`) -- Stage 8 isolates
            the relative-move mechanism from the language/embedding layer,
            same rationale `midepisode_regoal.py` already applies to
            Stage 5. `model.actor` is never accessed.
        env: The FetchReach-v4 env instance to roll out on.
        initial_goal_xyz: The goal active for the first `switch_step` steps,
            shape `(3,)`.
        switch_step: Number of steps to run against `initial_goal_xyz`
            before switching. Must be `>= 1` and `< max_steps`.
        direction: A key into `direction_vectors` naming the move direction.
        distance_m: Signed distance in meters to move from the achieved
            position at `switch_step`.
        max_steps: Total episode length (pre- + post-switch steps
            combined). Validated against the env's registered
            `max_episode_steps`.
        base_seed: Seed passed to `env.reset(seed=base_seed)`.
        box: The box the resolved target is clipped into.
        direction_vectors: Direction-name -> unit-vector mapping. Defaults
            to the production (provisional) `DIRECTION_UNIT_VECTORS`.

    Returns:
        A `RelativeMoveResult` with success against the resolved target,
        total step count, `switch_step`, the resolved target itself, and
        whether clipping changed the requested point.

    Raises:
        ValueError: If `direction` is not in `direction_vectors`, if
            `switch_step < 1`, if `switch_step >= max_steps`, or if
            `max_steps` exceeds the env's registered `max_episode_steps`.
    """
    if direction not in direction_vectors:
        msg = f"{direction!r} is not in the given direction_vectors mapping (known: {sorted(direction_vectors)})"
        raise ValueError(msg)
    if switch_step < 1:
        msg = f"switch_step must be >= 1 (got {switch_step}) -- a switch at step 0 isn't mid-episode"
        raise ValueError(msg)
    if switch_step >= max_steps:
        msg = f"switch_step ({switch_step}) must be < max_steps ({max_steps}) to leave a post-switch step"
        raise ValueError(msg)
    _ensure_within_env_step_limit(env, max_steps)

    initial_goal = np.asarray(initial_goal_xyz, dtype=np.float64)

    obs, _info = env.reset(seed=base_seed)

    obs, terminated, truncated, phase1_steps, _phase1_success = _run_goal_phase(
        model, env, obs, initial_goal, max_phase_steps=switch_step, terminated=False, truncated=False,
    )

    achieved_at_switch = np.array(obs["achieved_goal"], copy=True)
    raw_target = _relative_target_point(achieved_at_switch, direction, distance_m, direction_vectors)
    resolved_target = clip_to_box(raw_target, box=box)
    was_clipped = not np.allclose(raw_target, resolved_target)

    _obs, _terminated, _truncated, phase2_steps, is_success = _run_goal_phase(
        model,
        env,
        obs,
        resolved_target,
        max_phase_steps=max_steps - phase1_steps,
        terminated=terminated,
        truncated=truncated,
    )

    return RelativeMoveResult(
        success=is_success,
        n_steps=phase1_steps + phase2_steps,
        switch_step=switch_step,
        resolved_target_xyz=resolved_target,
        was_clipped=was_clipped,
    )
