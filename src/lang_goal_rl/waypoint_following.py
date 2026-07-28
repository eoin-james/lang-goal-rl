"""Stage 9: generalize stage 5's mid-episode goal switch to a chain of N >= 2 waypoints.

`midepisode_regoal.rollout_with_goal_switch` proved that swapping the policy's
goal input mid-episode (no `env.reset()`) works for exactly one switch,
literal-xyz goals. This module asks whether that mechanism holds for a whole
*chain* of switches: one continuous episode visiting `waypoints[0]`,
`waypoints[1]`, ..., `waypoints[N-1]` in order, each for its own step budget.

`rollout_with_waypoints` is a strict generalization, not a parallel
reimplementation: with `N=2` and a two-element `steps_per_leg`, it produces
the identical success outcome and step count as calling
`rollout_with_goal_switch` directly on the same two goals -- see the
regression test in `tests/lang_goal_rl/test_waypoint_following.py`'s
`TestEquivalenceWithMidepisodeRegoal`. Reuses
`midepisode_regoal._ensure_within_env_step_limit` for the same
budget-misconfiguration guard stage 5 already established, rather than
growing a second copy of it.

Literal-xyz goals only, deliberately -- unlike `rollout_with_goal_switch`
this module has no optional embedding-substitution mode. Stage 9 is testing
whether the re-goaling mechanism compounds cleanly over a longer chain, a
question fully orthogonal to stages 2-4's language/embedding layer, so there
is nothing here for an embedding mode to add.

Per-leg success, not just final-waypoint success: `per_waypoint_success[i]`
is judged from `info["is_success"]` gathered only during leg `i`'s own
steps, reset fresh at the start of each leg. A leg's success never leaks
into the next leg's judgment, and -- the one behavioral decision this module
makes that stage 5's two-phase code didn't need to -- an earlier leg failing
does NOT abort the rollout. Every remaining leg is still attempted with its
full step budget. Aborting early would hide exactly the information this
stage exists to produce: whether a policy that missed waypoint `i` can still
independently reach waypoint `i+1`, or whether error compounds down the
chain. Only the environment itself ending the episode (`terminated` or
`truncated`) can cut a later leg short; a merely-unreached goal never does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
import numpy.typing as npt

from lang_goal_rl.midepisode_regoal import _ensure_within_env_step_limit

if TYPE_CHECKING:
    import gymnasium as gym
    from stable_baselines3 import SAC


@dataclass(frozen=True)
class WaypointResult:
    """Outcome of one `rollout_with_waypoints` episode.

    Attributes:
        per_waypoint_success: One entry per waypoint, in order. Entry `i` is
            whether `info["is_success"]` was truthy on the final step of leg
            `i`, judged against `waypoints[i]` using only that leg's own
            steps.
        all_succeeded: `all(per_waypoint_success)`, provided as a single
            headline pass/fail so callers don't need to recompute it.
        n_steps: Total env steps run across every leg combined.
        leg_boundaries: The cumulative step index each leg ended at, one
            entry per waypoint. `leg_boundaries[-1] == n_steps`.
    """

    per_waypoint_success: tuple[bool, ...]
    all_succeeded: bool
    n_steps: int
    leg_boundaries: tuple[int, ...]


def _normalize_steps_per_leg(steps_per_leg: int | Sequence[int], n_legs: int) -> list[int]:
    """Expand a single int to `n_legs` equal budgets, or validate a per-leg sequence.

    Args:
        steps_per_leg: A single step budget applied to every leg, or one
            budget per leg.
        n_legs: Number of waypoints/legs in this rollout.

    Returns:
        A list of `n_legs` per-leg step budgets, each `>= 1`.

    Raises:
        ValueError: If a sequence is given with length != `n_legs`, or any
            budget is `< 1`.
    """
    if isinstance(steps_per_leg, int):
        budgets = [steps_per_leg] * n_legs
    else:
        budgets = list(steps_per_leg)
        if len(budgets) != n_legs:
            msg = (
                f"steps_per_leg has {len(budgets)} entries but there are {n_legs} waypoints "
                "-- pass a single int to apply to every leg, or one entry per waypoint"
            )
            raise ValueError(msg)
    if any(budget < 1 for budget in budgets):
        msg = f"steps_per_leg entries must all be >= 1 (got {budgets})"
        raise ValueError(msg)
    return budgets


def rollout_with_waypoints(
    model: SAC,
    env: gym.Env,
    *,
    waypoints: Sequence[npt.ArrayLike],
    steps_per_leg: int | Sequence[int],
    base_seed: int,
    max_steps: int | None = None,
) -> WaypointResult:
    """Roll out one episode visiting `waypoints` in order, no reset between legs.

    Generalizes `midepisode_regoal.rollout_with_goal_switch` from one switch
    to a chain of `N = len(waypoints)` legs: leg `i` targets `waypoints[i]`
    for `steps_per_leg[i]` steps (or the single `steps_per_leg` value, if an
    int, applied to every leg), then leg `i+1` begins immediately in the same
    episode. With `N=2` and a two-element `steps_per_leg`, this reduces to
    exactly `rollout_with_goal_switch`'s own result -- see this module's
    docstring and the equivalence regression test.

    Args:
        model: A trained SAC model. Only `model.predict` is called;
            `model.actor` is never accessed, matching
            `rollout_with_goal_switch`'s literal-xyz mode, so this works
            with any checkpoint including a plain stage-1 `MultiInputPolicy`.
        env: The FetchReach-v4 env instance to roll out on.
        waypoints: The ordered goals to visit, each shape `(3,)`. Must be
            non-empty.
        steps_per_leg: Step budget for each leg -- either one int applied to
            every leg, or a sequence with exactly `len(waypoints)` entries.
            Every budget must be `>= 1`.
        base_seed: Seed passed to `env.reset(seed=base_seed)`.
        max_steps: Optional explicit total-budget ceiling, validated against
            the env's registered `max_episode_steps` (see
            `midepisode_regoal._ensure_within_env_step_limit`) in place of
            the implied `sum(steps_per_leg)`. If given, must be `>=
            sum(steps_per_leg)` -- a smaller value would silently never be
            reached, since the actual run length is fully determined by the
            per-leg budgets, not by this ceiling. Defaults to
            `sum(steps_per_leg)`.

    Returns:
        A `WaypointResult` with per-leg success, the overall pass/fail,
        total steps, and cumulative leg boundaries.

    Raises:
        ValueError: If `waypoints` is empty, if `steps_per_leg` is
            malformed (see `_normalize_steps_per_leg`), if `max_steps` is
            given below `sum(steps_per_leg)`, or if the effective total
            budget exceeds the env's registered `max_episode_steps`.
    """
    if len(waypoints) == 0:
        msg = "waypoints must contain at least one goal"
        raise ValueError(msg)

    leg_budgets = _normalize_steps_per_leg(steps_per_leg, len(waypoints))
    total_budget = sum(leg_budgets)

    if max_steps is not None and max_steps < total_budget:
        msg = (
            f"max_steps ({max_steps}) is less than the sum of steps_per_leg ({total_budget}) "
            "-- the rollout's actual length is fully determined by steps_per_leg, so a smaller "
            "max_steps would never be reached; it exists only to validate a caller's expected "
            "total against the env's own step limit"
        )
        raise ValueError(msg)
    _ensure_within_env_step_limit(env, max_steps if max_steps is not None else total_budget)

    obs, _info = env.reset(seed=base_seed)

    n_steps = 0
    terminated = truncated = False
    per_waypoint_success: list[bool] = []
    leg_boundaries: list[int] = []

    for waypoint, leg_budget in zip(waypoints, leg_budgets, strict=True):
        goal = np.asarray(waypoint, dtype=np.float64)
        env.unwrapped.goal = goal.copy()
        obs["desired_goal"] = goal.copy()

        leg_end_step = n_steps + leg_budget
        is_success = False
        while n_steps < leg_end_step and not (terminated or truncated):
            action, _state = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            is_success = bool(info.get("is_success", is_success))
            n_steps += 1

        per_waypoint_success.append(is_success)
        leg_boundaries.append(n_steps)

    return WaypointResult(
        per_waypoint_success=tuple(per_waypoint_success),
        all_succeeded=all(per_waypoint_success),
        n_steps=n_steps,
        leg_boundaries=tuple(leg_boundaries),
    )
