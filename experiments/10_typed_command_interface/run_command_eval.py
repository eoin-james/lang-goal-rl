"""Stage 10 proof gate: goto/move/waypoint success through the real typed-command pipeline.

Every eval in this script goes through `parse_command` -> `CommandExecutor` ->
`clip_to_box` -> the real env/policy -- exactly the pipeline
`interactive_demo.run_commands` drives, never `relative_move.py`/
`waypoint_following.py`'s functions called directly. That's the point of
this stage: it's an *integration* check on top of stages 8/9's already-
validated mechanisms, not a re-derivation of them.

Five things measured per seed, in this order:

1. Literal-goal sanity check (same convention every prior stage uses) --
   confirms the reused checkpoint isn't one of the two documented
   SAC-collapse seeds before trusting anything else below.
2. `goto` eval: uniformly-sampled in-box points, issued as a single `goto
   X Y Z` command per episode, run through the full command pipeline, and
   compared against `rollout_fresh_with_budget` called directly on the
   identical (goal, seed, budget) -- a paired integration check, not just a
   raw success rate.
3. `move` eval: byte-for-byte the same combinations
   `08_relative_move_validation/run_relative_move_eval.py` uses (3 switch
   steps x 6 directions x 3 magnitudes x 20 episodes), but the post-switch
   target is resolved via `parse_command("move ...")` +
   `CommandExecutor.apply_command` instead of calling
   `relative_move.rollout_with_relative_move` -- if this stage's numbers
   diverge from stage 8's own, that's an integration bug, not noise.
4. `waypoints` eval: byte-for-byte the same combinations
   `09_waypoint_following/run_waypoint_eval.py` uses (2 sequence kinds x 3
   chain lengths x 2 budgets x 50 episodes), but the chain is driven by
   `parse_command("waypoints ...")` + `CommandExecutor.apply_command`/
   `advance` instead of calling `waypoint_following.rollout_with_waypoints`.
5. Stop-hold drift (new -- first real test of `StopCommand`'s design):
   mid-episode, issue `stop`, then measure how far the gripper drifts from
   its position-at-stop over the next K steps, at three stop timings.
6. Out-of-bounds `goto` clipping: a handful of deliberately way-out-of-box
   `goto` targets, confirming `clip_to_box` actually engages and the
   episode still runs to completion rather than crashing.

Malformed-input rejection is checked separately in
`check_malformed_input.py` (pure parser, no env/model needed).

Results are dumped to `runs/seed_<k>/results.json` for
`aggregate_and_report.py`; per-section summary lines are also printed so
`runs/seed_<k>/stdout.log` is independently readable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3 import SAC

from lang_goal_rl.command_executor import CommandExecutor
from lang_goal_rl.command_grammar import GotoCommand, MoveCommand, WaypointsCommand, parse_command
from lang_goal_rl.goal_region_vocabulary import MEASURED_GOAL_BOX, GoalBox, region_names, sample_region_goals
from lang_goal_rl.midepisode_regoal import _run_goal_phase, rollout_fresh_with_budget
from lang_goal_rl.relative_move import DIRECTION_UNIT_VECTORS, clip_to_box, compute_relative_goal

EXPERIMENT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = EXPERIMENT_DIR.parent / "01_uvfa_her_baseline" / "checkpoints"
"""Same 8-healthy-seed checkpoint set stages 8/9 reuse zero-shot -- see
ROADMAP.md's Known risks for why seeds 2 and 7 are never passed here."""

ENV_ID = "FetchReach-v4"
MAX_STEPS = 50
"""FetchReach-v4's registered `max_episode_steps` -- used for the default-length env
(sanity check, goto, move, out-of-bounds goto)."""

EXTENDED_MAX_EPISODE_STEPS = 100
"""Matches `09_waypoint_following/run_waypoint_eval.py`'s own extension: waypoint chains
(up to 5 legs x 18 steps = 90) and the stop-hold check's latest stop_step (40) + largest K
(20) = 60 both exceed the default 50-step limit -- see each module's own docstring for why
this doesn't change step-by-step dynamics, only how many steps the `TimeLimit` wrapper
allows before truncating."""

DEFAULT_SANITY_EPISODES = 50

# --- goto ---------------------------------------------------------------

N_GOTO_EPISODES = 100
GOTO_BASE_SEED = 30_000
GOTO_GOAL_SEED_OFFSET = 900_000
"""Offset applied before sampling a goto goal so that draw never reuses the episode's own
`env.reset(seed=...)` seed -- same convention as stage 8's `INITIAL_GOAL_SEED_OFFSET`."""


def literal_goal_sanity_check(model: SAC, env: gym.Env, n_episodes: int) -> float:
    """Roll out the policy on the env's own randomly-sampled goal, no override.

    Local copy of stage 8/9's own sanity-check helper (same `1000 + episode` held-out
    seed convention, same deterministic rollout loop) -- this project's convention is
    same-directory local copies rather than cross-experiment-directory imports.

    Args:
        model: The trained SAC checkpoint under test.
        env: The FetchReach-v4 env instance to roll out on.
        n_episodes: Number of held-out episodes to evaluate.

    Returns:
        Mean success rate over `n_episodes`.
    """
    successes = []
    for episode in range(n_episodes):
        obs, _info = env.reset(seed=1000 + episode)
        terminated = truncated = False
        is_success = False
        while not (terminated or truncated):
            action, _state = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            is_success = bool(info.get("is_success", is_success))
        successes.append(is_success)
    return float(np.mean(successes))


def sample_initial_goal(seed: int) -> np.ndarray:
    """Sample a varied xyz point from a randomly-chosen goal region (stage 8's helper, copied).

    Args:
        seed: Deterministic seed for both the region choice and the within-region point draw.

    Returns:
        An xyz point, shape `(3,)`.
    """
    rng = np.random.default_rng(seed)
    region = rng.choice(region_names())
    return sample_region_goals(str(region), 1, seed=seed)[0]


def sample_uniform_goal(seed: int, box: GoalBox = MEASURED_GOAL_BOX) -> np.ndarray:
    """Sample one xyz point uniformly within `box` -- FetchReach's own true goal distribution.

    Args:
        seed: Seed for the single uniform draw.
        box: The box to sample within.

    Returns:
        An xyz point, shape `(3,)`, guaranteed inside `box`.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(box.axis_min, box.axis_max)


def run_goto_eval(
    model: SAC, env: gym.Env, *, n_episodes: int, base_seed: int, box: GoalBox = MEASURED_GOAL_BOX,
) -> dict:
    """Run `n_episodes` `goto` commands through the real pipeline, paired against a direct baseline.

    Each episode: sample an in-box point, issue it as `parse_command("goto X Y Z")` ->
    `CommandExecutor.apply_command` -> `clip_to_box` -> a full-budget episode (this is the
    thing under test), then separately call `rollout_fresh_with_budget` on the identical
    (goal, seed, budget) directly -- bypassing the parser/executor entirely -- as the
    integration-check baseline. Both conditions target the same point sampled uniformly
    within `box`, so it is never clipped for this eval (out-of-bounds `goto` is a separate,
    dedicated check -- see `run_out_of_bounds_goto_eval`).

    Args:
        model: The trained SAC checkpoint under test.
        env: A FetchReach-v4 env instance at its default 50-step episode length.
        n_episodes: Number of (pipeline, baseline) episode pairs to run.
        base_seed: First episode's seed; episode `i` uses `base_seed + i` for both conditions.
        box: The box to sample goals within and clip into.

    Returns:
        A dict with per-episode success lists for both conditions and the aggregated rates.
    """
    pipeline_successes: list[bool] = []
    baseline_successes: list[bool] = []
    for episode_index in range(n_episodes):
        episode_seed = base_seed + episode_index
        goal = sample_uniform_goal(episode_seed + GOTO_GOAL_SEED_OFFSET, box=box)

        command = parse_command(f"goto {goal[0]} {goal[1]} {goal[2]}")
        if not isinstance(command, GotoCommand):
            msg = f"expected GotoCommand, got {type(command).__name__}"
            raise TypeError(msg)
        executor = CommandExecutor(box=box)

        obs, _info = env.reset(seed=episode_seed)
        achieved = np.array(obs["achieved_goal"], copy=True)
        executor.apply_command(command, current_achieved_xyz=achieved)
        target = clip_to_box(executor.target_for_step(), box=box)
        env.unwrapped.goal = target.copy()
        obs["desired_goal"] = target.copy()

        terminated = truncated = False
        is_success = False
        steps = 0
        while steps < MAX_STEPS and not (terminated or truncated):
            action, _state = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            is_success = bool(info.get("is_success", is_success))
            steps += 1
        pipeline_successes.append(is_success)

        baseline_success = rollout_fresh_with_budget(
            model, env, goal_xyz=goal, max_steps=MAX_STEPS, base_seed=episode_seed,
        )
        baseline_successes.append(baseline_success)

    return {
        "n_episodes": n_episodes,
        "pipeline_successes": pipeline_successes,
        "baseline_successes": baseline_successes,
        "pipeline_success_rate": float(np.mean(pipeline_successes)),
        "baseline_success_rate": float(np.mean(baseline_successes)),
    }


# --- move (exact stage-8 parity) ----------------------------------------

SWITCH_STEPS: tuple[int, ...] = (10, 25, 40)
DIRECTIONS: tuple[str, ...] = tuple(DIRECTION_UNIT_VECTORS.keys())
MAGNITUDES: dict[str, float] = {
    "small_5cm": 0.05,
    "medium_15cm": 0.15,
    "clip_forcing_35cm": 0.35,
}
MOVE_EPISODES_PER_COMBO = 20
MOVE_BASE_SEED = 40_000
MOVE_INITIAL_GOAL_SEED_OFFSET = 710_000
"""Distinct from stage 8's own `INITIAL_GOAL_SEED_OFFSET` (700_000) so this stage's episode
seeds never draw the exact same initial-goal sample stage 8 already used, even though both
scripts otherwise follow the identical convention."""


def run_move_combo_eval(
    model: SAC,
    env: gym.Env,
    *,
    switch_step: int,
    direction: str,
    magnitude_label: str,
    distance_m: float,
    n_episodes: int,
    base_seed: int,
    box: GoalBox = MEASURED_GOAL_BOX,
) -> dict:
    """Run one (switch_step, direction, magnitude) combo through the command pipeline.

    Structurally identical to stage 8's `run_combo_eval`: pre-switch phase toward a varied
    `initial_goal_xyz` (reusing `midepisode_regoal._run_goal_phase`, the same helper
    `relative_move.rollout_with_relative_move` itself uses for this exact phase -- see that
    module's docstring for the precedent of importing this private helper across a module
    boundary), then a post-switch target resolved via `parse_command("move ...")` ->
    `CommandExecutor.apply_command` (which calls `compute_relative_goal` internally, already
    clipped) rather than calling `rollout_with_relative_move` directly. The baseline is the
    same budget-matched fresh rollout stage 8 uses, judged against the identical resolved
    target.

    Args:
        model: The trained SAC checkpoint under test.
        env: A FetchReach-v4 env instance at its default 50-step episode length.
        switch_step: Step at which the `move` command is issued.
        direction: One of `DIRECTIONS` (a `KNOWN_DIRECTIONS` phrase).
        magnitude_label: Key into `MAGNITUDES`, echoed back for aggregation.
        distance_m: The magnitude in meters (`MAGNITUDES[magnitude_label]`).
        n_episodes: Number of episodes for this combo.
        base_seed: First episode's seed; episode `i` uses `base_seed + i`.
        box: The box the resolved move target is clipped into.

    Returns:
        A dict with the combo's identifying fields, per-episode success lists for both
        conditions, the clip-flag list, and the aggregated rates.
    """
    move_successes: list[bool] = []
    baseline_successes: list[bool] = []
    was_clipped_flags: list[bool] = []
    for episode_index in range(n_episodes):
        episode_seed = base_seed + episode_index
        initial_goal_xyz = sample_initial_goal(episode_seed + MOVE_INITIAL_GOAL_SEED_OFFSET)

        obs, _info = env.reset(seed=episode_seed)
        obs, terminated, truncated, phase1_steps, _phase1_success = _run_goal_phase(
            model, env, obs, initial_goal_xyz, max_phase_steps=switch_step, terminated=False, truncated=False,
        )
        achieved_at_switch = np.array(obs["achieved_goal"], copy=True)

        command = parse_command(f"move {direction} {distance_m}")
        if not isinstance(command, MoveCommand):
            msg = f"expected MoveCommand, got {type(command).__name__}"
            raise TypeError(msg)
        executor = CommandExecutor(box=box)
        executor.apply_command(command, current_achieved_xyz=achieved_at_switch)
        resolved_target = clip_to_box(executor.target_for_step(), box=box)

        raw_target = achieved_at_switch + distance_m * DIRECTION_UNIT_VECTORS[direction]
        was_clipped = not np.allclose(raw_target, resolved_target)

        _obs, _terminated, _truncated, phase2_steps, is_success = _run_goal_phase(
            model,
            env,
            obs,
            resolved_target,
            max_phase_steps=MAX_STEPS - phase1_steps,
            terminated=terminated,
            truncated=truncated,
        )
        move_successes.append(is_success)
        was_clipped_flags.append(was_clipped)

        baseline_success = rollout_fresh_with_budget(
            model, env, goal_xyz=resolved_target, max_steps=MAX_STEPS - switch_step, base_seed=episode_seed,
        )
        baseline_successes.append(baseline_success)

    return {
        "switch_step": switch_step,
        "direction": direction,
        "magnitude_label": magnitude_label,
        "distance_m": distance_m,
        "n_episodes": n_episodes,
        "move_successes": move_successes,
        "baseline_successes": baseline_successes,
        "was_clipped_flags": was_clipped_flags,
        "move_success_rate": float(np.mean(move_successes)),
        "baseline_success_rate": float(np.mean(baseline_successes)),
        "clip_rate": float(np.mean(was_clipped_flags)),
    }


# --- waypoints (exact stage-9 parity) -----------------------------------

TIGHT_BUDGET = 9
GENEROUS_BUDGET = 18
BUDGETS: dict[str, int] = {"tight": TIGHT_BUDGET, "generous": GENEROUS_BUDGET}
CHAIN_LENGTHS: tuple[int, ...] = (2, 3, 5)
SEQUENCE_KINDS: tuple[str, ...] = ("literal", "relative")
WAYPOINT_EPISODES_PER_CONDITION = 50
WAYPOINT_BASE_SEED = 50_000

LITERAL_REGION_SEED_OFFSET = 711_000
LITERAL_POINT_SEED_OFFSET = 811_000
RELATIVE_LEG0_SEED_OFFSET = 911_000
RELATIVE_DIRECTION_SEED_OFFSET = 961_000
RELATIVE_MOVE_DISTANCE_M = 0.15
"""Waypoint-generation constants, copied from `09_waypoint_following/run_waypoint_eval.py`'s
identical helpers with new seed offsets (this stage's own block, disjoint from stage 9's) --
same rationale, same sampling logic, so the (sequence_kind, chain_len, budget) numbers below
are directly comparable to stage 9's own."""


def generate_literal_waypoints(n_legs: int, seed: int) -> list[np.ndarray]:
    """Sample `n_legs` xyz waypoints from `n_legs` distinct goal regions (stage 9's helper, copied).

    Args:
        n_legs: Number of waypoints/legs to generate. Must be `<= 7`.
        seed: Deterministic seed for both the region choice and the within-region point draws.

    Returns:
        `n_legs` waypoints, in leg order.
    """
    names = region_names()
    rng = np.random.default_rng(seed + LITERAL_REGION_SEED_OFFSET)
    chosen_regions = rng.choice(np.array(names), size=n_legs, replace=False)
    return [
        sample_region_goals(str(region), 1, seed=seed + LITERAL_POINT_SEED_OFFSET + leg_index)[0]
        for leg_index, region in enumerate(chosen_regions)
    ]


def generate_relative_waypoints(
    n_legs: int, seed: int, distance_m: float = RELATIVE_MOVE_DISTANCE_M,
) -> list[np.ndarray]:
    """Chain relative moves off each leg's own precomputed target (stage 9's helper, copied).

    Leg 0 is a literal xyz point sampled from the "center" region; every subsequent leg is
    `current + distance_m * DIRECTION_UNIT_VECTORS[direction]`, clipped into
    `MEASURED_GOAL_BOX`, applied to the *previous leg's own target* -- the same stage-9 v1
    scope limit (legs precomputed before the episode starts, not truly live-relative).

    Args:
        n_legs: Number of waypoints/legs to generate.
        seed: Deterministic seed for leg 0's draw and every direction choice.
        distance_m: Signed distance moved per relative leg.

    Returns:
        `n_legs` waypoints, in leg order.
    """
    leg0 = sample_region_goals("center", 1, seed=seed + RELATIVE_LEG0_SEED_OFFSET)[0]
    waypoints = [leg0]
    direction_names = list(DIRECTION_UNIT_VECTORS.keys())
    rng = np.random.default_rng(seed + RELATIVE_DIRECTION_SEED_OFFSET)
    for _ in range(n_legs - 1):
        direction = str(rng.choice(np.array(direction_names)))
        waypoints.append(compute_relative_goal(waypoints[-1], direction, distance_m))
    return waypoints


def rollout_command_waypoints(
    model: SAC,
    env: gym.Env,
    *,
    waypoints: list[np.ndarray],
    steps_per_leg: int,
    base_seed: int,
    box: GoalBox = MEASURED_GOAL_BOX,
) -> tuple[list[bool], int]:
    """Roll out one waypoint chain through `parse_command`/`CommandExecutor`, no reset between legs.

    Mirrors `interactive_demo.run_commands`'s real step loop exactly: apply the parsed
    `WaypointsCommand` once, then every env step calls `executor.advance(...)` followed by a
    `clip_to_box` write of `executor.target_for_step()` into the env -- the same two calls
    `run_commands` makes every step, regardless of command type. Leg boundaries are fully
    deterministic here (`CommandExecutor`'s single `steps_per_leg` applies equally to every
    leg -- see its own docstring), so per-leg success is tracked by step-index // budget
    rather than by watching for a target change, which would be fragile if two legs ever
    happened to resolve to numerically identical targets.

    Per-leg success uses the *final* step's `info["is_success"]` within that leg (never a
    sticky OR) -- matching `waypoint_following.rollout_with_waypoints`'s own
    `is_success = bool(info.get("is_success", is_success))` semantics exactly, since
    `info` always carries this key for FetchReach so it always overwrites.

    Args:
        model: The trained SAC checkpoint under test.
        env: A FetchReach-v4 env instance with `max_episode_steps >= len(waypoints) *
            steps_per_leg`.
        waypoints: The ordered xyz legs to visit, each shape `(3,)`.
        steps_per_leg: Step budget per leg (a single int -- `CommandExecutor`'s constructor
            only accepts one budget for every leg, per stage 10's design).
        base_seed: Seed passed to `env.reset(seed=base_seed)`.
        box: The box every commanded goal is clipped into before being written into the env.

    Returns:
        `(per_leg_success, n_steps)` -- one success bool per waypoint, in leg order, and the
        total env steps actually run (may be less than `len(waypoints) * steps_per_leg` if
        the episode ends first).
    """
    n_legs = len(waypoints)
    leg_text = ", ".join(f"{w[0]} {w[1]} {w[2]}" for w in waypoints)
    command = parse_command(f"waypoints {leg_text}")
    if not isinstance(command, WaypointsCommand):
        msg = f"expected WaypointsCommand, got {type(command).__name__}"
        raise TypeError(msg)
    executor = CommandExecutor(box=box, steps_per_leg=steps_per_leg)

    obs, _info = env.reset(seed=base_seed)
    achieved = np.array(obs["achieved_goal"], copy=True)
    executor.apply_command(command, current_achieved_xyz=achieved)
    target = clip_to_box(executor.target_for_step(), box=box)
    env.unwrapped.goal = target.copy()
    obs["desired_goal"] = target.copy()

    total_budget = n_legs * steps_per_leg
    per_leg_success = [False] * n_legs
    current_leg = 0
    leg_success = False
    n_steps = 0
    terminated = truncated = False
    for step in range(total_budget):
        leg_at_step = step // steps_per_leg
        if leg_at_step != current_leg:
            per_leg_success[current_leg] = leg_success
            leg_success = False
            current_leg = leg_at_step

        action, _state = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        leg_success = bool(info.get("is_success", leg_success))
        executor.advance(achieved_xyz=obs["achieved_goal"], is_success=leg_success)
        target = clip_to_box(executor.target_for_step(), box=box)
        env.unwrapped.goal = target.copy()
        obs["desired_goal"] = target.copy()
        n_steps += 1
        if terminated or truncated:
            break

    per_leg_success[current_leg] = leg_success
    return per_leg_success, n_steps


def run_command_waypoint_condition(
    model: SAC,
    env: gym.Env,
    *,
    sequence_kind: str,
    chain_len: int,
    budget_name: str,
    n_episodes: int,
    base_seed: int,
    box: GoalBox = MEASURED_GOAL_BOX,
) -> dict:
    """Run one (sequence_kind, chain_len, budget) condition through the command pipeline.

    Args:
        model: The trained SAC checkpoint under test.
        env: A FetchReach-v4 env instance with `max_episode_steps` covering this condition's
            full `chain_len * budget`.
        sequence_kind: "literal" or "relative".
        chain_len: Number of waypoints/legs.
        budget_name: "tight" or "generous" -- key into `BUDGETS`.
        n_episodes: Number of episodes for this condition.
        base_seed: First episode's seed; episode `i` uses `base_seed + i`.
        box: The box every commanded goal is clipped into.

    Returns:
        A dict with per-leg chain/baseline success-rate lists and the whole-chain rate.
    """
    budget = BUDGETS[budget_name]
    per_leg_chain_successes: list[list[bool]] = [[] for _ in range(chain_len)]
    per_leg_baseline_successes: list[list[bool]] = [[] for _ in range(chain_len)]
    all_succeeded_flags: list[bool] = []

    for episode_index in range(n_episodes):
        episode_seed = base_seed + episode_index
        if sequence_kind == "literal":
            waypoints = generate_literal_waypoints(chain_len, seed=episode_seed)
        else:
            waypoints = generate_relative_waypoints(chain_len, seed=episode_seed)

        per_leg_success, _n_steps = rollout_command_waypoints(
            model, env, waypoints=waypoints, steps_per_leg=budget, base_seed=episode_seed, box=box,
        )
        all_succeeded_flags.append(all(per_leg_success))

        for leg_index in range(chain_len):
            per_leg_chain_successes[leg_index].append(per_leg_success[leg_index])
            baseline_success = rollout_fresh_with_budget(
                model, env, goal_xyz=waypoints[leg_index], max_steps=budget, base_seed=episode_seed,
            )
            per_leg_baseline_successes[leg_index].append(baseline_success)

    return {
        "sequence_kind": sequence_kind,
        "chain_len": chain_len,
        "budget_name": budget_name,
        "budget": budget,
        "n_episodes": n_episodes,
        "per_leg_chain_success_rate": [float(np.mean(x)) for x in per_leg_chain_successes],
        "per_leg_baseline_success_rate": [float(np.mean(x)) for x in per_leg_baseline_successes],
        "all_succeeded_rate": float(np.mean(all_succeeded_flags)),
    }


# --- stop-hold drift (new -- first real test) ---------------------------

STOP_STEPS: tuple[int, ...] = (10, 25, 40)
STOP_K_VALUES: tuple[int, ...] = (10, 20)
STOP_EPISODES_PER_CONDITION = 30
STOP_BASE_SEED = 60_000
STOP_INITIAL_GOAL_SEED_OFFSET = 712_000


def run_stop_hold_eval(
    model: SAC,
    env: gym.Env,
    *,
    stop_step: int,
    k_values: tuple[int, ...],
    n_episodes: int,
    base_seed: int,
    box: GoalBox = MEASURED_GOAL_BOX,
) -> dict[int, list[float]]:
    """Measure gripper drift after a `stop` command, at each K in `k_values`.

    Mirrors `run_commands`'s real step loop: drive toward a varied initial goal for
    `stop_step` steps, issue `stop` (freezing the hold target at the achieved position that
    instant), then keep stepping -- calling `executor.advance` and re-writing
    `executor.target_for_step()` into the env every step, exactly as the production loop
    does even though neither call changes anything for a single-goal `Stop` (no waypoint
    queue to advance through). Drift at K is the Euclidean distance between the achieved
    position K steps after the stop and the achieved position *at* the stop.

    Args:
        model: The trained SAC checkpoint under test.
        env: A FetchReach-v4 env instance with `max_episode_steps >= stop_step + max(k_values)`.
        stop_step: Step at which `stop` is issued.
        k_values: Post-stop step counts to measure drift at.
        n_episodes: Number of episodes to run.
        base_seed: First episode's seed; episode `i` uses `base_seed + i`.
        box: The box the hold target is clipped into (irrelevant in practice -- the
            achieved position is always in-box -- but kept for pipeline fidelity).

    Returns:
        Mapping from each K in `k_values` to the list of per-episode drift distances
        (only episodes that ran at least K post-stop steps before the env ended contribute
        to that K's list).
    """
    max_k = max(k_values)
    drifts: dict[int, list[float]] = {k: [] for k in k_values}

    for episode_index in range(n_episodes):
        episode_seed = base_seed + episode_index
        initial_goal_xyz = sample_initial_goal(episode_seed + STOP_INITIAL_GOAL_SEED_OFFSET)

        obs, _info = env.reset(seed=episode_seed)
        obs, terminated, truncated, _phase1_steps, _phase1_success = _run_goal_phase(
            model, env, obs, initial_goal_xyz, max_phase_steps=stop_step, terminated=False, truncated=False,
        )
        if terminated or truncated:
            continue  # episode ended before the stop point -- shouldn't happen given max_episode_steps, but guard

        achieved_at_stop = np.array(obs["achieved_goal"], copy=True)
        stop_command = parse_command("stop")
        executor = CommandExecutor(box=box)
        executor.apply_command(stop_command, current_achieved_xyz=achieved_at_stop)
        hold_target = clip_to_box(executor.target_for_step(), box=box)
        env.unwrapped.goal = hold_target.copy()
        obs["desired_goal"] = hold_target.copy()

        positions = [achieved_at_stop]
        step = 0
        while step < max_k and not (terminated or truncated):
            action, _state = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            executor.advance(achieved_xyz=obs["achieved_goal"], is_success=bool(info.get("is_success", False)))
            target = clip_to_box(executor.target_for_step(), box=box)
            env.unwrapped.goal = target.copy()
            obs["desired_goal"] = target.copy()
            positions.append(np.array(obs["achieved_goal"], copy=True))
            step += 1

        for k in k_values:
            if k < len(positions):
                drifts[k].append(float(np.linalg.norm(positions[k] - positions[0])))

    return drifts


# --- out-of-bounds goto clipping ----------------------------------------

OOB_GOTO_EPISODES = 10
OOB_BASE_SEED = 70_000
OOB_OVERSHOOT_FACTOR = 3.0
"""Multiplies the box's per-axis half-range so the requested point is guaranteed outside
`box` on the chosen axis regardless of sign -- comfortably beyond stage 8's own
`clip_forcing_35cm` margin, since this check is specifically about `goto` (which
`CommandExecutor` stores unclipped) rather than `move`."""


def run_out_of_bounds_goto_eval(
    model: SAC, env: gym.Env, *, n_episodes: int, base_seed: int, box: GoalBox = MEASURED_GOAL_BOX,
) -> list[dict]:
    """Issue deliberately way-out-of-box `goto` commands and confirm clipping, not a crash.

    Args:
        model: The trained SAC checkpoint under test.
        env: A FetchReach-v4 env instance at its default 50-step episode length.
        n_episodes: Number of out-of-bounds `goto` episodes to run.
        base_seed: First episode's seed; episode `i` uses `base_seed + i`.
        box: The box the raw (unclipped) `Goto` target should be clipped into.

    Returns:
        One dict per episode: the raw (unclipped) and clipped target, whether clipping
        actually changed the value, whether the episode ran to completion without raising,
        the resulting success flag, and steps run.
    """
    rng = np.random.default_rng(base_seed)
    results = []
    for episode_index in range(n_episodes):
        episode_seed = base_seed + episode_index
        axis = int(rng.integers(0, 3))
        sign = float(rng.choice([-1.0, 1.0]))
        raw_point = box.centroid.copy()
        raw_point[axis] += sign * (box.half_range[axis] * OOB_OVERSHOOT_FACTOR)

        command = parse_command(f"goto {raw_point[0]} {raw_point[1]} {raw_point[2]}")
        executor = CommandExecutor(box=box)
        obs, _info = env.reset(seed=episode_seed)
        achieved = np.array(obs["achieved_goal"], copy=True)
        executor.apply_command(command, current_achieved_xyz=achieved)
        raw_target = np.asarray(executor.target_for_step(), dtype=np.float64)
        clipped_target = clip_to_box(raw_target, box=box)
        was_clipped = not np.allclose(raw_target, clipped_target)

        env.unwrapped.goal = clipped_target.copy()
        obs["desired_goal"] = clipped_target.copy()

        crashed = False
        is_success = False
        steps = 0
        try:
            terminated = truncated = False
            while steps < MAX_STEPS and not (terminated or truncated):
                action, _state = model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, info = env.step(action)
                is_success = bool(info.get("is_success", is_success))
                steps += 1
        except Exception as error:  # noqa: BLE001 -- this check's whole point is "did anything raise", the exception itself is the signal
            crashed = True
            print(f"[out_of_bounds_goto] episode {episode_index}: CRASHED -- {type(error).__name__}: {error}")

        results.append(
            {
                "axis": axis,
                "sign": sign,
                "raw_target": raw_target.tolist(),
                "clipped_target": clipped_target.tolist(),
                "was_clipped": was_clipped,
                "crashed": crashed,
                "success": is_success,
                "n_steps": steps,
            }
        )
    return results


def main() -> None:
    """Load one seed's checkpoint zero-shot and run the full stage-10 command-pipeline eval suite."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Model checkpoint seed (0-9, never 2 or 7)")
    parser.add_argument("--sanity-episodes", type=int, default=DEFAULT_SANITY_EPISODES)
    args = parser.parse_args()

    checkpoint_path = CHECKPOINT_DIR / f"seed_{args.seed}.zip"
    gym.register_envs(gymnasium_robotics)

    # --- default-length env: sanity check, goto, move, out-of-bounds goto ---
    env = gym.make(ENV_ID)
    model = SAC.load(checkpoint_path, env=env)
    print(f"loaded checkpoint from {checkpoint_path} (zero-shot, no new training, seed={args.seed})")

    sanity_rate = literal_goal_sanity_check(model, env, args.sanity_episodes)
    print(
        f"sanity_check_success_rate={sanity_rate:.3f} over {args.sanity_episodes} episodes "
        "(literal control, full 50-step episode, no command pipeline)"
    )

    goto_result = run_goto_eval(model, env, n_episodes=N_GOTO_EPISODES, base_seed=GOTO_BASE_SEED + args.seed * 1000)
    print(
        f"goto: pipeline_success_rate={goto_result['pipeline_success_rate']:.3f} "
        f"baseline_success_rate={goto_result['baseline_success_rate']:.3f} over {N_GOTO_EPISODES} episodes"
    )

    move_results = []
    combo_index = 0
    for switch_step in SWITCH_STEPS:
        for direction in DIRECTIONS:
            for magnitude_label, distance_m in MAGNITUDES.items():
                base_seed = MOVE_BASE_SEED + args.seed * 100_000 + combo_index * 1000
                result = run_move_combo_eval(
                    model,
                    env,
                    switch_step=switch_step,
                    direction=direction,
                    magnitude_label=magnitude_label,
                    distance_m=distance_m,
                    n_episodes=MOVE_EPISODES_PER_COMBO,
                    base_seed=base_seed,
                )
                move_results.append(result)
                print(
                    f"move: switch_step={switch_step} direction={direction!r} "
                    f"magnitude={magnitude_label} ({distance_m}m) "
                    f"move_success_rate={result['move_success_rate']:.3f} "
                    f"baseline_success_rate={result['baseline_success_rate']:.3f} "
                    f"clip_rate={result['clip_rate']:.3f} over {MOVE_EPISODES_PER_COMBO} episodes"
                )
                combo_index += 1

    oob_result = run_out_of_bounds_goto_eval(
        model, env, n_episodes=OOB_GOTO_EPISODES, base_seed=OOB_BASE_SEED + args.seed * 1000,
    )
    n_clipped = sum(1 for r in oob_result if r["was_clipped"])
    n_crashed = sum(1 for r in oob_result if r["crashed"])
    print(
        f"out_of_bounds_goto: clipped={n_clipped}/{len(oob_result)} crashed={n_crashed}/{len(oob_result)} "
        f"over {OOB_GOTO_EPISODES} episodes"
    )
    env.close()

    # --- extended-length env: waypoints, stop-hold drift ---
    ext_env = gym.make(ENV_ID, max_episode_steps=EXTENDED_MAX_EPISODE_STEPS)
    ext_model = SAC.load(checkpoint_path, env=ext_env)

    waypoint_results = []
    condition_index = 0
    for sequence_kind in SEQUENCE_KINDS:
        for chain_len in CHAIN_LENGTHS:
            for budget_name in BUDGETS:
                base_seed = WAYPOINT_BASE_SEED + args.seed * 100_000 + condition_index * 10_000
                result = run_command_waypoint_condition(
                    ext_model,
                    ext_env,
                    sequence_kind=sequence_kind,
                    chain_len=chain_len,
                    budget_name=budget_name,
                    n_episodes=WAYPOINT_EPISODES_PER_CONDITION,
                    base_seed=base_seed,
                )
                waypoint_results.append(result)
                per_leg_chain = ", ".join(f"{rate:.3f}" for rate in result["per_leg_chain_success_rate"])
                per_leg_baseline = ", ".join(f"{rate:.3f}" for rate in result["per_leg_baseline_success_rate"])
                print(
                    f"waypoints: kind={sequence_kind} chain_len={chain_len} budget={budget_name}({result['budget']}) "
                    f"chain_per_leg=[{per_leg_chain}] baseline_per_leg=[{per_leg_baseline}] "
                    f"all_succeeded_rate={result['all_succeeded_rate']:.3f} over {WAYPOINT_EPISODES_PER_CONDITION} episodes"
                )
                condition_index += 1

    stop_results: dict[str, dict[str, list[float]]] = {}
    for stop_step_index, stop_step in enumerate(STOP_STEPS):
        base_seed = STOP_BASE_SEED + args.seed * 10_000 + stop_step_index * 1000
        drifts = run_stop_hold_eval(
            ext_model,
            ext_env,
            stop_step=stop_step,
            k_values=STOP_K_VALUES,
            n_episodes=STOP_EPISODES_PER_CONDITION,
            base_seed=base_seed,
        )
        stop_results[str(stop_step)] = {str(k): v for k, v in drifts.items()}
        summary = " ".join(
            f"drift_k{k}_mean={np.mean(v):.4f}_max={np.max(v):.4f}" if v else f"drift_k{k}=no_data"
            for k, v in drifts.items()
        )
        print(f"stop_hold: stop_step={stop_step} {summary} over {STOP_EPISODES_PER_CONDITION} episodes")
    ext_env.close()

    output = {
        "model_seed": args.seed,
        "checkpoint": str(checkpoint_path),
        "sanity_check_success_rate": sanity_rate,
        "sanity_check_episodes": args.sanity_episodes,
        "goto": goto_result,
        "move_combo_results": move_results,
        "waypoint_condition_results": waypoint_results,
        "stop_hold_drift": stop_results,
        "out_of_bounds_goto": oob_result,
    }
    results_path = EXPERIMENT_DIR / "runs" / f"seed_{args.seed}" / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(output, indent=2))
    print(f"results_saved={results_path}")


if __name__ == "__main__":
    main()
