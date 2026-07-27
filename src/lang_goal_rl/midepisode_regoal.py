"""Stage 5: inject a new instruction/goal mid-episode, without resetting.

Two rollout utilities, matched to the proof gate's budget-matched comparison
(see ROADMAP.md stage 5): a naive "run the full episode toward one goal, then
just start feeding a different goal partway through" would confound "did the
re-goaling mechanism work" with "did the policy have enough steps left to get
there" -- comparing a 30-steps-remaining swap against a full 50-step fresh
episode is not a fair test of re-goaling itself.

- `rollout_with_goal_switch`: the actual mechanism under test. Runs one
  episode, no reset, targeting `initial_goal_xyz` for `switch_step` steps,
  then `new_goal_xyz` for the remaining steps up to `max_steps` (the episode's
  *total* budget, matching stage 1-4's env default of 50 unless overridden).
- `rollout_fresh_with_budget`: the fair baseline. A fresh episode targeting
  `goal_xyz` from step 0, but capped at `max_steps` -- callers pass the
  *remaining* budget after a swap (`max_steps` of the swap rollout minus its
  `switch_step`) so both conditions get the identical number of steps to
  reach the same final goal, and the only difference is "was a different goal
  active before this one."

Goal-input mode: literal xyz by default (no embedding params), because stage
5 is testing the re-goaling mechanism itself, not the language/embedding
layer stages 2-4 already spent effort on. In this mode `env.unwrapped.goal`
(and the `desired_goal` slice of the `obs` dict handed to `model.predict`) is
overwritten directly with the target xyz -- the same "write straight into the
env's real goal state" mechanism `experiments/03_language_goal_projection/
train.py`'s `evaluate_language_goal` already validated for setting ground
truth after a reset, just applied mid-rollout instead. This works with any
SB3 `MultiInputPolicy` that doesn't do goal-embedding substitution, including
stage 1's plain SAC+HER checkpoint (default `CombinedExtractor`, no
`goal_encoder`) -- `model.actor` is never touched in this mode, so a
checkpoint without an `actor.features_extractor.goal_encoder` at all still
works (mirrors `episode_recording.record_episode`'s literal-mode guarantee).

Optional embedding mode (`initial_goal_embedding`/`new_goal_embedding`, both
required together): for a later experiment that wants to run this same
mid-episode-switch mechanism on top of a stage 2/3/4-style
`GoalEmbeddingExtractor` policy instead. Reuses `episode_recording.
_pin_desired_goal_embedding` -- the same monkeypatch stage 3's
`evaluate_language_goal` uses to substitute a fixed embedding for the
policy's desired-goal input -- rather than growing a third, independently
maintained copy of that mechanism. The env's ground-truth `goal` is *always*
xyz regardless of mode (FetchReach has no notion of an embedding-space goal
internally); the embedding params only change what the policy network sees,
never what decides success.

Success/failure in both functions comes from FetchReach's own
`info["is_success"]` (computed against whatever `env.unwrapped.goal` is at
the moment of that `step()` call) -- the same distance-based criterion every
other eval loop in this project already uses. `rollout_with_goal_switch`
resets its own success tracking at the switch, so a success recorded before
the switch (trivially true if `initial_goal_xyz` is the policy's starting
position) never counts toward the reported result -- only whether the *new*
goal was reached matters.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
import torch

from lang_goal_rl.episode_recording import _pin_desired_goal_embedding
from lang_goal_rl.goal_embedding_extractor import GoalEmbeddingExtractor

if TYPE_CHECKING:
    import gymnasium as gym
    from stable_baselines3 import SAC


@dataclass(frozen=True)
class GoalSwitchResult:
    """Outcome of one `rollout_with_goal_switch` episode.

    Attributes:
        success: Whether `info["is_success"]` was truthy on the *final*
            post-switch step. Any success during the pre-switch phase never
            reaches this field -- it's about the new goal only.
        n_steps: Total env steps run, pre- and post-switch combined. Equals
            `max_steps` unless the env truncated first (it won't, given
            `max_steps` is validated against the env's own registered
            episode length -- see `_ensure_within_env_step_limit`).
        switch_step: The `switch_step` argument, echoed back so a caller
            aggregating many results doesn't need to keep it separately.
    """

    success: bool
    n_steps: int
    switch_step: int


def _ensure_within_env_step_limit(env: gym.Env, max_steps: int) -> None:
    """Raise if `max_steps` exceeds the env's own registered `max_episode_steps`.

    A budget-matched comparison is meaningless if the requested budget is
    silently clipped by the env's `TimeLimit` wrapper before it's ever
    reached -- this catches that misconfiguration immediately rather than
    letting it produce a quietly-too-short episode. Skipped when the env
    doesn't expose a spec with this field (e.g. a bare test double), since
    there's no registered limit to check against.

    Args:
        env: The env `max_steps` will be applied to.
        max_steps: The requested step budget.

    Raises:
        ValueError: If the env has a registered `max_episode_steps` and
            `max_steps` exceeds it.
    """
    spec = getattr(env, "spec", None)
    registered_limit = getattr(spec, "max_episode_steps", None)
    if registered_limit is not None and max_steps > registered_limit:
        msg = (
            f"max_steps ({max_steps}) exceeds the env's registered "
            f"max_episode_steps ({registered_limit}) -- the env's own "
            "TimeLimit would truncate first, silently giving a shorter "
            "budget than requested."
        )
        raise ValueError(msg)


def _goal_input_context(
    model: SAC, embedding: torch.Tensor | None
) -> AbstractContextManager[None]:
    """Pin `model.actor.features_extractor`'s desired-goal output to `embedding`, or do nothing.

    `nullcontext()` when `embedding` is `None` so literal-xyz callers never
    touch `model.actor` at all -- required for compatibility with a plain
    stage-1 checkpoint that has no `features_extractor.goal_encoder`.

    Args:
        model: A trained SAC model. Only accessed (via `model.actor.
            features_extractor`) when `embedding` is not `None`.
        embedding: Fixed embedding to substitute, or `None` for no-op.

    Returns:
        A context manager; see `episode_recording._pin_desired_goal_embedding`.
    """
    if embedding is None:
        return nullcontext()
    features_extractor = cast("GoalEmbeddingExtractor", model.actor.features_extractor)
    return _pin_desired_goal_embedding(features_extractor, embedding)


def rollout_with_goal_switch(
    model: SAC,
    env: gym.Env,
    *,
    initial_goal_xyz: npt.ArrayLike,
    switch_step: int,
    new_goal_xyz: npt.ArrayLike,
    max_steps: int,
    base_seed: int,
    initial_goal_embedding: torch.Tensor | None = None,
    new_goal_embedding: torch.Tensor | None = None,
) -> GoalSwitchResult:
    """Roll out one episode targeting `initial_goal_xyz`, then `new_goal_xyz` from `switch_step` on.

    No `env.reset()` happens between the two phases -- this is a genuine
    mid-episode switch, not two separate episodes. `max_steps` is the
    episode's *total* budget (both phases combined), matching the env's
    default episode length unless a smaller budget is deliberately being
    tested; pair with `rollout_fresh_with_budget(max_steps=max_steps -
    switch_step, ...)` for the budget-matched baseline comparison the stage-5
    proof gate requires.

    Args:
        model: A trained SAC model. In literal-xyz mode (the default,
            `initial_goal_embedding`/`new_goal_embedding` both `None`),
            `model.actor` is never accessed -- any stage's checkpoint works,
            including stage 1's plain `MultiInputPolicy` with no
            goal-embedding extractor. In embedding mode, must have
            `model.actor.features_extractor` be a `GoalEmbeddingExtractor`
            (stage 2/3/4-style).
        env: The FetchReach-v4 env instance to roll out on.
        initial_goal_xyz: The goal active for the first `switch_step` steps,
            shape `(3,)`. Written into `env.unwrapped.goal` right after
            reset, so it overrides whatever the env would otherwise have
            sampled.
        switch_step: Number of steps to run against `initial_goal_xyz`
            before switching. Must be `>= 1` (a switch at step 0 is just a
            fresh episode, not a mid-episode switch) and `< max_steps`
            (there must be at least one post-switch step to judge).
        new_goal_xyz: The goal active from `switch_step` onward, shape
            `(3,)`. This is what `success` is judged against.
        max_steps: Total episode length (pre- + post-switch steps combined).
            Validated against the env's registered `max_episode_steps` --
            see `_ensure_within_env_step_limit`.
        base_seed: Seed passed to `env.reset(seed=base_seed)`. Callers
            running many episodes should vary this per call (e.g.
            `base_seed + episode_index`), the same convention used
            throughout this project's other eval loops.
        initial_goal_embedding: If given (together with
            `new_goal_embedding`), the policy's desired-goal *input* is
            additionally pinned to this fixed embedding for the pre-switch
            phase, via `episode_recording._pin_desired_goal_embedding`. The
            env's ground-truth goal is still `initial_goal_xyz` regardless.
        new_goal_embedding: Paired with `initial_goal_embedding` for the
            post-switch phase. Both or neither must be given.

    Returns:
        A `GoalSwitchResult` with whether the new goal was reached by the
        end of the episode, total step count, and the `switch_step` used.

    Raises:
        ValueError: If exactly one of `initial_goal_embedding`/
            `new_goal_embedding` is given, if `switch_step < 1`, if
            `switch_step >= max_steps`, or if `max_steps` exceeds the env's
            registered `max_episode_steps`.
    """
    if (initial_goal_embedding is None) != (new_goal_embedding is None):
        msg = "initial_goal_embedding and new_goal_embedding must both be given or both omitted"
        raise ValueError(msg)
    if switch_step < 1:
        msg = f"switch_step must be >= 1 (got {switch_step}) -- a switch at step 0 isn't mid-episode"
        raise ValueError(msg)
    if switch_step >= max_steps:
        msg = f"switch_step ({switch_step}) must be < max_steps ({max_steps}) to leave a post-switch step"
        raise ValueError(msg)
    _ensure_within_env_step_limit(env, max_steps)

    initial_goal = np.asarray(initial_goal_xyz, dtype=np.float64)
    new_goal = np.asarray(new_goal_xyz, dtype=np.float64)

    obs, _info = env.reset(seed=base_seed)
    env.unwrapped.goal = initial_goal.copy()
    obs["desired_goal"] = initial_goal.copy()

    n_steps = 0
    terminated = truncated = False
    with _goal_input_context(model, initial_goal_embedding):
        while n_steps < switch_step and not (terminated or truncated):
            action, _state = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(action)
            n_steps += 1

    env.unwrapped.goal = new_goal.copy()
    obs["desired_goal"] = new_goal.copy()
    is_success = False

    with _goal_input_context(model, new_goal_embedding):
        while not (terminated or truncated) and n_steps < max_steps:
            action, _state = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            is_success = bool(info.get("is_success", is_success))
            n_steps += 1

    return GoalSwitchResult(success=is_success, n_steps=n_steps, switch_step=switch_step)


def rollout_fresh_with_budget(
    model: SAC,
    env: gym.Env,
    *,
    goal_xyz: npt.ArrayLike,
    max_steps: int,
    base_seed: int,
) -> bool:
    """Roll out one fresh episode toward `goal_xyz`, capped at `max_steps`.

    The budget-matched baseline for `rollout_with_goal_switch`: no earlier
    goal is ever active, and the step budget is deliberately capped below
    the env's full episode length (rather than relying on the env's own
    `TimeLimit`) so it can be set equal to a swap rollout's *remaining*
    steps after `switch_step` -- holding total available time constant so
    the only difference between the two conditions is "was there an
    earlier, different goal."

    Args:
        model: A trained SAC model, literal-xyz-compatible (see
            `rollout_with_goal_switch`'s `model` argument for the same
            constraint -- this function never touches `model.actor`).
        env: The FetchReach-v4 env instance to roll out on.
        goal_xyz: The goal for the whole episode, shape `(3,)`. Written into
            `env.unwrapped.goal` right after reset.
        max_steps: Step budget for this episode -- typically a swap
            rollout's `max_steps - switch_step`, not the env's default
            episode length. Validated against the env's registered
            `max_episode_steps`.
        base_seed: Seed passed to `env.reset(seed=base_seed)`.

    Returns:
        Whether `info["is_success"]` was truthy on the final step.

    Raises:
        ValueError: If `max_steps` exceeds the env's registered
            `max_episode_steps`.
    """
    _ensure_within_env_step_limit(env, max_steps)
    goal = np.asarray(goal_xyz, dtype=np.float64)

    obs, _info = env.reset(seed=base_seed)
    env.unwrapped.goal = goal.copy()
    obs["desired_goal"] = goal.copy()

    terminated = truncated = False
    is_success = False
    n_steps = 0
    while not (terminated or truncated) and n_steps < max_steps:
        action, _state = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        is_success = bool(info.get("is_success", is_success))
        n_steps += 1

    return is_success
