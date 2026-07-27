"""Record a trained SB3 policy's rollout on a goal-conditioned env as a GIF.

This is the project's visual-proof utility: numbers in a report don't show
whether a trained policy actually reaches for the right place. `record_episode`
runs one real episode through `env.render()` (never a mocked renderer) and
encodes the frames with `imageio`. `record_episode_with_goal_switch` is the
same idea for an episode whose goal changes partway through, without a
reset — the visual-proof counterpart to `midepisode_regoal.
rollout_with_goal_switch`.

Two goal-embedding modes, matching what this project's stages actually need:

- Literal-goal mode (default, `goal_embedding_override=None`): the policy
  sees whatever `GoalEmbeddingExtractor.forward`
  (`goal_embedding_extractor.py`) normally computes from the env's real
  `desired_goal` — no patching, this module doesn't touch the model at all.
- Override mode (`goal_embedding_override` given): the policy's desired-goal
  input is pinned to a fixed embedding (e.g. a language projection) for
  every step of the episode. This reuses the exact monkeypatch mechanism
  stage 3 already validated in
  `experiments/03_language_goal_projection/train.py`'s
  `evaluate_language_goal` and `debug_language_eval.py`'s
  `run_with_override`: `model.actor.features_extractor.forward` is swapped
  for a version that substitutes the override wherever it would have
  computed `goal_encoder(desired_goal)`, and restored afterward.
  `achieved_goal` always goes through the real frozen `GoalEncoder`, in
  both modes, every step. `record_episode_with_goal_switch` extends this to
  two overrides, one per phase (see its own docstring).

Ground truth for success/failure is always whatever `info["is_success"]`
reports against the env's real state — this module never touches the env's
goal for `record_episode`. `record_episode_with_goal_switch` is the one
exception: it deliberately writes `initial_goal_xyz`/`new_goal_xyz` into
`env.unwrapped.goal` itself, because the whole point of a goal *switch* is
changing what the env's ground truth is — the embedding-override params
never affect that ground truth, only what the policy's features extractor
sees.
"""

from __future__ import annotations

import types
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import imageio.v2 as imageio
import numpy as np
import numpy.typing as npt
import torch

from lang_goal_rl.goal_embedding_extractor import GoalEmbeddingExtractor

if TYPE_CHECKING:
    from collections.abc import Iterator

    import gymnasium as gym
    from stable_baselines3 import SAC


@dataclass(frozen=True)
class EpisodeRecording:
    """Where a recorded episode's GIF landed, and whether it was a real success.

    Attributes:
        path: Where the GIF was written (`out_path`, unchanged).
        success: `True` iff the episode's final `info["is_success"]` was
            truthy. This is the real outcome — callers should use it to
            label the GIF honestly rather than assuming success.
        n_steps: Number of env steps the episode ran for. The GIF has
            `n_steps + 1` frames (an initial post-reset frame, plus one per
            step).
        total_travel: Sum of per-step Euclidean displacement of
            `obs["achieved_goal"]` (the gripper's xyz position) across the
            whole episode, in the env's native units. This is a measure of
            how much the gripper actually moved on screen, not just how far
            apart its start and end points are — a trajectory that moves out
            and back to near its starting point has a small start-to-end
            distance but a large `total_travel`. Demo-selection scripts use
            this to pick a visually compelling successful episode out of
            several real successes, rather than the first one found (see
            each stage's `make_demo.py` docstring).
    """

    path: Path
    success: bool
    n_steps: int
    total_travel: float


@contextmanager
def _pin_desired_goal_embedding(
    extractor: GoalEmbeddingExtractor,
    embedding: torch.Tensor,
) -> Iterator[None]:
    """Temporarily monkeypatch `extractor.forward` to return a fixed desired-goal embedding.

    Same substitution this project's stage 3 uses to feed a policy a
    language-projected embedding instead of `goal_encoder(desired_goal)`
    (`experiments/03_language_goal_projection/train.py`'s
    `evaluate_language_goal`) — pulled out here so `record_episode` and any
    future caller share one implementation instead of each re-deriving the
    monkeypatch. `achieved_goal` is untouched: it still runs through the
    real frozen `goal_encoder` every call.

    Args:
        extractor: The `GoalEmbeddingExtractor` instance to patch — normally
            `model.actor.features_extractor`.
        embedding: Fixed embedding to substitute, shape `(embed_dim,)`.
            Broadcast over the batch dimension on every call.

    Yields:
        Nothing; the patch is active for the duration of the `with` block
        and unconditionally restored on exit, including on exception.
    """
    original_forward = extractor.forward
    fixed = embedding.detach().to(torch.float32)

    def patched_forward(
        self: GoalEmbeddingExtractor, observations: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        flat_observation = self._flatten(observations["observation"])
        achieved_embedding = self.goal_encoder(observations["achieved_goal"])
        batch_size = achieved_embedding.shape[0]
        desired_embedding = fixed.expand(batch_size, -1)
        return torch.cat(
            [flat_observation, achieved_embedding, desired_embedding], dim=1
        )

    extractor.forward = types.MethodType(patched_forward, extractor)
    try:
        yield
    finally:
        extractor.forward = original_forward


def _goal_embedding_override_context(
    model: SAC, embedding: torch.Tensor | None
) -> AbstractContextManager[None]:
    """Pin `model.actor.features_extractor`'s desired-goal output to `embedding`, or do nothing.

    `nullcontext()` when `embedding` is `None` so literal-goal callers never
    touch `model.actor` at all -- required so a model without a
    `GoalEmbeddingExtractor` (e.g. a plain stage-1 checkpoint) still works.
    Shared by `record_episode` and `record_episode_with_goal_switch` so
    there's exactly one place that resolves the concrete
    `GoalEmbeddingExtractor` type (see the comment this replaced, inlined
    below where it's actually used).

    Args:
        model: A trained SAC model. Only accessed (via
            `model.actor.features_extractor`) when `embedding` is not
            `None`.
        embedding: Fixed embedding to substitute, or `None` for a no-op.

    Returns:
        A context manager; see `_pin_desired_goal_embedding`.
    """
    if embedding is None:
        return nullcontext()
    # SB3 types SAC.actor.features_extractor generically as
    # BaseFeaturesExtractor since any features-extractor class can be
    # plugged in via policy_kwargs. This project's build_model
    # (experiments/03_language_goal_projection/train.py) always constructs
    # the actor with features_extractor_class=GoalEmbeddingExtractor, so the
    # concrete runtime type is known even though the stub can't express it.
    # Only resolved here, inside the override branch, so literal-goal mode
    # never touches `model.actor` at all.
    features_extractor = cast("GoalEmbeddingExtractor", model.actor.features_extractor)
    return _pin_desired_goal_embedding(features_extractor, embedding)


def _sum_step_displacements(positions: list[np.ndarray]) -> float:
    """Sum the Euclidean distance between each consecutive pair of positions.

    Used to turn a per-step trace of `achieved_goal` into `total_travel` —
    deliberately a sum of per-step displacements rather than a start-to-end
    distance, since the latter can hide a trajectory that moves a lot but
    ends up near where it started.

    Args:
        positions: Per-step xyz positions in visit order, one per rendered
            frame (including the initial post-reset position).

    Returns:
        Total path length, or `0.0` if fewer than 2 positions were recorded.
    """
    if len(positions) < 2:
        return 0.0
    displacements = np.diff(np.stack(positions), axis=0)
    return float(np.linalg.norm(displacements, axis=1).sum())


def record_episode(
    env: gym.Env,
    model: SAC,
    *,
    out_path: Path,
    goal_embedding_override: torch.Tensor | None = None,
    max_steps: int | None = None,
    fps: int = 10,
) -> EpisodeRecording:
    """Roll out one episode with `model`'s policy on `env` and encode it as a GIF.

    Args:
        env: A `render_mode="rgb_array"` goal-conditioned env instance
            (this project only uses FetchReach-v4). Must already be built
            with that render mode — `env.render()` is called after every
            reset and step, and anything other than a real rgb array is
            treated as a misconfigured env, not a silently blank recording.
        model: A trained SAC model (this project's checkpoints are all
            SAC+HER, see `build_model` in
            `experiments/03_language_goal_projection/train.py`). Its policy
            is rolled out deterministically
            (`model.predict(obs, deterministic=True)`), matching every
            other eval protocol in this project. Only `model.actor` is
            touched, and only when `goal_embedding_override` is given.
        out_path: Where to write the GIF. Parent directories are created if
            missing.
        goal_embedding_override: If given, the policy's desired-goal input
            is pinned to this fixed embedding (shape `(embed_dim,)`) for
            every step, via `_pin_desired_goal_embedding` patching
            `model.actor.features_extractor`. If `None` (default), the
            policy sees whatever `GoalEmbeddingExtractor.forward` normally
            computes from the env's real `desired_goal` — this module
            doesn't touch the model at all in that case.
        max_steps: Safety cap on episode length. If the episode hasn't
            terminated or truncated by this many steps, recording stops
            anyway. `None` (default) relies solely on the env's own
            termination/truncation.
        fps: Frames per second to encode the GIF at.

    Returns:
        An `EpisodeRecording` with the GIF's path, whether the episode
        actually succeeded (from `info["is_success"]`), and how many steps
        it ran for.

    Raises:
        TypeError: If `env.render()` doesn't return an rgb array at any
            point — the env wasn't constructed with a render mode that
            produces frames.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[npt.ArrayLike] = []
    achieved_goal_positions: list[np.ndarray] = []
    with _goal_embedding_override_context(model, goal_embedding_override):
        obs, _info = env.reset()
        frames.append(_render_or_raise(env))
        achieved_goal_positions.append(
            np.asarray(obs["achieved_goal"], dtype=np.float64)
        )

        terminated = truncated = False
        is_success = False
        n_steps = 0
        while not (terminated or truncated) and (
            max_steps is None or n_steps < max_steps
        ):
            action, _state = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            frames.append(_render_or_raise(env))
            achieved_goal_positions.append(
                np.asarray(obs["achieved_goal"], dtype=np.float64)
            )
            is_success = bool(info.get("is_success", is_success))
            n_steps += 1

    # imageio's GIF (pillow) writer deprecated the `fps` kwarg in favor of
    # per-frame `duration` in milliseconds -- convert here so callers keep
    # the more intuitive `fps` unit without triggering a deprecation warning.
    imageio.mimsave(out_path, frames, duration=1000 / fps)
    return EpisodeRecording(
        path=out_path,
        success=is_success,
        n_steps=n_steps,
        total_travel=_sum_step_displacements(achieved_goal_positions),
    )


def record_episode_with_goal_switch(
    env: gym.Env,
    model: SAC,
    *,
    out_path: Path,
    initial_goal_xyz: npt.ArrayLike,
    switch_step: int,
    new_goal_xyz: npt.ArrayLike,
    initial_goal_embedding: torch.Tensor | None = None,
    new_goal_embedding: torch.Tensor | None = None,
    max_steps: int | None = None,
    fps: int = 10,
) -> EpisodeRecording:
    """Roll out one episode targeting `initial_goal_xyz`, then `new_goal_xyz` from `switch_step` on, as a GIF.

    Stage 5/6's visual-proof counterpart to `midepisode_regoal.
    rollout_with_goal_switch`: same mid-episode goal swap mechanism (no
    `env.reset()` between phases), but every step is rendered and encoded
    into `out_path`, matching `record_episode`'s render loop exactly. Ground
    truth is always `initial_goal_xyz`/`new_goal_xyz`, written directly into
    `env.unwrapped.goal` and `obs["desired_goal"]` right after reset and
    again at the switch -- the same "write straight into the env's real
    goal state" mechanism `rollout_with_goal_switch` and
    `evaluate_language_goal`
    (`experiments/03_language_goal_projection/train.py`) both already use.
    `initial_goal_embedding`/`new_goal_embedding` never change this ground
    truth; they only optionally change what the *policy* sees (see below).

    Two goal-input modes, matching what stages 5 and 6 each need:

    - Literal-xyz mode (default, both embedding args `None`): the policy
      sees whatever `GoalEmbeddingExtractor.forward` normally computes from
      the env's real `desired_goal` in both phases -- this is stage 5's
      mode, testing the re-goaling mechanism itself, not the embedding
      layer. `model.actor` is never touched.
    - Embedding-override mode (both embedding args given): the policy's
      desired-goal input is additionally pinned to `initial_goal_embedding`
      for the pre-switch phase and `new_goal_embedding` for the post-switch
      phase, via `_goal_embedding_override_context` /
      `_pin_desired_goal_embedding` -- the same monkeypatch `record_episode`
      and `rollout_with_goal_switch` already use. This is stage 6's mode
      (e.g. two different live English instructions before and after the
      switch). The env's ground-truth goal is still `initial_goal_xyz`/
      `new_goal_xyz` regardless -- a caller demoing a language instruction
      passes its known region centroid (or other ground-truth xyz) as the
      `*_goal_xyz` args, exactly as `evaluate_language_goal` does, rather
      than this function inventing a different notion of ground truth for
      embedding mode.

    Args:
        env: A `render_mode="rgb_array"` goal-conditioned env instance
            (this project only uses FetchReach-v4). `env.render()` is
            called after every reset and step.
        model: A trained SAC model, rolled out deterministically
            (`model.predict(obs, deterministic=True)`). In literal-xyz mode
            `model.actor` is never accessed -- any stage's checkpoint
            works. In embedding mode, `model.actor.features_extractor` must
            be a `GoalEmbeddingExtractor`.
        out_path: Where to write the GIF. Parent directories are created if
            missing.
        initial_goal_xyz: The goal active for the first `switch_step`
            steps, shape `(3,)`. Written into `env.unwrapped.goal` right
            after reset.
        switch_step: Number of steps to run against `initial_goal_xyz`
            before switching. Must be `>= 1` (a switch at step 0 is just a
            fresh episode, not a mid-episode switch) and, if `max_steps` is
            given, `< max_steps` (there must be at least one post-switch
            step to judge and record).
        new_goal_xyz: The goal active from `switch_step` onward, shape
            `(3,)`. This is what `success` is judged against.
        initial_goal_embedding: If given (together with
            `new_goal_embedding`), the policy's desired-goal input is
            additionally pinned to this fixed embedding for the pre-switch
            phase.
        new_goal_embedding: Paired with `initial_goal_embedding` for the
            post-switch phase. Both or neither must be given.
        max_steps: Safety cap on total episode length (pre- and
            post-switch steps combined). If the episode hasn't terminated
            or truncated by this many steps, recording stops anyway.
            `None` (default) relies solely on the env's own
            termination/truncation.
        fps: Frames per second to encode the GIF at.

    Returns:
        An `EpisodeRecording` with the GIF's path, whether the *new* goal
        was actually reached by the end of the episode (any success during
        the pre-switch phase never counts), and how many steps it ran for.

    Raises:
        ValueError: If exactly one of `initial_goal_embedding`/
            `new_goal_embedding` is given, if `switch_step < 1`, or if
            `max_steps` is given and `switch_step >= max_steps`.
        TypeError: If `env.render()` doesn't return an rgb array at any
            point.
    """
    if (initial_goal_embedding is None) != (new_goal_embedding is None):
        msg = "initial_goal_embedding and new_goal_embedding must both be given or both omitted"
        raise ValueError(msg)
    if switch_step < 1:
        msg = f"switch_step must be >= 1 (got {switch_step}) -- a switch at step 0 isn't mid-episode"
        raise ValueError(msg)
    if max_steps is not None and switch_step >= max_steps:
        msg = f"switch_step ({switch_step}) must be < max_steps ({max_steps}) to leave a post-switch step"
        raise ValueError(msg)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    initial_goal = np.asarray(initial_goal_xyz, dtype=np.float64)
    new_goal = np.asarray(new_goal_xyz, dtype=np.float64)

    frames: list[npt.ArrayLike] = []
    achieved_goal_positions: list[np.ndarray] = []
    obs, _info = env.reset()
    env.unwrapped.goal = initial_goal.copy()
    obs["desired_goal"] = initial_goal.copy()
    frames.append(_render_or_raise(env))
    achieved_goal_positions.append(np.asarray(obs["achieved_goal"], dtype=np.float64))

    n_steps = 0
    terminated = truncated = False
    with _goal_embedding_override_context(model, initial_goal_embedding):
        while n_steps < switch_step and not (terminated or truncated):
            action, _state = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(action)
            frames.append(_render_or_raise(env))
            achieved_goal_positions.append(
                np.asarray(obs["achieved_goal"], dtype=np.float64)
            )
            n_steps += 1

    env.unwrapped.goal = new_goal.copy()
    obs["desired_goal"] = new_goal.copy()
    is_success = False

    with _goal_embedding_override_context(model, new_goal_embedding):
        while not (terminated or truncated) and (
            max_steps is None or n_steps < max_steps
        ):
            action, _state = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            frames.append(_render_or_raise(env))
            achieved_goal_positions.append(
                np.asarray(obs["achieved_goal"], dtype=np.float64)
            )
            is_success = bool(info.get("is_success", is_success))
            n_steps += 1

    imageio.mimsave(out_path, frames, duration=1000 / fps)
    return EpisodeRecording(
        path=out_path,
        success=is_success,
        n_steps=n_steps,
        total_travel=_sum_step_displacements(achieved_goal_positions),
    )


def _render_or_raise(env: gym.Env) -> np.ndarray:
    """Call `env.render()` and fail loudly instead of recording a blank frame.

    Args:
        env: The env to render — must be built with a render mode that
            returns an array (e.g. `render_mode="rgb_array"`).

    Returns:
        The rendered frame.

    Raises:
        TypeError: If `env.render()` doesn't return an rgb array (e.g. it
            returns `None`, a string, or a list — any render mode other
            than `"rgb_array"`).
    """
    frame = env.render()
    if not isinstance(frame, np.ndarray):
        msg = (
            f"env.render() returned {type(frame).__name__}, not an rgb array — "
            "construct the env with render_mode='rgb_array'"
        )
        raise TypeError(msg)
    return frame
