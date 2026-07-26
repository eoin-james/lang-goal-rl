"""Record a trained SB3 policy's rollout on a goal-conditioned env as a GIF.

This is the project's visual-proof utility: numbers in a report don't show
whether a trained policy actually reaches for the right place. `record_episode`
runs one real episode through `env.render()` (never a mocked renderer) and
encodes the frames with `imageio`.

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
  both modes, every step.

Ground truth for success/failure is always whatever `info["is_success"]`
reports against the env's real state — this module never touches the env's
goal, only what the policy's features extractor sees.
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
    """

    path: Path
    success: bool
    n_steps: int


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

    override_context: AbstractContextManager[None] = nullcontext()
    if goal_embedding_override is not None:
        # SB3 types SAC.actor.features_extractor generically as
        # BaseFeaturesExtractor since any features-extractor class can be
        # plugged in via policy_kwargs. This project's build_model
        # (experiments/03_language_goal_projection/train.py) always
        # constructs the actor with
        # features_extractor_class=GoalEmbeddingExtractor, so the concrete
        # runtime type is known even though the stub can't express it. Only
        # resolved here, inside the override branch, so literal-goal mode
        # never touches `model.actor` at all.
        features_extractor = cast(
            "GoalEmbeddingExtractor", model.actor.features_extractor
        )
        override_context = _pin_desired_goal_embedding(
            features_extractor, goal_embedding_override
        )

    frames: list[npt.ArrayLike] = []
    with override_context:
        obs, _info = env.reset()
        frames.append(_render_or_raise(env))

        terminated = truncated = False
        is_success = False
        n_steps = 0
        while not (terminated or truncated) and (
            max_steps is None or n_steps < max_steps
        ):
            action, _state = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            frames.append(_render_or_raise(env))
            is_success = bool(info.get("is_success", is_success))
            n_steps += 1

    # imageio's GIF (pillow) writer deprecated the `fps` kwarg in favor of
    # per-frame `duration` in milliseconds -- convert here so callers keep
    # the more intuitive `fps` unit without triggering a deprecation warning.
    imageio.mimsave(out_path, frames, duration=1000 / fps)
    return EpisodeRecording(path=out_path, success=is_success, n_steps=n_steps)


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
