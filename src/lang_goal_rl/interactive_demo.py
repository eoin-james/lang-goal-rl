"""Interactive, visual stage-6 language-control demo.

A human types an English instruction and watches a trained FetchReach-v4 policy go for it in a
real-time window; typing a new instruction at any time -- including mid-episode, no CLI flags
needed -- redirects it live, via a background stdin-reading thread that feeds a queue the main
loop drains between simulation steps.

This merges two demos that briefly existed side by side after stage 6 landed:

- The original version of this file: had the live keyboard-redirect mechanism above, but
  disabled the real 50-step FetchReach-v4 episode limit (so it auto-reset forever) and never
  reported whether an episode actually succeeded -- a person watching it couldn't tell if the
  robot got there.
- `experiments/06_live_english_interface/live_demo.py` (a since-deleted draft): had honest
  success/failure reporting against the real episode limit, a match-quality diagnostic (is a
  typed sentence a confident match against the 84-sentence reference set, or an extrapolation?),
  and a headless-machine fallback to a live matplotlib window -- but only supported a fixed
  instruction (or one pre-declared switch) via CLI flags, not live typing.

This version keeps the live keyboard redirect and adds the other three:

- No `max_episode_steps` override: `gym.make` uses FetchReach-v4's own registered limit
  (confirmed below via `env.spec.max_episode_steps`, currently 50), so an episode really ends
  when the research environment says it does. Each ending prints whether `info["is_success"]`
  was ever true during that episode, before auto-resetting for the next one.
- Match-quality diagnostic: reuses `LiveGoalController.match_instruction`'s `GoalMatch.distance`
  (already computed for every lookup, no extra encoding) against an empirical "how far apart do
  two known reference sentences typically sit" baseline, so the terminal can flag a typed
  instruction as a confident match or a genuine extrapolation -- not just "matched something."
- Headless fallback: probes whether `render_mode="human"` can actually open a window before
  committing to it (window creation happens lazily inside `env.render()`'s first call, not at
  `gym.make()` time), falling back to a live-updating matplotlib window otherwise. Either way the
  user sees something live; the fallback is reported on stdout, never silently swapped in.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from stable_baselines3 import SAC

from lang_goal_rl.episode_recording import _pin_desired_goal_embedding
from lang_goal_rl.goal_embedding_extractor import GoalEmbeddingExtractor
from lang_goal_rl.goal_encoder import GoalEncoder
from lang_goal_rl.goal_region_vocabulary import (
    MEASURED_GOAL_BOX,
    region_names,
    sample_region_goals,
)
from lang_goal_rl.language_goal_projection import DEFAULT_N_TARGET_SAMPLES
from lang_goal_rl.live_goal_controller import GoalMatch, LiveGoalController

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "experiments"
    / "03_language_goal_projection"
    / "checkpoints"
    / "seed_0.zip"
)
DEFAULT_ENCODER = (
    REPO_ROOT
    / "experiments"
    / "02_contrastive_goal_embedding"
    / "artifacts"
    / "goal_encoder.pt"
)
ENV_ID = "FetchReach-v4"
CENTROID_SEED = 0
DEFAULT_FPS = 10.0
"""Matches every `record_episode*`/live-eval script's `fps` default elsewhere in this project --
paced slowly enough for a human to actually watch, not flash by."""


def _load_encoder(path: Path) -> GoalEncoder:
    encoder = GoalEncoder(goal_dim=3)
    encoder.load_state_dict(torch.load(path, map_location="cpu"))
    encoder.eval()
    return encoder


def _region_centroid(region_name: str) -> np.ndarray:
    region_seed = CENTROID_SEED + region_names().index(region_name)
    samples = sample_region_goals(
        region_name,
        DEFAULT_N_TARGET_SAMPLES,
        seed=region_seed,
        box=MEASURED_GOAL_BOX,
    )
    return samples.mean(axis=0)


def _leave_one_out_baseline_distance(reference_embeddings: np.ndarray) -> float:
    """Empirical "typical distance between two known sentences" baseline, no hardcoded cutoff.

    For each reference sentence, finds its own nearest *other* reference sentence's distance
    (leave-one-out 1-NN), then returns `mean + 2 * std` of that distribution. A typed
    instruction's own nearest-reference distance (`GoalMatch.distance`) is compared against this
    to decide whether to flag it as an unusually poor match -- built from the reference set's own
    geometry, the same data-derived (never invented-absolute-number) principle
    `semantic_neighbor_diagnostic.py` and the quality-gate's persona check already use.

    Args:
        reference_embeddings: The reference sentences' raw sentence-transformer embeddings,
            shape `(n_references, embed_dim)`.

    Returns:
        The `mean + 2 * std` leave-one-out nearest-neighbor distance across all reference rows.

    """
    pairwise = np.linalg.norm(
        reference_embeddings[:, None, :] - reference_embeddings[None, :, :], axis=2,
    )
    np.fill_diagonal(pairwise, np.inf)
    nearest_other_distances = pairwise.min(axis=1)
    return float(nearest_other_distances.mean() + 2 * nearest_other_distances.std())


def _describe_match_quality(match: GoalMatch, baseline_distance: float) -> str:
    """One-line, honest verdict: is `match` a confident lookup or a genuine extrapolation?

    Args:
        match: The typed instruction's `GoalMatch` from `LiveGoalController.match_instruction`.
        baseline_distance: `_leave_one_out_baseline_distance`'s empirical "typical" distance
            between two known reference sentences.

    Returns:
        A printable one-line description.

    """
    if match.distance <= baseline_distance:
        return (
            f"confident match (distance {match.distance:.3f} vs. typical "
            f"{baseline_distance:.3f} between two known sentences)"
        )
    return (
        f"extrapolation -- unusually far from anything in the reference vocabulary "
        f"(distance {match.distance:.3f} vs. typical {baseline_distance:.3f}); the nearest-"
        "neighbor lookup still returns its best guess, but treat this region assignment as "
        "a guess, not a confident match"
    )


class _LiveRenderer:
    """Renders one env frame per call, either to a native window or a live matplotlib fallback.

    Two backends, chosen once by `_build_renderer` and used identically thereafter, so the
    episode loop never branches on which backend is active:

    - Native (`render_mode="human"`): `env.render()` updates FetchReach-v4's own MuJoCo/GLFW
      window directly; this class's `render` just forwards the call.
    - Matplotlib fallback (`render_mode="rgb_array"`): `env.render()` returns an rgb array, which
      this class draws into a `plt.ion()` window via `imshow`, replacing the previous frame each
      call rather than accumulating a new figure per step.
    """

    def __init__(self, *, native: bool) -> None:
        """Store which backend is active; matplotlib's figure/axes are created lazily on first use.

        Args:
            native: `True` to forward `env.render()` untouched (human mode already draws to its
                own window); `False` to draw returned rgb arrays into a matplotlib window.

        """
        self._native = native
        self._figure = None
        self._image_artist = None

    def render(self, env: gym.Env) -> None:
        """Render one frame of `env`'s current state.

        Args:
            env: The env to render. In native mode this is expected to be built with
                `render_mode="human"`; in fallback mode, `render_mode="rgb_array"`.

        """
        frame = env.render()
        if self._native:
            return
        rgb_frame = cast("np.ndarray", frame)
        if self._figure is None:
            import matplotlib.pyplot as plt

            plt.ion()
            self._figure, axes = plt.subplots()
            axes.axis("off")
            self._image_artist = axes.imshow(rgb_frame)
        else:
            assert self._image_artist is not None, (
                "_image_artist is always set alongside _figure -- see the if-branch above"
            )
            self._image_artist.set_data(rgb_frame)
        self._figure.canvas.draw()
        self._figure.canvas.flush_events()

    def close(self) -> None:
        """Close the matplotlib window if one was opened; a no-op in native mode."""
        if self._figure is not None:
            import matplotlib.pyplot as plt

            plt.close(self._figure)


def _build_renderer(env_id: str) -> tuple[gym.Env, _LiveRenderer]:
    """Build the env + renderer pair, preferring a native live window, falling back honestly.

    Does one throwaway reset+render+close on a disposable `render_mode="human"` env to smoke-test
    whether this machine can actually open a GLFW window -- window creation happens lazily inside
    `env.render()`'s first call, not at `gym.make()` time, so `gym.make(render_mode="human")`
    succeeding proves nothing on its own. If the smoke test raises (no display attached, e.g. a
    headless CI box or an unforwarded SSH session), that exception is reported on stdout and the
    real env is built with `render_mode="rgb_array"` plus the matplotlib fallback instead -- the
    user still gets a live-updating window, just via a different route.

    Args:
        env_id: The Gymnasium env id to construct (this project only ever uses `"FetchReach-v4"`).

    Returns:
        A `(env, renderer)` pair, already reset-free (the caller still owns `env.reset()`), and
        the renderer already knows which backend to use.

    """
    try:
        smoke_env = gym.make(env_id, render_mode="human")
        smoke_env.reset()
        smoke_env.render()
        smoke_env.close()
    except Exception as error:  # noqa: BLE001 -- any failure here means "no usable native window", not a bug to fix
        print(
            f"[interactive_demo] render_mode='human' isn't usable on this machine "
            f"({type(error).__name__}: {error}) -- falling back to a live matplotlib window instead.",
        )
        return gym.make(env_id, render_mode="rgb_array"), _LiveRenderer(native=False)

    print("[interactive_demo] opened a native FetchReach-v4 window (render_mode='human').")
    return gym.make(env_id, render_mode="human"), _LiveRenderer(native=True)


def _goal_context(
    model: SAC, match: GoalMatch
) -> AbstractContextManager[None]:
    extractor = cast("GoalEmbeddingExtractor", model.actor.features_extractor)
    return _pin_desired_goal_embedding(extractor, match.embedding)


def _read_commands(commands: queue.SimpleQueue[str]) -> None:
    for line in sys.stdin:
        commands.put(line.strip())


def _apply_goal(env: gym.Env, observation: dict, match: GoalMatch) -> None:
    goal = _region_centroid(match.region_name)
    env.unwrapped.goal = goal.copy()
    observation["desired_goal"] = goal.copy()


def _print_match(instruction: str, match: GoalMatch, baseline_distance: float) -> None:
    print(f'\nInstruction: "{instruction}"')
    print(
        f'Matched: "{match.reference_instruction}" '
        f"-> {match.region_name} (distance={match.distance:.3f})"
    )
    print(f"Match quality: {_describe_match_quality(match, baseline_distance)}")


def _print_episode_result(
    *, instruction: str, success: bool, n_steps: int, max_steps: int
) -> None:
    verdict = "reached the target" if success else "did NOT reach the target"
    print(
        f'\nEpisode result: the robot {verdict} for "{instruction}" '
        f"({n_steps}/{max_steps} steps).",
    )


def run(
    *,
    checkpoint: Path,
    encoder_path: Path,
    seed: int,
    fps: float,
) -> None:
    """Run the visual environment while accepting live-typed English instructions."""
    gym.register_envs(gymnasium_robotics)
    env, renderer = _build_renderer(ENV_ID)
    assert env.spec is not None, f"{ENV_ID} must be a registered env with a spec"
    max_steps = env.spec.max_episode_steps
    assert max_steps is not None, f"{ENV_ID}'s spec must declare max_episode_steps"
    model = SAC.load(checkpoint, env=env)
    controller = LiveGoalController(_load_encoder(encoder_path))
    baseline_distance = _leave_one_out_baseline_distance(controller.reference_embeddings)

    print("Model ready. Type an English instruction and press Return.")
    print("Commands: reset, status, quit")
    first_instruction = input("> ").strip()
    while not first_instruction:
        first_instruction = input("> ").strip()
    if first_instruction.lower() == "quit":
        renderer.close()
        env.close()
        return

    current_instruction = first_instruction
    current_match = controller.match_instruction(current_instruction)
    _print_match(current_instruction, current_match, baseline_distance)

    observation, _ = env.reset(seed=seed)
    _apply_goal(env, observation, current_match)
    renderer.render(env)
    step_in_episode = 0
    episode_success = False

    commands: queue.SimpleQueue[str] = queue.SimpleQueue()
    reader = threading.Thread(target=_read_commands, args=(commands,), daemon=True)
    reader.start()
    print("\nType another instruction at any time to redirect.\n> ", end="", flush=True)

    episode = 1
    step_delay = 1.0 / fps
    try:
        while True:
            while not commands.empty():
                command = commands.get()
                lowered = command.lower()
                if lowered == "quit":
                    return
                if lowered == "status":
                    _print_match(current_instruction, current_match, baseline_distance)
                elif lowered == "reset":
                    observation, _ = env.reset(seed=seed + episode)
                    episode += 1
                    step_in_episode = 0
                    episode_success = False
                    _apply_goal(env, observation, current_match)
                    renderer.render(env)
                    print("\nEpisode reset.")
                elif command:
                    current_instruction = command
                    current_match = controller.match_instruction(command)
                    _apply_goal(env, observation, current_match)
                    _print_match(command, current_match, baseline_distance)
                print("> ", end="", flush=True)

            started = time.monotonic()
            with _goal_context(model, current_match):
                action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            renderer.render(env)
            step_in_episode += 1
            episode_success = bool(info.get("is_success", episode_success))

            if terminated or truncated:
                _print_episode_result(
                    instruction=current_instruction,
                    success=episode_success,
                    n_steps=step_in_episode,
                    max_steps=max_steps,
                )
                observation, _ = env.reset(seed=seed + episode)
                episode += 1
                step_in_episode = 0
                episode_success = False
                _apply_goal(env, observation, current_match)
                renderer.render(env)
                print("> ", end="", flush=True)

            remaining = step_delay - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        renderer.close()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Control FetchReach live with English instructions."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--encoder-path", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument(
        "--fps", type=float, default=DEFAULT_FPS, help="Maximum simulation steps per second."
    )
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be greater than zero")
    run(
        checkpoint=args.checkpoint,
        encoder_path=args.encoder_path,
        seed=args.seed,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
