"""Interactive, visual stage-6 language-control demo."""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import cast

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
INTERACTIVE_MAX_STEPS = 1_000_000
"""Effectively disables the research environment's 50-step evaluation limit.

The interactive demo remains live until the user enters ``reset`` or ``quit``.
"""


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


def _print_match(instruction: str, match: GoalMatch) -> None:
    print(f'\nInstruction: "{instruction}"')
    print(
        f'Matched: "{match.reference_instruction}" '
        f"-> {match.region_name} (distance={match.distance:.3f})"
    )


def run(
    *,
    checkpoint: Path,
    encoder_path: Path,
    seed: int,
    fps: float,
) -> None:
    """Run the visual environment while accepting Terminal instructions."""
    gym.register_envs(gymnasium_robotics)
    env = gym.make(
        ENV_ID,
        render_mode="human",
        max_episode_steps=INTERACTIVE_MAX_STEPS,
    )
    model = SAC.load(checkpoint, env=env)
    controller = LiveGoalController(_load_encoder(encoder_path))

    print("Model ready. Type an English instruction and press Return.")
    print("Commands: reset, status, quit")
    first_instruction = input("> ").strip()
    while not first_instruction:
        first_instruction = input("> ").strip()
    if first_instruction.lower() == "quit":
        env.close()
        return

    current_instruction = first_instruction
    current_match = controller.match_instruction(current_instruction)
    _print_match(current_instruction, current_match)

    observation, _ = env.reset(seed=seed)
    _apply_goal(env, observation, current_match)
    env.render()

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
                    _print_match(current_instruction, current_match)
                elif lowered == "reset":
                    observation, _ = env.reset(seed=seed + episode)
                    episode += 1
                    _apply_goal(env, observation, current_match)
                    env.render()
                    print("\nEpisode reset.")
                elif command:
                    current_instruction = command
                    current_match = controller.match_instruction(command)
                    _apply_goal(env, observation, current_match)
                    _print_match(command, current_match)
                print("> ", end="", flush=True)

            started = time.monotonic()
            with _goal_context(model, current_match):
                action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(action)
            env.render()

            if terminated or truncated:
                observation, _ = env.reset(seed=seed + episode)
                episode += 1
                _apply_goal(env, observation, current_match)
                env.render()

            remaining = step_delay - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Control FetchReach live with English instructions."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--encoder-path", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument(
        "--fps", type=float, default=20.0, help="Maximum simulation steps per second."
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
