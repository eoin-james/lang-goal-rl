"""Stage 4, attempt 3: measure each region's policy tolerance to embedding-space noise directly, no projection or sentence involved.

Attempt 2's reviewer diagnosed a bottleneck shift: among held-out
instructions correctly classified to their true region, "lower your arm
toward the floor" landed *closer* to its region's true centroid (distance
0.0151) than "shift your gripper toward the right edge" (0.0200) yet scored
0.000 RL success while the latter scored 1.000 on all 3 seeds -- meaning the
trained SAC policy's basin of attraction around each region's target
goal-embedding is nonuniform, and classification accuracy alone can't
predict RL success once classification is already decent. The reviewer's
prescribed Part A diagnostic (see `report.md`'s attempt-2 Reviewer verdict):
for each region, inject noise at several L2 magnitudes directly into the
exact centroid embedding used in training, and re-run the existing 3 SAC
checkpoints against each perturbed centroid via the same
`evaluate_language_goal` infrastructure stage 3/4 already use -- no
projection, no sentence-transformer, no new training. This isolates the
policy's own tolerance radius per region from everything upstream of it
(projection precision, sentence-embedding quality).

Each region's exact target embedding is `goal_region_vocabulary.
compute_region_target_embeddings` evaluated at `(n_samples=DEFAULT_N_TARGET_
SAMPLES, seed=CENTROID_TARGET_SEED)` over `region_names()` -- bit-identical
to the regression target `language_goal_projection.precompute_instruction_
targets` computed for every stage-3/4 projection checkpoint (same function,
same sample population, same region order), so this stays apples-to-apples
with everything else in this stage rather than inventing a separate notion
of "the region's target".

Ground truth (what decides success/failure) is `train.
compute_region_centroid(region_name)` -- the same fixed xyz centroid
`evaluate_language_goal` has judged every stage-3/4 language-goal eval
against since stage 3's attempt-4 fix (see `train.py`'s module docstring
for why this specific xyz point, not a resampled one, is the correct ground
truth). This script only ever changes what embedding the *policy* is shown
(via `evaluate_language_goal`'s `projected_embedding` monkeypatch
substitution) -- the ground truth judging success/failure is untouched.

This script evaluates one seed's checkpoint against every (region,
magnitude) combination and prints one `tolerance_success_rate=` line per
combo; `generate_report_attempt3.py` aggregates all 3 seeds' logs into the
region x magnitude table, tolerance-radius summary, and chart.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from stable_baselines3 import SAC

EXPERIMENT_DIR = Path(__file__).parent
STAGE3_DIR = EXPERIMENT_DIR.parent / "03_language_goal_projection"
sys.path.insert(0, str(STAGE3_DIR))

from train import (  # noqa: E402 -- STAGE3_DIR must be on sys.path first, see module docstring
    CENTROID_TARGET_SEED,
    DEFAULT_ENCODER_PATH,
    ENV_ID,
    evaluate_language_goal,
    load_frozen_encoder,
)

from lang_goal_rl.goal_region_vocabulary import compute_region_target_embeddings, region_names
from lang_goal_rl.language_goal_projection import DEFAULT_N_TARGET_SAMPLES

DEFAULT_CHECKPOINT_DIR = STAGE3_DIR / "checkpoints"

NOISE_MAGNITUDES: tuple[float, ...] = (0.0, 0.005, 0.010, 0.015, 0.020, 0.030, 0.050)
"""L2 magnitudes of the perturbation injected into each region's exact
target embedding. 0.0 is a sanity-check control -- zero perturbation should
reproduce `evaluate_language_goal`'s ~1.000 success (identical to a literal
goal-embedding baseline), confirming this script's eval plumbing itself
introduces no defect before trusting the nonzero-magnitude results."""

TOLERANCE_EVAL_BASE_SEED = 20_000
"""Base env-reset seed for this diagnostic, offset from every other stage
3/4 base seed already in use (`train.LITERAL_EVAL_BASE_SEED`=1000,
`train.LANGUAGE_EVAL_BASE_SEED`=5000, `eval_held_out.HELD_OUT_EVAL_BASE_
SEED`=9000) so no two scripts ever reuse the same env-reset seed range for a
different purpose. Combo `(region_index, magnitude_index)` gets
`TOLERANCE_EVAL_BASE_SEED + combo_index * n_episodes`, its own
self-contained, reproducible episode-seed block -- reused identically across
every seed's independent run (same convention `train.main` and
`eval_held_out.main` already use: the reset-seed range depends on the
instruction/region/magnitude being evaluated, not on which SAC seed is being
evaluated), so results are directly comparable seed-to-seed."""


def perturbation_vector(embed_dim: int, region_index: int, magnitude_index: int, magnitude: float) -> np.ndarray:
    """Draw an L2-norm-exactly-`magnitude` perturbation in a fixed random direction.

    The direction is an isotropic-Gaussian draw normalized to unit L2 norm,
    then scaled to `magnitude` -- so the returned vector's norm is exactly
    `magnitude` regardless of `embed_dim`, measuring *magnitude* of
    deviation specifically (what this diagnostic asks for), not raw
    per-component Gaussian variance (which would conflate the two and make
    the resulting norm dimension-dependent and not equal to `magnitude`).
    Seeded deterministically from `(region_index, magnitude_index)` so the
    exact same direction is redrawn on any re-run of this script.

    Args:
        embed_dim: Dimensionality of the goal-embedding space (16 for
            `GoalEncoder`'s default).
        region_index: This region's position in `region_names()` order.
        magnitude_index: This magnitude's position in `NOISE_MAGNITUDES`.
        magnitude: The target L2 norm of the returned vector.

    Returns:
        Array of shape (embed_dim,) with L2 norm equal to `magnitude` (the
        zero vector when `magnitude == 0.0`, since any direction scaled by
        zero is the zero vector).

    """
    rng = np.random.default_rng(region_index * len(NOISE_MAGNITUDES) + magnitude_index)
    direction = rng.normal(size=embed_dim)
    unit_direction = direction / np.linalg.norm(direction)
    return unit_direction * magnitude


def main() -> None:
    """Evaluate one seed's already-trained SAC checkpoint against every (region, noise magnitude) combo."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--encoder-path", type=Path, default=DEFAULT_ENCODER_PATH)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint or (DEFAULT_CHECKPOINT_DIR / f"seed_{args.seed}.zip")

    gym.register_envs(gymnasium_robotics)
    env = gym.make(ENV_ID)

    model = SAC.load(checkpoint_path, env=env)
    print(f"loaded checkpoint from {checkpoint_path} (no new training, seed={args.seed})")

    encoder = load_frozen_encoder(args.encoder_path)
    names = region_names()
    targets = compute_region_target_embeddings(
        encoder, names, n_samples=DEFAULT_N_TARGET_SAMPLES, seed=CENTROID_TARGET_SEED,
    )
    embed_dim = targets.shape[1]
    n_magnitudes = len(NOISE_MAGNITUDES)

    for region_index, region_name in enumerate(names):
        target = targets[region_index]
        for magnitude_index, magnitude in enumerate(NOISE_MAGNITUDES):
            perturbation = perturbation_vector(embed_dim, region_index, magnitude_index, magnitude)
            perturbed_embedding = target + torch.from_numpy(perturbation).float()

            combo_index = region_index * n_magnitudes + magnitude_index
            base_seed = TOLERANCE_EVAL_BASE_SEED + combo_index * args.n_episodes
            success_rate = evaluate_language_goal(
                model,
                env,
                region_name=region_name,
                projected_embedding=perturbed_embedding,
                n_episodes=args.n_episodes,
                base_seed=base_seed,
            )
            print(
                f"tolerance_success_rate={success_rate:.3f} seed={args.seed} "
                f'region="{region_name}" magnitude={magnitude:.3f} over {args.n_episodes} episodes',
            )

    env.close()


if __name__ == "__main__":
    main()
