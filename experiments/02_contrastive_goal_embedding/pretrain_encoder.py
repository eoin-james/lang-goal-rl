"""Stage 2 encoder pretraining: contrastively pretrain GoalEncoder on FetchReach's goal distribution.

Run once, before any RL seed. Produces a frozen, saved `GoalEncoder`
checkpoint that every RL training seed in this stage loads and reuses
unchanged — RL seed variance must not be confounded with
encoder-pretraining variance (the whole point of pretraining once instead
of per-seed).

Goal-pair sampling rationale: FetchReach's `desired_goal` on reset is drawn
from the env's actual target distribution (a small workspace region above
the table); `achieved_goal` on reset is just the gripper's roughly-fixed
starting position, so raw resets alone don't hand you a diverse *pair* of
distinct goals to contrast. Instead, a pool of true goal points is
collected once via many resets, and each contrastive step samples points
from that pool and builds an (anchor, positive) pair per point via two
independent small-noise augmentations — a standard InfoNCE-with-
augmentation setup. This directly targets the second half of stage 2's
proof gate ("distance-in-latent correlates with true task distance"):
pulling together two noisy views of the same true point, while every other
point in the batch serves as a negative, shapes the embedding to be locally
distance-preserving.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch

from lang_goal_rl.contrastive import info_nce_loss
from lang_goal_rl.goal_encoder import GoalEncoder

ENV_ID = "FetchReach-v4"

PRETRAIN_SEED = 0
"""Seed for both the goal-pool env resets and torch/numpy pretraining randomness.
Fixed and documented here because this encoder is pretrained exactly once and
then reused, unchanged, across all 10 RL seeds in this stage — reproducing
"the" stage-2 encoder means reproducing this seed."""

N_GOAL_SAMPLES = 2_000
"""Number of distinct `desired_goal` points collected (via env resets with
seeds `PRETRAIN_SEED .. PRETRAIN_SEED + N_GOAL_SAMPLES - 1`) to build the
pool of true goals pretraining draws from. A quick probe of 200 resets
showed FetchReach's goal distribution spans roughly 0.3 units per axis
with per-axis std ~0.085 (a small, bounded 3D box) — 2000 samples is far
more than enough to cover that box densely (>>embed_dim=16 and
>>batch_size=256), at negligible cost since collection is a bare
`env.reset()` with no stepping."""

AUGMENTATION_NOISE_STD = 0.01
"""Std of independent Gaussian noise added to each of the anchor/positive
views of a sampled goal point. ~12% of the goal distribution's per-axis std
(0.085), chosen so a noisy pair stays much closer to each other than to a
randomly different goal drawn into the same batch (typical distance between
two distinct goals is on the order of 0.1-0.2) — that gap is what lets the
InfoNCE objective learn a *locally* distance-preserving mapping rather than
just clustering same-pair-identity points with no relation to nearby-but-
distinct ones."""

N_PRETRAIN_STEPS = 2_000
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
LOG_EVERY = 200


def collect_goal_pool(n_samples: int, seed: int) -> np.ndarray:
    """Collect `n_samples` distinct desired-goal points from FetchReach's reset distribution.

    Args:
        n_samples: Number of resets to perform.
        seed: Base seed; reset `i` uses `seed + i`, giving deterministic,
            non-overlapping goal draws for a given `seed`.

    Returns:
        Array of shape (n_samples, 3) — FetchReach's `desired_goal` xyz per reset.
    """
    gym.register_envs(gymnasium_robotics)
    env = gym.make(ENV_ID)
    goals = np.empty((n_samples, 3), dtype=np.float32)
    for i in range(n_samples):
        obs, _info = env.reset(seed=seed + i)
        goals[i] = obs["desired_goal"]
    env.close()
    return goals


def pretrain(goal_pool: np.ndarray, seed: int) -> GoalEncoder:
    """Contrastively pretrain a GoalEncoder on noise-augmented pairs drawn from `goal_pool`.

    Args:
        goal_pool: Array of shape (n_pool, goal_dim) of true goal points, as
            returned by `collect_goal_pool`.
        seed: Seed for torch initialization and numpy sampling/augmentation.

    Returns:
        The pretrained `GoalEncoder`, left in eval-appropriate state (caller
        is responsible for freezing/using it downstream).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    encoder = GoalEncoder(goal_dim=goal_pool.shape[1])
    optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)

    n_pool = goal_pool.shape[0]
    final_loss = float("nan")
    for step in range(N_PRETRAIN_STEPS):
        idx = rng.integers(0, n_pool, size=BATCH_SIZE)
        base = goal_pool[idx]
        anchor_noise = rng.normal(0.0, AUGMENTATION_NOISE_STD, size=base.shape).astype(np.float32)
        positive_noise = rng.normal(0.0, AUGMENTATION_NOISE_STD, size=base.shape).astype(np.float32)

        anchors = encoder(torch.from_numpy(base + anchor_noise))
        positives = encoder(torch.from_numpy(base + positive_noise))
        loss = info_nce_loss(anchors, positives)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        final_loss = float(loss.item())
        if step % LOG_EVERY == 0 or step == N_PRETRAIN_STEPS - 1:
            print(f"pretrain step={step} info_nce_loss={final_loss:.4f}")

    return encoder


def main() -> None:
    """Collect the goal pool, pretrain the encoder once, and save its state dict."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "artifacts" / "goal_encoder.pt")
    args = parser.parse_args()

    goal_pool = collect_goal_pool(N_GOAL_SAMPLES, seed=PRETRAIN_SEED)
    print(f"collected goal pool: {goal_pool.shape[0]} points, seed={PRETRAIN_SEED}")

    encoder = pretrain(goal_pool, seed=PRETRAIN_SEED)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), args.out)
    print(f"saved pretrained encoder to {args.out}")


if __name__ == "__main__":
    main()
