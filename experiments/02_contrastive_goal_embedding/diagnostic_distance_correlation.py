"""Stage 2 diagnostic: does the frozen pretrained encoder preserve true xyz distance?

Loads the exact same frozen encoder checkpoint used for all 10 RL seeds
(`artifacts/goal_encoder.pt`), samples a fresh pool of goals from
FetchReach's reset distribution (seeds disjoint from both the pretraining
pool [0..1999] and the RL held-out eval seeds [1000..1049] used by
`train.py`), embeds them with the frozen encoder, and reports
`embedding_distance_correlation` between pairwise embedding distances and
pairwise true xyz distances. This is the numeric check behind the second
half of stage 2's proof gate: "distance-in-latent correlates with true task
distance."

Also saves the raw embeddings + true coordinates so `generate_report.py` can
build a 2D PCA projection chart from the same sample without re-sampling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lang_goal_rl.embedding_distance_correlation import embedding_distance_correlation
from lang_goal_rl.goal_encoder import GoalEncoder
from pretrain_encoder import collect_goal_pool

N_DIAGNOSTIC_SAMPLES = 500
"""Number of goals sampled for the correlation diagnostic. Large enough for a
stable pairwise-distance correlation estimate (500 points -> ~124,750 pairwise
distances) while staying cheap (bare env resets, no stepping)."""

DIAGNOSTIC_SEED = 5_000
"""Base seed for diagnostic goal sampling. Offset well clear of the
pretraining pool's seeds (0..1999) and the RL eval seeds (1000..1049) so this
diagnostic is measured on goals the encoder never saw during pretraining."""


def main() -> None:
    """Sample held-out goals, embed with the frozen encoder, and report the distance correlation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder-path",
        type=Path,
        default=Path(__file__).parent / "artifacts" / "goal_encoder.pt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "artifacts" / "diagnostic_embeddings.npz",
    )
    args = parser.parse_args()

    goals = collect_goal_pool(N_DIAGNOSTIC_SAMPLES, seed=DIAGNOSTIC_SEED)

    encoder = GoalEncoder(goal_dim=3)
    encoder.load_state_dict(torch.load(args.encoder_path, map_location="cpu"))
    encoder.eval()

    with torch.no_grad():
        embeddings = encoder(torch.from_numpy(goals)).numpy()

    correlation = embedding_distance_correlation(embeddings, goals)
    print(
        f"embedding_distance_correlation={correlation:.4f} over "
        f"{N_DIAGNOSTIC_SAMPLES} goals (seed={DIAGNOSTIC_SEED})"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, embeddings=embeddings, goals=goals)


if __name__ == "__main__":
    main()
