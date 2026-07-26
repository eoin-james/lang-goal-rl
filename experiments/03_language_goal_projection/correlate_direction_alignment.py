"""Attempt 3, part 2: does direction alignment actually predict RL success, or is that a confound?

Attempt 2's reviewer left an open question in `report.md`: attempt 2's per-instruction
RL success rates varied a lot (0.000-0.380 across the 14 fixed instructions), and
"center" region instructions scored noticeably higher than the rest. Is that variation
explained by how well each instruction's *projected* embedding points toward its
region's true direction in the frozen `GoalEncoder`'s space (directional accuracy) --
or is it a FetchReach-geometry confound (e.g. the fixed success radius just happening
to favor goals near the robot's reset position, independent of projection quality)?

This script settles that using data that already exists -- no new RL training or
evaluation. It:

1. Loads attempt 2's trained projection checkpoint (`artifacts/language_goal_projection_v2_fixed.pt`)
   and stage 2's frozen `GoalEncoder`, and computes
   `instruction_direction_diagnostic.measure_instruction_direction_alignment` against
   it -- the cosine similarity between attempt 2's projected output and each
   instruction's true region centroid.
2. Parses attempt 2's already-recorded per-instruction success rates directly from
   the raw eval logs (`runs_v2/seed_{0,1,2}/stdout.log`), averaged across all 3 seeds
   per instruction (not just seed 0's table in report.md) for a less noisy per-instruction
   estimate.
3. Computes the Pearson correlation between the two per-instruction series.

Output is printed for the caller to redirect to a log file (this project's practice
for diagnostic scripts, per the stage-3 attempt-1 reviewer's evidence-gap finding).
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path

import torch

from lang_goal_rl.goal_encoder import GoalEncoder
from lang_goal_rl.goal_region_vocabulary import ALL_INSTRUCTIONS, instruction_to_region
from lang_goal_rl.instruction_direction_diagnostic import measure_instruction_direction_alignment
from lang_goal_rl.language_embedding import encode_instructions
from lang_goal_rl.language_goal_projection import LanguageGoalProjection

EXPERIMENT_DIR = Path(__file__).parent
STAGE2_ENCODER_PATH = EXPERIMENT_DIR.parent / "02_contrastive_goal_embedding" / "artifacts" / "goal_encoder.pt"
ATTEMPT2_PROJECTION_PATH = EXPERIMENT_DIR / "artifacts" / "language_goal_projection_v2_fixed.pt"
ATTEMPT2_SEEDS = (0, 1, 2)

LANGUAGE_SUCCESS_RATE_RE = re.compile(
    r'^language_success_rate=([\d.]+) instruction="([^"]+)" region="([^"]+)" over (\d+) episodes$',
    re.MULTILINE,
)


def load_frozen_encoder(path: Path) -> GoalEncoder:
    """Load stage 2's pretrained `GoalEncoder` checkpoint, unchanged."""
    encoder = GoalEncoder(goal_dim=3)
    encoder.load_state_dict(torch.load(path, map_location="cpu"))
    encoder.eval()
    return encoder


def load_projection(path: Path) -> LanguageGoalProjection:
    """Load a `LanguageGoalProjection` checkpoint saved by `train_projection.py`."""
    checkpoint = torch.load(path, map_location="cpu")
    projection = LanguageGoalProjection(input_dim=checkpoint["input_dim"], embed_dim=checkpoint["embed_dim"])
    projection.load_state_dict(checkpoint["state_dict"])
    projection.eval()
    return projection


def parse_language_success_rates(log_path: Path) -> dict[str, float]:
    """Parse one seed's per-instruction language-goal success rates from its raw stdout log."""
    text = log_path.read_text()
    return {
        instruction: float(rate) for rate, instruction, _region, _n in LANGUAGE_SUCCESS_RATE_RE.findall(text)
    }


def pearson_r(xs: list[float], ys: list[float]) -> float:
    """Plain Pearson correlation coefficient between two equal-length series.

    No scipy dependency (not in `uv.lock`) -- this is the standard
    covariance-over-product-of-std-devs formula, nothing more.

    Args:
        xs: First series.
        ys: Second series, same length as `xs`.

    Returns:
        Pearson's r, in [-1.0, 1.0].

    Raises:
        ValueError: If the series have different lengths or fewer than 2 points.

    """
    if len(xs) != len(ys):
        msg = f"series length mismatch: {len(xs)} vs {len(ys)}"
        raise ValueError(msg)
    n = len(xs)
    if n < 2:
        msg = f"need at least 2 points to correlate, got {n}"
        raise ValueError(msg)

    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    std_x = statistics.pstdev(xs)
    std_y = statistics.pstdev(ys)
    if std_x == 0.0 or std_y == 0.0:
        msg = "cannot correlate a constant series (zero variance)"
        raise ValueError(msg)
    return cov / (n * std_x * std_y)


def main() -> None:
    """Compute and print the direction-alignment vs. success-rate correlation for attempt 2."""
    encoder = load_frozen_encoder(STAGE2_ENCODER_PATH)
    projection = load_projection(ATTEMPT2_PROJECTION_PATH)
    print(f"loaded attempt-2 projection from {ATTEMPT2_PROJECTION_PATH} (no retraining)")

    instructions = list(ALL_INSTRUCTIONS)
    region_names = [instruction_to_region(instruction) for instruction in instructions]
    sentence_embeddings = torch.from_numpy(encode_instructions(instructions))

    alignment = measure_instruction_direction_alignment(
        projection, encoder, sentence_embeddings, instructions, region_names,
    )
    cosine_by_instruction = alignment.as_dict()

    per_seed_rates = [
        parse_language_success_rates(EXPERIMENT_DIR / "runs_v2" / f"seed_{seed}" / "stdout.log")
        for seed in ATTEMPT2_SEEDS
    ]
    mean_success_by_instruction = {
        instruction: statistics.mean(rates[instruction] for rates in per_seed_rates) for instruction in instructions
    }

    print(f"\nPer-instruction: cosine similarity to true centroid (attempt-2 projection) "
          f"vs. mean success rate (attempt-2, {len(ATTEMPT2_SEEDS)} seeds x 50 episodes each)\n")
    print(f"{'instruction':<45} {'region':<16} {'cosine_sim':>10} {'mean_success':>13}")
    for instruction in instructions:
        print(
            f"{instruction:<45} {instruction_to_region(instruction):<16} "
            f"{cosine_by_instruction[instruction]:>10.4f} {mean_success_by_instruction[instruction]:>13.4f}",
        )

    cosine_series = [cosine_by_instruction[i] for i in instructions]
    success_series = [mean_success_by_instruction[i] for i in instructions]
    r = pearson_r(cosine_series, success_series)
    print(f"\nPearson r (cosine_similarity vs. mean_success_rate, n={len(instructions)} instructions) = {r:.4f}")

    center_cosines = [cosine_by_instruction[i] for i in instructions if instruction_to_region(i) == "center"]
    noncenter_cosines = [cosine_by_instruction[i] for i in instructions if instruction_to_region(i) != "center"]
    print(
        f"\nFor context (attempt 2's report flagged 'center' scoring higher than other regions): "
        f"center mean cosine_sim={statistics.mean(center_cosines):.4f}, "
        f"non-center mean cosine_sim={statistics.mean(noncenter_cosines):.4f}",
    )


if __name__ == "__main__":
    main()
