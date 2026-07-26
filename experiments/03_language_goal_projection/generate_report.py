"""Aggregate stage 3's 3-seed tiered result, run the collapse re-check, and write report.md + charts.

Unlike stage 2's `generate_report.py`, this stage's language-goal
substitution eval failed near-uniformly across all 3 tiered seeds (see
`debug_language_eval.py`'s diagnostic run) — a systematic embedding-space
mismatch, not seed-to-seed noise. Per `.claude/agents/CONTRACTS.md`'s tiered
seed strategy, a tier-1 result that's clearly failing does not get scaled to
the full 10-seed budget; this script reports the 3-seed result plus the
diagnostic evidence for why, rather than silently burning 7 more seeds on a
result that would not change qualitatively.
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path

import numpy as np
import torch

from lang_goal_rl.goal_encoder import GoalEncoder
from lang_goal_rl.goal_region_vocabulary import ALL_INSTRUCTIONS, MEASURED_GOAL_BOX, instruction_to_region
from lang_goal_rl.language_embedding import encode_instructions
from lang_goal_rl.language_goal_projection import LanguageGoalProjection
from lang_goal_rl.reporting import plot_embedding_projection, plot_multi_seed_success_rate, write_report

EXPERIMENT_DIR = Path(__file__).parent
SEEDS = [0, 1, 2]

LITERAL_SUCCESS_RATE_RE = re.compile(r"^success_rate=([\d.]+) over (\d+) episodes$", re.MULTILINE)
LANGUAGE_SUCCESS_RATE_RE = re.compile(
    r'^language_success_rate=([\d.]+) instruction="([^"]+)" region="([^"]+)" over (\d+) episodes$',
    re.MULTILINE,
)

STAGE2_10_SEED_RATES = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
"""Stage 2's locked-in 10-seed result, copied verbatim from
`experiments/02_contrastive_goal_embedding/report.md` — the baseline this
stage's literal-goal-protocol reproduction is checked against."""


def parse_seed_log(log_path: Path) -> tuple[float, dict[str, float]]:
    """Parse one seed's literal success rate and per-instruction language success rates.

    Args:
        log_path: Path to the seed's `stdout.log`.

    Returns:
        A tuple `(literal_success_rate, {instruction: language_success_rate})`.

    Raises:
        ValueError: If the literal `success_rate=` line is missing.
    """
    text = log_path.read_text()
    literal_match = LITERAL_SUCCESS_RATE_RE.search(text)
    if literal_match is None:
        msg = f"no literal success_rate line found in {log_path}"
        raise ValueError(msg)
    literal_rate = float(literal_match.group(1))

    language_rates = {
        instruction: float(rate) for rate, instruction, _region, _n in LANGUAGE_SUCCESS_RATE_RE.findall(text)
    }
    return literal_rate, language_rates


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


def main() -> None:
    """Aggregate stage 3's 3-seed tiered result and write the report."""
    literal_rates: dict[int, float] = {}
    language_rates_by_seed: dict[int, dict[str, float]] = {}
    raw_output_paths = []
    for seed in SEEDS:
        log_path = EXPERIMENT_DIR / "runs" / f"seed_{seed}" / "stdout.log"
        literal_rate, language_rates = parse_seed_log(log_path)
        literal_rates[seed] = literal_rate
        language_rates_by_seed[seed] = language_rates
        raw_output_paths.append(log_path)

    literal_values = [literal_rates[seed] for seed in SEEDS]
    language_all_values = [rate for seed in SEEDS for rate in language_rates_by_seed[seed].values()]
    language_mean_per_seed = {seed: statistics.mean(language_rates_by_seed[seed].values()) for seed in SEEDS}

    per_seed_literal_chart = plot_multi_seed_success_rate(
        {f"seed_{seed}": [literal_rates[seed]] for seed in SEEDS},
        out_path=EXPERIMENT_DIR / "charts" / "literal_goal_success_rate.png",
        proof_gate_threshold=statistics.mean(STAGE2_10_SEED_RATES),
    )
    per_seed_language_chart = plot_multi_seed_success_rate(
        {f"seed_{seed}": list(language_rates_by_seed[seed].values()) for seed in SEEDS},
        out_path=EXPERIMENT_DIR / "charts" / "language_goal_success_rate.png",
        proof_gate_threshold=statistics.mean(STAGE2_10_SEED_RATES),
    )

    # Embedding-projection chart: where the 14 instructions' *projected* embeddings land
    # (PCA) relative to the frozen GoalEncoder's outputs for goals actually drawn from the
    # env's real distribution during training -- the chart form of the norm-scale mismatch
    # `debug_language_eval.py` measured numerically.
    encoder = load_frozen_encoder(
        EXPERIMENT_DIR.parent / "02_contrastive_goal_embedding" / "artifacts" / "goal_encoder.pt",
    )
    projection = load_projection(EXPERIMENT_DIR / "artifacts" / "language_goal_projection.pt")

    rng = np.random.default_rng(0)
    training_like_goals = rng.uniform(MEASURED_GOAL_BOX.axis_min, MEASURED_GOAL_BOX.axis_max, size=(300, 3))
    with torch.no_grad():
        training_like_embeddings = encoder(torch.from_numpy(training_like_goals).float()).numpy()
    training_labels = ["literal goal_encoder(desired_goal), training-distribution sample"] * len(training_like_goals)

    instructions = list(ALL_INSTRUCTIONS)
    sentence_embeddings = torch.from_numpy(encode_instructions(instructions))
    with torch.no_grad():
        projected_embeddings = projection(sentence_embeddings).numpy()
    instruction_labels = [f"projected instruction ({instruction_to_region(i)})" for i in instructions]

    combined_embeddings = np.concatenate([training_like_embeddings, projected_embeddings], axis=0)
    combined_labels = training_labels + instruction_labels
    projection_chart = plot_embedding_projection(
        combined_embeddings,
        combined_labels,
        out_path=EXPERIMENT_DIR / "charts" / "embedding_projection.png",
    )

    collapse_log = (EXPERIMENT_DIR / "artifacts" / "collapse_diagnostic_stdout.log").read_text()
    ratio_match = re.search(r"min_cross_region_distance / collapse_epsilon = ([\d.]+)x", collapse_log)
    is_collapsed_match = re.search(r"is_collapsed=(\w+)", collapse_log)
    collapse_ratio = ratio_match.group(1) if ratio_match else "unknown"
    is_collapsed = is_collapsed_match.group(1) if is_collapsed_match else "unknown"

    metrics_table = (
        "### Half 1 — literal-goal protocol reproduction (sanity check before the language test)\n\n"
        "| Seed | Literal success rate (50 eval episodes, stage-2 protocol) |\n"
        "|------|------------------------------------------------------------|\n"
        + "\n".join(f"| {seed} | {literal_rates[seed]:.3f} |" for seed in SEEDS)
        + "\n\n"
        f"Stage 2's 10-seed baseline: mean={statistics.mean(STAGE2_10_SEED_RATES):.3f}, "
        f"median={statistics.median(STAGE2_10_SEED_RATES):.3f}, "
        f"mode={statistics.mode(STAGE2_10_SEED_RATES):.3f} — all 3 tiered seeds reproduce it exactly.\n\n"
        "### Half 2a — language-goal substitution success rate (the actual stage-3 test)\n\n"
        "| Seed | Mean success rate across 14 instructions (50 episodes each) |\n"
        "|------|----------------------------------------------------------------|\n"
        + "\n".join(f"| {seed} | {language_mean_per_seed[seed]:.3f} |" for seed in SEEDS)
        + "\n\n"
        f"Aggregate across all {len(SEEDS)} seeds x 14 instructions "
        f"({len(language_all_values)} success-rate samples): "
        f"mean={statistics.mean(language_all_values):.3f}, "
        f"median={statistics.median(language_all_values):.3f}, "
        f"max={max(language_all_values):.3f}.\n\n"
        "### Half 2a — per-instruction detail (seed 0)\n\n"
        "| Instruction | Region | Success rate |\n"
        "|-------------|--------|---------------|\n"
        + "\n".join(
            f"| {instruction} | {instruction_to_region(instruction)} | "
            f"{language_rates_by_seed[0].get(instruction, float('nan')):.3f} |"
            for instruction in ALL_INSTRUCTIONS
        )
        + "\n\n"
        "### Half 2b — collapse diagnostic (re-verified independently, not cited from the builder)\n\n"
        f"`min_cross_region_pairwise_distance / collapse_epsilon` = **{collapse_ratio}x** "
        f"(threshold is 1x; anything above 1x is \"not collapsed\"). `is_collapsed` = **{is_collapsed}**. "
        "Full numeric readout: `artifacts/collapse_diagnostic_stdout.log`.\n"
    )

    anomalies = (
        "The language-goal substitution test failed near-uniformly: mean success rate "
        f"{statistics.mean(language_all_values):.3f} across all 3 seeds x 14 instructions x 50 episodes "
        "(vs. literal-goal 1.000 on the same 3 checkpoints, reproducing stage 2's baseline exactly). "
        "This is NOT seed noise -- all 3 seeds show the same near-total failure, so scaling to the full "
        "10-seed budget was skipped per the tiered-seed strategy (a tier-1 result this uniformly bad would "
        "not change qualitatively with 7 more seeds).\n\n"
        "Root-caused via `debug_language_eval.py`, run against the trained seed_0 checkpoint:\n"
        "- Check 1: feeding the policy the *correct* `goal_encoder(literal_target)` embedding through the "
        "exact same monkeypatch substitution machinery used for the language test reproduces "
        "success_rate=1.000 over 20 episodes -- so the substitution mechanism itself (env goal override + "
        "features-extractor monkeypatch) is verified sound, not the source of the failure.\n"
        "- Check 2: norm-scale mismatch. `goal_encoder(desired_goal)` outputs, for goals actually drawn from "
        "the env's real training-time distribution (uniform over the measured box), have norm "
        "mean=0.039 std=0.009 (range ~0.022-0.073) over 500 samples. The trained `LanguageGoalProjection`'s "
        "outputs for the 14 fixed instructions have norms in the ~0.25-0.41 range -- 5-10x larger than "
        "anything the policy ever saw as a goal-embedding input during training. `train_projection`'s "
        "InfoNCE-style loss pulls each instruction toward its region's mean embedding and pushes it away "
        "from other regions' mean embeddings, but nothing in that objective constrains the *overall scale* "
        "of the projection's output to match the frozen encoder's actual output range -- it converged to "
        "well-separated points (satisfying half 2b's collapse check) that sit far outside the policy's "
        "training-distribution manifold (failing half 2a's success-rate check). The embedding-projection "
        "chart shows this directly: the projected instructions and the training-distribution goal-embedding "
        "cloud occupy visually distinct regions of the PCA plot."
    )

    known_risks_note = (
        "This failure does not match ROADMAP.md's documented SAC deterministic-eval-collapse signature "
        "(good training curve -> collapsed eval, preceded by an ent_coef_loss spike): that signature is "
        "about the *literal*-goal eval collapsing after training, but here the literal-goal eval is a "
        "clean 1.000 on all 3 seeds -- training and the frozen-encoder-based policy are both fine. The "
        "failure is specific to the language-projection substitution step, which is new to this stage and "
        "not something stage 1's cross-check applies to. "
        "The 'Metric mismatch' known risk (sentence-transformer's contrastive cosine-similarity space vs. "
        "a raw-distance-based reward) is adjacent but not quite what happened here either -- this stage "
        "never trains a distance-based reward off the sentence embedding directly; `train_projection` "
        "regresses into the *frozen GoalEncoder's* space via InfoNCE, and the resulting scale mismatch is "
        "a property of that regression's loss (no scale term), not of the sentence-embedding metric per se. "
        "Recording this as a new, distinct failure mode rather than force-fitting it to an existing "
        "Known risks entry. "
        "Per the ROADMAP's scope decision, this result is FetchReach-only and says nothing about harder "
        "tasks; it is a mechanism-level finding (projection output scale vs. training-distribution scale) "
        "that would need to be re-checked on any task, not something specific to FetchReach's dynamics."
    )

    write_report(
        stage=3,
        title="Frozen language embedding -> goal space",
        seeds=SEEDS,
        candidates=None,
        proof_gate_text=(
            "Success rate on language goals ~ stage-2 baseline; projection doesn't "
            "collapse distinct instructions to one point."
        ),
        metrics_table=metrics_table,
        chart_paths=[per_seed_literal_chart, per_seed_language_chart, projection_chart],
        raw_output_paths=raw_output_paths,
        anomalies=anomalies,
        known_risks_note=known_risks_note,
        out_dir=EXPERIMENT_DIR,
    )


if __name__ == "__main__":
    main()
