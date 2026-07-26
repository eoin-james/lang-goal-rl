"""Aggregate stage 2's 10-seed results, compare against stage 1, and write report.md + charts.

Parses each seed's final `success_rate=X.XXX` line from its stdout log (same
protocol as stage 1's `generate_report.py`), cross-checks any seed below the
0.98 gate against stage 1's known SAC deterministic-eval-collapse signature
(good training curve, then collapsed eval, preceded by an entropy-coefficient
spike — see ROADMAP.md's Known risks), and renders the report via the shared
`lang_goal_rl.reporting` module.
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path

import numpy as np

from lang_goal_rl.reporting import plot_embedding_projection, plot_multi_seed_success_rate, write_report

EXPERIMENT_DIR = Path(__file__).parent
SEEDS = list(range(10))
FINAL_SUCCESS_RATE_RE = re.compile(r"^success_rate=([\d.]+) over (\d+) episodes$", re.MULTILINE)
ENT_COEF_LOSS_RE = re.compile(r"ent_coef_loss\s*\|\s*(-?[\d.]+)")
TRAINING_SUCCESS_RATE_RE = re.compile(r"success_rate\s*\|\s*([\d.]+)")
ENT_COEF_SPIKE_THRESHOLD = 15.0
"""Matches the magnitude of the spikes documented in stage 1's report (19.6, 52.4)."""

# Stage 1's locked-in 10-seed result, copied verbatim from
# experiments/01_uvfa_her_baseline/report.md so this stage compares against
# the exact same numbers without re-parsing another stage's raw logs.
STAGE1_SEED_RATES = [1.000, 1.000, 0.000, 1.000, 1.000, 1.000, 1.000, 0.400, 1.000, 1.000]
STAGE1_REPORT_PATH = EXPERIMENT_DIR.parent / "01_uvfa_her_baseline" / "report.md"


def parse_final_success_rate(log_path: Path) -> float:
    """Extract the final reported success rate from a seed's stdout log.

    Args:
        log_path: Path to the seed's stdout.log.

    Returns:
        The final success rate as a float in [0, 1].

    Raises:
        ValueError: If the expected success_rate line isn't found in the log.
    """
    text = log_path.read_text()
    match = FINAL_SUCCESS_RATE_RE.search(text)
    if match is None:
        msg = f"no success_rate line found in {log_path}"
        raise ValueError(msg)
    return float(match.group(1))


def check_known_collapse_signature(log_path: Path) -> str:
    """Check a failed seed's log for stage 1's documented SAC eval-collapse signature.

    The signature (from ROADMAP.md's Known risks / stage 1's report): a good
    training-time success curve (rollout/success_rate climbing toward ~0.9+)
    followed by a collapsed deterministic eval score, preceded by an
    entropy-coefficient instability spike (`ent_coef_loss` jumping past ~15-50).

    Args:
        log_path: Path to the failed seed's stdout.log.

    Returns:
        A factual description of whether the signature was found, citing the
        peak training-time success_rate and peak |ent_coef_loss| observed.
    """
    text = log_path.read_text()
    training_rates = [float(m) for m in TRAINING_SUCCESS_RATE_RE.findall(text)]
    ent_coef_losses = [abs(float(m)) for m in ENT_COEF_LOSS_RE.findall(text)]

    peak_training_rate = max(training_rates) if training_rates else 0.0
    peak_ent_coef_loss = max(ent_coef_losses) if ent_coef_losses else 0.0

    good_training_curve = peak_training_rate >= 0.85
    spike_present = peak_ent_coef_loss >= ENT_COEF_SPIKE_THRESHOLD

    matches = good_training_curve and spike_present
    verdict = "MATCHES" if matches else "does NOT clearly match"
    return (
        f"{verdict} stage 1's known signature — peak training-time "
        f"success_rate={peak_training_rate:.3f} (good-curve threshold 0.85), "
        f"peak |ent_coef_loss|={peak_ent_coef_loss:.1f} (spike threshold "
        f"{ENT_COEF_SPIKE_THRESHOLD:.0f})."
    )


def distance_tercile_labels(goals: np.ndarray) -> list[str]:
    """Bucket goals into near/mid/far labels by distance from the goal-pool centroid.

    Used purely to color the PCA scatter with a true-space-meaningful label,
    so the projection can be visually checked for whether embedding structure
    tracks true distance.
    """
    distances = np.linalg.norm(goals - goals.mean(axis=0), axis=1)
    near_cut, far_cut = np.percentile(distances, [33.3, 66.7])
    labels = []
    for d in distances:
        if d <= near_cut:
            labels.append("near-centroid")
        elif d <= far_cut:
            labels.append("mid-centroid")
        else:
            labels.append("far-centroid")
    return labels


def main() -> None:
    """Aggregate stage 2's 10-seed results, run the known-risk cross-check, and write the report."""
    results: dict[str, list[float]] = {}
    raw_output_paths = []
    for seed in SEEDS:
        log_path = EXPERIMENT_DIR / "runs" / f"seed_{seed}" / "stdout.log"
        results[f"seed_{seed}"] = [parse_final_success_rate(log_path)]
        raw_output_paths.append(log_path)

    stage2_rates = [samples[0] for samples in results.values()]
    stage1_rates = STAGE1_SEED_RATES

    def summarize(rates: list[float]) -> dict[str, float]:
        return {
            "mean": statistics.mean(rates),
            "median": statistics.median(rates),
            "mode": statistics.mode(rates),
            "min": min(rates),
            "max": max(rates),
            "n_at_gate": sum(1 for r in rates if r >= 0.98),
        }

    s1 = summarize(stage1_rates)
    s2 = summarize(stage2_rates)

    per_seed_chart = plot_multi_seed_success_rate(
        results,
        out_path=EXPERIMENT_DIR / "charts" / "multi_seed_success_rate.png",
        proof_gate_threshold=s1["mean"],
    )
    comparison_chart = plot_multi_seed_success_rate(
        {"stage1_uvfa_her": stage1_rates, "stage2_contrastive_embedding": stage2_rates},
        out_path=EXPERIMENT_DIR / "charts" / "stage1_vs_stage2_comparison.png",
    )

    diagnostic_npz = np.load(EXPERIMENT_DIR / "artifacts" / "diagnostic_embeddings.npz")
    embeddings, goals = diagnostic_npz["embeddings"], diagnostic_npz["goals"]
    projection_chart = plot_embedding_projection(
        embeddings,
        distance_tercile_labels(goals),
        out_path=EXPERIMENT_DIR / "charts" / "embedding_projection.png",
    )

    diagnostic_text = (EXPERIMENT_DIR / "artifacts" / "diagnostic_stdout.log").read_text().strip()
    correlation_match = re.search(r"embedding_distance_correlation=([\d.-]+)", diagnostic_text)
    correlation = float(correlation_match.group(1)) if correlation_match else float("nan")

    metrics_table = (
        "### Stage 2 (contrastive embedding) — per seed\n\n"
        "| Seed | Success rate (50 eval episodes) |\n"
        "|------|----------------------------------|\n"
        + "\n".join(f"| {seed} | {results[f'seed_{seed}'][0]:.3f} |" for seed in SEEDS)
        + "\n\n"
        "### Stage 1 vs Stage 2 — aggregate comparison\n\n"
        "| Metric | Stage 1 (UVFA+HER, literal goal) | Stage 2 (contrastive embedding) |\n"
        "|--------|-----------------------------------|----------------------------------|\n"
        f"| Mean | {s1['mean']:.3f} | {s2['mean']:.3f} |\n"
        f"| **Median** | **{s1['median']:.3f}** | **{s2['median']:.3f}** |\n"
        f"| **Mode** | **{s1['mode']:.3f}** | **{s2['mode']:.3f}** |\n"
        f"| Min / Max | {s1['min']:.3f} / {s1['max']:.3f} | {s2['min']:.3f} / {s2['max']:.3f} |\n"
        f"| Seeds >= 0.98 | {s1['n_at_gate']}/10 | {s2['n_at_gate']}/10 |\n"
        "\n"
        "### Distance-in-latent diagnostic\n\n"
        f"`embedding_distance_correlation` = **{correlation:.4f}** "
        "(Pearson correlation between pairwise frozen-embedding distances "
        "and pairwise true xyz distances, measured on 500 held-out goals "
        "distinct from both the pretraining pool and the RL eval seeds — "
        "see `artifacts/diagnostic_stdout.log`).\n"
    )

    failed_seeds = [seed for seed in SEEDS if results[f"seed_{seed}"][0] < 0.98]
    if failed_seeds:
        signature_lines = [
            f"- seed {seed} (success_rate={results[f'seed_{seed}'][0]:.3f}): "
            + check_known_collapse_signature(EXPERIMENT_DIR / "runs" / f"seed_{seed}" / "stdout.log")
            for seed in failed_seeds
        ]
        failed_rates_text = ", ".join(f"{seed}={results[f'seed_{seed}'][0]:.3f}" for seed in failed_seeds)
        anomalies = (
            f"{len(failed_seeds)}/10 stage-2 seeds fell below the 0.98 gate: "
            f"{failed_rates_text}. "
            "Known-risk signature check per failed seed (stage 1 documented a "
            "~20% SAC deterministic-eval-collapse failure mode: good training "
            "curve, then collapsed eval, preceded by an entropy-coefficient "
            "spike):\n" + "\n".join(signature_lines)
        )
    else:
        anomalies = "All 10 stage-2 seeds reached >=0.98 success rate — no failures to cross-check."

    known_risks_note = (
        "Per ROADMAP.md's Known risks, stage 2 must compare against stage 1 "
        "using the same seed count (10, done here) at median/mode (see table "
        "above) rather than mean, and must check any failed seed against the "
        "documented ~20% SAC deterministic-eval-collapse signature before "
        "attributing it to the new contrastive-embedding component — see the "
        "Anomalies section for the per-failed-seed check. \"Metric mismatch\" "
        "(sentence-transformer/CLIP-text cosine-similarity space) is scoped "
        "to stage 3+ and does not apply to this stage's xyz-based contrastive "
        "encoder. \"Non-stationarity at stage 5\" does not apply — no "
        "mid-episode re-goaling happens here."
    )

    write_report(
        stage=2,
        title="Learned continuous goal embedding (contrastive pretraining)",
        seeds=SEEDS,
        candidates=None,
        proof_gate_text=(
            "Success rate matches stage-1 baseline within tolerance; "
            "distance-in-latent correlates with true task distance."
        ),
        metrics_table=metrics_table,
        chart_paths=[per_seed_chart, comparison_chart, projection_chart],
        raw_output_paths=raw_output_paths,
        anomalies=anomalies,
        known_risks_note=known_risks_note,
        out_dir=EXPERIMENT_DIR,
    )


if __name__ == "__main__":
    main()
