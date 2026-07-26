"""Shared chart and report-generation utilities.

Every experiment stage imports this module to produce its `report.md` and
`charts/*.png`. Keeping the plotting and report-rendering logic here (instead
of duplicated per-experiment) is what stops the report structure from
drifting stage to stage — see `.claude/agents/CONTRACTS.md` for the fixed
`report.md` section order this module renders.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt


def _ensure_parent(out_path: Path) -> Path:
    """Coerce to Path and create the parent directory if it doesn't exist."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def plot_training_curve(
    timesteps: Sequence[float],
    values: Sequence[float],
    *,
    ylabel: str,
    out_path: Path,
    seed: int | None = None,
) -> Path:
    """Plot a metric against training timesteps and save it as a PNG.

    Args:
        timesteps: X-axis timestep values.
        values: Y-axis metric values, same length as timesteps.
        ylabel: Label for the y-axis (e.g. "success rate").
        out_path: Destination PNG path; parent directories are created if
            missing.
        seed: Optional seed to annotate in the chart title.

    Returns:
        The path the PNG was written to (same as `out_path`).
    """
    out_path = _ensure_parent(out_path)

    fig, ax = plt.subplots()
    ax.plot(timesteps, values, marker="o")
    ax.set_xlabel("timesteps")
    ax.set_ylabel(ylabel)
    ax.set_title("Training curve" if seed is None else f"Training curve (seed {seed})")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_multi_seed_success_rate(
    results: dict[str, list[float]],
    *,
    out_path: Path,
    proof_gate_threshold: float | None = None,
) -> Path:
    """Plot per-seed mean success rate as a bar chart.

    Args:
        results: Mapping of seed/run label to a list of success-rate samples
            for that seed; each list is averaged into one bar.
        out_path: Destination PNG path; parent directories are created if
            missing.
        proof_gate_threshold: If given, draws a horizontal reference line at
            this value (the proof gate's required success rate).

    Returns:
        The path the PNG was written to (same as `out_path`).
    """
    out_path = _ensure_parent(out_path)

    labels = list(results.keys())
    means = [float(np.mean(samples)) for samples in results.values()]

    fig, ax = plt.subplots()
    ax.bar(labels, means)
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1.05)
    if proof_gate_threshold is not None:
        ax.axhline(proof_gate_threshold, color="red", linestyle="--", label="proof gate")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_embedding_projection(
    embeddings: npt.NDArray[np.floating],
    labels: Sequence[str],
    *,
    out_path: Path,
    n_components: int = 2,
) -> Path:
    """Project embeddings to 2D via numpy-only PCA (SVD) and scatter-plot them.

    No scikit-learn dependency — SVD on the mean-centered data gives the same
    principal components as PCA without pulling in a new library.

    Args:
        embeddings: Array of shape (n_samples, n_features).
        labels: Per-point category label, same length as `embeddings`; used
            to color the scatter and build a legend.
        out_path: Destination PNG path; parent directories are created if
            missing.
        n_components: Number of principal components to plot. Only 2 is
            supported (a 2D scatter).

    Returns:
        The path the PNG was written to (same as `out_path`).

    Raises:
        ValueError: If `n_components` is not 2.
    """
    if n_components != 2:
        msg = "only n_components=2 is supported for plotting"
        raise ValueError(msg)

    out_path = _ensure_parent(out_path)

    centered = embeddings - embeddings.mean(axis=0)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vt[:n_components].T

    fig, ax = plt.subplots()
    for label in sorted(set(labels)):
        mask = np.array([point_label == label for point_label in labels])
        ax.scatter(projected[mask, 0], projected[mask, 1], label=str(label))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_candidate_comparison(
    results: dict[str, list[float]],
    *,
    out_path: Path,
) -> Path:
    """Plot a candidate-vs-candidate success-rate comparison bar chart.

    Thin wrapper over `plot_multi_seed_success_rate` — a candidate comparison
    is structurally the same chart (one bar per label, averaged over
    samples), just without a proof-gate line.

    Args:
        results: Mapping of candidate name to a list of success-rate
            samples for that candidate.
        out_path: Destination PNG path.

    Returns:
        The path the PNG was written to (same as `out_path`).
    """
    return plot_multi_seed_success_rate(results, out_path=out_path)


def write_report(
    *,
    stage: int,
    title: str,
    seeds: list[int],
    candidates: list[str] | None = None,
    proof_gate_text: str,
    metrics_table: str,
    chart_paths: Sequence[Path],
    raw_output_paths: Sequence[Path],
    anomalies: str,
    known_risks_note: str = "none applicable",
    out_dir: Path,
) -> Path:
    """Render `report.md` in the fixed section order used by every stage.

    Args:
        stage: Stage number, matching the corresponding ROADMAP.md row.
        title: Stage title, e.g. "Goal-conditioned baseline (UVFA + HER)".
        seeds: Seeds run for this result, rendered verbatim into the header's
            "Seeds run" field (e.g. `[0, 1, 2]`).
        candidates: Candidate names compared in this result. `None` means a
            single locked-in approach — rendered as `"1 (locked-in)"`.
        proof_gate_text: Verbatim proof-gate text copied from ROADMAP.md.
        metrics_table: Pre-rendered markdown table for "Result summary".
        chart_paths: PNG paths to embed under "Charts", in order.
        raw_output_paths: Paths to raw stdout logs linked under "Raw output".
        anomalies: Factual, runner-reported anomaly text, or "none observed".
        known_risks_note: Which ROADMAP.md "Known risks" entries this result
            touches, or "none applicable" if none do.
        out_dir: Directory to write `report.md` into (created if missing).

    Returns:
        The path to the written `report.md`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now(tz=UTC).date().isoformat()
    candidates_text = "1 (locked-in)" if candidates is None else ", ".join(candidates)
    charts_section = "\n\n".join(f"![{path.name}]({path})" for path in chart_paths)
    raw_output_section = "\n".join(f"- [{path.name}]({path})" for path in raw_output_paths)

    report = f"""# Stage {stage}: {title}
**Date:** {today} **Seeds run:** {seeds} **Candidates:** {candidates_text}

## Proof gate (verbatim from ROADMAP.md)
> {proof_gate_text}

## Result summary
{metrics_table}

## Charts
{charts_section}

## Raw output
{raw_output_section}

## Anomalies (factual, not judged)
{anomalies}

## Known-risks cross-check
{known_risks_note}

## Reviewer verdict
_Left blank by the runner — filled in by the manager from the reviewer's
return._
"""

    report_path = out_dir / "report.md"
    report_path.write_text(report)
    return report_path
