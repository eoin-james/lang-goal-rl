"""Tests for the shared chart/report-generation module.

Covers what's testable without a real training run: each `plot_*` function
writes a valid PNG to the given path, and `write_report` renders a
`report.md` with every required section header present, in the fixed order,
with the caller-supplied content actually embedded.
"""

from pathlib import Path

import numpy as np
import pytest

from lang_goal_rl.reporting import (
    plot_candidate_comparison,
    plot_embedding_projection,
    plot_multi_seed_success_rate,
    plot_training_curve,
    write_report,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _assert_valid_png(path: Path) -> None:
    """Assert the file at path exists and starts with the PNG magic bytes."""
    assert path.exists()
    with path.open("rb") as f:
        assert f.read(8) == PNG_MAGIC


class TestPlotTrainingCurve:
    """plot_training_curve writes a PNG of a metric over timesteps."""

    def test_writes_valid_png_and_returns_out_path(self, tmp_path: Path) -> None:
        out_path = tmp_path / "curve.png"
        result = plot_training_curve(
            [0, 100, 200],
            [0.1, 0.5, 0.9],
            ylabel="success rate",
            out_path=out_path,
        )
        assert result == out_path
        _assert_valid_png(out_path)

    def test_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        out_path = tmp_path / "nested" / "dir" / "curve.png"
        result = plot_training_curve(
            [0, 100],
            [0.2, 0.4],
            ylabel="reward",
            out_path=out_path,
            seed=3,
        )
        assert result == out_path
        _assert_valid_png(out_path)


class TestPlotMultiSeedSuccessRate:
    """plot_multi_seed_success_rate writes a per-seed bar chart."""

    def test_writes_valid_png(self, tmp_path: Path) -> None:
        out_path = tmp_path / "multi_seed.png"
        results = {"seed_0": [0.8, 0.9], "seed_1": [0.85]}
        result = plot_multi_seed_success_rate(results, out_path=out_path)
        assert result == out_path
        _assert_valid_png(out_path)

    def test_writes_valid_png_with_proof_gate_threshold(self, tmp_path: Path) -> None:
        out_path = tmp_path / "multi_seed_gate.png"
        results = {"seed_0": [0.8, 0.9], "seed_1": [0.85]}
        result = plot_multi_seed_success_rate(
            results, out_path=out_path, proof_gate_threshold=0.95
        )
        assert result == out_path
        _assert_valid_png(out_path)


class TestPlotEmbeddingProjection:
    """plot_embedding_projection projects embeddings to 2D via numpy-only PCA."""

    def test_writes_valid_png(self, tmp_path: Path) -> None:
        out_path = tmp_path / "projection.png"
        rng = np.random.default_rng(0)
        embeddings = rng.normal(size=(12, 5))
        labels = ["a"] * 6 + ["b"] * 6
        result = plot_embedding_projection(embeddings, labels, out_path=out_path)
        assert result == out_path
        _assert_valid_png(out_path)

    def test_rejects_unsupported_component_count(self, tmp_path: Path) -> None:
        out_path = tmp_path / "projection.png"
        embeddings = np.zeros((4, 3))
        with pytest.raises(ValueError):
            plot_embedding_projection(
                embeddings, ["a", "a", "b", "b"], out_path=out_path, n_components=3
            )


class TestPlotCandidateComparison:
    """plot_candidate_comparison is a thin wrapper over the multi-seed plot."""

    def test_writes_valid_png(self, tmp_path: Path) -> None:
        out_path = tmp_path / "candidates.png"
        results = {"candidate_a": [0.7, 0.75], "candidate_b": [0.9, 0.92]}
        result = plot_candidate_comparison(results, out_path=out_path)
        assert result == out_path
        _assert_valid_png(out_path)


class TestWriteReport:
    """write_report renders report.md with the fixed section order."""

    REQUIRED_HEADERS_IN_ORDER = [
        "## Proof gate (verbatim from ROADMAP.md)",
        "## Result summary",
        "## Charts",
        "## Raw output",
        "## Anomalies (factual, not judged)",
        "## Known-risks cross-check",
        "## Reviewer verdict",
    ]

    def _build_report(
        self,
        tmp_path: Path,
        *,
        seeds: list[int] | None = None,
        candidates: list[str] | None = None,
        known_risks_note: str | None = None,
    ) -> tuple[Path, str]:
        chart_path = tmp_path / "charts" / "curve.png"
        chart_path.parent.mkdir(parents=True, exist_ok=True)
        chart_path.write_bytes(PNG_MAGIC)
        raw_output_path = tmp_path / "runs" / "seed_0" / "stdout.log"

        kwargs = {}
        if candidates is not None:
            kwargs["candidates"] = candidates
        if known_risks_note is not None:
            kwargs["known_risks_note"] = known_risks_note

        out_dir = tmp_path / "report_out"
        report_path = write_report(
            stage=1,
            title="Goal-conditioned baseline (UVFA + HER)",
            seeds=seeds if seeds is not None else [0, 1, 2],
            proof_gate_text=(
                "Near-100% success rate over held-out eval episodes on FetchReach"
            ),
            metrics_table="| seed | success_rate |\n|---|---|\n| 0 | 0.98 |\n",
            chart_paths=[chart_path],
            raw_output_paths=[raw_output_path],
            anomalies="none observed",
            out_dir=out_dir,
            **kwargs,
        )
        return report_path, report_path.read_text()

    def test_returns_path_to_report_md_in_out_dir(self, tmp_path: Path) -> None:
        report_path, _text = self._build_report(tmp_path)
        assert report_path == tmp_path / "report_out" / "report.md"
        assert report_path.exists()

    def test_contains_top_level_stage_and_title_header(self, tmp_path: Path) -> None:
        _report_path, text = self._build_report(tmp_path)
        assert "# Stage 1: Goal-conditioned baseline (UVFA + HER)" in text

    def test_contains_all_required_headers_in_order(self, tmp_path: Path) -> None:
        _report_path, text = self._build_report(tmp_path)
        indices = [text.index(header) for header in self.REQUIRED_HEADERS_IN_ORDER]
        assert indices == sorted(indices)

    def test_embeds_proof_gate_text_verbatim(self, tmp_path: Path) -> None:
        _report_path, text = self._build_report(tmp_path)
        assert "Near-100% success rate over held-out eval episodes on FetchReach" in text

    def test_embeds_metrics_table_verbatim(self, tmp_path: Path) -> None:
        _report_path, text = self._build_report(tmp_path)
        assert "| seed | success_rate |" in text
        assert "| 0 | 0.98 |" in text

    def test_embeds_anomalies_text(self, tmp_path: Path) -> None:
        _report_path, text = self._build_report(tmp_path)
        assert "none observed" in text

    def test_embeds_chart_path_reference(self, tmp_path: Path) -> None:
        _report_path, text = self._build_report(tmp_path)
        assert "curve.png" in text

    def test_embeds_raw_output_path_reference(self, tmp_path: Path) -> None:
        _report_path, text = self._build_report(tmp_path)
        assert "stdout.log" in text

    def test_header_renders_seeds_list(self, tmp_path: Path) -> None:
        _report_path, text = self._build_report(tmp_path, seeds=[0, 1, 2, 3, 4])
        assert "**Seeds run:** [0, 1, 2, 3, 4]" in text

    def test_header_defaults_candidates_to_locked_in_when_omitted(
        self, tmp_path: Path
    ) -> None:
        _report_path, text = self._build_report(tmp_path)
        assert "**Candidates:** 1 (locked-in)" in text

    def test_header_renders_explicit_candidates_list(self, tmp_path: Path) -> None:
        _report_path, text = self._build_report(
            tmp_path, candidates=["baseline_a", "baseline_b"]
        )
        assert "**Candidates:** baseline_a, baseline_b" in text

    def test_known_risks_section_defaults_to_none_applicable(
        self, tmp_path: Path
    ) -> None:
        _report_path, text = self._build_report(tmp_path)
        known_risks_start = text.index("## Known-risks cross-check")
        reviewer_start = text.index("## Reviewer verdict")
        section = text[known_risks_start:reviewer_start]
        assert "none applicable" in section

    def test_known_risks_section_renders_custom_note(self, tmp_path: Path) -> None:
        _report_path, text = self._build_report(
            tmp_path,
            known_risks_note=(
                "Touches the metric-mismatch risk: embeddings moved off "
                "sentence-transformers."
            ),
        )
        known_risks_start = text.index("## Known-risks cross-check")
        reviewer_start = text.index("## Reviewer verdict")
        section = text[known_risks_start:reviewer_start]
        assert "Touches the metric-mismatch risk" in section
