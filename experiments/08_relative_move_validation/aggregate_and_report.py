"""Aggregate all seeds' `run_relative_move_eval.py` output and write stage 8's `report.md`.

Reads `runs/seed_<k>/results.json` for every seed in `--seeds`, builds the
sanity-check table, the direction/magnitude/switch_step breakdown tables
(the plan's explicit requirement -- not one buried aggregate number), the
overall aggregate, renders charts, and calls
`lang_goal_rl.reporting.write_report(...)` -- the required deliverable per
`.claude/agents/experiment-runner.md`. Run after `launch_seeds.sh` (or an
equivalent per-seed launch) completes.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from lang_goal_rl.reporting import plot_candidate_comparison, plot_multi_seed_success_rate, write_report

EXPERIMENT_DIR = Path(__file__).parent
CHARTS_DIR = EXPERIMENT_DIR / "charts"
RELATIVE_CHARTS_DIR = Path("charts")
"""Relative counterpart of `CHARTS_DIR`, used only for the paths embedded in
`report.md` -- `uv run python <relative_script>.py` resolves `__file__` to
an absolute path, but the report's markdown links should stay relative
(matching every earlier stage's `report.md`/`evidence.md`, and portable if
the repo is moved). File I/O itself still goes through the absolute
`CHARTS_DIR`/`EXPERIMENT_DIR` so it's independent of the caller's cwd."""
RELATIVE_RUNS_DIR = Path("runs")

PROOF_GATE_TEXT = (
    "Reaches relative-move targets (multiple directions/magnitudes/switch-points) at a "
    "rate matching a budget-matched fresh baseline"
)

SANITY_COLLAPSE_THRESHOLD = 0.8
"""Below this, a seed's literal-goal sanity check is flagged as a possible
SAC deterministic-eval collapse (ROADMAP.md's Known risks -- seeds 2 and 7,
confirmed stage 1) rather than a stage-8 mechanism failure."""

KNOWN_COLLAPSE_SEEDS = (2, 7)


def load_seed_results(seeds: list[int]) -> dict[int, dict[str, Any]]:
    """Load every seed's `results.json`, keyed by seed.

    Args:
        seeds: Model checkpoint seeds to load.

    Returns:
        Mapping from seed to its parsed results dict.

    Raises:
        FileNotFoundError: If a seed's `results.json` is missing.
    """
    loaded = {}
    for seed in seeds:
        results_path = EXPERIMENT_DIR / "runs" / f"seed_{seed}" / "results.json"
        if not results_path.exists():
            msg = f"missing {results_path} -- did the seed's run finish?"
            raise FileNotFoundError(msg)
        loaded[seed] = json.loads(results_path.read_text())
    return loaded


def build_sanity_table(seed_results: dict[int, dict[str, Any]]) -> tuple[str, str]:
    """Build the literal-goal sanity-check markdown table and anomaly text.

    Args:
        seed_results: Per-seed loaded `results.json` dicts.

    Returns:
        `(markdown_table, anomaly_text)`.
    """
    rows = ["| Seed | Sanity success rate (literal control, full 50-step, no relative move) | Episodes |", "|---|---|---|"]
    rates = []
    anomalies = []
    for seed, result in sorted(seed_results.items()):
        rate = result["sanity_check_success_rate"]
        rates.append(rate)
        flag = " (known SAC collapse seed)" if seed in KNOWN_COLLAPSE_SEEDS else ""
        rows.append(f"| {seed}{flag} | {rate:.3f} | {result['sanity_check_episodes']} |")
        if rate < SANITY_COLLAPSE_THRESHOLD:
            anomalies.append(
                f"seed {seed}'s literal-goal sanity check scored {rate:.3f} "
                f"(< {SANITY_COLLAPSE_THRESHOLD}) -- resembles the known SAC "
                "deterministic-eval collapse signature (ROADMAP.md Known risks), "
                "not necessarily a stage-8 mechanism defect."
            )
    rows.append(f"| **Mean** | **{statistics.mean(rates):.3f}** | |")
    rows.append(f"| **Median** | **{statistics.median(rates):.3f}** | |")
    anomaly_text = "; ".join(anomalies) if anomalies else ""
    return "\n".join(rows), anomaly_text


def _healthy_seeds(seed_results: dict[int, dict[str, Any]]) -> list[int]:
    """Seeds whose sanity check did not resemble the SAC collapse signature."""
    return [
        seed
        for seed, result in sorted(seed_results.items())
        if result["sanity_check_success_rate"] >= SANITY_COLLAPSE_THRESHOLD
    ]


def _flatten_combos(seed_results: dict[int, dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    """Flatten every (seed, combo) pair for `seeds` into one list of combo dicts with a `seed` field."""
    flattened = []
    for seed in seeds:
        for combo in seed_results[seed]["combo_results"]:
            flattened.append({**combo, "seed": seed})
    return flattened


def build_breakdown_table(
    flattened_combos: list[dict[str, Any]], group_key: str, group_label: str,
) -> tuple[str, dict[str, list[float]], dict[str, list[float]], dict[str, list[float]]]:
    """Build one breakdown table (by direction, magnitude, or switch_step), aggregated across the other two dims and all seeds.

    Args:
        flattened_combos: Every (seed, combo) dict, from `_flatten_combos`.
        group_key: The combo dict field to group by (`"direction"`,
            `"magnitude_label"`, or `"switch_step"`).
        group_label: Human-readable column header for `group_key`.

    Returns:
        `(markdown_table, relative_move_rates_by_group, baseline_rates_by_group,
        clip_rates_by_group)` -- the three dicts are per-group lists of every
        matching combo's success/clip rate (one float per (seed, combo)
        entry), used both for the table's mean/median and for chart data.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for combo in flattened_combos:
        groups.setdefault(str(combo[group_key]), []).append(combo)

    rows = [
        f"| {group_label} | Relative-move mean | Relative-move median | Baseline mean | "
        "Baseline median | Clip rate | Episodes |",
        "|---|---|---|---|---|---|---|",
    ]
    relative_move_by_group: dict[str, list[float]] = {}
    baseline_by_group: dict[str, list[float]] = {}
    clip_by_group: dict[str, list[float]] = {}

    for group_name in sorted(groups, key=str):
        combos = groups[group_name]
        rm_rates = [c["relative_move_success_rate"] for c in combos]
        bl_rates = [c["baseline_success_rate"] for c in combos]
        clip_rates = [c["clip_rate"] for c in combos]
        n_episodes = sum(c["n_episodes"] for c in combos)
        rows.append(
            f"| {group_name} | {statistics.mean(rm_rates):.3f} | {statistics.median(rm_rates):.3f} | "
            f"{statistics.mean(bl_rates):.3f} | {statistics.median(bl_rates):.3f} | "
            f"{statistics.mean(clip_rates):.3f} | {n_episodes} |"
        )
        relative_move_by_group[group_name] = rm_rates
        baseline_by_group[group_name] = bl_rates
        clip_by_group[group_name] = clip_rates

    return "\n".join(rows), relative_move_by_group, baseline_by_group, clip_by_group


def build_overall_aggregate(flattened_combos: list[dict[str, Any]]) -> str:
    """Build the single overall relative-move-vs-baseline aggregate line.

    Args:
        flattened_combos: Every (seed, combo) dict, from `_flatten_combos`.

    Returns:
        Markdown text with the overall mean/median for both conditions and
        the overall clip rate, plus total episode count.
    """
    rm_rates = [c["relative_move_success_rate"] for c in flattened_combos]
    bl_rates = [c["baseline_success_rate"] for c in flattened_combos]
    clip_rates = [c["clip_rate"] for c in flattened_combos]
    total_episodes = sum(c["n_episodes"] for c in flattened_combos)
    return (
        f"**Overall (all switch_steps x directions x magnitudes, {len(flattened_combos)} combos, "
        f"{total_episodes} episodes):** relative-move mean={statistics.mean(rm_rates):.3f} "
        f"median={statistics.median(rm_rates):.3f}; budget-matched-baseline mean={statistics.mean(bl_rates):.3f} "
        f"median={statistics.median(bl_rates):.3f}; overall clip rate={statistics.mean(clip_rates):.3f}"
    )


def render_grouped_comparison_chart(
    relative_move_by_group: dict[str, list[float]],
    baseline_by_group: dict[str, list[float]],
    *,
    out_path: Path,
) -> Path:
    """Render one bar chart with `relative move` and `baseline` bars interleaved per group.

    `reporting.plot_candidate_comparison` only draws one bar per dict key (no
    grouped-bar support in this project's charting utility), so each group
    contributes two adjacent keys ("<group> (move)" / "<group> (baseline)")
    to a single flat dict -- the same one-bar-per-condition-label pattern
    stage 5/6's comparison charts already use, just with more labels.

    Args:
        relative_move_by_group: Per-group list of relative-move success
            samples, from `build_breakdown_table`.
        baseline_by_group: Same, for the budget-matched baseline.
        out_path: Destination PNG path.

    Returns:
        The path the PNG was written to.
    """
    combined: dict[str, list[float]] = {}
    for group_name in relative_move_by_group:
        combined[f"{group_name}\n(move)"] = relative_move_by_group[group_name]
        combined[f"{group_name}\n(baseline)"] = baseline_by_group[group_name]
    return plot_candidate_comparison(combined, out_path=out_path)


def main() -> None:
    """Aggregate all seeds' results and write stage 8's report.md."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 3])
    args = parser.parse_args()

    seed_results = load_seed_results(args.seeds)
    healthy_seeds = _healthy_seeds(seed_results)

    sanity_table, sanity_anomalies = build_sanity_table(seed_results)

    all_flattened = _flatten_combos(seed_results, sorted(seed_results.keys()))
    healthy_flattened = _flatten_combos(seed_results, healthy_seeds)

    direction_table, rm_by_direction, bl_by_direction, clip_by_direction = build_breakdown_table(
        healthy_flattened, "direction", "Direction"
    )
    magnitude_table, rm_by_magnitude, bl_by_magnitude, clip_by_magnitude = build_breakdown_table(
        healthy_flattened, "magnitude_label", "Magnitude"
    )
    switch_step_table, rm_by_switch_step, bl_by_switch_step, _clip_by_switch_step = build_breakdown_table(
        healthy_flattened, "switch_step", "Switch step"
    )
    overall_all_seeds = build_overall_aggregate(all_flattened)
    overall_healthy_seeds = build_overall_aggregate(healthy_flattened)

    metrics_table = f"""### Literal-goal sanity check (all seeds run, including known-collapse seeds 2/7 if present)

{sanity_table}

All breakdown tables below use only the healthy seeds ({healthy_seeds}) --
seeds resembling the known SAC deterministic-eval collapse signature are
excluded from the mechanism verdict per ROADMAP.md's stage-1 lesson
("compare against baselines using the same seed count, judge at
median/mode not mean, and check whether a failed seed shows this exact
signature before attributing a regression to the new component").

### Breakdown by direction (aggregated across switch_step, magnitude, healthy seeds)

{direction_table}

### Breakdown by magnitude (aggregated across switch_step, direction, healthy seeds)

{magnitude_table}

### Breakdown by switch_step (aggregated across direction, magnitude, healthy seeds)

{switch_step_table}

### Overall aggregate (proof-gate comparison)

{overall_healthy_seeds}

For completeness, the same aggregate including every seed run (collapse
seeds not excluded): {overall_all_seeds}
"""

    absolute_chart_paths = [
        plot_multi_seed_success_rate(
            {f"seed_{seed}": [result["sanity_check_success_rate"]] for seed, result in sorted(seed_results.items())},
            out_path=CHARTS_DIR / "sanity_check_success_rate.png",
            proof_gate_threshold=0.9,
        ),
        render_grouped_comparison_chart(
            rm_by_direction, bl_by_direction, out_path=CHARTS_DIR / "success_rate_by_direction.png"
        ),
        render_grouped_comparison_chart(
            rm_by_magnitude, bl_by_magnitude, out_path=CHARTS_DIR / "success_rate_by_magnitude.png"
        ),
        render_grouped_comparison_chart(
            rm_by_switch_step, bl_by_switch_step, out_path=CHARTS_DIR / "success_rate_by_switch_step.png"
        ),
        plot_multi_seed_success_rate(
            clip_by_magnitude, out_path=CHARTS_DIR / "clip_rate_by_magnitude.png",
        ),
    ]
    # write_report renders these paths verbatim into report.md's markdown links --
    # relativize so the report stays portable and matches every earlier stage's
    # convention (charts/*.png, runs/seed_<k>/stdout.log), independent of
    # whether `uv run` resolved `__file__` to an absolute path for this invocation.
    chart_paths = [RELATIVE_CHARTS_DIR / path.name for path in absolute_chart_paths]
    raw_output_paths = [RELATIVE_RUNS_DIR / f"seed_{seed}" / "stdout.log" for seed in sorted(seed_results)]

    clip_forcing_check = (
        "clip_forcing_35cm magnitude clip rate across healthy seeds: "
        f"{statistics.mean(clip_by_magnitude['clip_forcing_35cm']):.3f} "
        "(confirms was_clipped=True was actually forced, not merely assumed from the algebra "
        "in the magnitude's docstring)."
    )

    anomalies_text = clip_forcing_check
    if sanity_anomalies:
        anomalies_text += f"\n\n{sanity_anomalies}"
    else:
        anomalies_text += "\n\nNo sanity-check collapse observed on any seed run in this batch."

    known_risks_note = (
        "**Direction-sensitivity, not just distance (stage 4)**: the by-direction breakdown "
        "table above is the direct check this risk requires -- see report.md for whether any "
        "direction under- or over-performs its peers. **SAC deterministic-eval collapse "
        "(~20% of seeds, confirmed stage 1)**: checked via the sanity-check table above before "
        "trusting any relative-move result from that seed; healthy-seed breakdown tables "
        "exclude any seed matching the collapse signature. **Non-stationarity at stage 5**: "
        "not directly applicable here -- stage 8 tests a different mid-episode capability "
        "(relative move from an arbitrary achieved position, not a caller-supplied literal "
        "goal switch), though it shares the same budget-matched-baseline comparison "
        "methodology. **Region-vs-point / NN-lookup coverage density**: not applicable -- this "
        "stage uses exact literal xyz throughout, no embedding substitution engaged, "
        "deliberately isolating the relative-move mechanism from every embedding-layer "
        "confound stages 2-4 spent effort on."
    )

    report_path = write_report(
        stage=8,
        title="Relative-move validation",
        seeds=args.seeds,
        candidates=["relative_move", "budget_matched_baseline"],
        proof_gate_text=PROOF_GATE_TEXT,
        metrics_table=metrics_table,
        chart_paths=chart_paths,
        raw_output_paths=raw_output_paths,
        anomalies=anomalies_text,
        known_risks_note=known_risks_note,
        out_dir=EXPERIMENT_DIR,
    )
    print(f"report_written={report_path}")


if __name__ == "__main__":
    main()
