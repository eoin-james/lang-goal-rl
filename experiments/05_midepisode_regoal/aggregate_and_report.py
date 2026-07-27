"""Aggregate all seeds' `run_regoal_eval.py` output and write stage 5's `report.md`.

Reads `runs/seed_<k>/results.json` for every seed in `--seeds`, builds the
per-seed and cross-seed metrics tables, renders the charts, and calls
`lang_goal_rl.reporting.write_report(...)` -- the required deliverable per
`.claude/agents/experiment-runner.md`. Run after `launch_seeds.sh` completes.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from lang_goal_rl.reporting import (
    plot_candidate_comparison,
    plot_multi_seed_success_rate,
    write_report,
)

EXPERIMENT_DIR = Path(__file__).parent
CHARTS_DIR = EXPERIMENT_DIR / "charts"

PROOF_GATE_TEXT = (
    "Zero-shot goal-swap success rate vs. fresh-episode baseline; if it "
    "degrades, fine-tune with injected switches and re-measure."
)

SANITY_COLLAPSE_THRESHOLD = 0.8
"""Below this, a seed's literal-goal sanity check is flagged as a possible
SAC deterministic-eval collapse (see ROADMAP.md's Known risks) rather than a
stage-5 mechanism failure -- stage 1's own gate was "near-100%"."""


def load_seed_results(seeds: list[int]) -> dict[int, dict]:
    """Load every seed's `results.json`, keyed by seed.

    Args:
        seeds: Model checkpoint seeds to load.

    Returns:
        Mapping from seed to its parsed results dict.

    Raises:
        FileNotFoundError: If a seed's `results.json` is missing (e.g.
            `launch_seeds.sh` hasn't finished or that seed's run crashed).
    """
    loaded = {}
    for seed in seeds:
        results_path = EXPERIMENT_DIR / "runs" / f"seed_{seed}" / "results.json"
        if not results_path.exists():
            msg = (
                f"missing {results_path} -- did launch_seeds.sh finish for seed {seed}?"
            )
            raise FileNotFoundError(msg)
        loaded[seed] = json.loads(results_path.read_text())
    return loaded


def build_sanity_table(seed_results: dict[int, dict]) -> tuple[str, str]:
    """Build the checkpoint-provisioning sanity-check markdown table and anomaly text.

    Args:
        seed_results: Per-seed loaded `results.json` dicts.

    Returns:
        `(markdown_table, anomaly_text)`.
    """
    rows = [
        "| Seed | Sanity success rate (literal control, full 50-step, no swap) | Episodes |",
        "|---|---|---|",
    ]
    rates = []
    anomalies = []
    for seed, result in sorted(seed_results.items()):
        rate = result["sanity_check_success_rate"]
        rates.append(rate)
        rows.append(f"| {seed} | {rate:.3f} | {result['sanity_check_episodes']} |")
        if rate < SANITY_COLLAPSE_THRESHOLD:
            anomalies.append(
                f"seed {seed}'s literal-goal sanity check scored {rate:.3f} "
                f"(< {SANITY_COLLAPSE_THRESHOLD}) -- resembles the known SAC "
                "deterministic-eval collapse signature (ROADMAP.md Known "
                "risks), not necessarily a stage-5 mechanism defect."
            )
    rows.append(f"| **Mean** | **{statistics.mean(rates):.3f}** | |")
    rows.append(f"| **Median** | **{statistics.median(rates):.3f}** | |")
    anomaly_text = "; ".join(anomalies) if anomalies else ""
    return "\n".join(rows), anomaly_text


def build_per_seed_switch_step_table(seed_results: dict[int, dict]) -> str:
    """Build the per-seed, per-switch_step markdown table.

    Args:
        seed_results: Per-seed loaded `results.json` dicts.

    Returns:
        Markdown table string.
    """
    rows = [
        (
            "| Seed | switch_step | Swap success rate | Budget-matched baseline "
            "success rate | Full-budget reference success rate | Episodes |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for seed, result in sorted(seed_results.items()):
        for switch_result in result["switch_step_results"]:
            rows.append(
                f"| {seed} | {switch_result['switch_step']} | "
                f"{switch_result['swap_success_rate']:.3f} | "
                f"{switch_result['budget_matched_baseline_success_rate']:.3f} | "
                f"{switch_result['full_budget_reference_success_rate']:.3f} | "
                f"{switch_result['n_episodes']} |"
            )
    return "\n".join(rows)


def build_aggregate_table(
    seed_results: dict[int, dict], switch_steps: list[int]
) -> str:
    """Build the cross-seed mean/median table per switch_step -- the proof-gate comparison.

    Args:
        seed_results: Per-seed loaded `results.json` dicts.
        switch_steps: Switch-step values to aggregate over, in order.

    Returns:
        Markdown table string.
    """
    rows = [
        (
            "| switch_step | Swap mean | Swap median | Baseline mean | Baseline "
            "median | Full-budget mean | Full-budget median |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for switch_step in switch_steps:
        swap_rates = []
        baseline_rates = []
        fullbudget_rates = []
        for result in seed_results.values():
            matching = next(
                sr
                for sr in result["switch_step_results"]
                if sr["switch_step"] == switch_step
            )
            swap_rates.append(matching["swap_success_rate"])
            baseline_rates.append(matching["budget_matched_baseline_success_rate"])
            fullbudget_rates.append(matching["full_budget_reference_success_rate"])
        rows.append(
            f"| {switch_step} | {statistics.mean(swap_rates):.3f} | "
            f"{statistics.median(swap_rates):.3f} | {statistics.mean(baseline_rates):.3f} | "
            f"{statistics.median(baseline_rates):.3f} | {statistics.mean(fullbudget_rates):.3f} | "
            f"{statistics.median(fullbudget_rates):.3f} |"
        )
    return "\n".join(rows)


def render_charts(seed_results: dict[int, dict], switch_steps: list[int]) -> list[Path]:
    """Render the sanity-check bar chart and one swap/baseline/reference chart per switch_step.

    Args:
        seed_results: Per-seed loaded `results.json` dicts.
        switch_steps: Switch-step values to render one comparison chart each for.

    Returns:
        Chart paths, in the order they should appear in the report.
    """
    chart_paths = []

    sanity_results = {
        f"seed_{seed}": [result["sanity_check_success_rate"]]
        for seed, result in sorted(seed_results.items())
    }
    chart_paths.append(
        plot_multi_seed_success_rate(
            sanity_results,
            out_path=CHARTS_DIR / "sanity_check_success_rate.png",
            proof_gate_threshold=0.9,
        )
    )

    for switch_step in switch_steps:
        comparison_results: dict[str, list[float]] = {
            "swap": [],
            "budget_matched_baseline": [],
            "full_budget_reference": [],
        }
        for result in seed_results.values():
            matching = next(
                sr
                for sr in result["switch_step_results"]
                if sr["switch_step"] == switch_step
            )
            comparison_results["swap"].append(matching["swap_success_rate"])
            comparison_results["budget_matched_baseline"].append(
                matching["budget_matched_baseline_success_rate"]
            )
            comparison_results["full_budget_reference"].append(
                matching["full_budget_reference_success_rate"]
            )
        chart_paths.append(
            plot_candidate_comparison(
                comparison_results,
                out_path=CHARTS_DIR / f"switch_step_{switch_step}_comparison.png",
            )
        )

    return chart_paths


def main() -> None:
    """Aggregate all seeds' results and write stage 5's report.md."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    seed_results = load_seed_results(args.seeds)
    switch_steps = sorted(
        {
            sr["switch_step"]
            for r in seed_results.values()
            for sr in r["switch_step_results"]
        }
    )

    sanity_table, sanity_anomalies = build_sanity_table(seed_results)
    per_seed_table = build_per_seed_switch_step_table(seed_results)
    aggregate_table = build_aggregate_table(seed_results, switch_steps)

    metrics_table = f"""### Checkpoint-provisioning sanity check
(literal-goal control, reused checkpoints only -- see "Checkpoint provisioning" below)

{sanity_table}

### Proof-gate comparison: swap vs. budget-matched baseline (per seed, per switch_step)

{per_seed_table}

### Proof-gate comparison: cross-seed aggregate per switch_step

{aggregate_table}
"""

    chart_paths = render_charts(seed_results, switch_steps)
    raw_output_paths = [
        EXPERIMENT_DIR / "runs" / f"seed_{seed}" / "stdout.log"
        for seed in sorted(seed_results)
    ]

    checkpoint_provisioning_note = (
        "**Checkpoint provisioning (side task, not stage 5's own proof gate):** "
        "stage 1's `experiments/01_uvfa_her_baseline/train.py` never called "
        "`model.save(...)` despite being marked Done in ROADMAP.md, so no "
        "checkpoint existed on disk. `experiments/01_uvfa_her_baseline/"
        "provision_checkpoints.py` retrained 3 seeds using the exact same "
        "`build_model`/`evaluate` helpers and hyperparameters as `train.py` "
        "(imported directly, not copied) and added the one missing step -- "
        "`model.save(...)` -- persisting them to "
        "`experiments/01_uvfa_her_baseline/checkpoints/seed_<k>.zip`, with "
        "training logs under `experiments/01_uvfa_her_baseline/runs/seed_<k>/"
        "stdout.log`. This does not touch or supersede stage 1's own "
        "report.md/ROADMAP status -- it is purely a checkpoint-provisioning "
        "step in service of stage 5 (and any future stage needing a "
        "literal-goal policy). The sanity-check table above re-runs stage "
        "1's own literal-goal eval protocol against these freshly-provisioned "
        "checkpoints, to confirm they still perform the base task before any "
        "swap result is trusted."
    )

    anomalies_text = checkpoint_provisioning_note
    if sanity_anomalies:
        anomalies_text += f"\n\n{sanity_anomalies}"
    else:
        anomalies_text += "\n\nNo sanity-check collapse observed on any seed."

    known_risks_note = (
        "**Non-stationarity at stage 5**: this is exactly the risk this "
        "experiment measures -- see the swap-vs-baseline comparison above "
        "for whether the zero-shot goal-swap degrades relative to a fresh "
        "episode. **SAC deterministic-eval collapse (~20% of seeds, "
        "confirmed stage 1)**: checked via the sanity-check table above "
        "before trusting any swap result; see Anomalies for any seed that "
        "resembles the collapse signature. **Region-vs-point ground truth** "
        "and **NN-lookup coverage density**: not applicable here -- this "
        "stage uses exact literal xyz goals throughout (no embedding "
        "substitution engaged), deliberately isolating the re-goaling "
        "mechanism from every embedding-layer confound stages 2-4 spent "
        "effort on."
    )

    report_path = write_report(
        stage=5,
        title="Mid-episode re-goaling",
        seeds=args.seeds,
        candidates=["swap", "budget_matched_baseline", "full_budget_reference"],
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
