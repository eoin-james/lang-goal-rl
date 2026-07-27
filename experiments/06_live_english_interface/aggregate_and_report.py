"""Aggregate all seeds' `live_regoal_eval.py` output and write stage 6's `report.md`.

Reads `runs/seed_<k>/results.json` for every seed in `--seeds`, builds the
no-switch-control tables (including the stage-4 sanity cross-check), the
switch-test tables (task success + time-to-redirect distribution, Set A and
Set B kept separate throughout per the task brief), renders charts, and
calls `lang_goal_rl.reporting.write_report(...)` -- the required deliverable
per `.claude/agents/experiment-runner.md`. Run after `launch_seeds.sh`
completes.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from lang_goal_rl.reporting import (
    plot_candidate_comparison,
    plot_multi_seed_success_rate,
    write_report,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
CHARTS_DIR = EXPERIMENT_DIR / "charts"

PROOF_GATE_TEXT = "End-to-end demo across ad-hoc live phrasings: task success + time-to-redirect."

SANITY_COLLAPSE_THRESHOLD = 0.8
"""Below this, a seed's literal-goal sanity check is flagged as a possible
SAC deterministic-eval collapse (ROADMAP.md Known risks) rather than a
stage-6 mechanism failure -- same threshold stage 5's aggregator used."""

STAGE4_SET_A_REFERENCE_MEAN = 0.571
STAGE4_SET_A_REFERENCE_MEDIAN = 1.000
STAGE4_APPROX_TOLERANCE = 0.15
"""How close this experiment's Set-A no-switch-control mean must land to
stage 4's already-measured 0.571 (k=1, 84-sentence NN lookup, identical
mechanism via `LiveGoalController`) to count as "approximately matching" per
the task brief -- a wide-ish band since this run uses different eval seeds
than stage 4's own script, so exact reproduction isn't expected, only
mechanism agreement."""


def load_seed_results(seeds: list[int]) -> dict[int, dict]:
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
            msg = f"missing {results_path} -- did launch_seeds.sh finish for seed {seed}?"
            raise FileNotFoundError(msg)
        loaded[seed] = json.loads(results_path.read_text())
    return loaded


def build_sanity_table(seed_results: dict[int, dict]) -> tuple[str, str]:
    """Build the literal-goal sanity-check markdown table and anomaly text.

    Args:
        seed_results: Per-seed loaded `results.json` dicts.

    Returns:
        `(markdown_table, anomaly_text)`.

    """
    rows = ["| Seed | Literal sanity success rate | Episodes |", "|---|---|---|"]
    rates = []
    anomalies = []
    for seed, result in sorted(seed_results.items()):
        rate = result["literal_sanity_success_rate"]
        rates.append(rate)
        rows.append(f"| {seed} | {rate:.3f} | {result['sanity_episodes']} |")
        if rate < SANITY_COLLAPSE_THRESHOLD:
            anomalies.append(
                f"seed {seed}'s literal-goal sanity check scored {rate:.3f} "
                f"(< {SANITY_COLLAPSE_THRESHOLD}) -- resembles the known SAC "
                "deterministic-eval collapse signature (ROADMAP.md Known risks)."
            )
    rows.append(f"| **Mean** | **{statistics.mean(rates):.3f}** | |")
    rows.append(f"| **Median** | **{statistics.median(rates):.3f}** | |")
    return "\n".join(rows), ("; ".join(anomalies) if anomalies else "")


def build_control_table(seed_results: dict[int, dict], key: str) -> tuple[str, list[float], dict[str, list[float]]]:
    """Build a no-switch control's per-instruction markdown table, aggregated across seeds.

    Args:
        seed_results: Per-seed loaded `results.json` dicts.
        key: `"set_a_no_switch_control"` or `"set_b_no_switch_control"`.

    Returns:
        `(markdown_table, all_success_rate_samples, per_instruction_rates)` --
        `all_success_rate_samples` is every (instruction, seed) success rate
        sample flattened, and `per_instruction_rates` maps instruction text
        to its list of per-seed success rates, in first-seed's instruction
        order.

    """
    first_seed = min(seed_results)
    instructions = [row["instruction"] for row in seed_results[first_seed][key]]
    regions = [row["region"] for row in seed_results[first_seed][key]]
    n_episodes = seed_results[first_seed][key][0]["n_episodes"]

    per_instruction_rates: dict[str, list[float]] = {instruction: [] for instruction in instructions}
    for result in seed_results.values():
        for row in result[key]:
            per_instruction_rates[row["instruction"]].append(row["success_rate"])

    rows = [
        f"| Instruction | Region | Mean success rate ({len(seed_results)} seeds x {n_episodes} episodes) |",
        "|---|---|---|",
    ]
    all_samples: list[float] = []
    for instruction, region in zip(instructions, regions, strict=True):
        rates = per_instruction_rates[instruction]
        all_samples.extend(rates)
        rows.append(f'| {instruction} | {region} | {statistics.mean(rates):.3f} |')
    rows.append(
        f"| **Aggregate ({len(all_samples)} samples)** | | "
        f"**mean={statistics.mean(all_samples):.3f} median={statistics.median(all_samples):.3f}** |",
    )
    return "\n".join(rows), all_samples, per_instruction_rates


def build_switch_table(seed_results: dict[int, dict], key: str) -> tuple[str, dict]:
    """Build a switch test's per-pair markdown table (averaged across seeds) and its aggregate stats.

    Pairing is identical across all model seeds (deterministic
    `build_pairs`, independent of `--seed`), so each pair's success rate
    across seeds is a meaningful per-pair number, not noise from different
    episode conditions.

    Args:
        seed_results: Per-seed loaded `results.json` dicts.
        key: `"set_a_switch_episodes"` or `"set_b_switch_episodes"`.

    Returns:
        `(markdown_table, stats)` where `stats` has `task_success_rate`,
        `redirect_success_rate`, `time_to_redirect_samples` (every redirected
        episode's value, across all seeds and pairs), `n_episodes`.

    """
    first_seed = min(seed_results)
    n_pairs = len(seed_results[first_seed][key])

    rows = [
        (
            "| Pair | instr1 -> instr2 | Task success rate (3 seeds) | Redirected (n/3 seeds) | "
            "Time-to-redirect per seed that redirected |"
        ),
        "|---|---|---|---|---|",
    ]
    all_success: list[bool] = []
    all_ttr: list[int] = []
    all_redirected: list[bool] = []
    for pair_index in range(n_pairs):
        pair_successes = []
        pair_ttrs = []
        instr1 = instr2 = None
        for result in seed_results.values():
            episode = result[key][pair_index]
            instr1, instr2 = episode["instr1"], episode["instr2"]
            pair_successes.append(episode["success"])
            all_success.append(episode["success"])
            redirected = episode["time_to_redirect"] is not None
            all_redirected.append(redirected)
            if redirected:
                pair_ttrs.append(episode["time_to_redirect"])
                all_ttr.append(episode["time_to_redirect"])
        ttr_text = ", ".join(str(v) for v in pair_ttrs) if pair_ttrs else "did not redirect (any seed)"
        rows.append(
            f'| {pair_index} | "{instr1}" -> "{instr2}" | '
            f"{sum(pair_successes)}/{len(pair_successes)} | {len(pair_ttrs)}/{len(pair_successes)} | {ttr_text} |",
        )

    stats = {
        "task_success_rate": statistics.mean(1.0 if s else 0.0 for s in all_success),
        "redirect_success_rate": statistics.mean(1.0 if r else 0.0 for r in all_redirected),
        "time_to_redirect_samples": all_ttr,
        "n_episodes": len(all_success),
    }
    return "\n".join(rows), stats


def build_time_to_redirect_summary(label: str, stats: dict) -> str:
    """Build the time-to-redirect distribution summary line for one vocabulary set.

    Args:
        label: "Set A" or "Set B".
        stats: `build_switch_table`'s stats dict.

    Returns:
        A markdown line with the full distribution, not just a mean, per the
        task brief -- computed only over episodes that did redirect.

    """
    samples = stats["time_to_redirect_samples"]
    if not samples:
        return f"**{label}**: no episode redirected -- no time-to-redirect distribution to report."
    return (
        f"**{label}** ({len(samples)}/{stats['n_episodes']} episodes redirected): "
        f"mean={statistics.mean(samples):.2f} median={statistics.median(samples):.1f} "
        f"min={min(samples)} max={max(samples)} all_values={sorted(samples)}"
    )


def render_time_to_redirect_histogram(set_a_samples: list[int], set_b_samples: list[int]) -> Path:
    """Render a side-by-side histogram of Set A/Set B's time-to-redirect distributions.

    Custom chart (not one of `reporting.py`'s pre-built chart functions) --
    precedented by stage 4's `generate_report_attempt3.py`, which also drew
    its own matplotlib figure directly for a chart shape `reporting.py`
    doesn't provide (a distribution plot, not a per-label bar of means).

    Args:
        set_a_samples: Every Set-A episode's time-to-redirect value, for
            episodes that did redirect.
        set_b_samples: Same, for Set B.

    Returns:
        The path the PNG was written to.

    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHARTS_DIR / "time_to_redirect_distribution.png"

    max_value = max([*set_a_samples, *set_b_samples, 1])
    bins = range(0, max_value + 2)

    fig, ax = plt.subplots()
    ax.hist(set_a_samples, bins=bins, alpha=0.6, label=f"Set A (n={len(set_a_samples)})")
    ax.hist(set_b_samples, bins=bins, alpha=0.6, label=f"Set B (n={len(set_b_samples)})")
    ax.set_xlabel("steps from switch to first success (time-to-redirect)")
    ax.set_ylabel("count")
    ax.set_title("Time-to-redirect distribution (redirected episodes only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    """Aggregate all seeds' results and write stage 6's report.md."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    seed_results = load_seed_results(args.seeds)

    sanity_table, sanity_anomalies = build_sanity_table(seed_results)
    set_a_control_table, set_a_control_samples, _ = build_control_table(seed_results, "set_a_no_switch_control")
    set_b_control_table, set_b_control_samples, _ = build_control_table(seed_results, "set_b_no_switch_control")
    set_a_switch_table, set_a_switch_stats = build_switch_table(seed_results, "set_a_switch_episodes")
    set_b_switch_table, set_b_switch_stats = build_switch_table(seed_results, "set_b_switch_episodes")

    set_a_control_mean = statistics.mean(set_a_control_samples)
    set_a_control_median = statistics.median(set_a_control_samples)
    cross_check_delta = abs(set_a_control_mean - STAGE4_SET_A_REFERENCE_MEAN)
    cross_check_passed = (
        cross_check_delta <= STAGE4_APPROX_TOLERANCE
        and abs(set_a_control_median - STAGE4_SET_A_REFERENCE_MEDIAN) < 1e-9
    )
    cross_check_text = (
        f"Set-A no-switch-control mean={set_a_control_mean:.3f}, median={set_a_control_median:.3f} "
        f"vs. stage 4's already-measured 0.571 mean / 1.000 median (identical k=1 NN-lookup mechanism, "
        f"via LiveGoalController instead of stage 4's script calling nearest_neighbor_projection directly). "
        f"{'MATCHES (within tolerance) -- harness wiring confirmed correct before trusting Set B.' if cross_check_passed else 'DOES NOT MATCH -- see Anomalies, do not trust Set B results until resolved.'}"
    )

    ttr_chart = render_time_to_redirect_histogram(
        set_a_switch_stats["time_to_redirect_samples"], set_b_switch_stats["time_to_redirect_samples"],
    )

    # Short labels -- plot_candidate_comparison/plot_multi_seed_success_rate render
    # dict keys verbatim as x-tick labels with no rotation, so long snake_case keys
    # overlap illegibly on a 4-bar chart.
    comparison_results = {
        "A control": set_a_control_samples,
        "A switch": [1.0 if e["success"] else 0.0 for r in seed_results.values() for e in r["set_a_switch_episodes"]],
        "B control": set_b_control_samples,
        "B switch": [1.0 if e["success"] else 0.0 for r in seed_results.values() for e in r["set_b_switch_episodes"]],
    }
    comparison_chart = plot_candidate_comparison(
        comparison_results, out_path=CHARTS_DIR / "control_vs_switch_success_rate.png",
    )

    sanity_chart = plot_multi_seed_success_rate(
        {f"seed_{seed}": [result["literal_sanity_success_rate"]] for seed, result in sorted(seed_results.items())},
        out_path=CHARTS_DIR / "literal_sanity_success_rate.png",
        proof_gate_threshold=0.9,
    )

    metrics_table = f"""### Literal-goal sanity check (reused stage-3 checkpoints, no new training)

{sanity_table}

### Set A no-switch control (stage 4's 14 held-out paraphrases -- also this experiment's sanity cross-check)

{set_a_control_table}

**Cross-check against stage 4:** {cross_check_text}

### Set B no-switch control (7 brand-new phrasings, never used anywhere in this project before)

{set_b_control_table}

### Set A live mid-episode switch test (proof-gate metric)

{set_a_switch_table}

**Set A aggregate:** task_success_rate={set_a_switch_stats['task_success_rate']:.3f} redirect_success_rate={set_a_switch_stats['redirect_success_rate']:.3f} over {set_a_switch_stats['n_episodes']} episodes (14 pairs x 3 seeds)

**Set A time-to-redirect distribution** (computed only over episodes that redirected -- see "did not redirect" episodes counted separately above): {build_time_to_redirect_summary("Set A", set_a_switch_stats)}

### Set B live mid-episode switch test (proof-gate metric)

{set_b_switch_table}

**Set B aggregate:** task_success_rate={set_b_switch_stats['task_success_rate']:.3f} redirect_success_rate={set_b_switch_stats['redirect_success_rate']:.3f} over {set_b_switch_stats['n_episodes']} episodes (7 pairs x 3 seeds)

**Set B time-to-redirect distribution** (computed only over episodes that redirected): {build_time_to_redirect_summary("Set B", set_b_switch_stats)}
"""

    # write_report renders these paths verbatim into report.md's markdown links --
    # relativize to EXPERIMENT_DIR so the report stays portable (matches every
    # earlier stage's report.md, e.g. stage 5's "charts/sanity_check_success_rate.png"
    # rather than an absolute path baked in from wherever this script happened to run).
    chart_paths = [path.relative_to(EXPERIMENT_DIR) for path in (sanity_chart, comparison_chart, ttr_chart)]
    raw_output_paths = [
        (EXPERIMENT_DIR / "runs" / f"seed_{seed}" / "stdout.log").relative_to(EXPERIMENT_DIR)
        for seed in sorted(seed_results)
    ]

    anomalies_parts = []
    if sanity_anomalies:
        anomalies_parts.append(sanity_anomalies)
    else:
        anomalies_parts.append("No literal-sanity-check collapse observed on any seed.")
    if not cross_check_passed:
        anomalies_parts.append(
            "Set-A no-switch-control cross-check against stage 4 did not land within tolerance -- see the "
            "cross-check line above; treat Set B's results with caution until this is resolved."
        )
    anomalies_parts.append(
        "This experiment's `rollout_with_goal_switch_timed` (in `live_regoal_eval.py`) is a local, "
        "instrumented duplicate of `midepisode_regoal.rollout_with_goal_switch` -- the reusable function "
        "doesn't expose per-step success timing, which stage 6's time-to-redirect metric needs. Flagged as "
        "a candidate for promotion into `midepisode_regoal.py` if a future stage needs the same timing data "
        "(logged in `.claude/findings.md`)."
    )
    anomalies_text = " ".join(anomalies_parts)

    known_risks_note = (
        "**Nearest-neighbor lookup's generalization ceiling is bounded by reference-vocabulary coverage "
        "density**: this is exactly what Set B tests directly (brand-new phrasings never in the 84-sentence "
        "reference) -- see Set B's no-switch-control table for whether coverage gaps show up on this "
        "specific set of 7 new phrasings. **Non-stationarity / embedding noise interacting with a goal-swap "
        "(flagged explicitly in ROADMAP.md as untested going into stage 6)**: this experiment is the first "
        "direct measurement -- see the switch-vs-no-switch-control comparison chart for whether live "
        "language-pipeline goal-swapping degrades relative to the no-switch baseline. **Region-vs-point "
        "ground truth**: applied from the start here (compute_region_centroid, not a resampled point), per "
        "the lesson from stage 3. **SAC deterministic-eval collapse**: checked via the literal sanity table "
        "above before trusting any result."
    )

    report_path = write_report(
        stage=6,
        title="Live English interface",
        seeds=args.seeds,
        candidates=["set_a_no_switch_control", "set_a_switch", "set_b_no_switch_control", "set_b_switch"],
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
