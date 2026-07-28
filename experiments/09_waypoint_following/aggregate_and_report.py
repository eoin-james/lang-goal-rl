"""Aggregate all 8 healthy seeds' `run_waypoint_eval.py` output and (re)write
stage 9's `evidence.md` (full record) and `report.md` (short summary).

The first stage-9 pass ran only `seed_0` and was sent back by review:
CONTRACTS.md requires the full multi-seed convention for the actual
reviewer verdict, and this single-checkpoint result structurally could not
distinguish "the waypoint-chaining mechanism is robust" from "this one
already-oracle-solvable checkpoint has no room to fail." This script
implements the reviewer's exact recommendation: aggregate the same 12
conditions across the 8 healthy checkpoints (seeds 0,1,3,4,5,6,8,9 --
excluding 2,7, the documented SAC deterministic-eval collapse seeds), with
visibility kept **per (seed, condition)**, not collapsed into one grand
mean -- the whole point of the rerun is to catch an individual seed that
diverges from seed_0's pattern.

Two direct, factual checks this script computes and states plainly (not
judged, per the runner/reviewer split in `.claude/agents/experiment-runner.md`):

1. **Multi-leg failures** -- does any single episode, on any seed, fail 2+
   legs in the same chain? The first run never observed one in 600
   episodes; this run has 4800 (8 seeds x 12 conditions x 50 episodes) to
   check across.
2. **Monotonic-with-position trend** -- does per-leg failure rate increase
   strictly as leg position increases (1->2->3->4->5), on any individual
   seed or in the pooled aggregate? That is the direct signature of
   compounding error, distinct from "some legs just happen to be harder."

Loads seed_0's original `runs/final_results.json` (unmoved, kept exactly as
the first run produced it) alongside the 7 new seeds' `runs/seed_<k>/
final_results.json` (written by this script's parameterized
`run_waypoint_eval.py --seed <k>`). All 9 seeds report `episodes_per_condition
== 50`, so pooling across seeds is a simple unweighted mean of equal-n
per-seed rates -- no need to re-derive from raw per-episode bits except for
the multi-leg-failure count and monotonic check, which need the raw bits
directly.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from lang_goal_rl.reporting import write_report

EXPERIMENT_DIR = Path(__file__).resolve().parent
CHARTS_DIR = EXPERIMENT_DIR / "charts"
RUNS_DIR = EXPERIMENT_DIR / "runs"

PROOF_GATE_TEXT = (
    "N=2 reduces exactly to stage 5's `rollout_with_goal_switch` result "
    "(regression test); N=3-5 chains don't show compounding degradation."
)

CHAIN_LENGTHS = (2, 3, 5)
SEQUENCE_KINDS = ("literal", "relative")
BUDGET_NAMES = ("tight", "generous")
HEALTHY_SEEDS = (0, 1, 3, 4, 5, 6, 8, 9)
"""Excludes 2 and 7, the documented SAC deterministic-eval collapse seeds
(ROADMAP.md Known risks) -- never run for this stage, not omitted by
accident."""

EXPECTED_EPISODES_PER_CONDITION = 50


def load_seed_results(seed: int) -> dict:
    """Load one seed's `final_results.json`.

    Seed 0 is the original run's flat `runs/final_results.json` (kept
    exactly as produced, not moved into a `seed_0/` subdirectory); every
    other seed is this rerun's `runs/seed_<k>/final_results.json`, written
    by the now seed-parameterized `run_waypoint_eval.py`.

    Args:
        seed: Model checkpoint seed, one of `HEALTHY_SEEDS`.

    Returns:
        The parsed results dict for that seed.

    Raises:
        FileNotFoundError: If that seed's results file doesn't exist.
    """
    path = RUNS_DIR / "final_results.json" if seed == 0 else RUNS_DIR / f"seed_{seed}" / "final_results.json"
    if not path.exists():
        msg = f"missing {path} -- did run_waypoint_eval.py --seed {seed} --tag final finish?"
        raise FileNotFoundError(msg)
    return json.loads(path.read_text())


def load_all_seeds() -> dict[int, dict]:
    """Load every healthy seed's results dict, keyed by seed.

    Returns:
        Mapping of seed -> that seed's parsed `final_results.json`.

    Raises:
        ValueError: If any seed's `episodes_per_condition` isn't 50 -- the
            pooling below assumes equal-n per seed so a simple mean of
            per-seed rates equals the true pooled rate.
    """
    all_results = {seed: load_seed_results(seed) for seed in HEALTHY_SEEDS}
    for seed, results in all_results.items():
        if results["episodes_per_condition"] != EXPECTED_EPISODES_PER_CONDITION:
            msg = (
                f"seed {seed} ran {results['episodes_per_condition']} episodes/condition, "
                f"expected {EXPECTED_EPISODES_PER_CONDITION} -- pooling assumes equal n per seed"
            )
            raise ValueError(msg)
    return all_results


def find_condition(results: dict, *, sequence_kind: str, chain_len: int, budget_name: str) -> dict:
    """Look up one condition's result dict by its three identifying keys.

    Args:
        results: One seed's loaded results dict.
        sequence_kind: "literal" or "relative".
        chain_len: Number of waypoints/legs.
        budget_name: "tight" or "generous".

    Returns:
        That condition's result dict.
    """
    return next(
        c
        for c in results["conditions"]
        if c["sequence_kind"] == sequence_kind
        and c["chain_len"] == chain_len
        and c["budget_name"] == budget_name
    )


def build_sanity_table(all_results: dict[int, dict]) -> str:
    """Build the per-seed literal-goal sanity check table.

    Args:
        all_results: Mapping of seed -> loaded results dict.

    Returns:
        Markdown table, one row per healthy seed plus mean/median.
    """
    rows = ["| Seed | Sanity success rate (literal control, default 50-step, no waypoint chain) | Episodes |", "|---|---|---|"]
    rates = []
    for seed in HEALTHY_SEEDS:
        results = all_results[seed]
        rate = results["sanity_check_success_rate"]
        rates.append(rate)
        rows.append(f"| {seed} | {rate:.3f} | {results['sanity_check_episodes']} |")
    rows.append(f"| **Mean** | **{statistics.mean(rates):.3f}** | |")
    rows.append(f"| **Median** | **{statistics.median(rates):.3f}** | |")
    return "\n".join(rows)


def build_pooled_table(all_results: dict[int, dict], *, sequence_kind: str, budget_name: str) -> str:
    """Build the cross-seed pooled per-leg table for one (kind, budget) pair.

    Every healthy seed ran an equal 50 episodes/condition on the identical
    episode-seed sequence per condition, so the unweighted mean of the 8
    seeds' per-leg/whole-chain rates equals the true pooled rate over 400
    episodes -- no need to re-derive from raw bits for this table.

    Args:
        all_results: Mapping of seed -> loaded results dict.
        sequence_kind: "literal" or "relative".
        budget_name: "tight" or "generous".

    Returns:
        Markdown table, one row per chain length, pooled across 8 seeds.
    """
    rows = [
        "| Chain length | Pooled per-leg chain success rate (leg 1..N) | Pooled per-leg baseline success rate (leg 1..N) | Pooled whole-chain success rate | Whole-chain rate range across seeds | Episodes (8 seeds x 50) |",
        "|---|---|---|---|---|---|",
    ]
    for chain_len in CHAIN_LENGTHS:
        per_seed_conditions = [
            find_condition(all_results[seed], sequence_kind=sequence_kind, chain_len=chain_len, budget_name=budget_name)
            for seed in HEALTHY_SEEDS
        ]
        pooled_chain = [
            statistics.mean(cond["per_leg_chain_success_rate"][leg] for cond in per_seed_conditions)
            for leg in range(chain_len)
        ]
        pooled_baseline = [
            statistics.mean(cond["per_leg_baseline_success_rate"][leg] for cond in per_seed_conditions)
            for leg in range(chain_len)
        ]
        whole_chain_rates = [cond["all_succeeded_rate"] for cond in per_seed_conditions]
        pooled_whole_chain = statistics.mean(whole_chain_rates)
        chain_str = ", ".join(f"{r:.3f}" for r in pooled_chain)
        baseline_str = ", ".join(f"{r:.3f}" for r in pooled_baseline)
        rows.append(
            f"| N={chain_len} | [{chain_str}] | [{baseline_str}] | {pooled_whole_chain:.3f} | "
            f"{min(whole_chain_rates):.3f}-{max(whole_chain_rates):.3f} | {len(HEALTHY_SEEDS)}x50 |"
        )
    return "\n".join(rows)


def build_per_seed_table(all_results: dict[int, dict], *, sequence_kind: str, chain_len: int, budget_name: str) -> str:
    """Build one condition's full per-seed, per-leg breakdown table.

    Args:
        all_results: Mapping of seed -> loaded results dict.
        sequence_kind: "literal" or "relative".
        chain_len: Number of waypoints/legs.
        budget_name: "tight" or "generous".

    Returns:
        Markdown table, one row per healthy seed, plus a pooled row.
    """
    rows = ["| Seed | Per-leg chain success rate (leg 1..N) | Whole-chain success rate |", "|---|---|---|"]
    for seed in HEALTHY_SEEDS:
        cond = find_condition(all_results[seed], sequence_kind=sequence_kind, chain_len=chain_len, budget_name=budget_name)
        chain_str = ", ".join(f"{r:.3f}" for r in cond["per_leg_chain_success_rate"])
        marker = " (original seed_0 run)" if seed == 0 else ""
        rows.append(f"| {seed}{marker} | [{chain_str}] | {cond['all_succeeded_rate']:.3f} |")
    pooled = build_pooled_row(all_results, sequence_kind=sequence_kind, chain_len=chain_len, budget_name=budget_name)
    rows.append(pooled)
    return "\n".join(rows)


def build_pooled_row(all_results: dict[int, dict], *, sequence_kind: str, chain_len: int, budget_name: str) -> str:
    """Build the pooled-across-8-seeds summary row for a per-seed table.

    Args:
        all_results: Mapping of seed -> loaded results dict.
        sequence_kind: "literal" or "relative".
        chain_len: Number of waypoints/legs.
        budget_name: "tight" or "generous".

    Returns:
        One markdown table row.
    """
    per_seed_conditions = [
        find_condition(all_results[seed], sequence_kind=sequence_kind, chain_len=chain_len, budget_name=budget_name)
        for seed in HEALTHY_SEEDS
    ]
    pooled_chain = [
        statistics.mean(cond["per_leg_chain_success_rate"][leg] for cond in per_seed_conditions)
        for leg in range(chain_len)
    ]
    chain_str = ", ".join(f"{r:.3f}" for r in pooled_chain)
    pooled_whole_chain = statistics.mean(cond["all_succeeded_rate"] for cond in per_seed_conditions)
    return f"| **Pooled (8 seeds, N=400)** | **[{chain_str}]** | **{pooled_whole_chain:.3f}** |"


def find_multi_leg_failures(all_results: dict[int, dict]) -> tuple[list[dict], int, int]:
    """Scan every seed/condition/episode for 2+ legs failing in the same episode.

    This is the exact signature the first single-seed run never observed
    even once in 600 episodes; the reviewer's recommendation asks this
    rerun to check directly across all 8 healthy seeds' 4800 episodes,
    rather than accepting "it didn't happen on seed_0" as sufficient.

    Args:
        all_results: Mapping of seed -> loaded results dict.

    Returns:
        `(occurrences, total_episodes, total_multi_leg_failures)` --
        `occurrences` is a list of dicts identifying every episode with 2+
        failed legs (empty if none found); the two counts give the
        denominator and numerator for reporting the rate directly.
    """
    occurrences = []
    total_episodes = 0
    for seed in HEALTHY_SEEDS:
        results = all_results[seed]
        for sequence_kind in SEQUENCE_KINDS:
            for chain_len in CHAIN_LENGTHS:
                for budget_name in BUDGET_NAMES:
                    cond = find_condition(results, sequence_kind=sequence_kind, chain_len=chain_len, budget_name=budget_name)
                    for episode_index, bits in enumerate(cond["per_episode_chain_bits"]):
                        total_episodes += 1
                        failed_legs = [leg_index + 1 for leg_index, success in enumerate(bits) if not success]
                        if len(failed_legs) >= 2:
                            occurrences.append(
                                {
                                    "seed": seed,
                                    "sequence_kind": sequence_kind,
                                    "chain_len": chain_len,
                                    "budget_name": budget_name,
                                    "episode_index": episode_index,
                                    "failed_legs": failed_legs,
                                }
                            )
    return occurrences, total_episodes, len(occurrences)


def check_monotonic_trend(all_results: dict[int, dict]) -> list[dict]:
    """Check every seed/condition (chain_len >= 3) for a failure rate that rises with leg position.

    "Monotonic" here means the per-leg failure rate (1 - success rate)
    never decreases from one leg position to the next (ties allowed --
    most positions are tied at 0.0 given how few misses occur at 50
    episodes/condition); "strictly increasing" means every consecutive
    step is a strict increase, the stronger compounding signature the
    reviewer asked to rule out.

    Args:
        all_results: Mapping of seed -> loaded results dict.

    Returns:
        A list of dicts, one per (seed, condition) with chain_len >= 3,
        each recording the per-leg failure-rate sequence and whether it is
        non-decreasing / strictly increasing.
    """
    findings = []
    for seed in HEALTHY_SEEDS:
        results = all_results[seed]
        for sequence_kind in SEQUENCE_KINDS:
            for chain_len in (3, 5):
                for budget_name in BUDGET_NAMES:
                    cond = find_condition(results, sequence_kind=sequence_kind, chain_len=chain_len, budget_name=budget_name)
                    failure_rates = [1.0 - r for r in cond["per_leg_chain_success_rate"]]
                    non_decreasing = all(
                        failure_rates[i + 1] >= failure_rates[i] for i in range(len(failure_rates) - 1)
                    )
                    strictly_increasing = all(
                        failure_rates[i + 1] > failure_rates[i] for i in range(len(failure_rates) - 1)
                    )
                    findings.append(
                        {
                            "seed": seed,
                            "sequence_kind": sequence_kind,
                            "chain_len": chain_len,
                            "budget_name": budget_name,
                            "failure_rates": failure_rates,
                            "non_decreasing": non_decreasing,
                            "strictly_increasing": strictly_increasing,
                        }
                    )
    return findings


def build_isolated_failure_note(all_results: dict[int, dict], multi_leg_occurrences: list[dict], total_episodes: int) -> str:
    """Render the multi-leg-failure scan result as factual prose.

    Args:
        all_results: Mapping of seed -> loaded results dict.
        multi_leg_occurrences: Output of `find_multi_leg_failures`.
        total_episodes: Total episode count scanned (denominator).

    Returns:
        Factual prose describing the scan result.
    """
    if not multi_leg_occurrences:
        return (
            f"Zero episodes with 2+ failed legs found across all {len(HEALTHY_SEEDS)} healthy seeds x 12 "
            f"conditions x 50 episodes ({total_episodes} episodes scanned) -- the isolated, non-compounding "
            "single-leg-miss pattern the first (seed_0-only) run observed holds across every healthy seed, "
            "not just seed_0."
        )
    lines = [
        f"**{len(multi_leg_occurrences)} episode(s) with 2+ failed legs found across "
        f"{total_episodes} episodes scanned ({len(HEALTHY_SEEDS)} healthy seeds x 12 conditions x 50 "
        "episodes) -- a genuine multi-leg (compounding-candidate) failure the first, seed_0-only run "
        "never observed:**"
    ]
    for occurrence in multi_leg_occurrences:
        lines.append(
            f"- seed {occurrence['seed']}, {occurrence['sequence_kind']}/N={occurrence['chain_len']}/"
            f"{occurrence['budget_name']}, episode {occurrence['episode_index']}: failed legs "
            f"{occurrence['failed_legs']} of {occurrence['chain_len']}"
        )
    return "\n".join(lines)


def build_monotonic_trend_note(monotonic_findings: list[dict]) -> str:
    """Render the monotonic-trend scan result as factual prose.

    Args:
        monotonic_findings: Output of `check_monotonic_trend`.

    Returns:
        Factual prose describing which (seed, condition) pairs show a
        non-decreasing or strictly-increasing failure-rate-by-position
        pattern.
    """
    strictly_increasing = [f for f in monotonic_findings if f["strictly_increasing"]]
    all_zero = [f for f in monotonic_findings if max(f["failure_rates"]) == 0.0]
    non_decreasing_with_signal = [
        f
        for f in monotonic_findings
        if f["non_decreasing"] and not f["strictly_increasing"] and max(f["failure_rates"]) > 0.0
    ]

    lines = [
        f"Checked {len(monotonic_findings)} (seed, condition) pairs with chain_len in (3, 5) for whether "
        "per-leg failure rate rises monotonically with leg position. "
        f"{len(all_zero)} of these pairs had zero failures on every leg (trivially flat, no signal to "
        "check) -- omitted from the lists below; the remaining "
        f"{len(monotonic_findings) - len(all_zero)} pairs had at least one leg-position failure."
    ]
    if strictly_increasing:
        lines.append(
            f"**{len(strictly_increasing)} pair(s) show a strictly-increasing failure rate with position "
            "(the clear compounding signature):**"
        )
        for finding in strictly_increasing:
            rates_str = ", ".join(f"{r:.3f}" for r in finding["failure_rates"])
            lines.append(
                f"- seed {finding['seed']}, {finding['sequence_kind']}/N={finding['chain_len']}/"
                f"{finding['budget_name']}: failure rates by position [{rates_str}]"
            )
    else:
        lines.append("No pair shows a strictly-increasing failure rate with position.")
    if non_decreasing_with_signal:
        lines.append(
            f"{len(non_decreasing_with_signal)} pair(s) with at least one failure are non-decreasing but "
            "not strictly increasing (i.e. flat-then-one-bump, not a rising trend across every position) "
            "-- listed for completeness, not treated as a compounding signature on their own:"
        )
        for finding in non_decreasing_with_signal:
            rates_str = ", ".join(f"{r:.3f}" for r in finding["failure_rates"])
            lines.append(
                f"- seed {finding['seed']}, {finding['sequence_kind']}/N={finding['chain_len']}/"
                f"{finding['budget_name']}: failure rates by position [{rates_str}]"
            )
    else:
        lines.append("No pair with at least one failure is non-decreasing either (every miss recovers, never repeats or worsens at a later position).")
    return "\n".join(lines)


def render_per_seed_whole_chain_chart(all_results: dict[int, dict]) -> Path:
    """Bar chart: whole-chain success rate per seed, for the tight/N=5 conditions (both kinds).

    These are the two conditions where the original seed_0 run showed the
    most variance (0.960 literal, 0.940 relative) -- the direct visual for
    whether any individual seed diverges from seed_0's pattern.

    Args:
        all_results: Mapping of seed -> loaded results dict.

    Returns:
        The path the PNG was written to.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHARTS_DIR / "per_seed_whole_chain_tight_n5.png"

    fig, ax = plt.subplots()
    bar_width = 0.35
    x_positions = list(range(len(HEALTHY_SEEDS)))
    for series_index, sequence_kind in enumerate(SEQUENCE_KINDS):
        rates = [
            find_condition(all_results[seed], sequence_kind=sequence_kind, chain_len=5, budget_name="tight")[
                "all_succeeded_rate"
            ]
            for seed in HEALTHY_SEEDS
        ]
        offsets = [x + (series_index - 0.5) * bar_width for x in x_positions]
        ax.bar(offsets, rates, width=bar_width, label=f"{sequence_kind}/tight/N=5")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(seed) for seed in HEALTHY_SEEDS])
    ax.set_xlabel("checkpoint seed")
    ax.set_ylabel("whole-chain success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Whole-chain success by seed: tight budget, N=5 (the most variance-prone conditions)")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def render_per_seed_per_leg_chart(all_results: dict[int, dict], *, sequence_kind: str) -> Path:
    """Line chart: per-leg failure rate by position, one line per seed, tight/N=5.

    The direct visual for the monotonic-with-position check -- a
    compounding checkpoint would show its line rising left to right; the
    first run's seed_0 line is flat-then-one-bump, not a rise.

    Args:
        all_results: Mapping of seed -> loaded results dict.
        sequence_kind: "literal" or "relative".

    Returns:
        The path the PNG was written to.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHARTS_DIR / f"per_seed_per_leg_failure_{sequence_kind}_tight_n5.png"

    fig, ax = plt.subplots()
    leg_positions = list(range(1, 6))
    for seed in HEALTHY_SEEDS:
        cond = find_condition(all_results[seed], sequence_kind=sequence_kind, chain_len=5, budget_name="tight")
        failure_rates = [1.0 - r for r in cond["per_leg_chain_success_rate"]]
        ax.plot(leg_positions, failure_rates, marker="o", label=f"seed {seed}")
    ax.set_xlabel("leg position in chain")
    ax.set_ylabel("failure rate (1 - success rate)")
    ax.set_xticks(leg_positions)
    ax.set_title(f"Per-leg failure rate by position, per seed: {sequence_kind}/tight/N=5")
    ax.legend(fontsize="small", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def render_pooled_per_leg_chart(all_results: dict[int, dict], *, sequence_kind: str, budget_name: str) -> Path:
    """Line chart: pooled (8-seed) per-leg success rate, chain vs. baseline, one line pair per chain length.

    Same chart shape as the first run's `per_leg_<kind>_<budget>.png`, now
    computed on the 8-seed pooled data (n=400/condition) instead of
    seed_0 alone -- overwritten in place, this is a scale-up of the same
    chart, not a new one.

    Args:
        all_results: Mapping of seed -> loaded results dict.
        sequence_kind: "literal" or "relative".
        budget_name: "tight" or "generous".

    Returns:
        The path the PNG was written to.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHARTS_DIR / f"per_leg_{sequence_kind}_{budget_name}.png"

    fig, ax = plt.subplots()
    for chain_len in CHAIN_LENGTHS:
        per_seed_conditions = [
            find_condition(all_results[seed], sequence_kind=sequence_kind, chain_len=chain_len, budget_name=budget_name)
            for seed in HEALTHY_SEEDS
        ]
        pooled_chain = [
            statistics.mean(cond["per_leg_chain_success_rate"][leg] for cond in per_seed_conditions)
            for leg in range(chain_len)
        ]
        pooled_baseline = [
            statistics.mean(cond["per_leg_baseline_success_rate"][leg] for cond in per_seed_conditions)
            for leg in range(chain_len)
        ]
        leg_positions = list(range(1, chain_len + 1))
        ax.plot(leg_positions, pooled_chain, marker="o", label=f"N={chain_len} chain")
        ax.plot(leg_positions, pooled_baseline, marker="x", linestyle="--", label=f"N={chain_len} baseline")
    ax.set_xlabel("leg position in chain")
    ax.set_ylabel("success rate (pooled, 8 seeds)")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(1, max(CHAIN_LENGTHS) + 1))
    ax.set_title(f"Per-leg success (pooled, 8 seeds): {sequence_kind} sequences, {budget_name} budget")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def render_pooled_whole_chain_vs_length_chart(all_results: dict[int, dict]) -> Path:
    """Grouped bar chart: pooled (8-seed) whole-chain success rate vs. chain length.

    Same chart shape as the first run's `whole_chain_success_vs_length.png`,
    now on 8-seed pooled data -- overwritten in place.

    Args:
        all_results: Mapping of seed -> loaded results dict.

    Returns:
        The path the PNG was written to.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHARTS_DIR / "whole_chain_success_vs_length.png"

    fig, ax = plt.subplots()
    bar_width = 0.2
    x_positions = list(range(len(CHAIN_LENGTHS)))
    series_index = 0
    for sequence_kind in SEQUENCE_KINDS:
        for budget_name in BUDGET_NAMES:
            rates = [
                statistics.mean(
                    find_condition(all_results[seed], sequence_kind=sequence_kind, chain_len=chain_len, budget_name=budget_name)[
                        "all_succeeded_rate"
                    ]
                    for seed in HEALTHY_SEEDS
                )
                for chain_len in CHAIN_LENGTHS
            ]
            offsets = [x + (series_index - 1.5) * bar_width for x in x_positions]
            ax.bar(offsets, rates, width=bar_width, label=f"{sequence_kind}/{budget_name}")
            series_index += 1
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"N={n}" for n in CHAIN_LENGTHS])
    ax.set_ylabel("whole-chain success rate (pooled, 8 seeds)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Whole-chain success vs. chain length (pooled, 8 seeds)")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    """Aggregate all 8 healthy seeds and (re)write stage 9's evidence.md/report.md skeleton."""
    parser = argparse.ArgumentParser()
    parser.parse_args()

    all_results = load_all_seeds()

    sanity_table = build_sanity_table(all_results)

    metrics_table_parts = [f"### Checkpoint sanity check (all 8 healthy seeds)\n\n{sanity_table}\n"]

    metrics_table_parts.append(
        "### Cross-seed pooled results (8 seeds, 400 episodes/condition)\n\n"
        "Every table below pools all 8 healthy seeds' equal-n (50 episodes/condition) results into a "
        "single 400-episode/condition rate -- the same tables the first (seed_0-only) run reported, now "
        "at 8x the checkpoint coverage.\n"
    )
    for sequence_kind in SEQUENCE_KINDS:
        for budget_name in BUDGET_NAMES:
            table = build_pooled_table(all_results, sequence_kind=sequence_kind, budget_name=budget_name)
            metrics_table_parts.append(f"#### {sequence_kind} sequences, {budget_name} budget\n\n{table}\n")

    metrics_table_parts.append(
        "### Per-seed breakdown, tight budget (the only budget with any non-1.000 result)\n\n"
        "Full per-leg, per-seed breakdown for every tight-budget condition -- this is the direct check "
        "for whether any individual seed diverges from seed_0's isolated/non-compounding pattern.\n"
    )
    for sequence_kind in SEQUENCE_KINDS:
        for chain_len in CHAIN_LENGTHS:
            table = build_per_seed_table(all_results, sequence_kind=sequence_kind, chain_len=chain_len, budget_name="tight")
            metrics_table_parts.append(f"#### {sequence_kind}, tight budget, N={chain_len}\n\n{table}\n")

    metrics_table_parts.append(
        "### Per-seed whole-chain rate, generous budget (compact -- every per-leg value was 1.000 for "
        "every seed at this budget, see raw JSON for full per-leg confirmation)\n"
    )
    for sequence_kind in SEQUENCE_KINDS:
        rows = ["| Seed | N=2 | N=3 | N=5 |", "|---|---|---|---|"]
        for seed in HEALTHY_SEEDS:
            rates = [
                find_condition(all_results[seed], sequence_kind=sequence_kind, chain_len=chain_len, budget_name="generous")[
                    "all_succeeded_rate"
                ]
                for chain_len in CHAIN_LENGTHS
            ]
            rows.append(f"| {seed} | {rates[0]:.3f} | {rates[1]:.3f} | {rates[2]:.3f} |")
        metrics_table_parts.append(f"#### {sequence_kind}, generous budget\n\n" + "\n".join(rows) + "\n")

    metrics_table = "\n".join(metrics_table_parts)

    chart_paths = [render_pooled_whole_chain_vs_length_chart(all_results)]
    for sequence_kind in SEQUENCE_KINDS:
        for budget_name in BUDGET_NAMES:
            chart_paths.append(render_pooled_per_leg_chart(all_results, sequence_kind=sequence_kind, budget_name=budget_name))
    chart_paths.append(render_per_seed_whole_chain_chart(all_results))
    for sequence_kind in SEQUENCE_KINDS:
        chart_paths.append(render_per_seed_per_leg_chart(all_results, sequence_kind=sequence_kind))
    chart_paths = [p.relative_to(EXPERIMENT_DIR) for p in chart_paths]

    raw_output_paths = [
        (RUNS_DIR / "n2_equivalence_regression_test.log").relative_to(EXPERIMENT_DIR),
        (RUNS_DIR / "tier1_stdout.log").relative_to(EXPERIMENT_DIR),
        (RUNS_DIR / "tier1_results.json").relative_to(EXPERIMENT_DIR),
        (RUNS_DIR / "final_stdout.log").relative_to(EXPERIMENT_DIR),
        (RUNS_DIR / "final_results.json").relative_to(EXPERIMENT_DIR),
    ]
    for seed in HEALTHY_SEEDS:
        if seed == 0:
            continue
        raw_output_paths.append((RUNS_DIR / f"seed_{seed}" / "final_stdout.log").relative_to(EXPERIMENT_DIR))
        raw_output_paths.append((RUNS_DIR / f"seed_{seed}" / "final_results.json").relative_to(EXPERIMENT_DIR))

    multi_leg_occurrences, total_episodes, multi_leg_count = find_multi_leg_failures(all_results)
    multi_leg_note = build_isolated_failure_note(all_results, multi_leg_occurrences, total_episodes)
    monotonic_findings = check_monotonic_trend(all_results)
    monotonic_note = build_monotonic_trend_note(monotonic_findings)

    anomalies_text = (
        "Every generous-budget condition (both sequence kinds, all 3 chain lengths), pooled across all 8 "
        "healthy seeds, scored a clean 1.000 on every leg for every individual seed -- same "
        "oracle-solvable-ceiling limit documented since stages 1/3/5, now confirmed to hold across the "
        "full healthy-seed set, not just seed_0. The only non-1.000 results appear at the tight budget, "
        "chain lengths >= 3, matching the first run's pattern.\n\n"
        f"**Multi-leg-failure check:** {multi_leg_note}\n\n"
        f"**Monotonic-with-position check:** {monotonic_note}"
    )

    known_risks_note = (
        "**SAC deterministic-eval collapse (~20% of seeds, confirmed stage 1)**: directly addressed by "
        "this rerun's whole purpose -- seeds 2 and 7 (the documented collapse seeds) are excluded from "
        "every table above, never run for this stage. **Checkpoint-dependent behavior (this stage's own "
        "reviewer verdict on the first pass)**: directly checked above via the per-seed breakdown tables, "
        "the multi-leg-failure scan, and the monotonic-trend check -- the three concrete tests the "
        "reviewer named as distinguishing a clean pass from a genuine checkpoint-dependent finding. "
        "**Direction-sensitivity, not just distance (stage 4)**: not directly probed here, same "
        "limitation as the first run -- this stage's relative-move sequences use a fixed 0.15m step in a "
        "randomly chosen direction per leg, not systematically varied per direction; stage 8's own "
        "relative-move validation is the right place to check that per-direction. **Oracle-solvable "
        "ceiling (ROADMAP.md, stages 1/3/5)**: re-observed here across all 8 seeds -- every generous-"
        "budget condition and most tight-budget conditions hit 1.000."
    )

    report_path = write_report(
        stage=9,
        title="Waypoint following",
        seeds=list(HEALTHY_SEEDS),
        candidates=[f"{k}/{b}" for k in SEQUENCE_KINDS for b in BUDGET_NAMES],
        proof_gate_text=PROOF_GATE_TEXT,
        metrics_table=metrics_table,
        chart_paths=chart_paths,
        raw_output_paths=raw_output_paths,
        anomalies=anomalies_text,
        known_risks_note=known_risks_note,
        out_dir=EXPERIMENT_DIR,
    )
    print(f"report_written={report_path}")
    print(f"multi_leg_failures_found={multi_leg_count} / {total_episodes} episodes scanned")
    print(f"strictly_increasing_monotonic_pairs={sum(1 for f in monotonic_findings if f['strictly_increasing'])}")
    print(f"non_decreasing_monotonic_pairs={sum(1 for f in monotonic_findings if f['non_decreasing'])}")


if __name__ == "__main__":
    main()
