"""Aggregate `run_waypoint_eval.py`'s output and write stage 9's `report.md`.

Reads `runs/<tag>_results.json` (this experiment's single checkpoint, no
per-seed sharding -- see `run_waypoint_eval.py`'s module docstring for why),
builds the per-leg-position success tables the stage-9 proof gate needs
(chain vs. budget-matched baseline, broken out by leg position, never
collapsed into one "did it finish" number), renders charts, and calls
`lang_goal_rl.reporting.write_report(...)` -- the required deliverable per
`.claude/agents/experiment-runner.md`. Run after `run_waypoint_eval.py`'s
final-tier pass completes.

Two custom chart types beyond `reporting.py`'s pre-built functions (same
precedent stage 4/6 already established for chart shapes the shared helpers
don't provide): a per-leg-position line chart (chain vs. baseline, one line
per chain length) and a whole-chain-success-vs-chain-length grouped bar
chart -- the direct visual for "does success degrade as the chain gets
longer," which is this stage's central question.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from lang_goal_rl.reporting import write_report

EXPERIMENT_DIR = Path(__file__).resolve().parent
CHARTS_DIR = EXPERIMENT_DIR / "charts"

PROOF_GATE_TEXT = (
    "N=2 reduces exactly to stage 5's `rollout_with_goal_switch` result "
    "(regression test); N=3-5 chains don't show compounding degradation."
)

CHAIN_LENGTHS = (2, 3, 5)
SEQUENCE_KINDS = ("literal", "relative")
BUDGET_NAMES = ("tight", "generous")


def load_results(tag: str) -> dict:
    """Load `runs/<tag>_results.json`.

    Args:
        tag: Tier tag used by `run_waypoint_eval.py` (e.g. "tier1", "final").

    Returns:
        The parsed results dict.

    Raises:
        FileNotFoundError: If that tag's results file doesn't exist.
    """
    results_path = EXPERIMENT_DIR / "runs" / f"{tag}_results.json"
    if not results_path.exists():
        msg = f"missing {results_path} -- did run_waypoint_eval.py --tag {tag} finish?"
        raise FileNotFoundError(msg)
    return json.loads(results_path.read_text())


def find_condition(results: dict, *, sequence_kind: str, chain_len: int, budget_name: str) -> dict:
    """Look up one condition's result dict by its three identifying keys.

    Args:
        results: A loaded `run_waypoint_eval.py` results dict.
        sequence_kind: "literal" or "relative".
        chain_len: Number of waypoints/legs.
        budget_name: "tight" or "generous".

    Returns:
        That condition's result dict.

    Raises:
        StopIteration: If no matching condition was run.
    """
    return next(
        c
        for c in results["conditions"]
        if c["sequence_kind"] == sequence_kind
        and c["chain_len"] == chain_len
        and c["budget_name"] == budget_name
    )


def build_per_leg_table(results: dict, *, sequence_kind: str, budget_name: str) -> str:
    """Build one markdown table: rows = chain lengths, columns = per-leg chain vs. baseline success.

    Args:
        results: A loaded `run_waypoint_eval.py` results dict.
        sequence_kind: "literal" or "relative".
        budget_name: "tight" or "generous".

    Returns:
        Markdown table string, one row per chain length tested.
    """
    rows = ["| Chain length | Per-leg chain success rate (leg 1..N) | Per-leg baseline success rate (leg 1..N) | Whole-chain success rate | Episodes |", "|---|---|---|---|---|"]
    for chain_len in CHAIN_LENGTHS:
        cond = find_condition(results, sequence_kind=sequence_kind, chain_len=chain_len, budget_name=budget_name)
        chain_str = ", ".join(f"{r:.3f}" for r in cond["per_leg_chain_success_rate"])
        baseline_str = ", ".join(f"{r:.3f}" for r in cond["per_leg_baseline_success_rate"])
        rows.append(
            f"| N={chain_len} | [{chain_str}] | [{baseline_str}] | "
            f"{cond['all_succeeded_rate']:.3f} | {cond['n_episodes']} |"
        )
    return "\n".join(rows)


def build_isolated_failure_note(results: dict) -> str:
    """Inspect every tight-budget condition's raw per-episode bits for compounding vs. isolated failures.

    For every episode that didn't fully succeed, records which leg(s) it
    failed at. A "compounding" signature would show multiple different legs
    failing within the *same* episode (a miss early in the chain dragging
    down every later leg); an "isolated" signature shows each failing
    episode missing exactly one leg and recovering immediately on the next
    one.

    Args:
        results: A loaded `run_waypoint_eval.py` results dict.

    Returns:
        A factual, per-condition breakdown of every failing episode found,
        or a statement that no failures were observed.
    """
    lines = []
    any_failure = False
    for sequence_kind in SEQUENCE_KINDS:
        for chain_len in CHAIN_LENGTHS:
            for budget_name in BUDGET_NAMES:
                cond = find_condition(
                    results, sequence_kind=sequence_kind, chain_len=chain_len, budget_name=budget_name
                )
                failing_episodes = [
                    (episode_index, bits)
                    for episode_index, bits in enumerate(cond["per_episode_chain_bits"])
                    if not all(bits)
                ]
                if not failing_episodes:
                    continue
                any_failure = True
                for episode_index, bits in failing_episodes:
                    failed_legs = [leg_index + 1 for leg_index, success in enumerate(bits) if not success]
                    n_failed = len(failed_legs)
                    shape = "isolated (1 leg)" if n_failed == 1 else f"COMPOUNDING ({n_failed} legs)"
                    lines.append(
                        f"- {sequence_kind}/N={chain_len}/{budget_name}, episode {episode_index}: "
                        f"failed leg(s) {failed_legs} of {chain_len} -- {shape}"
                    )
    if not any_failure:
        return "No chain-rollout failures observed in any condition (all 1.000)."
    header = (
        "Every non-1.000 condition's individual failing episodes, inspected "
        "for whether a miss at one leg drags down subsequent legs in the "
        "*same* episode (compounding) or is an isolated single-leg miss "
        "that the next leg recovers from cleanly:\n"
    )
    return header + "\n".join(lines)


def render_per_leg_chart(results: dict, *, sequence_kind: str, budget_name: str) -> Path:
    """Line chart: per-leg success rate (chain vs. baseline), one line pair per chain length.

    Args:
        results: A loaded `run_waypoint_eval.py` results dict.
        sequence_kind: "literal" or "relative".
        budget_name: "tight" or "generous".

    Returns:
        The path the PNG was written to.
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHARTS_DIR / f"per_leg_{sequence_kind}_{budget_name}.png"

    fig, ax = plt.subplots()
    for chain_len in CHAIN_LENGTHS:
        cond = find_condition(results, sequence_kind=sequence_kind, chain_len=chain_len, budget_name=budget_name)
        leg_positions = list(range(1, chain_len + 1))
        ax.plot(leg_positions, cond["per_leg_chain_success_rate"], marker="o", label=f"N={chain_len} chain")
        ax.plot(
            leg_positions,
            cond["per_leg_baseline_success_rate"],
            marker="x",
            linestyle="--",
            label=f"N={chain_len} baseline",
        )
    ax.set_xlabel("leg position in chain")
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(1, max(CHAIN_LENGTHS) + 1))
    ax.set_title(f"Per-leg success: {sequence_kind} sequences, {budget_name} budget")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def render_whole_chain_vs_length_chart(results: dict) -> Path:
    """Grouped bar chart: whole-chain success rate vs. chain length, one group per (kind, budget).

    The direct visual for this stage's central question -- does success
    degrade as the chain gets longer -- across every combination tested.

    Args:
        results: A loaded `run_waypoint_eval.py` results dict.

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
                find_condition(results, sequence_kind=sequence_kind, chain_len=chain_len, budget_name=budget_name)[
                    "all_succeeded_rate"
                ]
                for chain_len in CHAIN_LENGTHS
            ]
            offsets = [x + (series_index - 1.5) * bar_width for x in x_positions]
            ax.bar(offsets, rates, width=bar_width, label=f"{sequence_kind}/{budget_name}")
            series_index += 1
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"N={n}" for n in CHAIN_LENGTHS])
    ax.set_ylabel("whole-chain success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Whole-chain success vs. chain length")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    """Aggregate the final-tier results and write stage 9's report.md."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="final")
    parser.add_argument("--tier1-tag", type=str, default="tier1")
    args = parser.parse_args()

    results = load_results(args.tag)
    tier1_results = load_results(args.tier1_tag)

    checkpoint_relpath = Path(results["checkpoint"]).relative_to(EXPERIMENT_DIR.parent.parent)
    metrics_table_parts = [
        f"### Checkpoint sanity check\nliteral-goal control, default 50-step episode, no waypoint chain: "
        f"**{results['sanity_check_success_rate']:.3f}** over {results['sanity_check_episodes']} episodes "
        f"(checkpoint: `{checkpoint_relpath}`)\n"
    ]
    for sequence_kind in SEQUENCE_KINDS:
        for budget_name in BUDGET_NAMES:
            table = build_per_leg_table(results, sequence_kind=sequence_kind, budget_name=budget_name)
            metrics_table_parts.append(
                f"### {sequence_kind} sequences, {budget_name} budget "
                f"({results['episodes_per_condition']} episodes/condition)\n\n{table}\n"
            )

    metrics_table = "\n".join(metrics_table_parts)

    chart_paths = [render_whole_chain_vs_length_chart(results)]
    for sequence_kind in SEQUENCE_KINDS:
        for budget_name in BUDGET_NAMES:
            chart_paths.append(render_per_leg_chart(results, sequence_kind=sequence_kind, budget_name=budget_name))
    chart_paths = [p.relative_to(EXPERIMENT_DIR) for p in chart_paths]

    raw_output_paths = [
        (EXPERIMENT_DIR / "runs" / "tier1_stdout.log").relative_to(EXPERIMENT_DIR),
        (EXPERIMENT_DIR / "runs" / "tier1_results.json").relative_to(EXPERIMENT_DIR),
        (EXPERIMENT_DIR / "runs" / "final_stdout.log").relative_to(EXPERIMENT_DIR),
        (EXPERIMENT_DIR / "runs" / "final_results.json").relative_to(EXPERIMENT_DIR),
    ]

    isolated_failure_note = build_isolated_failure_note(results)

    anomalies_text = (
        "Tier-1 (15 episodes/condition) and the final tier (50 episodes/condition) are numerically "
        "consistent wherever both have enough resolution to compare (e.g. relative/N=5/tight: 14/15=0.933 "
        "vs. 47/50=0.940) -- no sign of a fluke result at the smaller tier. Every generous-budget condition "
        "(both sequence kinds, all 3 chain lengths) scored a clean 1.000 on every leg, with zero baseline "
        "failures anywhere in the whole experiment -- this checkpoint sits at an oracle-solvable ceiling for "
        "this task at this budget, the same informativeness limit ROADMAP.md already documents for stages "
        "1/3/5's 1.000 scores (not a new finding, just re-observed here). The only non-1.000 results appear "
        "at the tight budget, and only for chain lengths >= 3; see the per-leg tables and "
        f"whole_chain_success_vs_length.png. {isolated_failure_note}"
    )

    known_risks_note = (
        "**Direction-sensitivity, not just distance (stage 4, carried into Phase 2a's known risks)**: not "
        "directly probed here -- this stage's relative-move sequences use a fixed 0.15m step in a randomly "
        "chosen direction per leg (not systematically varied per direction), so a direction-specific failure "
        "mode would not necessarily surface in this result; stage 8's own relative-move validation is the "
        "right place to check that per-direction. **Oracle-solvable ceiling (ROADMAP.md, stages 1/3/5)**: "
        "directly re-observed here -- every generous-budget condition and most tight-budget conditions hit "
        "1.000, so this result's informativeness is concentrated in the small number of tight-budget, "
        "longer-chain conditions that show any variance at all; a harder task or a smaller tight-budget "
        "value would give this test more room to actually fail if the mechanism were going to."
    )

    report_path = write_report(
        stage=9,
        title="Waypoint following",
        seeds=[0],
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
    print("--- tier1 vs final sanity spot-check ---")
    print(f"tier1 sanity_check_success_rate={tier1_results['sanity_check_success_rate']:.3f}")
    print(f"final sanity_check_success_rate={results['sanity_check_success_rate']:.3f}")


if __name__ == "__main__":
    main()
