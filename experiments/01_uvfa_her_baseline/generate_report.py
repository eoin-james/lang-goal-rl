"""Aggregate the 10-seed stage-1 retrofit results and write report.md + charts.

Parses each seed's final `success_rate=X.XXX` line from its stdout log,
builds the multi-seed bar chart, and renders report.md via the shared
`lang_goal_rl.reporting` module per `.claude/agents/CONTRACTS.md`.

This is the follow-up batch: seeds 0-4 were the original retrofit run:
0=1.000, 1=1.000, 2=0.000, 3=1.000, 4=1.000. The reviewer returned
INCONCLUSIVE and recommended running seeds 5-9 before any stage-1 verdict.
`write_report` always renders a blank "Reviewer verdict" placeholder, so
this script's caller re-inserts the manager's existing verdict text after
regeneration — see the note in that section of report.md.
"""

import re
from pathlib import Path

from lang_goal_rl.reporting import plot_multi_seed_success_rate, write_report

EXPERIMENT_DIR = Path(__file__).parent
SEEDS = list(range(10))
FINAL_SUCCESS_RATE_RE = re.compile(r"^success_rate=([\d.]+) over (\d+) episodes$", re.MULTILINE)


def parse_final_success_rate(log_path: Path) -> float:
    """Extract the final reported success rate from a seed's stdout log.

    Args:
        log_path: Path to the seed's stdout.log, containing the script's
            terminal `success_rate=X.XXX over N episodes` print.

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


def main() -> None:
    """Aggregate per-seed success rates across all 10 seeds and write the stage-1 report."""
    results: dict[str, list[float]] = {}
    raw_output_paths = []
    for seed in SEEDS:
        log_path = EXPERIMENT_DIR / "runs" / f"seed_{seed}" / "stdout.log"
        results[f"seed_{seed}"] = [parse_final_success_rate(log_path)]
        raw_output_paths.append(log_path)

    rates = [samples[0] for samples in results.values()]
    mean_rate = sum(rates) / len(rates)
    min_rate, max_rate = min(rates), max(rates)
    n_at_gate = sum(1 for r in rates if r >= 0.98)

    chart_path = plot_multi_seed_success_rate(
        results,
        out_path=EXPERIMENT_DIR / "charts" / "multi_seed_success_rate.png",
        proof_gate_threshold=0.9,
    )

    metrics_table = (
        "| Seed | Success rate (50 eval episodes) |\n"
        "|------|----------------------------------|\n"
        + "\n".join(f"| {seed} | {results[f'seed_{seed}'][0]:.3f} |" for seed in SEEDS)
        + f"\n| **Mean (10 seeds)** | **{mean_rate:.3f}** |\n"
        + f"| Min / Max | {min_rate:.3f} / {max_rate:.3f} |\n"
        + f"| Seeds >= 0.98 | {n_at_gate}/10 |\n"
    )

    original_anomalies = (
        "Seed 2 returned success_rate=0.000 over 50 eval episodes, while "
        "seeds 0, 1, 3, 4 all returned 1.000 — a single total-failure seed "
        "among four perfect seeds. This is a genuine per-seed result, not a "
        "run artifact: seed_2/stdout.log shows the same training shape as "
        "the other seeds (success_rate climbing steadily during training, "
        "reaching ~0.98-0.99 by the end of the training-time success_rate "
        "logging), then the held-out eval loop reports 0 successes out of "
        "50. This looks like an eval-time policy collapse or a "
        "deterministic-action failure mode specific to this seed's learned "
        "policy, not a data or logging bug. Flagging for reviewer judgment "
        "on whether this counts as a proof-gate failure.\n\n"
        "Wall-clock check (requested by coordinator): the coordinator "
        "reported ~46 minutes wall-clock for all 5 seeds and asked whether "
        "the launch serialized them. Checked directly: `runs/seed_*/stdout.log` "
        "were all created at the same second (2026-07-24 15:59:38) and all "
        "last-modified within 1 second of each other (16:03:08-16:03:09) — "
        "total wall-clock for all 5 seeds to complete was ~3.5 minutes, "
        "matching the ~3 min/seed expectation for true concurrency (if "
        "serialized, 5 x ~3 min would be ~15 min minimum, not 3.5). `ps aux` "
        "taken shortly after launch showed all 5 `python train.py` "
        "processes running simultaneously at ~100% CPU each (i.e. pinned to "
        "one core each, consistent with OMP_NUM_THREADS=1/MKL_NUM_THREADS=1 "
        "actually taking effect — no oversubscription). The launch script "
        "used `cmd & ... ; wait` for all 5 backgrounded processes in one "
        "shell, which is genuinely concurrent, not serialized, and no "
        "concurrency-cap bug was found (cap = min(5, cores-2) = 5 on this "
        "10-core machine, so running all 5 at once is correct, not a bug). "
        "The reported 46-minute figure does not match the file-timestamp "
        "evidence and most likely reflects elapsed time in the surrounding "
        "session (message/notification delivery lag) rather than actual "
        "training wall-clock."
    )

    follow_up_lines = [str(seed) + f"={results[f'seed_{seed}'][0]:.3f}" for seed in range(5, 10)]
    n_failed_follow_up = sum(1 for r in rates[5:10] if r < 0.98)
    follow_up_note = (
        "\n\nFollow-up batch (seeds 5-9, run after reviewer returned "
        "INCONCLUSIVE on seeds 0-4): " + ", ".join(follow_up_lines) + ". "
        + (
            f"{n_failed_follow_up} of the 5 new seeds fell below 0.98."
            if n_failed_follow_up
            else "All 5 new seeds reached >=0.98 — no additional failures in this batch."
        )
        + " First launch attempt for this batch was killed mid-training by "
        "the runner's tool-call timeout (5 processes were still running at "
        "~1 minute in, no success_rate line in any log) and had to be "
        "relaunched as an explicit background job; the logs below are from "
        "the second, completed launch."
    )

    anomalies = original_anomalies + follow_up_note

    write_report(
        stage=1,
        title="Goal-conditioned baseline (UVFA + HER)",
        seeds=SEEDS,
        candidates=None,
        proof_gate_text=(
            "Near-100% success rate over held-out eval episodes on FetchReach."
        ),
        metrics_table=metrics_table,
        chart_paths=[chart_path],
        raw_output_paths=raw_output_paths,
        anomalies=anomalies,
        known_risks_note=(
            "None of the ROADMAP.md \"Known risks\" entries apply to this "
            "stage. \"Metric mismatch\" is scoped to stage 3+ (sentence-"
            "transformer/CLIP-text embeddings replacing literal xyz goals) "
            "and \"Non-stationarity at stage 5\" is scoped to mid-episode "
            "re-goaling — neither is in play for a stage-1 literal-goal "
            "SAC+HER baseline."
        ),
        out_dir=EXPERIMENT_DIR,
    )


if __name__ == "__main__":
    main()
