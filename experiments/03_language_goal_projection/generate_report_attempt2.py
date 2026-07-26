"""Stage 3, attempt 2: aggregate the scale-fix retest and write its `report.md` section.

Attempt 1 (see `report.md`'s "Attempt 1" section, preserved verbatim) FAILed
the proof gate's success-rate half after the projection's output landed 5-10x
outside the frozen `GoalEncoder`'s real operating-norm range. The rl-builder
added an explicit norm-matching term to `train_projection`'s loss
(`combined_projection_loss`, see `language_goal_projection.py`) plus a
fail-fast `check_projection_norm_range` to catch a repeat before spending an
RL eval budget on it. This script aggregates attempt 2's re-test: the fixed
projection's fail-fast check, its collapse re-check, and the language-goal
substitution eval re-run against the *same 3 already-trained SAC checkpoints*
(no new RL training).

Writes its own `write_report(...)`-shaped section into a scratch location
(`artifacts/attempt2_report_scratch/report.md`), which the runner then splices
into the top-level `report.md` alongside attempt 1's preserved content --
see the runner's task instructions for why this stage keeps both attempts
visible rather than overwriting attempt 1's data (same pattern as stage 1's
two-pass retrofit).
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path

from lang_goal_rl.goal_region_vocabulary import instruction_to_region
from lang_goal_rl.reporting import write_report

EXPERIMENT_DIR = Path(__file__).parent
SEEDS = [0, 1, 2]

LITERAL_SUCCESS_RATE_RE = re.compile(r"^success_rate=([\d.]+) over (\d+) episodes$", re.MULTILINE)
LANGUAGE_SUCCESS_RATE_RE = re.compile(
    r'^language_success_rate=([\d.]+) instruction="([^"]+)" region="([^"]+)" over (\d+) episodes$',
    re.MULTILINE,
)

STAGE2_10_SEED_RATES = [1.0] * 10
"""Stage 2's locked-in 10-seed result, copied verbatim from
`experiments/02_contrastive_goal_embedding/report.md`."""

ATTEMPT1_LANGUAGE_MEAN = 0.002
"""Attempt 1's aggregate language-goal success rate mean, copied verbatim from
the "Attempt 1" section of `report.md` -- the pre-fix baseline this retest is
compared against."""

ATTEMPT1_NORM_RANGE = "0.25-0.41"
"""Attempt 1's projected-instruction norm range, copied verbatim from
`debug_language_eval.py`'s Check 2 as cited in `report.md`."""

ATTEMPT1_COLLAPSE_RATIO = "143.85"
"""Attempt 1's collapse-diagnostic ratio, copied verbatim from
`artifacts/collapse_diagnostic_stdout.log`."""


def parse_seed_log(log_path: Path) -> tuple[float, dict[str, float]]:
    """Parse one seed's literal success rate and per-instruction language success rates.

    Args:
        log_path: Path to the seed's `stdout.log` under `runs_v2/seed_<k>/`.

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


def main() -> None:
    """Aggregate attempt 2's 3-seed retest and write its scratch report section."""
    literal_rates: dict[int, float] = {}
    language_rates_by_seed: dict[int, dict[str, float]] = {}
    raw_output_paths = []
    for seed in SEEDS:
        log_path = EXPERIMENT_DIR / "runs_v2" / f"seed_{seed}" / "stdout.log"
        literal_rate, language_rates = parse_seed_log(log_path)
        literal_rates[seed] = literal_rate
        language_rates_by_seed[seed] = language_rates
        raw_output_paths.append(log_path)
    raw_output_paths.append(EXPERIMENT_DIR / "artifacts" / "norm_range_check.log")
    raw_output_paths.append(EXPERIMENT_DIR / "artifacts" / "collapse_diagnostic_v2_stdout.log")

    language_all_values = [rate for seed in SEEDS for rate in language_rates_by_seed[seed].values()]
    language_mean_per_seed = {seed: statistics.mean(language_rates_by_seed[seed].values()) for seed in SEEDS}

    norm_check_log = (EXPERIMENT_DIR / "artifacts" / "norm_range_check.log").read_text()
    norm_passed = "PASSED=True" in norm_check_log
    collapse_log = (EXPERIMENT_DIR / "artifacts" / "collapse_diagnostic_v2_stdout.log").read_text()
    ratio_match = re.search(r"min_cross_region_distance / collapse_epsilon = ([\d.]+)x", collapse_log)
    is_collapsed_match = re.search(r"is_collapsed=(\w+)", collapse_log)
    collapse_ratio = ratio_match.group(1) if ratio_match else "unknown"
    is_collapsed = is_collapsed_match.group(1) if is_collapsed_match else "unknown"

    instructions_seed0 = list(language_rates_by_seed[0].items())

    metrics_table = (
        "### Fail-fast norm-range check (run immediately after retraining the projection, before any RL eval)\n\n"
        f"Full readout saved to `artifacts/norm_range_check.log`. `PASSED={norm_passed}`. All 14 instructions' "
        f"projected norms now fall inside the frozen `GoalEncoder`'s 2x reference band (mean=0.0393, "
        f"bounds=[0.0196, 0.0786]) -- versus attempt 1's {ATTEMPT1_NORM_RANGE} range (5-10x outside it). "
        "Per the task instructions, the RL eval below only ran because this check passed first.\n\n"
        "### Collapse re-check (fixed projection, run before the RL eval)\n\n"
        f"Full readout saved to `artifacts/collapse_diagnostic_v2_stdout.log`. "
        f"`min_cross_region_pairwise_distance / collapse_epsilon` = **{collapse_ratio}x** (threshold is 1x). "
        f"`is_collapsed` = **{is_collapsed}**. Lower than attempt 1's {ATTEMPT1_COLLAPSE_RATIO}x (the fixed "
        "projection's outputs now sit at the encoder's true, much smaller scale, so absolute pairwise distances "
        "shrank too) but still well clear of the collapse threshold.\n\n"
        "### Literal-goal protocol reproduction (unchanged from attempt 1 -- same 3 saved SAC checkpoints, "
        "no retraining)\n\n"
        "| Seed | Literal success rate (50 eval episodes, stage-2 protocol) |\n"
        "|------|------------------------------------------------------------|\n"
        + "\n".join(f"| {seed} | {literal_rates[seed]:.3f} |" for seed in SEEDS)
        + "\n\n"
        "Confirms the 3 checkpoints are untouched and still reproduce stage 2's baseline exactly -- the retest "
        "changes only the projection checkpoint, nothing about the trained SAC policies.\n\n"
        "### Language-goal substitution success rate (the actual retest)\n\n"
        "| Seed | Mean success rate across 14 instructions (50 episodes each) |\n"
        "|------|----------------------------------------------------------------|\n"
        + "\n".join(f"| {seed} | {language_mean_per_seed[seed]:.3f} |" for seed in SEEDS)
        + "\n\n"
        f"Aggregate across all {len(SEEDS)} seeds x 14 instructions ({len(language_all_values)} success-rate "
        f"samples): mean=**{statistics.mean(language_all_values):.3f}**, "
        f"median=**{statistics.median(language_all_values):.3f}**, max=**{max(language_all_values):.3f}**, "
        f"min=**{min(language_all_values):.3f}**.\n\n"
        f"**Comparison to stage-2 baseline** (mean=median=mode={statistics.mean(STAGE2_10_SEED_RATES):.3f} over "
        "10 seeds, per ROADMAP Known risks' judge-at-median/mode guidance): "
        f"{statistics.median(language_all_values):.3f} median vs. {statistics.mean(STAGE2_10_SEED_RATES):.3f} -- "
        "the gate's \"~ stage-2 baseline\" bar is not met.\n\n"
        f"**Comparison to this stage's attempt-1 FAIL** (mean={ATTEMPT1_LANGUAGE_MEAN:.3f} across the identical "
        "3-seeds x 14-instructions x 50-episodes protocol): mean improved "
        f"{ATTEMPT1_LANGUAGE_MEAN:.3f} -> {statistics.mean(language_all_values):.3f}, a "
        f"~{statistics.mean(language_all_values) / ATTEMPT1_LANGUAGE_MEAN:.0f}x increase in absolute terms. The "
        "scale fix produced a real, measurable improvement -- the fail-fast check confirms the specific defect "
        "it targeted (output norm 5-10x outside the reference range) is gone -- but the result is still nowhere "
        "near the proof gate's bar.\n\n"
        "### Per-instruction detail (seed 0)\n\n"
        "| Instruction | Region | Success rate |\n"
        "|-------------|--------|---------------|\n"
        + "\n".join(
            f"| {instruction} | {instruction_to_region(instruction)} | {rate:.3f} |"
            for instruction, rate in instructions_seed0
        )
        + "\n"
    )

    anomalies = (
        "The norm-scale fix diagnosed in attempt 1 is confirmed fixed at the source: the fail-fast check "
        "(`artifacts/norm_range_check.log`) shows all 14 projected-instruction norms now inside the frozen "
        f"encoder's real 2x reference band, versus attempt 1's {ATTEMPT1_NORM_RANGE} range (5-10x outside it). "
        f"The collapse check still passes ({collapse_ratio}x margin, "
        "`artifacts/collapse_diagnostic_v2_stdout.log`), so distinct instructions are still not collapsing to "
        "one point.\n\n"
        "Despite both diagnostics now passing, the RL success-rate half of the gate is still far below the "
        f"stage-2 baseline: aggregate mean {statistics.mean(language_all_values):.3f} / median "
        f"{statistics.median(language_all_values):.3f} across 3 seeds x 14 instructions, vs. a required ~1.000. "
        f"This is a real improvement over attempt 1's {ATTEMPT1_LANGUAGE_MEAN:.3f} mean (getting the scale "
        "right clearly helped some), but the projection still is not landing the policy's desired-goal input "
        "close enough to what the frozen `GoalEncoder` would have produced for the true literal target -- "
        "getting the *norm* right is necessary but evidently not sufficient; the InfoNCE separation term has "
        "no explicit pressure to pull the *direction* of each projected point toward its region's true "
        "centroid beyond what a noisy per-step positive-sample estimate provides. Literal-goal control is "
        "unchanged and still a clean 1.000 on all 3 seeds (same checkpoints, no retraining), so this is not a "
        "policy regression -- it is still specific to the projection's output landing in the wrong place "
        "within the correct scale band. \"center\" region instructions score noticeably higher than the rest "
        "(seed 0: 0.140/0.380 vs. mostly 0.000-0.100 elsewhere), consistent with \"center\" being the region "
        "closest to the overall goal-space centroid the InfoNCE target is pulled toward on average -- a "
        "directional-accuracy gap, not a scale gap.\n\n"
        "Per the tiered-seed strategy, this 3-seed result is uniform enough (per-seed means "
        f"{min(language_mean_per_seed.values()):.3f}-{max(language_mean_per_seed.values()):.3f}) that scaling "
        "to the full 10-seed budget would not change the qualitative picture -- skipped for the same reason "
        "attempt 1 skipped it."
    )

    known_risks_note = (
        "Same framing as attempt 1: this result is not the documented SAC deterministic-eval-collapse "
        "signature (literal eval is still a clean 1.000 on all 3 seeds, using the same unretrained "
        "checkpoints), and it is not quite the \"Metric mismatch\" known risk either (the projection regresses "
        "into the frozen `GoalEncoder`'s space via InfoNCE + the new norm-matching term, not a raw "
        "sentence-embedding distance reward). The specific defect attempt 1 diagnosed (loss-structural scale "
        "invariance) is now directly falsified by the fail-fast check's numbers -- the fix worked at removing "
        "that defect -- but a *second*, previously-masked defect is now visible: even at the correct scale, "
        "the projection's output direction does not track its region's true centroid closely enough for the "
        "frozen-encoder-conditioned policy to succeed. Recording this as a new, distinct residual gap rather "
        "than folding it into the \"Metric mismatch\" entry, for the same reason attempt 1 declined to "
        "force-fit its finding there."
    )

    write_report(
        stage=3,
        title="Frozen language embedding -> goal space (Attempt 2: scale-fix retest)",
        seeds=SEEDS,
        candidates=None,
        proof_gate_text=(
            "Success rate on language goals ~ stage-2 baseline; projection doesn't "
            "collapse distinct instructions to one point."
        ),
        metrics_table=metrics_table,
        chart_paths=[
            EXPERIMENT_DIR / "charts" / "language_goal_success_rate_v2.png",
            EXPERIMENT_DIR / "charts" / "embedding_projection_v2.png",
        ],
        raw_output_paths=raw_output_paths,
        anomalies=anomalies,
        known_risks_note=known_risks_note,
        out_dir=EXPERIMENT_DIR / "artifacts" / "attempt2_report_scratch",
    )


if __name__ == "__main__":
    main()
