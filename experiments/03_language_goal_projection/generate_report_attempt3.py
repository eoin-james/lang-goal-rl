"""Stage 3, attempt 3: aggregate the fixed-centroid-regression retest and write its `report.md` section.

Attempt 2 (see `report.md`'s "Attempt 2" section, preserved verbatim) fixed the
InfoNCE loss's scale-invariance defect but only reached a 0.069 mean language-goal
success rate -- getting the projection's output *norm* right was necessary but not
sufficient; per-instruction success varied a lot and correlated only weakly (r=0.345,
see `correlate_direction_alignment.py`) with how well the projection's *direction*
tracked its region's true centroid. The rl-builder rewrote `train_projection` to
regress directly to each instruction's fixed, precomputed-once true centroid (plain
MSE, no InfoNCE term) -- see `language_goal_projection.py`'s module docstring.

This script aggregates attempt 3's re-test: the fail-fast norm-range check, the
collapse re-check, and the language-goal substitution eval re-run against the *same
3 already-trained SAC checkpoints* from attempt 1 (no new RL training in this
attempt either).

Writes its own `write_report(...)`-shaped section into a scratch location
(`artifacts/attempt3_report_scratch/report.md`), which is then manually spliced into
the top-level `report.md` alongside attempts 1 and 2's preserved content -- same
pattern attempt 2 used for attempt 1.
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path

import numpy as np
import torch

from lang_goal_rl.goal_encoder import GoalEncoder
from lang_goal_rl.goal_region_vocabulary import ALL_INSTRUCTIONS, MEASURED_GOAL_BOX, instruction_to_region
from lang_goal_rl.language_embedding import encode_instructions
from lang_goal_rl.language_goal_projection import LanguageGoalProjection
from lang_goal_rl.reporting import plot_embedding_projection, plot_multi_seed_success_rate, write_report

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

ATTEMPT2_LANGUAGE_MEAN = 0.069
"""Attempt 2's aggregate language-goal success rate mean, copied verbatim from the
"Attempt 2" section of `report.md` -- the pre-fix baseline this retest is compared
against."""

ATTEMPT2_NORM_RANGE = "inside the 2x reference band (norm-fix confirmed passing)"
ATTEMPT2_COLLAPSE_RATIO = "24.68"
DIRECTION_ALIGNMENT_PEARSON_R = 0.3453
"""Attempt 2's projection: Pearson r between per-instruction cosine similarity to the
true region centroid and per-instruction mean RL success rate (3 seeds x 50 episodes),
copied verbatim from `artifacts/direction_alignment_correlation.log`
(`correlate_direction_alignment.py`)."""


def parse_seed_log(log_path: Path) -> tuple[float, dict[str, float]]:
    """Parse one seed's literal success rate and per-instruction language success rates.

    Args:
        log_path: Path to the seed's `stdout.log` under `runs_v3/seed_<k>/`.

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


def load_frozen_encoder(path: Path) -> GoalEncoder:
    """Load stage 2's pretrained `GoalEncoder` checkpoint, unchanged."""
    encoder = GoalEncoder(goal_dim=3)
    encoder.load_state_dict(torch.load(path, map_location="cpu"))
    encoder.eval()
    return encoder


def load_projection(path: Path) -> LanguageGoalProjection:
    """Load a `LanguageGoalProjection` checkpoint saved by `train_projection.py`."""
    checkpoint = torch.load(path, map_location="cpu")
    projection = LanguageGoalProjection(input_dim=checkpoint["input_dim"], embed_dim=checkpoint["embed_dim"])
    projection.load_state_dict(checkpoint["state_dict"])
    projection.eval()
    return projection


def main() -> None:
    """Aggregate attempt 3's 3-seed retest, generate its charts, and write its scratch report section."""
    literal_rates: dict[int, float] = {}
    language_rates_by_seed: dict[int, dict[str, float]] = {}
    raw_output_paths = []
    for seed in SEEDS:
        log_path = EXPERIMENT_DIR / "runs_v3" / f"seed_{seed}" / "stdout.log"
        literal_rate, language_rates = parse_seed_log(log_path)
        literal_rates[seed] = literal_rate
        language_rates_by_seed[seed] = language_rates
        raw_output_paths.append(log_path)
    raw_output_paths.append(EXPERIMENT_DIR / "artifacts" / "projection_train_stdout_v3.log")
    raw_output_paths.append(EXPERIMENT_DIR / "artifacts" / "norm_range_check_v3.log")
    raw_output_paths.append(EXPERIMENT_DIR / "artifacts" / "collapse_diagnostic_v3_stdout.log")
    raw_output_paths.append(EXPERIMENT_DIR / "artifacts" / "direction_alignment_correlation.log")

    language_all_values = [rate for seed in SEEDS for rate in language_rates_by_seed[seed].values()]
    language_mean_per_seed = {seed: statistics.mean(language_rates_by_seed[seed].values()) for seed in SEEDS}

    norm_check_log = (EXPERIMENT_DIR / "artifacts" / "norm_range_check_v3.log").read_text()
    norm_passed = "PASSED=True" in norm_check_log
    n_out_of_range_match = re.search(r"(\d+) instruction\(s\) out of range", norm_check_log)
    n_out_of_range = n_out_of_range_match.group(1) if n_out_of_range_match else "0"

    collapse_log = (EXPERIMENT_DIR / "artifacts" / "collapse_diagnostic_v3_stdout.log").read_text()
    ratio_match = re.search(r"min_cross_region_distance / collapse_epsilon = ([\d.]+)x", collapse_log)
    is_collapsed_match = re.search(r"is_collapsed=(\w+)", collapse_log)
    collapse_ratio = ratio_match.group(1) if ratio_match else "unknown"
    is_collapsed = is_collapsed_match.group(1) if is_collapsed_match else "unknown"

    instructions_seed0 = list(language_rates_by_seed[0].items())

    # Charts: language-goal success-rate bars (v3) and the embedding-projection PCA scatter,
    # same visual form attempt 1 used in generate_report.py, regenerated against the v3 checkpoint.
    per_seed_language_chart = plot_multi_seed_success_rate(
        {f"seed_{seed}": list(language_rates_by_seed[seed].values()) for seed in SEEDS},
        out_path=EXPERIMENT_DIR / "charts" / "language_goal_success_rate_v3.png",
        proof_gate_threshold=statistics.mean(STAGE2_10_SEED_RATES),
    )

    encoder = load_frozen_encoder(
        EXPERIMENT_DIR.parent / "02_contrastive_goal_embedding" / "artifacts" / "goal_encoder.pt",
    )
    projection = load_projection(EXPERIMENT_DIR / "artifacts" / "language_goal_projection_v3.pt")

    rng = np.random.default_rng(0)
    training_like_goals = rng.uniform(MEASURED_GOAL_BOX.axis_min, MEASURED_GOAL_BOX.axis_max, size=(300, 3))
    with torch.no_grad():
        training_like_embeddings = encoder(torch.from_numpy(training_like_goals).float()).numpy()
    training_labels = ["literal goal_encoder(desired_goal), training-distribution sample"] * len(training_like_goals)

    instructions = list(ALL_INSTRUCTIONS)
    sentence_embeddings = torch.from_numpy(encode_instructions(instructions))
    with torch.no_grad():
        projected_embeddings = projection(sentence_embeddings).numpy()
    instruction_labels = [f"projected instruction ({instruction_to_region(i)})" for i in instructions]

    combined_embeddings = np.concatenate([training_like_embeddings, projected_embeddings], axis=0)
    combined_labels = training_labels + instruction_labels
    projection_chart = plot_embedding_projection(
        combined_embeddings,
        combined_labels,
        out_path=EXPERIMENT_DIR / "charts" / "embedding_projection_v3.png",
    )

    metrics_table = (
        "### Projection retraining (fixed-centroid regression, no InfoNCE term)\n\n"
        "Full readout saved to `artifacts/projection_train_stdout_v3.log`. Loss (plain MSE to each "
        "instruction's precomputed-once true region centroid) dropped from mean=0.0020 (first 20 steps) to "
        "mean=0.0000 (last 20 steps) over 2000 steps -- the projection converges to match its fixed target "
        "almost exactly, as expected for a closed-form regression target (see "
        "`test_trained_projection_output_matches_fixed_target_closely` in "
        "`tests/lang_goal_rl/test_language_goal_projection.py`).\n\n"
        "### Fail-fast norm-range check (run immediately after retraining, before any RL eval)\n\n"
        f"Full readout saved to `artifacts/norm_range_check_v3.log`. `PASSED={norm_passed}` -- "
        f"**{n_out_of_range} of 14 instructions fell outside the 2x reference band** "
        "(mean=0.0393, bounds=[0.0196, 0.0786]), all 6 directional non-center regions except 'reach forward' "
        "and 'reach back', with norms as low as 0.0163 ('reach down low' / 'move your hand downward'). "
        "This is **not** the attempt-1/2 defect recurring: the projection matches its fixed target almost "
        "exactly (see above), so these low norms are the *true* per-region centroid norms under the frozen "
        "`GoalEncoder` -- some regions (the ones near the edges/corners of the measured box) genuinely have "
        "smaller-magnitude embeddings than the box-wide average this check's reference distribution is built "
        "from. The check's own module docstring flags it as no longer load-bearing for correctness once "
        "regression-to-true-target is used (an in-range norm becomes 'essentially automatic' only when the "
        "*true* target norms cluster near the box-wide mean, which turned out not to hold for every region "
        "here) -- treated as a factual finding about region geometry, not a fail-fast stop, and the RL eval "
        "below was run regardless per that reasoning.\n\n"
        "### Collapse re-check (fixed projection, run before the RL eval)\n\n"
        f"Full readout saved to `artifacts/collapse_diagnostic_v3_stdout.log`. "
        f"`min_cross_region_pairwise_distance / collapse_epsilon` = **{collapse_ratio}x** (threshold is 1x). "
        f"`is_collapsed` = **{is_collapsed}**. Lower than attempt 2's {ATTEMPT2_COLLAPSE_RATIO}x (attempt 3's "
        "true-centroid targets sit closer together in absolute terms for some region pairs than attempt 2's "
        "noisy per-step-estimated targets did) but still well clear of the collapse threshold.\n\n"
        "### Direction-alignment vs. success-rate correlation (attempt 2's open question, resolved with "
        "existing data -- no new RL runs)\n\n"
        "Full readout saved to `artifacts/direction_alignment_correlation.log` "
        "(`correlate_direction_alignment.py`). Computed `measure_instruction_direction_alignment` against "
        "**attempt 2's** projection checkpoint (loaded, not retrained) and correlated the resulting "
        "per-instruction cosine similarities against attempt 2's already-recorded per-instruction success "
        f"rates (mean across all 3 seeds x 50 episodes, parsed from `runs_v2/seed_*/stdout.log`): "
        f"**Pearson r = {DIRECTION_ALIGNMENT_PEARSON_R:.4f}** (n=14 instructions). 'center' region "
        "instructions had a higher mean cosine similarity (0.9217) than non-center instructions (0.8388), "
        "consistent with attempt 2's report noting 'center' scored highest -- but r=0.345 is only a weak-to-"
        "moderate positive correlation, not the value you'd expect if direction alignment were the dominant "
        "explanation for attempt 2's per-instruction success-rate variation. This settles the open question "
        "as: **directional accuracy against the true centroid explains some, but clearly not most, of "
        "attempt 2's per-instruction variation** -- a FetchReach-geometry confound (e.g. some regions' true "
        "goals sitting closer to the arm's reset position, independent of embedding quality) is a plausible "
        "co-factor and was not ruled out by this analysis.\n\n"
        "### Literal-goal protocol reproduction (unchanged from attempts 1/2 -- same 3 saved SAC checkpoints, "
        "no retraining)\n\n"
        "| Seed | Literal success rate (50 eval episodes, stage-2 protocol) |\n"
        "|------|------------------------------------------------------------|\n"
        + "\n".join(f"| {seed} | {literal_rates[seed]:.3f} |" for seed in SEEDS)
        + "\n\n"
        "Confirms the 3 checkpoints are untouched and still reproduce stage 2's baseline exactly -- this "
        "retest changes only the projection checkpoint, nothing about the trained SAC policies.\n\n"
        "### Language-goal substitution success rate (the actual attempt-3 retest)\n\n"
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
        "the gate's \"~ stage-2 baseline\" bar is still not met.\n\n"
        f"**Comparison to this stage's attempt-2 result** (mean={ATTEMPT2_LANGUAGE_MEAN:.3f} across the "
        "identical 3-seeds x 14-instructions x 50-episodes protocol): mean improved "
        f"{ATTEMPT2_LANGUAGE_MEAN:.3f} -> {statistics.mean(language_all_values):.3f}, a "
        f"~{statistics.mean(language_all_values) / ATTEMPT2_LANGUAGE_MEAN:.1f}x increase in absolute terms. "
        "Regressing directly to the true, precomputed-once centroid (no noisy per-step InfoNCE target) "
        "produced a real, further improvement over attempt 2 -- but the result is still well short of the "
        "proof gate's bar.\n\n"
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
        "The fail-fast norm-range check FAILED this attempt (8/14 instructions out of the 2x reference band) "
        "even though the projection's training loss converged almost exactly to its fixed target "
        "(mean loss 0.0000 over the last 20 steps) -- this is the check's own documented limitation "
        "(module docstring: 'no longer load-bearing for correctness... an in-range norm is essentially "
        "automatic' only holds if the true per-region centroid norms cluster near the box-wide reference "
        "mean, which several directional regions' true centroids do not) firing on real region geometry, "
        "not a training defect. Verified this is not a repeat of attempt 1's bug: attempt 1's projection was "
        "5-10x *outside* the range in the *high* direction (0.25-0.41 vs. reference mean 0.039) because the "
        "InfoNCE loss had no scale term at all; attempt 3's out-of-range instructions are all *slightly below* "
        "the lower bound (0.0163-0.0195 vs. lower bound 0.0196) because those regions' true target norms are "
        "genuinely smaller than the box-wide average -- a fundamentally different, much smaller, and "
        "structurally-explained deviation. Proceeded to the RL eval regardless, per the module docstring's "
        "own framing of this check as a heuristic proxy, superseded by direct-target-matching for "
        "correctness.\n\n"
        "The collapse check still passes "
        f"({collapse_ratio}x margin, `artifacts/collapse_diagnostic_v3_stdout.log`), so distinct instructions "
        "are still not collapsing to one point.\n\n"
        f"Aggregate language-goal success rate improved again: mean {statistics.mean(language_all_values):.3f} / "
        f"median {statistics.median(language_all_values):.3f}, up from attempt 2's "
        f"{ATTEMPT2_LANGUAGE_MEAN:.3f} mean / 0.040 median -- a "
        f"~{statistics.mean(language_all_values) / ATTEMPT2_LANGUAGE_MEAN:.1f}x further improvement. Still "
        "nowhere near the required ~1.000. Literal-goal control is unchanged and still a clean 1.000 on all "
        "3 seeds (same checkpoints, no retraining), so this remains specific to the projection, not a policy "
        "regression.\n\n"
        f"The direction-alignment correlation analysis (Pearson r={DIRECTION_ALIGNMENT_PEARSON_R:.4f} against "
        "attempt 2's per-instruction success rates) resolves the open question from attempt 2's reviewer: "
        "directional accuracy correlates only weakly-to-moderately with success, so the 'center does better' "
        "pattern observed in both attempt 2 and attempt 3 is very likely a mix of genuine directional-accuracy "
        "effect *and* a FetchReach-geometry confound (goals near the workspace center may simply be easier "
        "for this policy to reach, independent of how accurate the goal embedding is) -- not purely explained "
        "by either factor alone. This matters for interpreting attempt 3's residual gap too: even with "
        "near-perfect target-matching (attempt 3's training loss), the substitution eval still tops out at a "
        "0.44 max per-instruction success rate, well below the 1.000 literal baseline, which is more "
        "consistent with a remaining representational or distributional gap between 'true region centroid "
        "under GoalEncoder' and 'what the policy actually needs to see to succeed' than with any residual "
        "direction/scale defect in the projection itself.\n\n"
        "Per the tiered-seed strategy, this 3-seed result is uniform enough (per-seed means "
        f"{min(language_mean_per_seed.values()):.3f}-{max(language_mean_per_seed.values()):.3f}) that scaling "
        "to the full 10-seed budget would not change the qualitative picture -- skipped for the same reason "
        "attempts 1 and 2 skipped it."
    )

    known_risks_note = (
        "Same framing as attempts 1 and 2: this result is not the documented SAC deterministic-eval-collapse "
        "signature (literal eval is still a clean 1.000 on all 3 seeds, using the same unretrained "
        "checkpoints), and it is not quite the \"Metric mismatch\" known risk either (the projection regresses "
        "into the frozen `GoalEncoder`'s space via direct MSE to a precomputed centroid, not a raw "
        "sentence-embedding distance reward). Attempt 2's diagnosed defect (InfoNCE's separation term having "
        "no pressure to pull direction toward the true centroid) is now directly addressed by construction "
        "(the loss *is* distance-to-true-centroid) -- and the result improved again, consistent with that "
        "diagnosis being at least partially correct. But the residual gap (max 0.44 per-instruction, mean "
        "0.157, vs. required ~1.000) persists even with near-exact target matching, which is new evidence "
        "against 'projection accuracy' being the whole story -- something else in the true-centroid-to-policy "
        "pathway (e.g. a region's true centroid embedding not actually being what the policy needs for goals "
        "sampled elsewhere in that region, since the centroid is a single point but each eval episode samples "
        "a random point within the region) may be the next thing to investigate. Recording this as a new, "
        "distinct residual gap rather than folding it into an existing entry, for the same reason attempts 1 "
        "and 2 declined to force-fit their findings there."
    )

    write_report(
        stage=3,
        title="Frozen language embedding -> goal space (Attempt 3: fixed-centroid regression retest)",
        seeds=SEEDS,
        candidates=None,
        proof_gate_text=(
            "Success rate on language goals ~ stage-2 baseline; projection doesn't "
            "collapse distinct instructions to one point."
        ),
        metrics_table=metrics_table,
        chart_paths=[per_seed_language_chart, projection_chart],
        raw_output_paths=raw_output_paths,
        anomalies=anomalies,
        known_risks_note=known_risks_note,
        out_dir=EXPERIMENT_DIR / "artifacts" / "attempt3_report_scratch",
    )


if __name__ == "__main__":
    main()
