"""Stage 4, attempt 2: aggregate the data-augmentation retest and write its `report.md` section.

Attempt 1 (see `report.md`'s top-level section, preserved verbatim) FAILed:
held-out RL success collapsed to near-zero (mean 0.024, median 0.000) against
a 1.000 training-vocabulary baseline, diagnosed as `LanguageGoalProjection`
memorizing its 14-sentence training vocabulary rather than learning a
generalizing rule -- independently confirmed by a zero-training
nearest-neighbor ceiling test (0.714 vs. the trained MLP's 0.286 semantic-
neighbor accuracy on the same held-out set). The reviewer's recommended fix,
in order: NN-ceiling test (done, attempt 1) -> data augmentation (this
attempt) -> smoothness regularization only if augmentation alone isn't
enough.

This attempt retrains the projection on `augmented_training_vocabulary`'s
70-sentence vocabulary (`train_projection_augmented.py`, unchanged
hyperparameters from the run that trained `language_goal_projection_v3.pt`)
and reruns all three of attempt 1's parts against the identical, unchanged
held-out test set (`held_out_paraphrases`), plus a new sanity check: does the
retrained projection still ace the *original* 14-sentence stage-3 vocabulary
it no longer trains on directly (see this script's "regression check"
section -- it does not, a real finding, not a bug in this script).

Writes its own `write_report(...)`-shaped section into a scratch location
(`artifacts/attempt2_report_scratch/report.md`), manually spliced into the
top-level `report.md` alongside attempt 1's preserved content -- same
pattern stage 3's attempts 2-4 used.
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from diagnose_open_vocab import (
    STAGE2_ENCODER_PATH,
    load_frozen_encoder,
    load_projection,
    run_compositional_diagnostic,
)
from diagnose_open_vocab_v2 import AUGMENTED_PROJECTION_PATH, run_semantic_neighbor_diagnostic_v2
from lang_goal_rl.augmented_training_vocabulary import AUGMENTED_INSTRUCTIONS, augmented_instruction_to_region
from lang_goal_rl.goal_region_vocabulary import ALL_INSTRUCTIONS, instruction_to_region
from lang_goal_rl.held_out_paraphrases import (
    COMPOSITIONAL_INSTRUCTIONS,
    compositional_texts,
    held_out_region_names,
    held_out_texts,
)
from lang_goal_rl.language_embedding import encode_instructions
from lang_goal_rl.reporting import plot_embedding_projection, plot_multi_seed_success_rate, write_report

if TYPE_CHECKING:
    from lang_goal_rl.semantic_neighbor_diagnostic import CompositionalPlacement, SemanticNeighborReport

EXPERIMENT_DIR = Path(__file__).parent
SEEDS = [0, 1, 2]

LITERAL_SUCCESS_RATE_RE = re.compile(r"^success_rate=([\d.]+) over (\d+) episodes$", re.MULTILINE)
LANGUAGE_SUCCESS_RATE_RE = re.compile(
    r'^language_success_rate=([\d.]+) instruction="([^"]+)" region="([^"]+)" over (\d+) episodes$',
    re.MULTILINE,
)

STAGE3_FINAL_SUCCESS_RATE = 1.000
"""Stage 3's final (attempt 4) language-goal success rate on its own 14
*training* instructions, copied verbatim from `experiments/
03_language_goal_projection/report.md`'s Attempt 4 section -- unchanged from
`generate_report.py`'s constant of the same name."""

ATTEMPT1_NEIGHBOR_ACCURACY = 0.286
"""Attempt 1's semantic-neighbor diagnostic accuracy (4/14), copied verbatim
from `report.md`'s top-level (attempt 1) Part 1 section."""

ATTEMPT1_LANGUAGE_MEAN = 0.024
ATTEMPT1_LANGUAGE_MEDIAN = 0.000
ATTEMPT1_LANGUAGE_NONZERO = "1/42"
"""Attempt 1's aggregate held-out RL success-rate stats, copied verbatim from
`report.md`'s top-level (attempt 1) Part 2 section."""

NN_CEILING_K1_ACCURACY = 0.714
"""The zero-training nearest-neighbor ceiling test's k=1 semantic-neighbor
accuracy (10/14), copied verbatim from `report.md`'s attempt-1 "Part 4"
section -- the upper-bound reference this attempt's own accuracy is compared
against, alongside attempt 1's trained-MLP figure."""


def parse_seed_log(log_path: Path) -> tuple[float, dict[str, float]]:
    """Parse one seed's literal success rate and per-instruction language success rates.

    Args:
        log_path: Path to a seed's `stdout.log` (either `eval_held_out.py`'s
            or `eval_training_vocab_regression.py`'s output -- both scripts
            print in the identical `success_rate=`/`language_success_rate=`
            format `train.py`'s eval functions were built around).

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


def _semantic_neighbor_table(report: SemanticNeighborReport) -> str:
    """Render the semantic-neighbor diagnostic's per-instruction breakdown as a markdown table."""
    lines = [
        "| Instruction | True region | Nearest region | Correct | Margin (true-region minus nearest-region distance; 0 "
        "when correct, positive means the wrong region won by that much) |",
        "|-------------|-------------|----------------|---------|----------------------------------------------------------"
        "-----------------------------------------------------------------|",
    ]
    for result in report.results:
        true_distance = result.distances_by_region[result.true_region_name]
        nearest_distance = result.distances_by_region[result.nearest_region_name]
        margin = true_distance - nearest_distance
        verdict = "yes" if result.is_correct else "NO"
        lines.append(
            f"| {result.instruction} | {result.true_region_name} | {result.nearest_region_name} | "
            f"{verdict} | {margin:.4f} |",
        )
    return "\n".join(lines)


def _compositional_table(placements: tuple[CompositionalPlacement, ...]) -> str:
    """Render the compositional placement diagnostic as a markdown table."""
    lines = [
        "| Instruction | Components | Nearest region | Nearest is a component | Component balance (1.0=equidistant) |",
        "|-------------|------------|-----------------|--------------------------|----------------------------------------|",
    ]
    for placement in placements:
        lines.append(
            f"| {placement.instruction} | {' / '.join(placement.component_region_names)} | "
            f"{placement.nearest_region_name} | {placement.nearest_is_component} | "
            f"{placement.component_distance_balance:.3f} |",
        )
    return "\n".join(lines)


def main() -> None:  # noqa: PLR0914 -- report-assembly script, many small named locals aid readability over a dict blob
    """Aggregate attempt 2's data-augmentation retest and write its `report.md` section."""
    # --- Part 1 + Part 3: re-run in-process against the augmented-vocabulary projection (no RL) ---
    projection = load_projection(AUGMENTED_PROJECTION_PATH)
    encoder = load_frozen_encoder(STAGE2_ENCODER_PATH)
    neighbor_report = run_semantic_neighbor_diagnostic_v2(projection)
    compositional_placements = run_compositional_diagnostic(projection, encoder)

    # --- Part 2: RL success rate on the unchanged held-out phrasings (parsed from already-run logs) ---
    literal_rates: dict[int, float] = {}
    language_rates_by_seed: dict[int, dict[str, float]] = {}
    raw_output_paths = []
    for seed in SEEDS:
        log_path = EXPERIMENT_DIR / "runs" / "attempt2" / f"seed_{seed}" / "stdout.log"
        literal_rate, language_rates = parse_seed_log(log_path)
        literal_rates[seed] = literal_rate
        language_rates_by_seed[seed] = language_rates
        raw_output_paths.append(log_path)

    language_all_values = [rate for seed in SEEDS for rate in language_rates_by_seed[seed].values()]
    language_mean_per_seed = {seed: statistics.mean(language_rates_by_seed[seed].values()) for seed in SEEDS}
    language_mean = statistics.mean(language_all_values)
    language_median = statistics.median(language_all_values)
    n_nonzero = sum(1 for value in language_all_values if value > 0)
    per_instruction_mean = {
        instruction: statistics.mean(language_rates_by_seed[seed][instruction] for seed in SEEDS)
        for instruction in held_out_texts()
    }

    # --- Regression check: does the new projection still ace the ORIGINAL 14 stage-3 training instructions? ---
    regression_literal_rates: dict[int, float] = {}
    regression_rates_by_seed: dict[int, dict[str, float]] = {}
    regression_raw_output_paths = []
    for seed in SEEDS:
        log_path = EXPERIMENT_DIR / "runs" / "attempt2" / "regression_check" / f"seed_{seed}" / "stdout.log"
        literal_rate, language_rates = parse_seed_log(log_path)
        regression_literal_rates[seed] = literal_rate
        regression_rates_by_seed[seed] = language_rates
        regression_raw_output_paths.append(log_path)

    regression_all_values = [rate for seed in SEEDS for rate in regression_rates_by_seed[seed].values()]
    regression_mean_per_seed = {seed: statistics.mean(regression_rates_by_seed[seed].values()) for seed in SEEDS}
    regression_mean = statistics.mean(regression_all_values)
    regression_median = statistics.median(regression_all_values)
    regression_nonzero = sum(1 for value in regression_all_values if value > 0)
    regression_per_instruction_mean = {
        instruction: statistics.mean(regression_rates_by_seed[seed][instruction] for seed in SEEDS)
        for instruction in ALL_INSTRUCTIONS
    }

    # --- Charts ---
    held_out_chart = plot_multi_seed_success_rate(
        {f"seed_{seed}": list(language_rates_by_seed[seed].values()) for seed in SEEDS},
        out_path=EXPERIMENT_DIR / "charts" / "held_out_success_rate_v2.png",
        proof_gate_threshold=STAGE3_FINAL_SUCCESS_RATE,
    )
    regression_chart = plot_multi_seed_success_rate(
        {f"seed_{seed}": list(regression_rates_by_seed[seed].values()) for seed in SEEDS},
        out_path=EXPERIMENT_DIR / "charts" / "stage3_vocab_regression_check_v2.png",
        proof_gate_threshold=STAGE3_FINAL_SUCCESS_RATE,
    )

    augmented_train_sentence_embeddings = torch.from_numpy(encode_instructions(AUGMENTED_INSTRUCTIONS))
    original_train_sentence_embeddings = torch.from_numpy(encode_instructions(ALL_INSTRUCTIONS))
    held_out_sentence_embeddings = torch.from_numpy(encode_instructions(held_out_texts()))
    compositional_sentence_embeddings = torch.from_numpy(encode_instructions(compositional_texts()))
    with torch.no_grad():
        augmented_train_projected = projection(augmented_train_sentence_embeddings)
        original_train_projected = projection(original_train_sentence_embeddings)
        held_out_projected = projection(held_out_sentence_embeddings)
        compositional_projected = projection(compositional_sentence_embeddings)

    augmented_mapping = augmented_instruction_to_region()
    all_embeddings = np.concatenate(
        [
            augmented_train_projected.numpy(),
            original_train_projected.numpy(),
            held_out_projected.numpy(),
            compositional_projected.numpy(),
        ],
        axis=0,
    )
    all_labels = (
        [f"{augmented_mapping[instruction]}·aug-train" for instruction in AUGMENTED_INSTRUCTIONS]
        + [f"{instruction_to_region(instruction)}·orig-train" for instruction in ALL_INSTRUCTIONS]
        + [f"{region}·held-out" for region in held_out_region_names()]
        + ["compositional"] * len(COMPOSITIONAL_INSTRUCTIONS)
    )
    embedding_chart = plot_embedding_projection(
        all_embeddings,
        all_labels,
        out_path=EXPERIMENT_DIR / "charts" / "embedding_projection_open_vocab_v2.png",
    )

    n_correct = sum(1 for r in neighbor_report.results if r.is_correct)
    n_total = len(neighbor_report.results)

    metrics_table = (
        "### What changed\n\n"
        "`LanguageGoalProjection` was retrained (`train_projection_augmented.py`) on "
        "`augmented_training_vocabulary.AUGMENTED_INSTRUCTIONS` -- 70 sentences, 10 diverse phrasings per region "
        "-- instead of `goal_region_vocabulary.ALL_INSTRUCTIONS` (14 sentences, 2 per region). Every "
        "hyperparameter (`n_steps=2000`, `learning_rate=1e-3`, `n_target_samples=1000`, `box=MEASURED_GOAL_BOX`, "
        "`seed=0`) is unchanged from the run that trained `language_goal_projection_v3.pt` -- confirmed from "
        "`report.md`'s attempt-3 section (loss dropped to 0.0000 over 2000 steps, matching this retrain's "
        "`runs/attempt2/projection_train_stdout.log`). Saved as a new checkpoint "
        "(`artifacts/language_goal_projection_v5_augmented.pt`) so v3 stays available for comparison/provenance. "
        "No SAC policy was retrained -- all RL evals below reuse the same 3 stage-3 checkpoints "
        "(`03_language_goal_projection/checkpoints/seed_{0,1,2}.zip`) attempt 1 used.\n\n"
        "### Part 1 -- Semantic-neighbor diagnostic (no RL; frozen sentence-transformer + augmented "
        "projection only)\n\n"
        "Reference set: the 70 *augmented-training* instructions' own projected embeddings (through the new "
        "`language_goal_projection_v5_augmented.pt`) -- same reference-set choice as attempt 1 (the training "
        "instructions' own projected output geometry, not a separately-computed centroid), just over the "
        "larger vocabulary this projection actually trained on. Query set is unchanged: the same 14 "
        "`held_out_paraphrases.HELD_OUT_PARAPHRASES`.\n\n"
        f"**Aggregate accuracy: {neighbor_report.accuracy:.3f} ({n_correct}/{n_total}) -- vs. attempt 1's "
        f"{ATTEMPT1_NEIGHBOR_ACCURACY:.3f} (4/14) and the NN-ceiling's {NN_CEILING_K1_ACCURACY:.3f} (10/14, k=1).**\n\n"
        f"{_semantic_neighbor_table(neighbor_report)}\n\n"
        "### Part 2 -- RL success rate on held-out phrasings (the actual generalization test)\n\n"
        "Same 3 already-trained SAC checkpoints as attempt 1 -- no retraining. Only the projection checkpoint "
        "changed (v3 -> v5_augmented). Ground truth judged against each instruction's region centroid "
        "(`train.compute_region_centroid`), unchanged from attempt 1.\n\n"
        "| Seed | Literal success rate (50 eval episodes, stage-2/3 protocol) | Mean held-out language "
        "success rate (14 instructions x 50 episodes) |\n"
        "|------|------------------------------------------------------------|------------------------------"
        "----------------------------------|\n"
        + "\n".join(
            f"| {seed} | {literal_rates[seed]:.3f} | {language_mean_per_seed[seed]:.3f} |" for seed in SEEDS
        )
        + "\n\n"
        f"Aggregate across {len(SEEDS)} seeds x 14 held-out instructions ({len(language_all_values)} "
        f"success-rate samples): mean=**{language_mean:.3f}**, median=**{language_median:.3f}**, "
        f"max=**{max(language_all_values):.3f}**, min=**{min(language_all_values):.3f}**, "
        f"nonzero samples=**{n_nonzero}/{len(language_all_values)}**.\n\n"
        f"**Comparison to attempt 1** (mean={ATTEMPT1_LANGUAGE_MEAN:.3f}, median={ATTEMPT1_LANGUAGE_MEDIAN:.3f}, "
        f"nonzero={ATTEMPT1_LANGUAGE_NONZERO}): mean {ATTEMPT1_LANGUAGE_MEAN:.3f} -> {language_mean:.3f}, "
        f"nonzero samples {ATTEMPT1_LANGUAGE_NONZERO} -> {n_nonzero}/{len(language_all_values)}. **Comparison to "
        f"stage 3's training-vocabulary baseline** ({STAGE3_FINAL_SUCCESS_RATE:.3f}): {language_median:.3f} "
        f"median vs. {STAGE3_FINAL_SUCCESS_RATE:.3f} -- still far short. Literal-goal control stays a clean "
        "1.000 on all 3 seeds (same checkpoints, unchanged), so both this attempt's improvement and its "
        "remaining gap are specific to the projection, not a policy or checkpoint change.\n\n"
        "### Per-instruction detail (mean success rate across all 3 seeds)\n\n"
        "| Instruction | Region | Held-out RL success rate | Semantic-neighbor verdict |\n"
        "|-------------|--------|---------------------------|-----------------------------|\n"
        + "\n".join(
            f"| {instruction} | {region} | {per_instruction_mean[instruction]:.3f} | "
            f"{'correct' if next(r for r in neighbor_report.results if r.instruction == instruction).is_correct else 'WRONG'} |"
            for instruction, region in zip(held_out_texts(), held_out_region_names(), strict=True)
        )
        + "\n\n"
        "### Part 3 -- Compositional instructions (no single ground-truth region; reported honestly, "
        "no forced verdict)\n\n"
        f"{_compositional_table(compositional_placements)}\n\n"
        "### Sanity check -- does the augmented-vocabulary projection still ace the ORIGINAL stage-3 "
        "training vocabulary?\n\n"
        "The 70-sentence `augmented_training_vocabulary` has **zero string overlap** with the original 14 "
        "`goal_region_vocabulary.ALL_INSTRUCTIONS` (independently verified: `set(AUGMENTED_INSTRUCTIONS) & "
        "set(ALL_INSTRUCTIONS) == set()`) -- it replaces, rather than extends, the original training sentences. "
        "This check evaluates the new projection on the original 14 instructions it no longer trains on "
        "directly, using the same 3 SAC checkpoints and the same fixed-centroid ground truth as Part 2.\n\n"
        "| Seed | Literal success rate (50 eval episodes) | Mean success rate on ORIGINAL 14 training "
        "instructions (50 episodes each) |\n"
        "|------|-------------------------------------------|-------------------------------------------------"
        "------------------------|\n"
        + "\n".join(
            f"| {seed} | {regression_literal_rates[seed]:.3f} | {regression_mean_per_seed[seed]:.3f} |"
            for seed in SEEDS
        )
        + "\n\n"
        f"Aggregate across {len(SEEDS)} seeds x 14 original instructions ({len(regression_all_values)} "
        f"success-rate samples): mean=**{regression_mean:.3f}**, median=**{regression_median:.3f}**, "
        f"max=**{max(regression_all_values):.3f}**, min=**{min(regression_all_values):.3f}**, "
        f"nonzero samples=**{regression_nonzero}/{len(regression_all_values)}**. This is **not** a clean "
        f"~1.000 reproduction of stage 3's attempt-4 baseline -- see Anomalies below.\n\n"
        "| Instruction | Region | Success rate on original vocabulary |\n"
        "|-------------|--------|----------------------------------------|\n"
        + "\n".join(
            f"| {instruction} | {instruction_to_region(instruction)} | "
            f"{regression_per_instruction_mean[instruction]:.3f} |"
            for instruction in ALL_INSTRUCTIONS
        )
        + "\n\n"
        "### Before/after comparison\n\n"
        "| Metric | Attempt 1 (14-sentence vocab) | Attempt 2 (70-sentence vocab) | NN-ceiling (k=1, zero-training) |\n"
        "|--------|-------------------------------|-------------------------------|-----------------------------------|\n"
        f"| Semantic-neighbor accuracy | {ATTEMPT1_NEIGHBOR_ACCURACY:.3f} (4/14) | "
        f"{neighbor_report.accuracy:.3f} ({n_correct}/14) | {NN_CEILING_K1_ACCURACY:.3f} (10/14) |\n"
        f"| Held-out RL success (mean) | {ATTEMPT1_LANGUAGE_MEAN:.3f} | {language_mean:.3f} | "
        "n/a -- geometry-only test, no RL policy involved |\n"
        f"| Held-out RL success (median) | {ATTEMPT1_LANGUAGE_MEDIAN:.3f} | {language_median:.3f} | n/a |\n"
        f"| Held-out RL nonzero samples | {ATTEMPT1_LANGUAGE_NONZERO} | {n_nonzero}/{len(language_all_values)} | "
        "n/a |\n"
    )

    anomalies = (
        f"Semantic-neighbor accuracy more than doubled: {ATTEMPT1_NEIGHBOR_ACCURACY:.3f} (4/14, attempt 1) -> "
        f"{neighbor_report.accuracy:.3f} ({n_correct}/14, attempt 2) -- closer to, though still below, the "
        f"NN-ceiling's {NN_CEILING_K1_ACCURACY:.3f} (10/14). Held-out RL success improved in the same direction "
        f"but by a smaller margin: mean {ATTEMPT1_LANGUAGE_MEAN:.3f} -> {language_mean:.3f}, nonzero samples "
        f"{ATTEMPT1_LANGUAGE_NONZERO} -> {n_nonzero}/{len(language_all_values)} -- still far short of the "
        f"{STAGE3_FINAL_SUCCESS_RATE:.3f} training-vocabulary baseline and still a FAIL-magnitude gap by any "
        "reasonable reading, not graceful degradation. Data augmentation helped -- direction, not magnitude, "
        "is the honest summary.\n\n"
        "**The regression check surfaced a result the task did not anticipate:** the augmented-vocabulary "
        "projection does NOT reproduce stage 3's ~1.000 success rate on the ORIGINAL 14 training instructions "
        f"(mean={regression_mean:.3f}, median={regression_median:.3f}, nonzero={regression_nonzero}/"
        f"{len(regression_all_values)} -- roughly the same order of magnitude as this attempt's own held-out "
        "result, not a clean pass). Root cause, verified directly (not inferred): "
        "`set(AUGMENTED_INSTRUCTIONS) & set(ALL_INSTRUCTIONS) == set()` -- the 70-sentence augmented vocabulary "
        "shares no exact strings with the original 14, so it functions as a *replacement* training set, not an "
        "*extension* of it. From the retrained projection's point of view, the original 14 sentences are now "
        "just as unseen as `held_out_paraphrases` -- which is exactly why their success rate lands in the same "
        "range as the held-out set's, rather than at 1.000. This is evidence *for*, not against, the "
        "generalization diagnosis: a projection trained on 70 diverse phrasings generalizes moderately to *any* "
        "unseen phrasing (original-14 or held-out-14 alike) rather than memorizing one specific closed set "
        "perfectly and failing everywhere else, which is qualitatively the behavior change data augmentation "
        "was supposed to produce -- it just hasn't (yet) produced enough of it to clear either bar.\n\n"
        "Per-instruction pattern: `'move your hand to the right'`/`'reach toward the right side'`/`'shift your "
        "gripper toward the right edge'` (all `reach right`) are disproportionately represented among this "
        "attempt's nonzero successes across both the held-out and regression-check evals -- 5 of the 10 total "
        "nonzero samples across both evals involve a `reach right` instruction, versus 7 regions sharing "
        "roughly equal representation in each vocabulary. Not enough samples to generalize from, but worth "
        "watching if a future attempt investigates per-region variance.\n\n"
        "Compositional placement changed direction on both instructions: `'reach up and to the left'` now lands "
        f"nearest `'reach up high'` with balance {compositional_placements[0].component_distance_balance:.3f} "
        "(near-equidistant between its two components, up from attempt 1's 0.346, which was skewed hard toward "
        f"one side); `'reach forward and down'` now lands nearest `'reach forward'` with balance "
        f"{compositional_placements[1].component_distance_balance:.3f} (up from attempt 1's 0.716). Both moved "
        "toward a more balanced placement between their two named components, consistent with a smoother, less "
        "memorized output geometry -- reported factually, not as a pass/fail signal (per "
        "`held_out_paraphrases.py`'s design, compositional instructions have no single ground-truth region)."
    )

    known_risks_note = (
        "Directly extends ROADMAP.md's 'Projection-layer overfitting to a minimal vocabulary' entry (added "
        "after attempt 1): the reviewer's prescribed fix (data augmentation) produced a real, measured "
        "improvement in the predicted direction (semantic-neighbor accuracy 0.286 -> "
        f"{neighbor_report.accuracy:.3f}, held-out RL mean 0.024 -> {language_mean:.3f}) but did not close the "
        "gap to the proof gate. Not the documented SAC deterministic-eval-collapse signature (literal eval is "
        "a clean 1.000 on all 3 seeds throughout, same checkpoints as attempt 1). Not the 'Metric mismatch' "
        "known risk (same frozen sentence-transformer, same frozen GoalEncoder, only the projection's training "
        "vocabulary changed). New finding worth adding to Known risks: augmenting a fixed-vocabulary projection's "
        "training set without deliberately including the *original* training sentences turns those original "
        "sentences into held-out data for the retrained projection -- if a future stage needs both 'ace the "
        "original vocabulary' and 'generalize to new phrasing' simultaneously, the training set must include "
        "both, not just a larger disjoint replacement set."
    )

    write_report(
        stage=4,
        title="Open vocabulary (Attempt 2: data augmentation fix)",
        seeds=SEEDS,
        candidates=None,
        proof_gate_text=(
            "Graceful degradation on unseen phrasing; semantic neighbors land near each other in goal space."
        ),
        metrics_table=metrics_table,
        chart_paths=[held_out_chart, regression_chart, embedding_chart],
        raw_output_paths=[
            EXPERIMENT_DIR / "runs" / "attempt2" / "projection_train_stdout.log",
            *raw_output_paths,
            *regression_raw_output_paths,
            EXPERIMENT_DIR / "artifacts" / "semantic_neighbor_diagnostic_v2_stdout.log",
        ],
        anomalies=anomalies,
        known_risks_note=known_risks_note,
        out_dir=EXPERIMENT_DIR / "artifacts" / "attempt2_report_scratch",
    )
    print(f"wrote {EXPERIMENT_DIR / 'artifacts' / 'attempt2_report_scratch' / 'report.md'}")


if __name__ == "__main__":
    main()
