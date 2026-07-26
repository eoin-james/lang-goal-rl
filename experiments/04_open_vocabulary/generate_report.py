"""Stage 4: aggregate all three parts (semantic-neighbor, RL held-out eval, compositional) and write `report.md`.

Re-runs `diagnose_open_vocab.py`'s two diagnostics in-process (cheap,
deterministic, no RL -- see that module's docstring for why this is not a
duplicate of its saved log) to build the semantic-neighbor and compositional
tables/chart. Parses `eval_held_out.py`'s per-seed stdout logs
(`runs/seed_<k>/stdout.log`, already run and saved -- this script performs
no RL itself) for the held-out RL success-rate half, mirroring stage 3's
`generate_report_attempt4.py` regex-parsing pattern.
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
    STAGE3_PROJECTION_PATH,
    load_frozen_encoder,
    load_projection,
    run_compositional_diagnostic,
    run_semantic_neighbor_diagnostic,
)
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
03_language_goal_projection/report.md`'s Attempt 4 section -- the baseline
this stage's held-out (never-trained-on) phrasing result is compared
against."""


def parse_seed_log(log_path: Path) -> tuple[float, dict[str, float]]:
    """Parse one seed's literal success rate and per-instruction held-out language success rates.

    Args:
        log_path: Path to the seed's `stdout.log` under `runs/seed_<k>/`.

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
    """Aggregate all three stage-4 parts and write `report.md`."""
    # --- Part 1: semantic-neighbor diagnostic (re-run in-process, no RL) ---
    projection = load_projection(STAGE3_PROJECTION_PATH)
    encoder = load_frozen_encoder(STAGE2_ENCODER_PATH)
    neighbor_report = run_semantic_neighbor_diagnostic(projection)

    # --- Part 3: compositional placement (re-run in-process, no RL) ---
    compositional_placements = run_compositional_diagnostic(projection, encoder)

    # --- Part 2: RL success rate on held-out phrasings (parsed from already-run logs) ---
    literal_rates: dict[int, float] = {}
    language_rates_by_seed: dict[int, dict[str, float]] = {}
    raw_output_paths = []
    for seed in SEEDS:
        log_path = EXPERIMENT_DIR / "runs" / f"seed_{seed}" / "stdout.log"
        literal_rate, language_rates = parse_seed_log(log_path)
        literal_rates[seed] = literal_rate
        language_rates_by_seed[seed] = language_rates
        raw_output_paths.append(log_path)

    language_all_values = [rate for seed in SEEDS for rate in language_rates_by_seed[seed].values()]
    language_mean_per_seed = {seed: statistics.mean(language_rates_by_seed[seed].values()) for seed in SEEDS}
    language_mean = statistics.mean(language_all_values)
    language_median = statistics.median(language_all_values)
    n_nonzero = sum(1 for value in language_all_values if value > 0)

    # Per-instruction mean across the 3 seeds (row-aligned with held_out_texts()/held_out_region_names()).
    per_instruction_mean = {
        instruction: statistics.mean(language_rates_by_seed[seed][instruction] for seed in SEEDS)
        for instruction in held_out_texts()
    }

    # --- Chart 1: per-seed held-out RL success rate vs. stage-3's final (training-vocabulary) baseline ---
    rl_chart = plot_multi_seed_success_rate(
        {f"seed_{seed}": list(language_rates_by_seed[seed].values()) for seed in SEEDS},
        out_path=EXPERIMENT_DIR / "charts" / "held_out_success_rate.png",
        proof_gate_threshold=STAGE3_FINAL_SUCCESS_RATE,
    )

    # --- Chart 2: embedding-projection PCA, held-out + compositional relative to the training vocabulary ---
    train_sentence_embeddings = torch.from_numpy(encode_instructions(ALL_INSTRUCTIONS))
    held_out_sentence_embeddings = torch.from_numpy(encode_instructions(held_out_texts()))
    compositional_sentence_embeddings = torch.from_numpy(encode_instructions(compositional_texts()))
    with torch.no_grad():
        train_projected = projection(train_sentence_embeddings)
        held_out_projected = projection(held_out_sentence_embeddings)
        compositional_projected = projection(compositional_sentence_embeddings)

    all_embeddings = np.concatenate(
        [
            train_projected.numpy(),
            held_out_projected.numpy(),
            compositional_projected.numpy(),
        ],
        axis=0,
    )
    all_labels = (
        [f"{instruction_to_region(instruction)}·train" for instruction in ALL_INSTRUCTIONS]
        + [f"{region}·held-out" for region in held_out_region_names()]
        + ["compositional"] * len(COMPOSITIONAL_INSTRUCTIONS)
    )
    embedding_chart = plot_embedding_projection(
        all_embeddings,
        all_labels,
        out_path=EXPERIMENT_DIR / "charts" / "embedding_projection_open_vocab.png",
    )

    metrics_table = (
        "### Part 1 -- Semantic-neighbor diagnostic (no RL; frozen sentence-transformer + "
        "stage-3 projection only)\n\n"
        "Reference set: the 14 *training* instructions' own projected embeddings "
        "(`goal_region_vocabulary.ALL_INSTRUCTIONS` run through the same, unchanged "
        "`language_goal_projection_v3.pt`) -- chosen over region centroids because the proof gate "
        "asks whether the projection's actual output geometry for real sentences places semantic "
        "neighbors near each other, not whether it lands near a separately-computed idealized average "
        "(see `diagnose_open_vocab.py`'s module docstring; independently re-checked against region "
        "centroids instead and the accuracy did not change, 0.286 either way).\n\n"
        f"**Aggregate accuracy: {neighbor_report.accuracy:.3f} ({sum(1 for r in neighbor_report.results if r.is_correct)}/"
        f"{len(neighbor_report.results)}) -- vs. a 1/7 ≈ 0.143 random-region-assignment baseline "
        "(2x chance, but far from reliable).**\n\n"
        f"{_semantic_neighbor_table(neighbor_report)}\n\n"
        "### Part 2 -- RL success rate on held-out phrasings (the actual generalization test)\n\n"
        "Same 3 already-trained SAC checkpoints and same stage-3 fixed-centroid-regression projection "
        "checkpoint as stage 3's attempt 4 -- no retraining. Ground truth judged against each "
        "instruction's region centroid (`train.compute_region_centroid`), applying the region-vs-point "
        "lesson from the start, per `ROADMAP.md`'s Known risks.\n\n"
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
        f"**Comparison to stage 3's final (training-vocabulary) baseline** ({STAGE3_FINAL_SUCCESS_RATE:.3f}, "
        "attempt 4, same checkpoints and same projection, 14 *trained-on* instructions): "
        f"{language_median:.3f} median vs. {STAGE3_FINAL_SUCCESS_RATE:.3f} -- generalization to unseen "
        "phrasing collapses almost entirely; literal-goal control stays a clean 1.000 on all 3 seeds "
        "(same checkpoints, unchanged), so this is specific to the held-out projections landing off-target, "
        "not a policy or checkpoint regression.\n\n"
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
        f"{_compositional_table(compositional_placements)}\n"
    )

    anomalies = (
        f"Held-out RL success rate collapsed to near-zero: {n_nonzero}/{len(language_all_values)} samples "
        f"nonzero (mean {language_mean:.3f}, median {language_median:.3f}), against a {STAGE3_FINAL_SUCCESS_RATE:.3f} "
        "training-vocabulary baseline using the identical checkpoints, projection, and eval protocol -- the only "
        "variable that changed is which 14 instructions were projected. This tracks the semantic-neighbor "
        f"diagnostic's finding directly: only {sum(1 for r in neighbor_report.results if r.is_correct)}/"
        f"{len(neighbor_report.results)} held-out paraphrases' projected embeddings land nearest their own true "
        "region, so most held-out instructions send the policy toward the wrong region entirely -- well outside "
        "FetchReach's 0.05m success radius, not a near-miss. Literal-goal control is unchanged and a clean 1.000 "
        "on all 3 seeds, ruling out a policy or checkpoint problem; this is specific to the projection's "
        "direction accuracy for sentence embeddings it was never trained on.\n\n"
        "The one nonzero held-out sample ('raise your arm as high as it will go', seed 1, "
        f"{language_rates_by_seed[1]['raise your arm as high as it will go']:.3f}) "
        "is also the semantic-neighbor diagnostic's clearest correct classification for that region (both "
        "'reach up high' held-out paraphrases classify correctly) -- consistent with, not contradicting, the "
        "overall pattern rather than a random outlier.\n\n"
        "Compositional instructions ('reach up and to the left', 'reach forward and down') both land nearest "
        "one of their two named component regions (not a third, unrelated region), and both skew toward one "
        "component more than the other (balance 0.346 and 0.716) rather than sitting exactly on the midline "
        "between them -- the projection resolves a compositional phrase to 'closer to one direction' rather "
        "than an even blend or a random unrelated point, even though it was never built or trained to handle "
        "composition."
    )

    known_risks_note = (
        "Not the documented SAC deterministic-eval-collapse signature (literal eval is a clean 1.000 on all 3 "
        "seeds, same checkpoints as stages 2/3, no retraining). Not the 'Metric mismatch' known risk either "
        "(nothing about the sentence-embedding or distance-reward metric changed -- same frozen "
        "sentence-transformer, same frozen GoalEncoder, same projection checkpoint as stage 3's passing "
        "attempt 4). This result directly applies the region-vs-point eval-protocol lesson from ROADMAP.md's "
        "Known risks from the start (ground truth judged against `compute_region_centroid`, never a resampled "
        "point) -- so the near-zero held-out success rate is not a repeat of that defect; it reflects a new, "
        "distinct finding this stage exists to surface: `LanguageGoalProjection`, trained via direct regression "
        "on a closed 14-instruction vocabulary with no generalization pressure (no held-out validation set, no "
        "regularization term encouraging smooth interpolation between training points), does not generalize "
        "its *direction* accuracy to unseen phrasings -- it graceful-degrades on the diagnostic (28.6% vs. "
        "14.3% chance, better than random) far more than it does on the actual RL task (a region miss of even "
        "a few centimeters misses FetchReach's tight 0.05m success radius entirely). Worth tracking as a new "
        "Known risks entry before stage 5/6 build on this projection unchanged."
    )

    report_path = write_report(
        stage=4,
        title="Open vocabulary",
        seeds=SEEDS,
        candidates=None,
        proof_gate_text=(
            "Graceful degradation on unseen phrasing; semantic neighbors land near each other in goal space."
        ),
        metrics_table=metrics_table,
        chart_paths=[rl_chart, embedding_chart],
        raw_output_paths=[
            *raw_output_paths,
            EXPERIMENT_DIR / "artifacts" / "semantic_neighbor_diagnostic_stdout.log",
        ],
        anomalies=anomalies,
        known_risks_note=known_risks_note,
        out_dir=EXPERIMENT_DIR,
    )
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
