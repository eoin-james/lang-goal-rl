"""Stage 11 proof-gate evaluation: command-type classifier held-out accuracy.

Pure NLP classification -- no RL policy, no env, no episode seeds. "Seeds"
here means random seeds for the *classifier's training procedure* (weight
init via `torch.manual_seed`), not env/policy seeds; there is no `checkpoints/
seed_<k>.zip` to load and no rollout involved anywhere in this script.

Trains `CommandTypeClassifier` (`command_type_classifier.py`) on the labeled
examples from `command_type_vocabulary.build_command_type_training_set` and
evaluates on the disjoint held-out examples from
`command_type_held_out_vocabulary.build_command_type_held_out_set`, per the
Phase 2b plan's proof gate:

    On the held-out set: (a) >=90% overall top-1 accuracy across the 5
    classes, and (b) 0% of held-out UNSUPPORTED sentences classified as
    anything actionable (MOVE/GOTO_NAMED_REGION/STOP/RESET) -- reported as a
    separate, stricter sub-metric.

ATTEMPT 2 (`runs_v2/`): rl-builder redesigned `command_type_vocabulary.py`
and `command_type_held_out_vocabulary.py` (2026-07-24 fix documented in both
modules' docstrings) after attempt 1 (`runs/`, see `report.md`/
`evidence.md`'s dated attempt-1 section) failed the gate at ~67-69% overall
accuracy with held-out MOVE at a flat 0%, because `GOTO_NAMED_REGION`'s old
training phrasing ("angle your hand toward the front") was structurally
identical to a natural MOVE sentence. The new vocabulary gives MOVE a
magnitude-cued relative-displacement convention ("shift left a bit") and
GOTO_NAMED_REGION an absolute-destination convention ("go to the far left
side"); `check_cross_class_embedding_overlap` (new in
`command_type_vocabulary.py`) confirms the old vocabulary's GOTO->MOVE
nearest-neighbor collision rate was 11.9%, the new one's is 0.0%. Attempt 2
cleared gate (a) (94.2-96.2% overall, 12/12 runs) but only partially cleared
gate (b): STOP was flat at 75% (6/8, every run) on two phrasings ("cut it
out immediately", "no more movement please"), and UNSUPPORTED's sub-gate
failed 10/12 runs on one recurring miss ("calculate the square root of
nine" -> RESET, no math-question training examples existed).

ATTEMPT 3 (this run, writes to `runs_v3/`): rl-builder closed both attempt-2
gaps additively, training-set-only, no architecture change and no held-out
change. `_STOP_PHRASINGS` grew 18->26 (added idiomatic/negation cessation
phrasings with no explicit stop-keyword, e.g. "cut it out", "no more of
that" -- the training set previously had zero examples of that shape, only
direct stop-keyword imperatives). `_UNSUPPORTED_PHRASINGS` grew 26->31
(added math/calculation phrasings, e.g. "solve this equation for x", "find
the square root of sixteen" -- the training set previously had no semantic
anchor for arithmetic requests at all). `command_type_held_out_vocabulary.py`
is completely unchanged from attempt 2 -- same 52 held-out examples,
including the exact same "cut it out immediately", "no more movement
please", and "calculate the square root of nine" that failed before. Class
sizes are now GOTO_NAMED_REGION=84, MOVE=90, STOP=26, RESET=18,
UNSUPPORTED=31 -- this script still reads them live from the vocabulary
modules rather than hardcoding counts.

This script otherwise reruns the identical handful of training
configurations tried in attempts 1 and 2 against the updated training
vocabulary -- per the task brief, the goal is to confirm whether the
additive data fix alone clears the gate, not to explore new
hyperparameters. `train_command_type_classifier`'s own hyperparameters (200
epochs, lr=1e-3, plain unweighted cross-entropy) were only ever exercised as
an informal smoke check by the module's author, not against this gate. This
script also has explicit latitude (see the task brief) to try a handful of
different training configurations -- different epoch counts, learning
rates, and a class-balanced loss weighting computed here (not added to
`command_type_classifier.py`, since editing `src/` is out of scope for this
stage) -- to see whether the gate is reachable, without ever touching the
held-out set itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as f  # noqa: N812 -- mirrors command_type_classifier.py's own alias

from lang_goal_rl.command_type_classifier import (
    CommandTypeClassifier,
    classify_command_type,
    train_command_type_classifier,
)
from lang_goal_rl.command_type_held_out_vocabulary import build_command_type_held_out_set
from lang_goal_rl.command_type_vocabulary import CommandType, build_command_type_training_set
from lang_goal_rl.language_embedding import encode_instructions
from lang_goal_rl.reporting import plot_multi_seed_success_rate, plot_training_curve

torch.set_num_threads(1)  # CONTRACTS.md concurrency convention -- avoid oversubscribing cores

OUT_DIR = Path(__file__).parent
# Attempt 1's `runs/` (pre-vocabulary-fix, FAILED) and attempt 2's `runs_v2/`
# (partial pass -- gate (a) only) are left untouched -- this rerun writes to
# a separate versioned directory, the same convention stage 3 used across
# its own multiple attempts (runs/, runs_v2/, runs_v3/, ...).
RUNS_DIR = OUT_DIR / "runs_v3"
CHARTS_DIR = OUT_DIR / "charts"
CLASS_ORDER: tuple[CommandType, ...] = tuple(CommandType)
ACTIONABLE_TYPES = {CommandType.MOVE, CommandType.GOTO_NAMED_REGION, CommandType.STOP, CommandType.RESET}
TRAINING_PROCEDURE_SEEDS = (0, 1, 2)


def _confusion_matrix(true_labels: list[CommandType], pred_labels: list[CommandType]) -> np.ndarray:
    """Build a 5x5 confusion matrix, rows=true class, cols=predicted class, ordered by `CLASS_ORDER`."""
    index_of = {command_type: index for index, command_type in enumerate(CLASS_ORDER)}
    matrix = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=int)
    for true_label, pred_label in zip(true_labels, pred_labels, strict=True):
        matrix[index_of[true_label], index_of[pred_label]] += 1
    return matrix


def _evaluate(classifier: CommandTypeClassifier, held_out_examples: list, held_out_embeddings: torch.Tensor) -> dict:
    """Evaluate a trained classifier against the held-out set and compute every proof-gate metric.

    Args:
        classifier: Trained `CommandTypeClassifier`.
        held_out_examples: The 52 `LabeledCommandExample`s (for ground truth).
        held_out_embeddings: Precomputed frozen embeddings, same order as `held_out_examples`.

    Returns:
        Dict with overall_accuracy, per_class_accuracy, confusion_matrix (list of lists),
        unsupported_actionable_count/rate, and per-example predictions for the raw log.
    """
    with torch.no_grad():
        logits = classifier(held_out_embeddings)
    predicted_indices = torch.argmax(logits, dim=1).tolist()
    predicted_labels = [CLASS_ORDER[index] for index in predicted_indices]
    true_labels = [example.command_type for example in held_out_examples]

    correct = [pred == true for pred, true in zip(predicted_labels, true_labels, strict=True)]
    overall_accuracy = sum(correct) / len(correct)

    per_class_accuracy = {}
    for command_type in CLASS_ORDER:
        class_mask = [true == command_type for true in true_labels]
        n_class = sum(class_mask)
        if n_class == 0:
            per_class_accuracy[command_type.value] = None
            continue
        class_correct = sum(c for c, m in zip(correct, class_mask, strict=True) if m)
        per_class_accuracy[command_type.value] = class_correct / n_class

    unsupported_predictions = [
        pred for pred, true in zip(predicted_labels, true_labels, strict=True) if true == CommandType.UNSUPPORTED
    ]
    n_unsupported = len(unsupported_predictions)
    unsupported_actionable_count = sum(1 for pred in unsupported_predictions if pred in ACTIONABLE_TYPES)
    unsupported_actionable_rate = unsupported_actionable_count / n_unsupported if n_unsupported else None

    confusion_matrix = _confusion_matrix(true_labels, predicted_labels)

    return {
        "overall_accuracy": overall_accuracy,
        "n_held_out": len(true_labels),
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion_matrix.tolist(),
        "n_unsupported": n_unsupported,
        "unsupported_actionable_count": unsupported_actionable_count,
        "unsupported_actionable_rate": unsupported_actionable_rate,
        "predictions": [
            {"text": example.text, "true": example.command_type.value, "predicted": pred.value}
            for example, pred in zip(held_out_examples, predicted_labels, strict=True)
        ],
    }


def _plot_confusion_matrix(matrix: list[list[int]], *, out_path: Path, title: str) -> Path:
    """Render a `CLASS_ORDER`-ordered confusion matrix as an annotated heatmap PNG.

    No shared `reporting.py` helper exists for a confusion matrix (that
    module's chart functions are all success-rate/embedding-scatter shaped),
    so this is a small, stage-11-local plotting function -- in scope here
    since it lives in `experiments/`, not `src/`.

    Args:
        matrix: 5x5 confusion matrix, rows=true class, cols=predicted class,
            ordered by `CLASS_ORDER`.
        out_path: Destination PNG path; parent directories are created if
            missing.
        title: Chart title (e.g. names the config the matrix was drawn from).

    Returns:
        The path the PNG was written to (same as `out_path`).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [command_type.value for command_type in CLASS_ORDER]
    array = np.array(matrix)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(array, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            text_color = "white" if array[row, col] > array.max() / 2 else "black"
            ax.text(col, row, str(array[row, col]), ha="center", va="center", color=text_color)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _train_weighted(
    sentence_embeddings: torch.Tensor,
    labels: list[CommandType],
    *,
    epochs: int,
    learning_rate: float,
    seed: int,
    class_weights: torch.Tensor,
) -> tuple[CommandTypeClassifier, list[float]]:
    """Class-balanced-loss variant of `train_command_type_classifier`.

    Mirrors `command_type_classifier.train_command_type_classifier`'s training
    loop exactly, but passes `weight=class_weights` into `cross_entropy` --
    inverse-frequency weighting so the 90-example MOVE class and the
    15-example STOP/RESET classes don't get equal total gradient mass just
    because they're equal-magnitude single training steps. This is deliberately
    NOT added to `command_type_classifier.py` itself (out of scope for this
    stage -- src/ changes are rl-builder's domain), just a locally-defined
    alternative training procedure to test whether class-balancing moves the
    held-out accuracy needle at all.

    Args:
        sentence_embeddings: Frozen training embeddings, shape (n_examples, input_dim).
        labels: Ground-truth `CommandType` per example, same order as `sentence_embeddings`.
        epochs: Number of optimizer steps.
        learning_rate: Adam learning rate.
        seed: Seed for the classifier's weight initialization.
        class_weights: Per-class weight tensor, ordered by `CLASS_ORDER`, passed to
            `torch.nn.functional.cross_entropy`'s `weight` argument.

    Returns:
        A tuple `(classifier, loss_history)`, same shape as `train_command_type_classifier`'s return.
    """
    torch.manual_seed(seed)
    classifier = CommandTypeClassifier(input_dim=sentence_embeddings.shape[1])
    index_of = {command_type: index for index, command_type in enumerate(CLASS_ORDER)}
    target_indices = torch.tensor([index_of[label] for label in labels], dtype=torch.long)
    frozen_embeddings = sentence_embeddings.detach().to(torch.float32)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    loss_history: list[float] = []
    for _epoch in range(epochs):
        logits = classifier(frozen_embeddings)
        loss = f.cross_entropy(logits, target_indices, weight=class_weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))
    return classifier, loss_history


def main() -> None:
    """Run every tuning config across `TRAINING_PROCEDURE_SEEDS`, evaluate against the proof gate, and log raw output."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    train_examples = list(build_command_type_training_set(seed=0))
    held_out_examples = list(build_command_type_held_out_set())

    train_texts = [example.text for example in train_examples]
    held_out_texts = [example.text for example in held_out_examples]
    train_labels = [example.command_type for example in train_examples]

    train_embeddings = torch.from_numpy(encode_instructions(train_texts))
    held_out_embeddings = torch.from_numpy(encode_instructions(held_out_texts))

    # Inverse-frequency class weights for the class-balanced config, computed live from
    # the training set's own label counts (whatever they are this attempt) -- not a
    # hardcoded guess, so a vocabulary-side class-count change never goes stale here.
    index_of = {command_type: index for index, command_type in enumerate(CLASS_ORDER)}
    class_counts = np.zeros(len(CLASS_ORDER))
    for label in train_labels:
        class_counts[index_of[label]] += 1
    class_weights = torch.tensor(class_counts.sum() / (len(CLASS_ORDER) * class_counts), dtype=torch.float32)

    configs = {
        "baseline_200ep_lr1e-3": {"epochs": 200, "learning_rate": 1e-3, "weighted": False},
        "more_epochs_1000ep_lr1e-3": {"epochs": 1000, "learning_rate": 1e-3, "weighted": False},
        "lower_lr_2000ep_lr5e-4": {"epochs": 2000, "learning_rate": 5e-4, "weighted": False},
        "class_balanced_500ep_lr1e-3": {"epochs": 500, "learning_rate": 1e-3, "weighted": True},
    }

    all_results: dict[str, list[dict]] = {}
    for config_name, config in configs.items():
        config_results = []
        for seed in TRAINING_PROCEDURE_SEEDS:
            if config["weighted"]:
                classifier, loss_history = _train_weighted(
                    train_embeddings,
                    train_labels,
                    epochs=config["epochs"],
                    learning_rate=config["learning_rate"],
                    seed=seed,
                    class_weights=class_weights,
                )
            else:
                classifier, loss_history = train_command_type_classifier(
                    train_embeddings,
                    train_labels,
                    epochs=config["epochs"],
                    learning_rate=config["learning_rate"],
                    seed=seed,
                )

            eval_result = _evaluate(classifier, held_out_examples, held_out_embeddings)
            eval_result["config"] = config_name
            eval_result["seed"] = seed
            eval_result["final_train_loss"] = loss_history[-1]
            eval_result["loss_history"] = loss_history
            config_results.append(eval_result)

            run_path = RUNS_DIR / f"{config_name}_seed_{seed}.json"
            run_path.write_text(json.dumps(eval_result, indent=2))

            print(
                f"[{config_name}][seed={seed}] "
                f"overall_top1_accuracy={eval_result['overall_accuracy']:.4f} "
                f"({eval_result['overall_accuracy'] * eval_result['n_held_out']:.0f}/{eval_result['n_held_out']}) "
                f"unsupported_actionable_rate={eval_result['unsupported_actionable_rate']:.4f} "
                f"({eval_result['unsupported_actionable_count']}/{eval_result['n_unsupported']}) "
                f"final_train_loss={eval_result['final_train_loss']:.4f}",
            )
        all_results[config_name] = config_results

    summary_path = RUNS_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                config_name: [
                    {
                        "seed": r["seed"],
                        "overall_accuracy": r["overall_accuracy"],
                        "unsupported_actionable_rate": r["unsupported_actionable_rate"],
                        "per_class_accuracy": r["per_class_accuracy"],
                        "final_train_loss": r["final_train_loss"],
                    }
                    for r in results
                ]
                for config_name, results in all_results.items()
            },
            indent=2,
        ),
    )

    # Sanity check demonstrating classify_command_type end-to-end on one held-out example
    # using the last-trained classifier (a smoke check, not part of any reported metric).
    sample_text = held_out_examples[0].text
    sample_prediction = classify_command_type(sample_text, classifier)
    print(f"[sanity] classify_command_type({sample_text!r}) -> {sample_prediction.value}")

    # Charts (v3 suffix -- attempts 1 and 2's charts/*.png are left in place,
    # same versioning convention as runs_v3/ above).
    accuracy_by_config = {
        config_name: [r["overall_accuracy"] for r in results] for config_name, results in all_results.items()
    }
    plot_multi_seed_success_rate(
        accuracy_by_config,
        out_path=CHARTS_DIR / "held_out_accuracy_by_config_v3.png",
        proof_gate_threshold=0.90,
    )

    baseline_seed0 = next(r for r in all_results["baseline_200ep_lr1e-3"] if r["seed"] == 0)
    _plot_confusion_matrix(
        baseline_seed0["confusion_matrix"],
        out_path=CHARTS_DIR / "confusion_matrix_baseline_v3.png",
        title="Confusion matrix -- baseline_200ep_lr1e-3, seed 0 (attempt 3, coverage-gap fix)",
    )
    plot_training_curve(
        list(range(len(baseline_seed0["loss_history"]))),
        baseline_seed0["loss_history"],
        ylabel="training loss (cross-entropy)",
        out_path=CHARTS_DIR / "training_loss_baseline_seed0_v3.png",
        seed=0,
    )


if __name__ == "__main__":
    main()
