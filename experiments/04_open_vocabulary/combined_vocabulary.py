"""Stage 4, attempt 4: the combined 84-sentence reference set (14 original + 70 augmented).

Attempt 3's reviewer verdict recommended a decisive next test: bypass
`LanguageGoalProjection`'s trained MLP entirely and use the zero-training
`nearest_neighbor_projection` (k=1) as the inference-time mapping, drawing on
every known sentence at once -- the original 14 `goal_region_vocabulary.
ALL_INSTRUCTIONS` plus the 70 `augmented_training_vocabulary.
AUGMENTED_INSTRUCTIONS` -- rather than either vocabulary alone. Attempt 2
already confirmed the two sets are string-disjoint (`set(AUGMENTED_
INSTRUCTIONS) & set(ALL_INSTRUCTIONS) == set()`), so concatenating them is a
safe, non-overlapping 84-sentence union, not a dedup problem.

This module is the one place that builds that combined set and its two
derived artifacts (raw 384-dim sentence embeddings, fixed 16-dim region-
centroid regression targets) so `nn_lookup_classification.py` (geometry-only
check) and `eval_nn_lookup_held_out.py` (RL eval) both build it identically
instead of duplicating the logic and risking drift between the two scripts.

Every target is computed via `precompute_instruction_targets` at the exact
`(n_samples=1000, seed=0)` pair every other stage-3/4 target computation used
(`CENTROID_N_SAMPLES`/`CENTROID_SEED` below, matching `nn_ceiling_test.py`'s
and `train.py`'s same-named constants) -- both the original 14's and the
augmented 70's regions are listed in the same canonical `region_names()`
order (center, forward, back, left, right, up, down) at their first
occurrence in the combined list, so `compute_region_target_embeddings`'s
per-unique-region seed offsets land on the identical centroids every prior
script computed, not a separately invented approximation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from lang_goal_rl.augmented_training_vocabulary import AUGMENTED_INSTRUCTIONS, augmented_instruction_to_region
from lang_goal_rl.goal_encoder import GoalEncoder
from lang_goal_rl.goal_region_vocabulary import ALL_INSTRUCTIONS, instruction_to_region
from lang_goal_rl.language_embedding import encode_instructions
from lang_goal_rl.language_goal_projection import precompute_instruction_targets

EXPERIMENT_DIR = Path(__file__).parent
STAGE2_ENCODER_PATH = EXPERIMENT_DIR.parent / "02_contrastive_goal_embedding" / "artifacts" / "goal_encoder.pt"

CENTROID_N_SAMPLES = 1000
"""Matches `language_goal_projection.DEFAULT_N_TARGET_SAMPLES` and every
prior stage-3/4 script's `CENTROID_N_SAMPLES` -- see module docstring."""

CENTROID_SEED = 0
"""Matches `train_projection.PROJECTION_SEED` and every prior stage-3/4
script's `CENTROID_SEED` -- see module docstring."""


def load_frozen_encoder(path: Path = STAGE2_ENCODER_PATH) -> GoalEncoder:
    """Load stage 2's pretrained `GoalEncoder` checkpoint, unchanged.

    Args:
        path: Path to the state dict saved by stage 2's `pretrain_encoder.py`.

    Returns:
        The loaded `GoalEncoder`, in eval mode.

    """
    encoder = GoalEncoder(goal_dim=3)
    encoder.load_state_dict(torch.load(path, map_location="cpu"))
    encoder.eval()
    return encoder


def combined_instructions_and_regions() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the 84 combined instructions and their row-aligned region names.

    84 = 14 (`goal_region_vocabulary.ALL_INSTRUCTIONS`) + 70
    (`augmented_training_vocabulary.AUGMENTED_INSTRUCTIONS`), original-14
    first. The two source vocabularies are disjoint strings (verified in
    attempt 2; re-asserted by callers that care, e.g. `main()` in the two
    scripts that import this module).

    Returns:
        A tuple `(instructions, region_names)`, both length 84, row-aligned.

    """
    augmented_map = augmented_instruction_to_region()
    instructions = tuple(ALL_INSTRUCTIONS) + tuple(AUGMENTED_INSTRUCTIONS)
    regions = tuple(instruction_to_region(instruction) for instruction in ALL_INSTRUCTIONS) + tuple(
        augmented_map[instruction] for instruction in AUGMENTED_INSTRUCTIONS
    )
    return instructions, regions


def build_combined_reference(
    encoder: GoalEncoder,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Raw sentence embeddings + fixed region-centroid targets for the combined 84-sentence set.

    Args:
        encoder: Stage 2's frozen `GoalEncoder`, used to compute each row's
            fixed region-centroid regression target.

    Returns:
        A tuple `(raw_embeddings, targets)`: shapes (84, 384) and (84, 16),
        row-aligned with `combined_instructions_and_regions()`.

    """
    instructions, regions = combined_instructions_and_regions()
    raw_embeddings = encode_instructions(instructions)
    targets = precompute_instruction_targets(
        encoder, regions, n_samples=CENTROID_N_SAMPLES, seed=CENTROID_SEED,
    ).numpy()
    return raw_embeddings, targets
