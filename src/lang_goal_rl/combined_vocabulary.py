"""The 84-sentence combined reference vocabulary: stage 3's 14 + stage 4's augmented 70.

Stage 4 (open vocabulary) passed by replacing a trained, overfit projection
MLP with a zero-training k=1 nearest-neighbor lookup over this exact
84-sentence set -- `goal_region_vocabulary.ALL_INSTRUCTIONS` (14) unioned
with `augmented_training_vocabulary.AUGMENTED_INSTRUCTIONS` (70), disjoint
strings, original 14 first (see ROADMAP.md's "Resolution (attempt 4...)"
note). That combination first existed only as a one-off script,
`experiments/04_open_vocabulary/combined_vocabulary.py`, written for a single
experiment run. Stage 6 depends on the same 84-sentence set as a
foundational, reusable building block (via `LiveGoalController`, see
`live_goal_controller.py`), not a one-off -- so this module promotes it into
`src/lang_goal_rl/` with tests, keeping the exact sentence set unchanged.

`load_frozen_encoder` (the experiment script's helper for loading stage 2's
checkpoint from a hardcoded experiment-relative path) is deliberately *not*
promoted here: it's experiment-specific path plumbing, not reusable vocabulary
logic. Every function below instead takes an already-loaded `GoalEncoder` as
a parameter, so this module has no dependency on `experiments/` at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from lang_goal_rl.augmented_training_vocabulary import AUGMENTED_INSTRUCTIONS, augmented_instruction_to_region
from lang_goal_rl.goal_region_vocabulary import ALL_INSTRUCTIONS, instruction_to_region
from lang_goal_rl.language_embedding import encode_instructions
from lang_goal_rl.language_goal_projection import DEFAULT_N_TARGET_SAMPLES, precompute_instruction_targets

if TYPE_CHECKING:
    from lang_goal_rl.goal_encoder import GoalEncoder

N_COMBINED_INSTRUCTIONS = 84
"""14 (`goal_region_vocabulary.ALL_INSTRUCTIONS`) + 70
(`augmented_training_vocabulary.AUGMENTED_INSTRUCTIONS`). The two source
vocabularies are string-disjoint (verified by stage 4's
`test_augmented_training_vocabulary.py`), so this is a plain union, not a
dedup count."""


def combined_instructions_and_regions() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the 84 combined instructions and their row-aligned region names.

    Original 14 first, then the augmented 70 -- matching stage 4's already-
    validated ordering, so any caller relying on row position (e.g. a fixed
    seed offset derived from first-occurrence order in
    `precompute_instruction_targets`) reproduces stage 4's exact numbers.

    Returns:
        A tuple `(instructions, region_names)`, both length
        `N_COMBINED_INSTRUCTIONS`, row-aligned.

    """
    augmented_region_map = augmented_instruction_to_region()
    instructions = ALL_INSTRUCTIONS + AUGMENTED_INSTRUCTIONS
    regions = tuple(instruction_to_region(instruction) for instruction in ALL_INSTRUCTIONS) + tuple(
        augmented_region_map[instruction] for instruction in AUGMENTED_INSTRUCTIONS
    )
    return instructions, regions


def build_combined_reference(
    goal_encoder: GoalEncoder,
    *,
    n_samples: int = DEFAULT_N_TARGET_SAMPLES,
    seed: int = 0,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Raw sentence embeddings + fixed region-centroid targets for the combined 84-sentence set.

    Both arrays are computed in one shot -- 84 sentences through the frozen
    sentence-transformer, and each sentence's region centroid through
    `goal_encoder` -- so a caller (e.g. `LiveGoalController`) can precompute
    and cache this once rather than re-deriving it per lookup.

    Args:
        goal_encoder: A frozen `GoalEncoder`, used to compute each row's
            fixed region-centroid regression target (see
            `language_goal_projection.precompute_instruction_targets`).
        n_samples: xyz samples averaged per unique region when computing its
            fixed centroid target -- see `precompute_instruction_targets`.
        seed: Base seed for the region-centroid sampling.

    Returns:
        A tuple `(raw_embeddings, targets)`: shapes
        `(N_COMBINED_INSTRUCTIONS, 384)` and
        `(N_COMBINED_INSTRUCTIONS, goal_encoder.embed_dim)`, row-aligned with
        `combined_instructions_and_regions()`.

    """
    instructions, regions = combined_instructions_and_regions()
    raw_embeddings = encode_instructions(instructions)
    targets = precompute_instruction_targets(
        goal_encoder, regions, n_samples=n_samples, seed=seed,
    ).numpy()
    return raw_embeddings, targets.astype(np.float32)
