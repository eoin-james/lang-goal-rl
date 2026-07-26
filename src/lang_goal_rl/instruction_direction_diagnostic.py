"""Diagnostic: how well a projection's output direction tracks its instruction's TRUE region centroid.

Attempt 2's reviewer, reviewing `experiments/03_language_goal_projection/report.md`,
flagged an open question this repo hadn't answered yet: attempt 2's
per-instruction RL success rates varied a lot (0.000 to 0.380 across the 14
fixed instructions), and "center" region instructions scored noticeably
higher than the rest. Is that variation actually explained by how well each
instruction's projected embedding points toward its region's true direction
in the frozen `GoalEncoder`'s space -- or is it a confound (e.g.
FetchReach's fixed success radius just happening to favor goals near the
robot's reset position, independent of projection quality)?

This module computes the measurement needed to investigate that -- per
instruction, the cosine similarity between the projection's output and a
large, non-stochastic sample of its region's true centroid (the same kind
of fixed target `language_goal_projection.precompute_instruction_targets`
now trains against, see that module's "Attempt 3" docstring section). It
deliberately measures *direction only*: cosine similarity is invariant to
positive rescaling, so this is orthogonal to (not a duplicate of)
`language_goal_projection.check_projection_norm_range`'s scale check.

Correlating the resulting per-instruction similarities against attempt 2's
already-collected per-instruction success rates is left to the caller (the
experiment-runner) -- no new RL training or evaluation is needed for that,
since attempt 2's success-rate data already exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as f  # noqa: N812 -- mirrors contrastive.py's common torch.nn.functional alias

from lang_goal_rl.goal_region_vocabulary import MEASURED_GOAL_BOX
from lang_goal_rl.language_goal_projection import DEFAULT_N_TARGET_SAMPLES, precompute_instruction_targets

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lang_goal_rl.goal_encoder import GoalEncoder
    from lang_goal_rl.goal_region_vocabulary import GoalBox
    from lang_goal_rl.language_goal_projection import LanguageGoalProjection


@dataclass(frozen=True)
class InstructionDirectionAlignment:
    """Per-instruction cosine similarity between a projection's output and its region's true centroid.

    Attributes:
        instructions: The instructions measured, in input order.
        region_names: Each instruction's region, same order as `instructions`.
        cosine_similarities: Cosine similarity for each instruction, same
            order. Ranges from -1.0 (exactly opposite direction) through 0.0
            (orthogonal) to 1.0 (exactly the same direction) -- unaffected
            by either vector's magnitude.

    """

    instructions: tuple[str, ...]
    region_names: tuple[str, ...]
    cosine_similarities: tuple[float, ...]

    def as_dict(self) -> dict[str, float]:
        """Map each instruction to its cosine similarity.

        Convenience for a caller correlating this against a separate
        per-instruction success-rate mapping (e.g. attempt 2's already
        collected RL results), keyed the same way.
        """
        return dict(zip(self.instructions, self.cosine_similarities, strict=True))


def cosine_similarity_to_true_centroid(
    projected_embeddings: torch.Tensor,
    true_centroids: torch.Tensor,
    instructions: Sequence[str],
    region_names: Sequence[str],
) -> InstructionDirectionAlignment:
    """Cosine similarity per row, given precomputed projected outputs and their true centroids.

    Args:
        projected_embeddings: A projection's output, shape (n_instructions, embed_dim).
        true_centroids: Each instruction's true region centroid, shape
            (n_instructions, embed_dim), row-aligned with
            `projected_embeddings` (e.g. via
            `language_goal_projection.precompute_instruction_targets`).
        instructions: Label for each row, same length and order.
        region_names: Each instruction's region name, same length and order.

    Returns:
        An `InstructionDirectionAlignment`.

    Raises:
        ValueError: If `projected_embeddings`, `true_centroids`,
            `instructions`, and `region_names` don't all share the same row
            count.

    """
    n = projected_embeddings.shape[0]
    if true_centroids.shape[0] != n or len(instructions) != n or len(region_names) != n:
        msg = (
            f"row count mismatch: projected_embeddings has {n} rows, true_centroids has "
            f"{true_centroids.shape[0]}, instructions has {len(instructions)}, region_names has "
            f"{len(region_names)}"
        )
        raise ValueError(msg)

    with torch.no_grad():
        similarities = f.cosine_similarity(projected_embeddings, true_centroids, dim=1)

    return InstructionDirectionAlignment(
        instructions=tuple(instructions),
        region_names=tuple(region_names),
        cosine_similarities=tuple(float(x) for x in similarities.tolist()),
    )


def measure_instruction_direction_alignment(
    projection: LanguageGoalProjection,
    goal_encoder: GoalEncoder,
    sentence_embeddings: torch.Tensor,
    instructions: Sequence[str],
    region_names: Sequence[str],
    *,
    box: GoalBox = MEASURED_GOAL_BOX,
    n_samples: int = DEFAULT_N_TARGET_SAMPLES,
    seed: int = 0,
) -> InstructionDirectionAlignment:
    """Run the full pipeline: project the vocabulary, compute true centroids, measure cosine similarity.

    Args:
        projection: The (typically trained) `LanguageGoalProjection` to check.
        goal_encoder: Stage 2's frozen encoder, used to compute each
            instruction's true region centroid.
        sentence_embeddings: Frozen sentence embeddings, shape
            (n_instructions, input_dim), row-aligned with `instructions`.
        instructions: Instruction text, one per row of `sentence_embeddings`.
        region_names: Each instruction's region name, same order.
        box: Goal box to sample regions within.
        n_samples: xyz samples averaged per unique region when computing its
            true centroid (see `precompute_instruction_targets`) -- a large,
            non-stochastic sample, not the noisy per-step estimate attempts
            1/2 used.
        seed: Base seed for region sampling.

    Returns:
        An `InstructionDirectionAlignment`.

    """
    with torch.no_grad():
        projected_embeddings = projection(sentence_embeddings)

    true_centroids = precompute_instruction_targets(
        goal_encoder, region_names, box=box, n_samples=n_samples, seed=seed,
    )

    return cosine_similarity_to_true_centroid(projected_embeddings, true_centroids, instructions, region_names)
