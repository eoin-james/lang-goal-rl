"""LiveGoalController: stage 6's real-time text -> embedding -> goal loop.

Stage 6 is the capstone stage, wiring together every prior stage's already-
Done mechanism for live use:

1. Stage 3's frozen sentence-transformer (`language_embedding.
   encode_instructions`) turns an arbitrary English sentence into a 384-dim
   embedding.
2. Stage 4's confirmed-best zero-training k=1 nearest-neighbor lookup
   (`nearest_neighbor_projection.nearest_neighbor_projection`) maps that
   384-dim embedding to a 16-dim goal-space embedding, using the 84-sentence
   combined reference vocabulary (`combined_vocabulary.py`) as the known
   points to search. k=1 is the default because stage 4 measured it as
   strictly better than blending (k=3): k=1 always lands exactly on a known
   region centroid with zero directional deviation, while a blend never
   does -- see ROADMAP.md's "Resolution (attempt 4...)" note.

This module owns only the sentence -> goal-embedding step. It does not run
the policy (stage 2's `GoalEmbeddingExtractor`-based SAC) or the mid-episode
substitution (stage 5's `midepisode_regoal.py` /
`episode_recording._pin_desired_goal_embedding`) -- an experiment-runner's
live demo loop wires this class's output into those.

Why a class, not a bare function: the 84-sentence reference's raw sentence
embeddings and fixed region-centroid targets (`combined_vocabulary.
build_combined_reference`) are the same for every query -- re-encoding 84
sentences through the sentence-transformer on every live utterance would add
needless latency to what's supposed to be a live interface. `__init__`
computes and caches that reference exactly once; `instruction_to_goal_embedding`
only ever encodes the one new instruction it's given.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from lang_goal_rl.combined_vocabulary import build_combined_reference
from lang_goal_rl.language_embedding import encode_instructions
from lang_goal_rl.language_goal_projection import DEFAULT_N_TARGET_SAMPLES
from lang_goal_rl.nearest_neighbor_projection import nearest_neighbor_projection

if TYPE_CHECKING:
    from lang_goal_rl.goal_encoder import GoalEncoder

DEFAULT_K = 1
"""Stage 4's confirmed-best mode: exact nearest-neighbor copy-through, no
blending -- see the module docstring."""


class LiveGoalController:
    """Live English instruction -> 16-dim goal embedding, via a cached k-NN reference lookup.

    The 84-sentence combined reference vocabulary's raw sentence embeddings
    and fixed region-centroid targets are computed once at construction (see
    the module docstring) and reused for every call to
    `instruction_to_goal_embedding`.
    """

    def __init__(
        self,
        goal_encoder: GoalEncoder,
        *,
        k: int = DEFAULT_K,
        n_target_samples: int = DEFAULT_N_TARGET_SAMPLES,
        seed: int = 0,
    ) -> None:
        """Precompute and cache the 84-sentence reference the k-NN lookup searches.

        Args:
            goal_encoder: A frozen `GoalEncoder`, used once here to compute
                each reference sentence's fixed region-centroid target (see
                `combined_vocabulary.build_combined_reference`). Never
                mutated.
            k: Number of nearest reference sentences to blend per lookup.
                Defaults to `DEFAULT_K` (1) -- see the module docstring for
                why k=1 is stage 4's confirmed-best mode.
            n_target_samples: xyz samples averaged per unique region when
                computing its fixed centroid target -- see
                `combined_vocabulary.build_combined_reference`.
            seed: Base seed for the region-centroid sampling.

        """
        self._k = k
        self._reference_embeddings, self._reference_targets = build_combined_reference(
            goal_encoder, n_samples=n_target_samples, seed=seed,
        )

    def instruction_to_goal_embedding(self, instruction: str) -> torch.Tensor:
        """Map one live English instruction to its 16-dim goal embedding.

        Encodes only `instruction` itself through the frozen sentence-
        transformer -- the 84-sentence reference this looks up against was
        already computed and cached in `__init__`, not recomputed here.

        Args:
            instruction: An arbitrary English instruction. Not required to
                be one of the 84 reference sentences -- an exact reference
                match (distance 0) is handled correctly by
                `nearest_neighbor_projection` as a degenerate case, but any
                unseen phrasing is equally valid input.

        Returns:
            Tensor of shape `(goal_encoder.embed_dim,)`: the k-NN-blended
            target of the nearest reference instruction(s) to `instruction`
            in raw 384-dim sentence-embedding space. With the default `k=1`
            this is an exact copy of the single nearest reference's known
            target, not an approximation.

        """
        query_embedding = encode_instructions([instruction])[0]
        blended = nearest_neighbor_projection(
            query_embedding, self._reference_embeddings, self._reference_targets, k=self._k,
        )
        return torch.from_numpy(blended).to(torch.float32)
