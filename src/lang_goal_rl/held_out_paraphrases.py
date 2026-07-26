"""Stage 4's held-out vocabulary: new phrasings of stage 3's 7 regions, never trained on.

Stage 3's fixed 14-instruction vocabulary (`goal_region_vocabulary.ALL_INSTRUCTIONS`)
is what `language_goal_projection.train_projection` regresses toward. Stage
4's proof gate ("graceful degradation on unseen phrasing") needs a second,
disjoint set of instructions that are never fed to `train_projection` --
only encoded through the frozen sentence-transformer and projection at eval
time, so a caller can measure whether the projection generalizes to wording
it has never seen rather than just memorizing 14 fixed strings.

Two kinds of held-out instruction live here:

1. `HELD_OUT_PARAPHRASES` -- 2 genuinely new phrasings per existing region
   (verb/structure changes, not single-word synonym swaps -- see
   `tests/lang_goal_rl/test_held_out_paraphrases.py`'s trivial-edit check).
   Each has a real, known ground-truth region, so a caller can score
   "did the held-out phrasing land nearest its own region" directly (see
   `semantic_neighbor_diagnostic.diagnose_semantic_neighbors`).

2. `COMPOSITIONAL_INSTRUCTIONS` -- instructions naming two of the 7 regions
   at once (e.g. "reach up and to the left"). The current region design has
   no single region representing a diagonal/combined direction, so these
   have no ground-truth region to score against -- that's not a gap to
   patch here by inventing new region machinery, it's the honest shape of
   the question stage 4 asks: what does the *existing* pipeline do with a
   phrase it has no dedicated target for? `component_region_names` records
   which two regions the instruction combines, so
   `semantic_neighbor_diagnostic.diagnose_compositional_placement` can
   report where the projected embedding actually lands relative to those
   two regions' centroids, without asserting a pass/fail verdict.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeldOutParaphrase:
    """One held-out instruction and the existing region it's a new phrasing of.

    Attributes:
        text: The held-out English instruction. Never present in
            `goal_region_vocabulary.ALL_INSTRUCTIONS`.
        region_name: The existing region (see `goal_region_vocabulary.region_names`)
            this phrasing's ground truth is. Used as the expected answer in
            `semantic_neighbor_diagnostic.diagnose_semantic_neighbors`.

    """

    text: str
    region_name: str


HELD_OUT_PARAPHRASES: tuple[HeldOutParaphrase, ...] = (
    HeldOutParaphrase("settle into the middle of the workspace", "center"),
    HeldOutParaphrase("return your hand to a neutral position", "center"),
    HeldOutParaphrase("push your arm out in front of you", "reach forward"),
    HeldOutParaphrase("extend forward away from your body", "reach forward"),
    HeldOutParaphrase("draw your hand back toward yourself", "reach back"),
    HeldOutParaphrase("retreat away from the front of the workspace", "reach back"),
    HeldOutParaphrase("swing your arm over to the left", "reach left"),
    HeldOutParaphrase("shift your gripper toward the left edge", "reach left"),
    HeldOutParaphrase("swing your arm over to the right", "reach right"),
    HeldOutParaphrase("shift your gripper toward the right edge", "reach right"),
    HeldOutParaphrase("raise your arm as high as it will go", "reach up high"),
    HeldOutParaphrase("extend upward toward the ceiling", "reach up high"),
    HeldOutParaphrase("lower your arm toward the floor", "reach down low"),
    HeldOutParaphrase("drop your gripper down low", "reach down low"),
)
"""14 held-out phrasings, 2 per existing region, disjoint from
`goal_region_vocabulary.ALL_INSTRUCTIONS`. Never used to train the
projection -- test-only."""


@dataclass(frozen=True)
class CompositionalInstruction:
    """An instruction naming two existing regions at once, with no single ground-truth region.

    Attributes:
        text: The compositional English instruction.
        component_region_names: The two existing regions this instruction
            combines, in the order they're mentioned in `text`.

    """

    text: str
    component_region_names: tuple[str, str]


COMPOSITIONAL_INSTRUCTIONS: tuple[CompositionalInstruction, ...] = (
    CompositionalInstruction(
        "reach up and to the left", ("reach up high", "reach left")
    ),
    CompositionalInstruction(
        "reach forward and down", ("reach forward", "reach down low")
    ),
)
"""Instructions the current 7-region design has no dedicated target for.
Deliberately not given a `region_name` -- scoring these against a single
"correct" region would misrepresent what an honest open-vocabulary eval is
testing. See `semantic_neighbor_diagnostic.diagnose_compositional_placement`
for how these are measured instead."""


def held_out_texts() -> tuple[str, ...]:
    """Return every `HELD_OUT_PARAPHRASES` text, in `HELD_OUT_PARAPHRASES` order."""
    return tuple(paraphrase.text for paraphrase in HELD_OUT_PARAPHRASES)


def held_out_region_names() -> tuple[str, ...]:
    """Return every `HELD_OUT_PARAPHRASES` ground-truth region name, row-aligned with `held_out_texts`."""
    return tuple(paraphrase.region_name for paraphrase in HELD_OUT_PARAPHRASES)


def compositional_texts() -> tuple[str, ...]:
    """Return every `COMPOSITIONAL_INSTRUCTIONS` text, in `COMPOSITIONAL_INSTRUCTIONS` order."""
    return tuple(instruction.text for instruction in COMPOSITIONAL_INSTRUCTIONS)
