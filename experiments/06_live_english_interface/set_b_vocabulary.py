"""Stage 6's Set B: 7 brand-new phrasings, never used anywhere in this project before.

Stage 6's proof gate is "ad-hoc live phrasings" -- Set A
(`lang_goal_rl.held_out_paraphrases.HELD_OUT_PARAPHRASES`) is a reuse of
stage 4's already-measured 14 held-out paraphrases, which is valuable as a
sanity cross-check (this experiment's harness should approximately reproduce
stage 4's 0.571 mean/1.000 median k=1 result under the same mechanism) but
does not, by itself, test genuinely novel input the way "ad-hoc" implies --
those 14 phrasings have technically been measured once before. Set B is the
part of this experiment that actually tests novelty: one new instruction per
region (7 total, spanning every region in
`goal_region_vocabulary.region_names()`), worded differently from anything
in `goal_region_vocabulary.ALL_INSTRUCTIONS`,
`augmented_training_vocabulary.AUGMENTED_INSTRUCTIONS`, or
`held_out_paraphrases.HELD_OUT_PARAPHRASES`/`.COMPOSITIONAL_INSTRUCTIONS`.

Disjointness is checked the same way stage 4's own vocabulary additions were
checked (see `tests/lang_goal_rl/test_augmented_training_vocabulary.py`):
exact case-insensitive match, and the "trivial single-word edit" guard (same
word count, differing at exactly one word position) -- `verify_disjoint`
below runs both checks. This module lives under `experiments/` (not
`src/lang_goal_rl/`), so unlike `augmented_training_vocabulary.py` it has no
mirrored `tests/` file; `live_regoal_eval.py` calls `verify_disjoint()` as a
runtime guard before running any RL eval, raising immediately if a future
edit to these 7 sentences reintroduces an overlap.
"""

from __future__ import annotations

from dataclasses import dataclass

from lang_goal_rl.augmented_training_vocabulary import AUGMENTED_INSTRUCTIONS
from lang_goal_rl.goal_region_vocabulary import ALL_INSTRUCTIONS
from lang_goal_rl.held_out_paraphrases import compositional_texts, held_out_texts


@dataclass(frozen=True)
class SetBInstruction:
    """One Set B instruction and the region it targets.

    Attributes:
        text: The brand-new English instruction.
        region_name: The region (`goal_region_vocabulary.region_names()`)
            this instruction's ground truth is.

    """

    text: str
    region_name: str


SET_B_INSTRUCTIONS: tuple[SetBInstruction, ...] = (
    SetBInstruction("keep the robotic hand hovering exactly at the workspace's midpoint", "center"),
    SetBInstruction("push the end effector forward, away from the robot's base", "reach forward"),
    SetBInstruction("bring the arm back in, closer to the robot's chassis", "reach back"),
    SetBInstruction("carry the hand across the workspace toward its left boundary", "reach left"),
    SetBInstruction(
        "swing the arm over so the gripper reaches the right-hand side of the space", "reach right",
    ),
    SetBInstruction("send the arm climbing toward the highest point it can reach", "reach up high"),
    SetBInstruction("let the arm descend toward the lowest point it can reach", "reach down low"),
)
"""7 instructions, one per region, worded distinctly from every existing
vocabulary this project has used (stage 3's training set, stage 4's
augmented training set, stage 4's held-out/compositional test set). See
`verify_disjoint` for the exact check."""


def set_b_texts() -> tuple[str, ...]:
    """Return every `SET_B_INSTRUCTIONS` text, in `SET_B_INSTRUCTIONS` order."""
    return tuple(instruction.text for instruction in SET_B_INSTRUCTIONS)


def set_b_region_names() -> tuple[str, ...]:
    """Return every `SET_B_INSTRUCTIONS` region name, row-aligned with `set_b_texts`."""
    return tuple(instruction.region_name for instruction in SET_B_INSTRUCTIONS)


def _is_trivial_single_word_edit(a: str, b: str) -> bool:
    """Return whether `a` and `b` have equal word count and differ at exactly one word position."""
    a_words, b_words = a.split(), b.split()
    if len(a_words) != len(b_words):
        return False
    differing_positions = sum(1 for x, y in zip(a_words, b_words, strict=True) if x.lower() != y.lower())
    return differing_positions == 1


def verify_disjoint() -> list[str]:
    """Check Set B against every prior-stage vocabulary for exact or trivial-edit overlap.

    Reuses the same two checks stage 4's own vocabulary additions were
    verified with (`test_augmented_training_vocabulary.py`): case-insensitive
    exact match, and same-word-count-differs-by-one-word. Checked against the
    union of `goal_region_vocabulary.ALL_INSTRUCTIONS` (14, stage 3's
    training set), `augmented_training_vocabulary.AUGMENTED_INSTRUCTIONS`
    (70, stage 4's training set), `held_out_paraphrases.HELD_OUT_PARAPHRASES`
    and `.COMPOSITIONAL_INSTRUCTIONS` (stage 4's fixed test set) -- i.e.
    everything this project has ever used as a phrasing, not just the
    combined 84-sentence reference `LiveGoalController` searches.

    Returns:
        A list of human-readable problem descriptions; empty if Set B is
        fully disjoint from every prior vocabulary.

    """
    reference_texts = (
        tuple(ALL_INSTRUCTIONS)
        + tuple(AUGMENTED_INSTRUCTIONS)
        + tuple(held_out_texts())
        + tuple(compositional_texts())
    )
    reference_lower = {text.lower() for text in reference_texts}

    problems = []
    for candidate in set_b_texts():
        if candidate.lower() in reference_lower:
            problems.append(f"{candidate!r} is an exact duplicate of an existing instruction")
        for reference in reference_texts:
            if _is_trivial_single_word_edit(candidate, reference):
                problems.append(f"{candidate!r} is a trivial single-word edit of {reference!r}")

    texts = set_b_texts()
    for i, a in enumerate(texts):
        for j, b in enumerate(texts):
            if i != j and _is_trivial_single_word_edit(a, b):
                problems.append(f"{a!r} is a trivial single-word edit of its own Set B sibling {b!r}")

    return problems
