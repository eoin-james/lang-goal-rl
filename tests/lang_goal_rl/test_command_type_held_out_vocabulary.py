"""Tests for stage 11's held-out command-type vocabulary.

`build_command_type_held_out_set` is never fed to
`command_type_classifier.train_command_type_classifier` -- it exists so a
caller (the experiment-runner) can measure the trained classifier's accuracy
on genuinely unseen phrasing, not phrasing it memorized. These tests check
disjointness against the training vocabulary: no exact-text collision and no
trivial single-word edit of a training phrasing (the same guard
`test_held_out_paraphrases.py` / `test_augmented_training_vocabulary.py` use
for stage 3/4's vocabulary). Also checks (2026-07-24 fix) that
GOTO_NAMED_REGION's held-out set no longer reuses
`held_out_paraphrases.HELD_OUT_PARAPHRASES`'s directional-verb phrasings --
see `command_type_held_out_vocabulary.py`'s docstring.
"""

from __future__ import annotations

from lang_goal_rl.command_type_held_out_vocabulary import (
    build_command_type_held_out_set,
    goto_named_region_held_out_examples,
    move_held_out_examples,
    reset_held_out_examples,
    stop_held_out_examples,
    unsupported_held_out_examples,
)
from lang_goal_rl.command_type_vocabulary import (
    CommandType,
    LabeledCommandExample,
    build_command_type_training_set,
)
from lang_goal_rl.goal_region_vocabulary import region_names
from lang_goal_rl.held_out_paraphrases import held_out_texts


def _is_trivial_single_word_edit(a: str, b: str) -> bool:
    """True iff `a` and `b` have the same word count and differ in exactly one word position.

    Same cheap approximation `test_held_out_paraphrases.py` and
    `test_augmented_training_vocabulary.py` use for "genuinely different
    wording, not a lazy synonym swap".
    """
    a_words = a.split()
    b_words = b.split()
    if len(a_words) != len(b_words):
        return False
    differing_positions = sum(1 for x, y in zip(a_words, b_words, strict=True) if x != y)
    return differing_positions == 1


class TestGotoNamedRegionHeldOutExamples:
    """GOTO_NAMED_REGION held-out: fresh absolute-destination phrasings, 2 per region."""

    def test_count_matches_2_per_region(self) -> None:
        assert len(goto_named_region_held_out_examples()) == 2 * len(region_names())

    def test_every_example_is_labeled_goto_named_region(self) -> None:
        examples = goto_named_region_held_out_examples()
        assert all(example.command_type is CommandType.GOTO_NAMED_REGION for example in examples)

    def test_all_texts_are_unique(self) -> None:
        texts = [example.text for example in goto_named_region_held_out_examples()]
        assert len(texts) == len(set(texts))

    def test_no_longer_reuses_held_out_paraphrases_directional_phrasing(self) -> None:
        """The original version relabeled `held_out_paraphrases.HELD_OUT_PARAPHRASES`
        wholesale -- that directional-verb phrasing collided with MOVE (see
        `command_type_held_out_vocabulary.py`'s docstring). This class's held-out
        texts must now be disjoint from that Phase-1 held-out set.
        """
        texts = {example.text for example in goto_named_region_held_out_examples()}
        assert texts.isdisjoint(set(held_out_texts()))

    def test_avoids_the_reach_angle_swing_directional_verb_pattern(self) -> None:
        texts = [example.text for example in goto_named_region_held_out_examples()]
        for text in texts:
            assert not text.lower().startswith(("reach ", "angle ", "swing ")), text


class TestPerClassHeldOutExamples:
    """MOVE / STOP / RESET / UNSUPPORTED held-out sets: correctly labeled, non-empty, unique."""

    def test_move_examples_are_labeled_move_and_nonempty(self) -> None:
        examples = move_held_out_examples()
        assert len(examples) > 0
        assert all(example.command_type is CommandType.MOVE for example in examples)

    def test_move_examples_never_use_the_angle_or_swing_verbs(self) -> None:
        """These two words are exactly what collided with the old GOTO_NAMED_REGION
        training data (see `command_type_vocabulary.py`'s docstring)."""
        for example in move_held_out_examples():
            text = example.text.lower()
            assert "angle" not in text
            assert "swing" not in text

    def test_stop_examples_are_labeled_stop_and_nonempty(self) -> None:
        examples = stop_held_out_examples()
        assert len(examples) > 0
        assert all(example.command_type is CommandType.STOP for example in examples)

    def test_reset_examples_are_labeled_reset_and_nonempty(self) -> None:
        examples = reset_held_out_examples()
        assert len(examples) > 0
        assert all(example.command_type is CommandType.RESET for example in examples)

    def test_unsupported_examples_are_labeled_unsupported_and_nonempty(self) -> None:
        examples = unsupported_held_out_examples()
        assert len(examples) > 0
        assert all(example.command_type is CommandType.UNSUPPORTED for example in examples)


class TestBuildCommandTypeHeldOutSet:
    """build_command_type_held_out_set: all 5 classes, genuinely disjoint from training."""

    def test_returns_labeled_command_example_instances(self) -> None:
        examples = build_command_type_held_out_set()
        assert all(isinstance(example, LabeledCommandExample) for example in examples)

    def test_every_class_is_represented(self) -> None:
        examples = build_command_type_held_out_set()
        present = {example.command_type for example in examples}
        assert present == set(CommandType)

    def test_all_texts_are_unique(self) -> None:
        texts = [example.text for example in build_command_type_held_out_set()]
        assert len(texts) == len(set(texts))

    def test_no_exact_text_collides_with_the_training_vocabulary(self) -> None:
        held_out_texts_set = {example.text.lower() for example in build_command_type_held_out_set()}
        training_texts_set = {example.text.lower() for example in build_command_type_training_set()}
        assert held_out_texts_set.isdisjoint(training_texts_set)

    def test_no_held_out_text_is_a_trivial_single_word_edit_of_a_training_text(self) -> None:
        held_out = build_command_type_held_out_set()
        training = build_command_type_training_set()
        for held_out_example in held_out:
            for training_example in training:
                assert not _is_trivial_single_word_edit(held_out_example.text, training_example.text), (
                    f"{held_out_example.text!r} is a single-word edit of "
                    f"training text {training_example.text!r}"
                )
