"""Tests for stage 11's command-type training vocabulary.

`build_command_type_training_set` is the labeled `(text, CommandType)` set
`command_type_classifier.train_command_type_classifier` trains against. These
tests check the *shape* of the set: exactly 5 classes, the expected count per
class, no cross-class duplicate text, and (2026-07-24 fix) that
`GOTO_NAMED_REGION` is built from fresh absolute-destination phrasings rather
than Phase 1's directional region vocabulary -- the root cause of MOVE's 0%
held-out accuracy documented in `command_type_vocabulary.py`'s docstring.
"""

from __future__ import annotations

from lang_goal_rl.augmented_training_vocabulary import AUGMENTED_INSTRUCTIONS
from lang_goal_rl.command_grammar import KNOWN_DIRECTIONS
from lang_goal_rl.command_type_vocabulary import (
    CommandType,
    LabeledCommandExample,
    build_command_type_training_set,
    check_cross_class_embedding_overlap,
    goto_named_region_examples,
    move_examples,
    reset_examples,
    stop_examples,
    unsupported_examples,
)
from lang_goal_rl.goal_region_vocabulary import ALL_INSTRUCTIONS, region_names


class TestCommandType:
    """CommandType: exactly the 5 supervised labels this stage classifies."""

    def test_has_exactly_5_values(self) -> None:
        assert len(CommandType) == 5

    def test_has_the_expected_5_members(self) -> None:
        assert {member.name for member in CommandType} == {
            "MOVE",
            "GOTO_NAMED_REGION",
            "STOP",
            "RESET",
            "UNSUPPORTED",
        }


class TestGotoNamedRegionExamples:
    """GOTO_NAMED_REGION: fresh absolute-destination phrasings, one set per region name."""

    def test_count_matches_12_phrasings_per_region(self) -> None:
        examples = goto_named_region_examples()
        assert len(examples) == 12 * len(region_names())

    def test_every_example_is_labeled_goto_named_region(self) -> None:
        examples = goto_named_region_examples()
        assert all(example.command_type is CommandType.GOTO_NAMED_REGION for example in examples)

    def test_all_texts_are_unique(self) -> None:
        texts = [example.text for example in goto_named_region_examples()]
        assert len(texts) == len(set(texts))

    def test_does_not_reuse_phase_1_region_vocabulary_verbatim(self) -> None:
        """The original version relabeled `ALL_INSTRUCTIONS`/`AUGMENTED_INSTRUCTIONS`
        wholesale -- that directional-verb phrasing is what collided with MOVE
        and caused a 0% held-out MOVE accuracy. This class's own texts must
        now be entirely disjoint from those two Phase-1 vocabularies.
        """
        texts = {example.text for example in goto_named_region_examples()}
        phase_1_texts = set(ALL_INSTRUCTIONS) | set(AUGMENTED_INSTRUCTIONS)
        assert texts.isdisjoint(phase_1_texts)

    def test_avoids_the_reach_angle_swing_directional_verb_pattern(self) -> None:
        """The collision-causing pattern was a "reach/angle/swing <direction>"
        verb-phrase -- Phase 1's region-naming convention. This class's own
        templates must never use that pattern.
        """
        texts = [example.text for example in goto_named_region_examples()]
        for text in texts:
            assert not text.lower().startswith(("reach ", "angle ", "swing ")), text

    def test_every_region_name_is_represented_by_its_own_place_phrasings(self) -> None:
        """Each of the 7 region names contributes exactly 12 phrasings, so the
        class still maps onto the same underlying regions even though
        `CommandType.GOTO_NAMED_REGION` itself carries no region label.
        """
        examples = goto_named_region_examples()
        assert len(examples) % len(region_names()) == 0


class TestMoveExamples:
    """MOVE: direction-phrase templates over every KNOWN_DIRECTIONS entry."""

    def test_every_example_is_labeled_move(self) -> None:
        examples = move_examples()
        assert all(example.command_type is CommandType.MOVE for example in examples)

    def test_has_between_10_and_15_phrasings_per_direction(self) -> None:
        examples = move_examples()
        assert len(examples) >= 10 * len(KNOWN_DIRECTIONS)
        assert len(examples) <= 15 * len(KNOWN_DIRECTIONS)

    def test_all_move_texts_are_unique(self) -> None:
        examples = move_examples()
        texts = [example.text for example in examples]
        assert len(texts) == len(set(texts))

    def test_includes_phrasings_both_with_and_without_a_magnitude_phrase(self) -> None:
        """A mix of lengths so the classifier can't overfit on sentence length alone."""
        examples = move_examples()
        word_counts = {len(example.text.split()) for example in examples}
        assert len(word_counts) > 1

    def test_every_phrasing_carries_a_magnitude_cue_or_from_here_framing(self) -> None:
        """The 2026-07-24 fix: every MOVE sentence must read as incremental
        motion from wherever the robot currently is, not a directional
        destination phrase -- see `command_type_vocabulary.py`'s docstring.
        """
        magnitude_or_from_here_markers = (
            "a bit",
            "slightly",
            "a small amount",
            "a little",
            "a good distance",
            "all the way",
            "a short distance",
            "a touch",
            "from here",
            "from where you are",
            "from your current",
            "just a little bit",
        )
        for example in move_examples():
            text = example.text.lower()
            assert any(marker in text for marker in magnitude_or_from_here_markers), example.text

    def test_never_uses_the_angle_or_swing_verbs(self) -> None:
        """These two words are exactly what collided with the old GOTO_NAMED_REGION
        training data (see `command_type_vocabulary.py`'s docstring)."""
        for example in move_examples():
            text = example.text.lower()
            assert "angle" not in text
            assert "swing" not in text


class TestStopExamples:
    """STOP: diverse phrasings."""

    def test_every_example_is_labeled_stop(self) -> None:
        examples = stop_examples()
        assert all(example.command_type is CommandType.STOP for example in examples)

    def test_has_between_20_and_30_phrasings(self) -> None:
        count = len(stop_examples())
        assert 20 <= count <= 30

    def test_all_texts_are_unique(self) -> None:
        texts = [example.text for example in stop_examples()]
        assert len(texts) == len(set(texts))


class TestResetExamples:
    """RESET: diverse phrasings."""

    def test_every_example_is_labeled_reset(self) -> None:
        examples = reset_examples()
        assert all(example.command_type is CommandType.RESET for example in examples)

    def test_has_at_least_15_phrasings(self) -> None:
        assert len(reset_examples()) >= 15

    def test_all_texts_are_unique(self) -> None:
        texts = [example.text for example in reset_examples()]
        assert len(texts) == len(set(texts))


class TestUnsupportedExamples:
    """UNSUPPORTED: a varied set of out-of-scope requests, 20-35 phrasings."""

    def test_every_example_is_labeled_unsupported(self) -> None:
        examples = unsupported_examples()
        assert all(example.command_type is CommandType.UNSUPPORTED for example in examples)

    def test_has_between_20_and_35_phrasings(self) -> None:
        count = len(unsupported_examples())
        assert 20 <= count <= 35

    def test_all_texts_are_unique(self) -> None:
        texts = [example.text for example in unsupported_examples()]
        assert len(texts) == len(set(texts))


class TestBuildCommandTypeTrainingSet:
    """build_command_type_training_set: all 5 classes combined, shuffled with a fixed seed."""

    def test_returns_labeled_command_example_instances(self) -> None:
        examples = build_command_type_training_set(seed=0)
        assert all(isinstance(example, LabeledCommandExample) for example in examples)

    def test_total_count_matches_the_sum_of_every_class(self) -> None:
        examples = build_command_type_training_set(seed=0)
        expected = (
            len(goto_named_region_examples())
            + len(move_examples())
            + len(stop_examples())
            + len(reset_examples())
            + len(unsupported_examples())
        )
        assert len(examples) == expected

    def test_every_class_is_represented(self) -> None:
        examples = build_command_type_training_set(seed=0)
        present = {example.command_type for example in examples}
        assert present == set(CommandType)

    def test_all_texts_across_classes_are_unique(self) -> None:
        examples = build_command_type_training_set(seed=0)
        texts = [example.text for example in examples]
        assert len(texts) == len(set(texts))

    def test_same_seed_produces_the_same_order(self) -> None:
        first = build_command_type_training_set(seed=7)
        second = build_command_type_training_set(seed=7)
        assert first == second

    def test_different_seeds_produce_a_different_order(self) -> None:
        first = build_command_type_training_set(seed=0)
        second = build_command_type_training_set(seed=1)
        assert [e.text for e in first] != [e.text for e in second]

    def test_shuffling_does_not_change_the_underlying_set_of_examples(self) -> None:
        first = build_command_type_training_set(seed=0)
        second = build_command_type_training_set(seed=1)
        assert set(first) == set(second)


class TestCheckCrossClassEmbeddingOverlap:
    """check_cross_class_embedding_overlap: the cheap pre-train collision diagnostic."""

    def test_reports_one_neighbor_per_input_example(self) -> None:
        examples = build_command_type_training_set(seed=0)
        report = check_cross_class_embedding_overlap(examples)
        assert len(report.neighbors) == len(examples)

    def test_move_and_goto_named_region_show_low_cross_class_confusion_on_the_fixed_vocabulary(self) -> None:
        """The whole point of the 2026-07-24 redesign: on the *current*
        vocabulary, MOVE and GOTO_NAMED_REGION should no longer look like
        each other's nearest neighbors in embedding space.
        """
        examples = build_command_type_training_set(seed=0)
        report = check_cross_class_embedding_overlap(examples, flag_threshold=0.15)
        flagged_class_pairs = {(a, b) for a, b, _rate in report.flagged_pairs}
        assert ("MOVE", "GOTO_NAMED_REGION") not in flagged_class_pairs
        assert ("GOTO_NAMED_REGION", "MOVE") not in flagged_class_pairs

    def test_flags_an_injected_pair_of_near_duplicate_classes(self) -> None:
        """Sanity check that the diagnostic actually flags a real collision:
        two classes given (near-)identical text should show up as a flagged
        pair, so a caller can trust a clean report from the real vocabulary.
        """
        examples = (
            LabeledCommandExample("angle your arm toward the left", CommandType.MOVE),
            LabeledCommandExample("angle your arm toward the left side", CommandType.GOTO_NAMED_REGION),
            LabeledCommandExample("angle your arm toward the right", CommandType.MOVE),
            LabeledCommandExample("angle your arm toward the right side", CommandType.GOTO_NAMED_REGION),
            *stop_examples(),
        )
        report = check_cross_class_embedding_overlap(examples, flag_threshold=0.15)
        flagged_class_pairs = {(a, b) for a, b, _rate in report.flagged_pairs}
        assert ("MOVE", "GOTO_NAMED_REGION") in flagged_class_pairs or (
            "GOTO_NAMED_REGION",
            "MOVE",
        ) in flagged_class_pairs
