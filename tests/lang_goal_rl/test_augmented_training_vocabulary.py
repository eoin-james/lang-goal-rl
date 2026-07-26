"""Tests for stage 4's augmented training vocabulary (data-augmentation fix).

`AUGMENTED_REGIONS` replaces the fixed 2-phrasings-per-region training set
(`goal_region_vocabulary.REGIONS`, still used by the already-Done stage 3
report and left untouched) with 10 diverse phrasings per region, needed so
`LanguageGoalProjection` has enough examples per class to generalize instead
of memorizing (see ROADMAP.md's "Projection-layer overfitting" risk).

These tests check the *shape* of the augmented set: exactly 7 regions x 10
phrasings, no internal duplicates, and -- the load-bearing check -- zero
overlap with stage 4's fixed held-out/compositional test set
(`held_out_paraphrases`), so retraining on this set and re-measuring against
the existing held-out phrases stays an apples-to-apples comparison with the
already-measured 0.286 MLP baseline and 0.714 nearest-neighbor ceiling.
"""

from __future__ import annotations

from lang_goal_rl.augmented_training_vocabulary import (
    AUGMENTED_INSTRUCTIONS,
    AUGMENTED_REGIONS,
    augmented_instruction_to_region,
)
from lang_goal_rl.goal_region_vocabulary import region_names
from lang_goal_rl.held_out_paraphrases import compositional_texts, held_out_texts


class TestAugmentedRegions:
    """AUGMENTED_REGIONS: 7 regions x 10 diverse phrasings each."""

    def test_there_are_exactly_7_regions(self) -> None:
        assert len(AUGMENTED_REGIONS) == 7

    def test_region_names_match_the_existing_7_region_names_exactly(self) -> None:
        assert {region.name for region in AUGMENTED_REGIONS} == set(region_names())

    def test_region_names_are_unique(self) -> None:
        names = [region.name for region in AUGMENTED_REGIONS]
        assert len(names) == len(set(names))

    def test_every_region_has_exactly_10_phrasings(self) -> None:
        for region in AUGMENTED_REGIONS:
            assert len(region.instructions) == 10, (
                f"region {region.name!r} has {len(region.instructions)} phrasings, expected 10"
            )


class TestAugmentedInstructions:
    """AUGMENTED_INSTRUCTIONS: flat union of every region's phrasings."""

    def test_has_70_total_instructions(self) -> None:
        assert len(AUGMENTED_INSTRUCTIONS) == 70

    def test_matches_the_union_of_region_instructions(self) -> None:
        expected = {
            instruction for region in AUGMENTED_REGIONS for instruction in region.instructions
        }
        assert set(AUGMENTED_INSTRUCTIONS) == expected

    def test_no_duplicate_phrasing_case_insensitive(self) -> None:
        lowered = [instruction.lower() for instruction in AUGMENTED_INSTRUCTIONS]
        assert len(lowered) == len(set(lowered))

    def test_zero_overlap_with_held_out_paraphrases_case_insensitive(self) -> None:
        held_out_lower = {text.lower() for text in held_out_texts()}
        augmented_lower = {text.lower() for text in AUGMENTED_INSTRUCTIONS}
        assert augmented_lower.isdisjoint(held_out_lower)

    def test_zero_overlap_with_compositional_instructions_case_insensitive(self) -> None:
        compositional_lower = {text.lower() for text in compositional_texts()}
        augmented_lower = {text.lower() for text in AUGMENTED_INSTRUCTIONS}
        assert augmented_lower.isdisjoint(compositional_lower)

    def test_no_augmented_text_is_a_trivial_single_word_edit_of_a_held_out_or_compositional_text(
        self,
    ) -> None:
        # Same trivial-edit guard as test_held_out_paraphrases.py's
        # test_no_held_out_text_is_a_trivial_single_word_edit_of_an_existing_instruction,
        # adapted here: every augmented phrasing should differ from every
        # held-out/compositional text in more than one word position when
        # the two have the same word count (the shape a lazy single-word
        # swap would take) -- otherwise retraining on this set would leak
        # a near-copy of the fixed held-out test set into training.
        reference_texts = tuple(held_out_texts()) + tuple(compositional_texts())
        for augmented in AUGMENTED_INSTRUCTIONS:
            augmented_words = augmented.split()
            for reference in reference_texts:
                reference_words = reference.split()
                if len(reference_words) != len(augmented_words):
                    continue
                differing_positions = sum(
                    1 for a, b in zip(augmented_words, reference_words, strict=True) if a != b
                )
                assert differing_positions != 1, (
                    f"{augmented!r} is a single-word edit of held-out/compositional text {reference!r}"
                )


class TestAugmentedInstructionToRegion:
    """augmented_instruction_to_region resolves every augmented instruction to its region."""

    def test_returns_a_dict_covering_every_augmented_instruction(self) -> None:
        mapping = augmented_instruction_to_region()
        assert set(mapping.keys()) == set(AUGMENTED_INSTRUCTIONS)

    def test_every_value_is_a_real_region_name(self) -> None:
        mapping = augmented_instruction_to_region()
        valid_regions = set(region_names())
        assert all(region in valid_regions for region in mapping.values())

    def test_mapping_matches_each_regions_own_instructions(self) -> None:
        mapping = augmented_instruction_to_region()
        for region in AUGMENTED_REGIONS:
            for instruction in region.instructions:
                assert mapping[instruction] == region.name
