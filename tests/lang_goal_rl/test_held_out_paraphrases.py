"""Tests for stage 4's held-out paraphrase / compositional instruction set.

These instructions are never used to train the projection (see
`language_goal_projection.train_projection`) -- only to test generalization
in stage 4's proof gate. The tests here check the *shape* of the set (every
paraphrase maps to a real stage-3 region, nothing collides with the
training vocabulary, compositional instructions name two distinct real
regions) rather than anything about embedding geometry, which is
`semantic_neighbor_diagnostic`'s job.
"""

from __future__ import annotations

from lang_goal_rl.goal_region_vocabulary import ALL_INSTRUCTIONS, region_names
from lang_goal_rl.held_out_paraphrases import (
    COMPOSITIONAL_INSTRUCTIONS,
    HELD_OUT_PARAPHRASES,
    compositional_texts,
    held_out_region_names,
    held_out_texts,
)


class TestHeldOutParaphrases:
    """HELD_OUT_PARAPHRASES: new phrasings of stage 3's 7 existing regions."""

    def test_every_paraphrase_names_a_real_region(self) -> None:
        valid_regions = set(region_names())
        assert all(
            paraphrase.region_name in valid_regions
            for paraphrase in HELD_OUT_PARAPHRASES
        )

    def test_every_region_has_at_least_two_held_out_paraphrases(self) -> None:
        counts: dict[str, int] = {}
        for paraphrase in HELD_OUT_PARAPHRASES:
            counts[paraphrase.region_name] = counts.get(paraphrase.region_name, 0) + 1
        for region in region_names():
            assert counts.get(region, 0) >= 2, (
                f"region {region!r} has fewer than 2 held-out paraphrases"
            )

    def test_no_held_out_text_collides_with_the_training_vocabulary(self) -> None:
        texts = held_out_texts()
        assert set(texts).isdisjoint(ALL_INSTRUCTIONS)
        assert len(texts) == len(HELD_OUT_PARAPHRASES)

    def test_held_out_texts_are_unique(self) -> None:
        texts = held_out_texts()
        assert len(texts) == len(set(texts))

    def test_held_out_texts_and_region_names_are_row_aligned(self) -> None:
        texts = held_out_texts()
        regions = held_out_region_names()
        assert len(texts) == len(regions) == len(HELD_OUT_PARAPHRASES)
        for paraphrase, text, region in zip(
            HELD_OUT_PARAPHRASES, texts, regions, strict=True
        ):
            assert paraphrase.text == text
            assert paraphrase.region_name == region

    def test_no_held_out_text_is_a_trivial_single_word_edit_of_an_existing_instruction(
        self,
    ) -> None:
        # A cheap approximation of "genuinely different wording, not a
        # trivial synonym swap": every held-out phrasing should differ from
        # every training instruction in more than one word position when
        # the two have the same word count (the shape a lazy single-word
        # swap would take).
        for paraphrase in HELD_OUT_PARAPHRASES:
            new_words = paraphrase.text.split()
            for existing in ALL_INSTRUCTIONS:
                existing_words = existing.split()
                if len(existing_words) != len(new_words):
                    continue
                differing_positions = sum(
                    1 for a, b in zip(new_words, existing_words, strict=True) if a != b
                )
                assert differing_positions != 1, (
                    f"{paraphrase.text!r} is a single-word edit of existing instruction {existing!r}"
                )


class TestCompositionalInstructions:
    """COMPOSITIONAL_INSTRUCTIONS: instructions combining two existing regions."""

    def test_there_is_at_least_one_compositional_instruction(self) -> None:
        assert len(COMPOSITIONAL_INSTRUCTIONS) >= 1

    def test_every_compositional_instruction_names_exactly_two_distinct_real_regions(
        self,
    ) -> None:
        valid_regions = set(region_names())
        for instruction in COMPOSITIONAL_INSTRUCTIONS:
            assert len(instruction.component_region_names) == 2
            assert (
                instruction.component_region_names[0]
                != instruction.component_region_names[1]
            )
            assert set(instruction.component_region_names) <= valid_regions

    def test_compositional_texts_do_not_collide_with_training_or_held_out_vocabulary(
        self,
    ) -> None:
        texts = set(compositional_texts())
        assert texts.isdisjoint(ALL_INSTRUCTIONS)
        assert texts.isdisjoint(held_out_texts())

    def test_compositional_texts_are_unique(self) -> None:
        texts = compositional_texts()
        assert len(texts) == len(set(texts))
