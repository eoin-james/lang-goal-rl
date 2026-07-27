"""Tests for the promoted 84-sentence combined reference vocabulary.

Stage 4 (open vocabulary, Done) built this combined set --
`goal_region_vocabulary.ALL_INSTRUCTIONS` (14) + `augmented_training_
vocabulary.AUGMENTED_INSTRUCTIONS` (70) -- as a one-off script in
`experiments/04_open_vocabulary/combined_vocabulary.py` to feed the k=1
nearest-neighbor lookup that passed stage 4's proof gate. Stage 6 depends on
that exact same 84-sentence set as a foundational building block (via
`LiveGoalController`, see `test_live_goal_controller.py`), so this module
promotes it into `src/lang_goal_rl/` with tests -- these tests exist to
pin down that promotion kept the *exact* 84-sentence set stage 4 already
validated, not a redefinition.
"""

from __future__ import annotations

import numpy as np
import torch

from lang_goal_rl.augmented_training_vocabulary import AUGMENTED_INSTRUCTIONS, augmented_instruction_to_region
from lang_goal_rl.combined_vocabulary import build_combined_reference, combined_instructions_and_regions
from lang_goal_rl.goal_encoder import GoalEncoder
from lang_goal_rl.goal_region_vocabulary import ALL_INSTRUCTIONS, instruction_to_region
from lang_goal_rl.language_goal_projection import precompute_instruction_targets


class TestCombinedInstructionsAndRegions:
    """combined_instructions_and_regions: the exact 84-sentence union, row-aligned to region names."""

    def test_returns_exactly_84_instructions(self) -> None:
        instructions, _regions = combined_instructions_and_regions()
        assert len(instructions) == 84

    def test_regions_are_row_aligned_with_instructions(self) -> None:
        instructions, regions = combined_instructions_and_regions()
        assert len(regions) == len(instructions)

    def test_no_duplicate_instructions(self) -> None:
        instructions, _regions = combined_instructions_and_regions()
        assert len(instructions) == len(set(instructions))

    def test_matches_the_union_of_all_instructions_and_augmented_instructions_exactly(self) -> None:
        instructions, _regions = combined_instructions_and_regions()
        expected = set(ALL_INSTRUCTIONS) | set(AUGMENTED_INSTRUCTIONS)
        assert set(instructions) == expected

    def test_the_original_14_instructions_come_before_the_augmented_70(self) -> None:
        instructions, _regions = combined_instructions_and_regions()
        assert instructions[:14] == ALL_INSTRUCTIONS
        assert instructions[14:] == AUGMENTED_INSTRUCTIONS

    def test_every_region_matches_the_instructions_own_source_vocabulary_mapping(self) -> None:
        instructions, regions = combined_instructions_and_regions()
        augmented_map = augmented_instruction_to_region()
        for instruction, region in zip(instructions, regions, strict=True):
            if instruction in ALL_INSTRUCTIONS:
                assert region == instruction_to_region(instruction)
            else:
                assert region == augmented_map[instruction]


class TestBuildCombinedReference:
    """build_combined_reference: raw sentence embeddings + fixed region-centroid targets, once."""

    def test_returns_84_raw_embeddings_of_dimension_384(self) -> None:
        encoder = GoalEncoder(goal_dim=3, embed_dim=16, hidden_dim=8)
        raw_embeddings, _targets = build_combined_reference(encoder, n_samples=20, seed=0)
        assert raw_embeddings.shape == (84, 384)

    def test_returns_84_targets_matching_the_encoders_embed_dim(self) -> None:
        encoder = GoalEncoder(goal_dim=3, embed_dim=16, hidden_dim=8)
        _raw_embeddings, targets = build_combined_reference(encoder, n_samples=20, seed=0)
        assert targets.shape == (84, 16)

    def test_raw_embeddings_are_float32(self) -> None:
        encoder = GoalEncoder(goal_dim=3, embed_dim=16, hidden_dim=8)
        raw_embeddings, _targets = build_combined_reference(encoder, n_samples=20, seed=0)
        assert raw_embeddings.dtype == np.float32

    def test_targets_are_float32(self) -> None:
        encoder = GoalEncoder(goal_dim=3, embed_dim=16, hidden_dim=8)
        _raw_embeddings, targets = build_combined_reference(encoder, n_samples=20, seed=0)
        assert targets.dtype == np.float32

    def test_is_deterministic_for_a_given_seed(self) -> None:
        encoder = GoalEncoder(goal_dim=3, embed_dim=16, hidden_dim=8)
        first_raw, first_targets = build_combined_reference(encoder, n_samples=20, seed=3)
        second_raw, second_targets = build_combined_reference(encoder, n_samples=20, seed=3)
        np.testing.assert_array_equal(first_raw, second_raw)
        np.testing.assert_array_equal(first_targets, second_targets)

    def test_targets_match_a_direct_call_to_precompute_instruction_targets(self) -> None:
        """Sanity check this reuses stage 3's grounded target machinery
        (via the full 84-row region list) rather than reimplementing it.
        """
        encoder = GoalEncoder(goal_dim=3, embed_dim=16, hidden_dim=8)
        _instructions, regions = combined_instructions_and_regions()

        expected = precompute_instruction_targets(encoder, regions, n_samples=20, seed=0).numpy()
        _raw_embeddings, actual = build_combined_reference(encoder, n_samples=20, seed=0)

        np.testing.assert_array_equal(expected, actual)

    def test_leaves_goal_encoder_parameters_unchanged(self) -> None:
        encoder = GoalEncoder(goal_dim=3, embed_dim=16, hidden_dim=8)
        before = [p.clone() for p in encoder.parameters()]

        build_combined_reference(encoder, n_samples=20, seed=0)

        after = list(encoder.parameters())
        for p_before, p_after in zip(before, after, strict=True):
            assert torch.equal(p_before, p_after)
