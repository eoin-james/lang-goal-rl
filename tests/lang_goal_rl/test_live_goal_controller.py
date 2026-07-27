"""Tests for LiveGoalController: stage 6's live text -> goal-embedding component.

Ties together three already-Done pieces for live use: `language_embedding.
encode_instructions` (stage 3's frozen sentence-transformer), `nearest_
neighbor_projection.nearest_neighbor_projection` at k=1 (stage 4's confirmed-
best zero-training lookup, see ROADMAP.md's "Resolution (attempt 4...)"
note), and the 84-sentence combined reference vocabulary promoted in
`combined_vocabulary.py`. The class exists specifically to precompute and
cache that 84-sentence reference *once*, at construction -- re-encoding 84
sentences on every live utterance would defeat the point of a "live"
interface. These tests check that caching actually happens (not just that
the pipeline produces a plausible-looking number).
"""

from __future__ import annotations

import torch

import lang_goal_rl.combined_vocabulary as combined_vocabulary_module
import lang_goal_rl.live_goal_controller as live_goal_controller_module
from lang_goal_rl.combined_vocabulary import build_combined_reference, combined_instructions_and_regions
from lang_goal_rl.goal_encoder import GoalEncoder
from lang_goal_rl.live_goal_controller import LiveGoalController


def _make_encoder() -> GoalEncoder:
    return GoalEncoder(goal_dim=3, embed_dim=16, hidden_dim=8)


class TestLiveGoalControllerCaching:
    """The 84-sentence reference is encoded once at construction, never re-encoded per call."""

    def test_the_84_sentence_reference_is_encoded_exactly_once_regardless_of_call_count(
        self, monkeypatch
    ) -> None:  # noqa: ANN001
        from lang_goal_rl import language_embedding

        call_sizes: list[int] = []
        original_encode = language_embedding.encode_instructions

        def counting_encode(instructions, **kwargs):  # noqa: ANN001, ANN003, ANN202
            call_sizes.append(len(instructions))
            return original_encode(instructions, **kwargs)

        monkeypatch.setattr(combined_vocabulary_module, "encode_instructions", counting_encode)
        monkeypatch.setattr(live_goal_controller_module, "encode_instructions", counting_encode)

        controller = LiveGoalController(_make_encoder(), n_target_samples=20, seed=0)
        controller.instruction_to_goal_embedding("reach up high")
        controller.instruction_to_goal_embedding("reach up high")
        controller.instruction_to_goal_embedding("move your hand downward")

        # Exactly one batch call encoded all 84 reference sentences (at
        # construction); every later call encodes only the single new
        # instruction (size 1), never re-encoding the 84-sentence reference.
        assert call_sizes.count(84) == 1
        assert call_sizes == [84, 1, 1, 1]


class TestLiveGoalControllerLookup:
    """instruction_to_goal_embedding: sentence -> 384-dim embedding -> k=1 NN -> 16-dim goal embedding."""

    def test_result_has_the_encoders_embed_dim_shape(self) -> None:
        controller = LiveGoalController(_make_encoder(), n_target_samples=20, seed=0)

        result = controller.instruction_to_goal_embedding("reach toward the ceiling somehow")

        assert result.shape == (16,)

    def test_result_is_a_float32_torch_tensor(self) -> None:
        controller = LiveGoalController(_make_encoder(), n_target_samples=20, seed=0)

        result = controller.instruction_to_goal_embedding("reach toward the ceiling somehow")

        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32

    def test_a_known_reference_sentence_maps_to_exactly_its_own_cached_target_at_k1(self) -> None:
        """k=1 exact-copy semantics (stage 4's confirmed-best mode, see
        ROADMAP.md): querying with a sentence that is itself one of the 84
        reference instructions has distance 0 to its own row, so the nearest
        neighbor is that row and the blend degenerates to an exact copy of
        its known target -- not an approximation.
        """
        encoder = _make_encoder()
        controller = LiveGoalController(encoder, k=1, n_target_samples=20, seed=0)
        instructions, _regions = combined_instructions_and_regions()
        known_sentence = "reach up high"
        assert known_sentence in instructions
        row_index = instructions.index(known_sentence)

        _raw_embeddings, expected_targets = build_combined_reference(encoder, n_samples=20, seed=0)
        expected = torch.from_numpy(expected_targets[row_index]).to(torch.float32)

        result = controller.instruction_to_goal_embedding(known_sentence)

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=0)

    def test_default_k_is_one(self) -> None:
        encoder = _make_encoder()
        default_controller = LiveGoalController(encoder, n_target_samples=20, seed=0)
        explicit_k1_controller = LiveGoalController(encoder, k=1, n_target_samples=20, seed=0)

        default_result = default_controller.instruction_to_goal_embedding("move your hand upward")
        explicit_result = explicit_k1_controller.instruction_to_goal_embedding("move your hand upward")

        torch.testing.assert_close(default_result, explicit_result, atol=1e-6, rtol=0)

    def test_leaves_goal_encoder_parameters_unchanged(self) -> None:
        encoder = _make_encoder()
        before = [p.clone() for p in encoder.parameters()]

        controller = LiveGoalController(encoder, n_target_samples=20, seed=0)
        controller.instruction_to_goal_embedding("reach up high")

        after = list(encoder.parameters())
        for p_before, p_after in zip(before, after, strict=True):
            assert torch.equal(p_before, p_after)
