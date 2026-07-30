"""Tests for the frozen sentence-transformer wrapper.

Requires the `all-MiniLM-L6-v2` model to be available (downloaded once from
the Hugging Face Hub on first use, then cached locally by
`sentence-transformers`). No RL/gym dependency here — this module only turns
strings into vectors.
"""

from __future__ import annotations

import numpy as np

from lang_goal_rl.language_embedding import (
    DEFAULT_MODEL_REVISION,
    LANGUAGE_EMBED_DIM,
    encode_instructions,
)


class TestDefaultModelRevision:
    """The model revision is pinned to guard against silent upstream changes."""

    def test_is_a_full_length_git_commit_hash(self) -> None:
        assert len(DEFAULT_MODEL_REVISION) == 40
        assert all(c in "0123456789abcdef" for c in DEFAULT_MODEL_REVISION)


class TestEncodeInstructions:
    """encode_instructions maps a list of strings to fixed-size embedding vectors."""

    def test_output_shape_matches_batch_size_and_embed_dim(self) -> None:
        embeddings = encode_instructions(["move your hand to the left", "reach up high"])
        assert embeddings.shape == (2, LANGUAGE_EMBED_DIM)

    def test_output_is_float32_numpy_array(self) -> None:
        embeddings = encode_instructions(["move your hand to the left"])
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.dtype == np.float32

    def test_different_sentences_get_different_embeddings(self) -> None:
        embeddings = encode_instructions(["move your hand to the left", "reach up high"])
        assert not np.allclose(embeddings[0], embeddings[1])

    def test_is_deterministic_for_the_same_sentence(self) -> None:
        first = encode_instructions(["reach down low"])
        second = encode_instructions(["reach down low"])
        np.testing.assert_array_equal(first, second)
