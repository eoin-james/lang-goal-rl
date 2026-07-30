"""Frozen sentence-transformer wrapper: English instruction -> fixed-size embedding.

`all-MiniLM-L6-v2` is the frozen pretrained language embedding stage 3's
projection layer maps into stage 2's goal space. It's picked over a larger
sentence-transformer or a CLIP-text encoder because: (a) it's the standard,
widely-used small sentence-transformer (384-dim, ~80MB) — plenty of capacity
for a closed, ~14-instruction fixed vocabulary (see
`goal_region_vocabulary.ALL_INSTRUCTIONS`), and (b) VLM-RM/LIV-style reward
pipelines lean on the encoder's *frozen* semantic geometry, so a bigger model
buys nothing at this stage's scope while costing more to load and run. The
model is never fine-tuned here — only the projection layer downstream of it
learns.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
"""Pinned commit hash for `DEFAULT_MODEL_NAME`'s `main` branch (reproducibility:
guards against a silent upstream change to the model's default revision)."""
LANGUAGE_EMBED_DIM = 384
"""Output dimensionality of `all-MiniLM-L6-v2`. This is the projection
layer's expected input size (see `language_goal_projection.py`)."""


@lru_cache(maxsize=1)
def _load_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """Load and cache a frozen `SentenceTransformer` by name.

    Cached (not reloaded per call) since loading is comparatively expensive
    and every call in this process uses the same frozen model.
    """
    return SentenceTransformer(model_name, revision=DEFAULT_MODEL_REVISION)


def encode_instructions(
    instructions: Sequence[str], *, model_name: str = DEFAULT_MODEL_NAME,
) -> npt.NDArray[np.float32]:
    """Embed a batch of instructions with the frozen sentence-transformer.

    Args:
        instructions: English instruction strings to embed.
        model_name: Hugging Face Hub name of the sentence-transformer model.

    Returns:
        Array of shape (len(instructions), LANGUAGE_EMBED_DIM), dtype float32.

    """
    model = _load_model(model_name)
    embeddings = model.encode(list(instructions), convert_to_numpy=True)
    return embeddings.astype(np.float32)
