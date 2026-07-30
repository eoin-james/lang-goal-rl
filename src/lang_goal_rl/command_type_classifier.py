"""CommandTypeClassifier: frozen sentence embedding -> `CommandType` (stage 11).

The first learned piece of Phase 2b's English-to-typed-command pipeline: a
small classification head on top of the same frozen `all-MiniLM-L6-v2`
sentence embedding `language_goal_projection.LanguageGoalProjection` already
uses (`language_embedding.py`). Same Linear-ReLU-Linear, 64-hidden shape as
that module -- this stage only swaps the regression head for a 5-way
softmax head, since the underlying "small trained head on a frozen 384-dim
embedding" problem shape is identical.

This module deliberately stops at "which of the 5 command types". Deciding
what to *do* with an `UNSUPPORTED` classification (reject, ask for
clarification, fall back to a default) is one layer up, in a later stage's
dispatch logic -- `classify_command_type` never raises on well-formed
English, by design; returning `UNSUPPORTED` for out-of-scope text is the
correct, ordinary outcome here, not an error condition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as f  # noqa: N812 -- mirrors contrastive.py's common torch.nn.functional alias
from torch import nn

from lang_goal_rl.command_type_vocabulary import CommandType
from lang_goal_rl.language_embedding import encode_instructions

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_INPUT_DIM = 384
"""Matches `language_embedding.LANGUAGE_EMBED_DIM` (`all-MiniLM-L6-v2`'s output size)."""

_COMMAND_TYPE_TO_INDEX: dict[CommandType, int] = {
    command_type: index for index, command_type in enumerate(CommandType)
}
_INDEX_TO_COMMAND_TYPE: dict[int, CommandType] = {
    index: command_type for command_type, index in _COMMAND_TYPE_TO_INDEX.items()
}


class CommandTypeClassifier(nn.Module):
    """Small MLP mapping a (batch, input_dim) sentence embedding to (batch, n_command_types) logits.

    Same Linear-ReLU-Linear shape as `LanguageGoalProjection`
    (`language_goal_projection.py`), for the same reason: a single hidden
    layer gives enough capacity to separate 5 classes over a closed,
    template-generated vocabulary without over-parameterizing -- a deeper
    network would mostly add overfitting risk here, not useful capacity.
    """

    def __init__(self, input_dim: int = DEFAULT_INPUT_DIM, hidden_dim: int = 64) -> None:
        """Build the classifier's layers.

        Args:
            input_dim: Dimensionality of the frozen sentence embedding
                (384 for `all-MiniLM-L6-v2`).
            hidden_dim: Width of the single hidden layer.

        """
        super().__init__()
        self.input_dim = input_dim
        n_classes = len(CommandType)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, sentence_embeddings: torch.Tensor) -> torch.Tensor:
        """Map a batch of sentence embeddings to unnormalized class logits.

        Args:
            sentence_embeddings: Tensor of shape (batch, input_dim).

        Returns:
            Tensor of shape (batch, len(CommandType)) -- raw logits, not
            softmax-normalized (use `torch.argmax` or `F.softmax` as needed).

        """
        return self.net(sentence_embeddings.to(torch.float32))


def train_command_type_classifier(
    sentence_embeddings: torch.Tensor,
    labels: Sequence[CommandType],
    *,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    seed: int = 0,
    classifier: CommandTypeClassifier | None = None,
) -> tuple[CommandTypeClassifier, list[float]]:
    """Train a `CommandTypeClassifier` via plain cross-entropy.

    Small and fast by design: this is a 384->64->5 MLP over at most a few
    hundred short sentences, expected to train in seconds.

    Args:
        sentence_embeddings: Precomputed frozen sentence embeddings, shape
            (n_examples, input_dim) -- e.g. via
            `language_embedding.encode_instructions` on
            `command_type_vocabulary.build_command_type_training_set`'s
            texts.
        labels: Ground-truth `CommandType` for each row of
            `sentence_embeddings`, same length and order.
        epochs: Number of optimizer steps.
        learning_rate: Adam learning rate for `classifier`'s parameters.
        seed: Seed for `classifier`'s initialization when none is supplied.
        classifier: An existing `CommandTypeClassifier` to continue training.
            If `None`, a fresh one is constructed sized to
            `sentence_embeddings`'s dimension.

    Returns:
        A tuple `(classifier, loss_history)`: the trained module and the
        cross-entropy loss recorded at every epoch.

    """
    torch.manual_seed(seed)
    resolved_classifier = classifier or CommandTypeClassifier(input_dim=sentence_embeddings.shape[1])

    target_indices = torch.tensor(
        [_COMMAND_TYPE_TO_INDEX[label] for label in labels], dtype=torch.long,
    )
    frozen_sentence_embeddings = sentence_embeddings.detach().to(torch.float32)

    optimizer = torch.optim.Adam(resolved_classifier.parameters(), lr=learning_rate)
    loss_history: list[float] = []
    for _epoch in range(epochs):
        logits = resolved_classifier(frozen_sentence_embeddings)
        loss = f.cross_entropy(logits, target_indices)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    return resolved_classifier, loss_history


def classify_command_type(text: str, classifier: CommandTypeClassifier) -> CommandType:
    """Classify one sentence's command type, end to end.

    Embeds `text` with the frozen sentence-transformer
    (`language_embedding.encode_instructions`) and returns the classifier's
    argmax class. Never raises on well-formed English input -- an
    out-of-scope or nonsensical sentence is expected to classify as
    `CommandType.UNSUPPORTED`, which is this function's correct, ordinary
    output for that input, not an exception. Rejecting `UNSUPPORTED` text is
    a later stage's dispatch-logic decision, not this function's.

    Args:
        text: The English sentence to classify.
        classifier: A trained (or untrained) `CommandTypeClassifier`.

    Returns:
        The predicted `CommandType`.

    """
    embedding = encode_instructions([text])
    sentence_embedding = torch.from_numpy(embedding).to(torch.float32)
    with torch.no_grad():
        logits = classifier(sentence_embedding)
    predicted_index = int(torch.argmax(logits, dim=1).item())
    return _INDEX_TO_COMMAND_TYPE[predicted_index]
