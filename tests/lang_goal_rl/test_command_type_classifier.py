"""Tests for stage 11's command-type classifier.

`CommandTypeClassifier` maps a frozen sentence-transformer embedding
(384-dim, see `language_embedding.LANGUAGE_EMBED_DIM`) to a 5-way softmax
over `command_type_vocabulary.CommandType` -- the same
Linear-ReLU-Linear-64-hidden shape as `LanguageGoalProjection`
(`language_goal_projection.py`), sized for a classification head instead of
a regression head. `train_command_type_classifier` fits it with plain
cross-entropy on precomputed sentence embeddings, and `classify_command_type`
is the end-to-end text -> `CommandType` entry point the experiment-runner and
any later dispatch logic call. These tests stay offline/fast for training
(synthetic embeddings), and only call the real frozen encoder for
`classify_command_type`'s own contract (never raises, always returns a valid
label).
"""

from __future__ import annotations

import torch

from lang_goal_rl.command_type_classifier import (
    CommandTypeClassifier,
    classify_command_type,
    train_command_type_classifier,
)
from lang_goal_rl.command_type_vocabulary import CommandType


class TestCommandTypeClassifier:
    """CommandTypeClassifier is a trainable nn.Module mapping input_dim -> 5-way logits."""

    def test_forward_output_shape_matches_number_of_command_types(self) -> None:
        classifier = CommandTypeClassifier(input_dim=384, hidden_dim=64)
        batch = torch.randn(5, 384)
        output = classifier(batch)
        assert output.shape == (5, len(CommandType))

    def test_default_input_dim_matches_minilm_output(self) -> None:
        classifier = CommandTypeClassifier()
        batch = torch.randn(3, 384)
        assert classifier(batch).shape == (3, len(CommandType))

    def test_is_a_real_trainable_module_with_parameters(self) -> None:
        classifier = CommandTypeClassifier(input_dim=8, hidden_dim=6)
        parameters = list(classifier.parameters())
        assert len(parameters) > 0
        assert all(p.requires_grad for p in parameters)

    def test_forward_pass_produces_finite_output(self) -> None:
        classifier = CommandTypeClassifier(input_dim=8, hidden_dim=6)
        batch = torch.randn(10, 8)
        assert torch.isfinite(classifier(batch)).all()


class TestTrainCommandTypeClassifier:
    """train_command_type_classifier: standard cross-entropy training loop."""

    def test_returns_a_classifier_and_a_loss_history(self) -> None:
        torch.manual_seed(0)
        embeddings = torch.randn(10, 8)
        labels = [
            CommandType.MOVE,
            CommandType.GOTO_NAMED_REGION,
            CommandType.STOP,
            CommandType.RESET,
            CommandType.UNSUPPORTED,
        ] * 2

        classifier, loss_history = train_command_type_classifier(
            embeddings,
            labels,
            epochs=5,
            classifier=CommandTypeClassifier(input_dim=8, hidden_dim=6),
        )

        assert isinstance(classifier, CommandTypeClassifier)
        assert len(loss_history) == 5

    def test_training_reduces_loss_on_a_separable_synthetic_case(self) -> None:
        """Each class gets its own one-hot-ish direction in embedding space --
        a trivially separable synthetic case, just enough to check the
        training loop actually descends, not to measure real accuracy (that's
        the experiment-runner's job on the real vocabulary).
        """
        torch.manual_seed(0)
        n_classes = len(CommandType)
        embeddings = torch.eye(n_classes).repeat(4, 1)
        labels = list(CommandType) * 4

        _classifier, loss_history = train_command_type_classifier(
            embeddings,
            labels,
            epochs=200,
            learning_rate=0.05,
            classifier=CommandTypeClassifier(input_dim=n_classes, hidden_dim=6),
        )

        early_mean = sum(loss_history[:10]) / 10
        late_mean = sum(loss_history[-10:]) / 10
        assert late_mean < early_mean

    def test_constructs_a_default_classifier_when_none_is_provided(self) -> None:
        torch.manual_seed(0)
        embeddings = torch.randn(5, 384)
        labels = [CommandType.MOVE, CommandType.STOP, CommandType.RESET, CommandType.UNSUPPORTED,
                  CommandType.GOTO_NAMED_REGION]

        classifier, _loss_history = train_command_type_classifier(embeddings, labels, epochs=2)

        assert classifier(embeddings).shape == (5, len(CommandType))


class TestClassifyCommandType:
    """classify_command_type: real text -> CommandType, never raises."""

    def test_returns_a_valid_command_type_for_arbitrary_text(self) -> None:
        classifier = CommandTypeClassifier()
        for text in ["move forward", "stop", "pick up the block", "", "asdkfj qweoi", "reset the episode"]:
            result = classify_command_type(text, classifier)
            assert isinstance(result, CommandType)

    def test_never_raises_on_well_formed_or_malformed_input(self) -> None:
        classifier = CommandTypeClassifier()
        odd_inputs = [
            "move forward",
            "!!!???",
            "a" * 500,
            "1234567890",
            "reach up high please and thank you very much",
        ]
        for text in odd_inputs:
            classify_command_type(text, classifier)
