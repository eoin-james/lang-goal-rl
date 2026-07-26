"""SB3 features extractor that swaps literal goals for learned embeddings.

Sits between FetchReach's raw Dict observation and the policy/critic
networks. `observation` passes through flattened and unchanged.
`achieved_goal` and `desired_goal` are each run through the *same*
`GoalEncoder` instance (shared weights, so both live in one comparable
distance metric) before being concatenated on. HER's `HerReplayBuffer` and
`compute_reward` sit upstream of this extractor in the SB3 pipeline and
operate on literal xyz throughout — this class only changes what the
policy/critic networks see, never what gets stored or relabeled.

Training-mode decision: the encoder is expected to arrive *pretrained*
(via `contrastive.info_nce_loss` on sampled goal pairs) and is frozen by
default here, rather than trained end-to-end through the RL loss. End-to-end
backprop is simpler to wire up, but nothing in the RL objective requires
the resulting embedding to preserve xyz distance — the policy only needs
*an* invertible-enough encoding, not a distance-preserving one. The
contrastive pretraining objective is the only mechanism in this codebase
that directly targets the proof gate's "distance-in-latent correlates with
true task distance" requirement, so pretrain-then-freeze is what actually
tries to satisfy it. `freeze_encoder=False` is left available for the
experiment-runner to A/B against, but frozen is the default.

Why the encoder is copied, not referenced: SB3 builds a separate
`GoalEmbeddingExtractor` instance per network (actor, critic,
critic_target) from the same `features_extractor_kwargs` dict. Storing a
caller-supplied `goal_encoder` by reference would alias the same tensors
across all three instances, which corrupts SAC's target-network
`polyak_update` (it silently drifts "frozen" weights — see the inline
comment in `__init__` for the confirmed repro). Every instance therefore
gets its own `copy.deepcopy` of the resolved encoder.
"""

from __future__ import annotations

import copy

import torch
from gymnasium import spaces
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from lang_goal_rl.goal_encoder import DEFAULT_EMBED_DIM, GoalEncoder


class GoalEmbeddingExtractor(BaseFeaturesExtractor):
    """Features extractor: raw `observation` + embedded `achieved_goal`/`desired_goal`."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        goal_encoder: GoalEncoder | None = None,
        embed_dim: int = DEFAULT_EMBED_DIM,
        hidden_dim: int = 64,
        freeze_encoder: bool = True,
    ) -> None:
        """Build the extractor.

        Args:
            observation_space: FetchReach-shaped Dict space with
                `observation`, `achieved_goal`, `desired_goal` keys.
            goal_encoder: An existing (typically contrastively pretrained)
                `GoalEncoder` to reuse. Deep-copied into this extractor
                rather than referenced directly — see the class docstring's
                "Why the encoder is copied" note. If `None`, a fresh one is
                constructed with `embed_dim`/`hidden_dim`.
            embed_dim: Embedding dimensionality, used only when
                `goal_encoder` is `None`.
            hidden_dim: Encoder hidden width, used only when `goal_encoder`
                is `None`.
            freeze_encoder: If `True` (default), the goal encoder's
                parameters are set to `requires_grad=False` so RL training
                cannot drift the pretrained embedding geometry. Set `False`
                to fine-tune the encoder jointly with the policy/critic.
        """
        obs_dim = get_flattened_obs_dim(observation_space.spaces["observation"])
        goal_dim = get_flattened_obs_dim(observation_space.spaces["achieved_goal"])

        # nn.Module forbids assigning submodules before its __init__() runs,
        # so the encoder is resolved into a local first and only attached to
        # `self` once `super().__init__()` (below) has set up module state.
        #
        # Deep-copy rather than reference a caller-supplied encoder: SB3
        # constructs one GoalEmbeddingExtractor per network (actor, critic,
        # critic_target) from the *same* `features_extractor_kwargs` dict,
        # so a passed-in encoder object would otherwise be aliased across
        # all three. That aliasing breaks SAC's target-network
        # `polyak_update(critic.parameters(), critic_target.parameters(), tau)`:
        # if both sides of that zip resolve to the same underlying tensor,
        # the in-place `mul_` then `add_` sequence mutates the "frozen"
        # weights by a tiny amount every target-update step (confirmed via
        # an end-to-end SAC+HER smoke run: weights drifted ~1e-3 after 200
        # timesteps despite `requires_grad=False`). Deep-copying gives every
        # network its own tensor, which also means freezing here never
        # mutates the caller's own object.
        resolved_encoder = copy.deepcopy(goal_encoder) if goal_encoder is not None else GoalEncoder(
            goal_dim=goal_dim, embed_dim=embed_dim, hidden_dim=hidden_dim
        )

        super().__init__(
            observation_space,
            features_dim=obs_dim + 2 * resolved_encoder.embed_dim,
        )

        self.goal_encoder = resolved_encoder
        if freeze_encoder:
            for parameter in self.goal_encoder.parameters():
                parameter.requires_grad = False

        self._flatten = nn.Flatten()

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate the flattened observation with both goal embeddings.

        Args:
            observations: Dict of tensors keyed `observation`,
                `achieved_goal`, `desired_goal`, each shaped (batch, dim).

        Returns:
            Tensor of shape (batch, features_dim).
        """
        flat_observation = self._flatten(observations["observation"])
        achieved_embedding = self.goal_encoder(observations["achieved_goal"])
        desired_embedding = self.goal_encoder(observations["desired_goal"])
        return torch.cat([flat_observation, achieved_embedding, desired_embedding], dim=1)
