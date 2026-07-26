"""Tests for GoalEmbeddingExtractor, the SB3 features-extractor integration point.

Sits between the raw Dict observation (`observation`/`achieved_goal`/
`desired_goal`, FetchReach's shape) and the policy/critic networks: passes
`observation` through unchanged (flattened) and routes `achieved_goal` /
`desired_goal` through a shared `GoalEncoder` so the policy only ever sees
the learned embedding, never the literal xyz goal. HER's replay buffer and
`compute_reward` sit upstream of this and are untouched — they still see
literal xyz.
"""

import numpy as np
import torch
from gymnasium import spaces

from lang_goal_rl.goal_encoder import GoalEncoder
from lang_goal_rl.goal_embedding_extractor import GoalEmbeddingExtractor

OBS_DIM = 10
GOAL_DIM = 3


def _fetch_reach_observation_space() -> spaces.Dict:
    """Build a Dict observation space shaped like FetchReach-v4's."""
    return spaces.Dict(
        {
            "observation": spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float64),
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(GOAL_DIM,), dtype=np.float64),
            "desired_goal": spaces.Box(-np.inf, np.inf, shape=(GOAL_DIM,), dtype=np.float64),
        }
    )


def _sample_observations(batch_size: int) -> dict[str, torch.Tensor]:
    return {
        "observation": torch.rand(batch_size, OBS_DIM, dtype=torch.float64),
        "achieved_goal": torch.rand(batch_size, GOAL_DIM, dtype=torch.float64),
        "desired_goal": torch.rand(batch_size, GOAL_DIM, dtype=torch.float64),
    }


class TestGoalEmbeddingExtractor:
    """GoalEmbeddingExtractor concatenates raw observation with goal embeddings."""

    def test_features_dim_is_observation_dim_plus_two_embeddings(self) -> None:
        extractor = GoalEmbeddingExtractor(_fetch_reach_observation_space(), embed_dim=16)
        assert extractor.features_dim == OBS_DIM + 2 * 16

    def test_forward_output_shape_matches_features_dim(self) -> None:
        extractor = GoalEmbeddingExtractor(_fetch_reach_observation_space(), embed_dim=16)
        observations = _sample_observations(batch_size=5)
        output = extractor(observations)
        assert output.shape == (5, extractor.features_dim)

    def test_observation_slice_passes_through_unchanged(self) -> None:
        extractor = GoalEmbeddingExtractor(_fetch_reach_observation_space(), embed_dim=16)
        observations = _sample_observations(batch_size=3)
        output = extractor(observations)
        raw_observation_slice = output[:, :OBS_DIM].to(torch.float64)
        assert torch.allclose(raw_observation_slice, observations["observation"])

    def test_achieved_and_desired_goal_share_the_same_encoder(self) -> None:
        extractor = GoalEmbeddingExtractor(_fetch_reach_observation_space(), embed_dim=16)
        same_goal = torch.rand(1, GOAL_DIM, dtype=torch.float64)
        observations = {
            "observation": torch.rand(1, OBS_DIM, dtype=torch.float64),
            "achieved_goal": same_goal,
            "desired_goal": same_goal,
        }
        output = extractor(observations)
        achieved_embedding = output[:, OBS_DIM : OBS_DIM + 16]
        desired_embedding = output[:, OBS_DIM + 16 :]
        # Same literal goal fed through both slots must land at the same
        # embedding, since achieved_goal and desired_goal are the same
        # underlying quantity (xyz position) and must share a distance
        # metric to be comparable.
        assert torch.allclose(achieved_embedding, desired_embedding)

    def test_accepts_a_pretrained_goal_encoder_and_copies_its_weights(self) -> None:
        pretrained = GoalEncoder(goal_dim=GOAL_DIM, embed_dim=16)
        extractor = GoalEmbeddingExtractor(
            _fetch_reach_observation_space(), goal_encoder=pretrained
        )
        # A deep copy, not the same object: see
        # test_independent_extractor_instances_do_not_share_encoder_tensors
        # for why aliasing the caller's object across multiple extractor
        # instances (as SB3 constructs for actor/critic/critic_target) is
        # unsafe.
        assert extractor.goal_encoder is not pretrained
        goal = torch.rand(1, GOAL_DIM, dtype=torch.float64)
        assert torch.equal(extractor.goal_encoder(goal), pretrained(goal))

    def test_does_not_mutate_the_caller_supplied_encoder_object(self) -> None:
        pretrained = GoalEncoder(goal_dim=GOAL_DIM, embed_dim=16)
        GoalEmbeddingExtractor(
            _fetch_reach_observation_space(), goal_encoder=pretrained, freeze_encoder=True
        )
        # Freezing must apply only to the extractor's own copy — the
        # caller's original object may still be in use elsewhere (e.g. a
        # separate contrastive pretraining loop) and must keep its own
        # requires_grad state untouched.
        assert all(p.requires_grad for p in pretrained.parameters())

    def test_freezes_pretrained_encoder_parameters_by_default(self) -> None:
        pretrained = GoalEncoder(goal_dim=GOAL_DIM, embed_dim=16)
        extractor = GoalEmbeddingExtractor(
            _fetch_reach_observation_space(), goal_encoder=pretrained
        )
        assert all(not p.requires_grad for p in extractor.goal_encoder.parameters())

    def test_independent_extractor_instances_do_not_share_encoder_tensors(self) -> None:
        # SB3 constructs a separate GoalEmbeddingExtractor for the actor,
        # critic, and critic_target from the same `features_extractor_kwargs`
        # dict, so the same `goal_encoder` object is handed to every
        # constructor call. If the extractor stored that object as-is,
        # SAC's target-network polyak_update (which walks
        # zip(critic.parameters(), critic_target.parameters())) would
        # silently drift the "frozen" weights by operating on the same
        # underlying tensor twice. Each extractor must own an independent
        # tensor even when built from the same pretrained encoder object.
        pretrained = GoalEncoder(goal_dim=GOAL_DIM, embed_dim=16)
        extractor_a = GoalEmbeddingExtractor(
            _fetch_reach_observation_space(), goal_encoder=pretrained
        )
        extractor_b = GoalEmbeddingExtractor(
            _fetch_reach_observation_space(), goal_encoder=pretrained
        )
        assert extractor_a.goal_encoder is not extractor_b.goal_encoder
        for param_a, param_b in zip(
            extractor_a.goal_encoder.parameters(),
            extractor_b.goal_encoder.parameters(),
            strict=True,
        ):
            assert param_a.data_ptr() != param_b.data_ptr()

    def test_can_leave_encoder_trainable_when_freeze_is_disabled(self) -> None:
        pretrained = GoalEncoder(goal_dim=GOAL_DIM, embed_dim=16)
        extractor = GoalEmbeddingExtractor(
            _fetch_reach_observation_space(),
            goal_encoder=pretrained,
            freeze_encoder=False,
        )
        assert all(p.requires_grad for p in extractor.goal_encoder.parameters())
