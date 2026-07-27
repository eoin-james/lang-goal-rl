"""Tests for midepisode_regoal: stage 5's mid-episode goal-switch mechanism.

Uses the real FetchReach-v4 env throughout (MuJoCo dynamics are deterministic
given a seed and a fixed action sequence, confirmed via a manual probe before
writing these tests) so the goal-swap and success-judging logic run against
real env state, never a mocked env. Only the *policy* (`model.predict`) is a
lightweight stub — these are stage-5 mechanism tests, not proof-gate RL runs,
so no trained checkpoint is required. Two stub flavors:

- `_RecordingStubModel`: takes a fixed action every step (default zero, i.e.
  no-op) and records each `obs["desired_goal"]` it was asked to predict on —
  this is the spy that proves the goal-swap actually reaches the policy's
  input at the right step, not just that the function ran without erroring.
- `_EmbeddingRecordingStubModel`: same, but with a real `GoalEmbeddingExtractor`
  as `actor.features_extractor` and a probe call into it each step, to prove
  the optional embedding-substitution path (reusing
  `episode_recording._pin_desired_goal_embedding`) patches and restores
  correctly around the switch.

Success/failure cases use a synthetic achieved-goal target rather than an
end-to-end run whose outcome can't be independently verified: FetchReach's
`distance_threshold` is 0.05m, and a zero action moves `achieved_goal` by
~2e-4m per step (measured directly), so setting the post-switch goal equal to
a just-observed `achieved_goal` guarantees success, and offsetting it by 10m
guarantees failure, regardless of what any real policy would have done.
"""

from __future__ import annotations

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import pytest
import torch

from lang_goal_rl.episode_recording import _pin_desired_goal_embedding
from lang_goal_rl.goal_embedding_extractor import GoalEmbeddingExtractor
from lang_goal_rl.midepisode_regoal import (
    GoalSwitchResult,
    rollout_fresh_with_budget,
    rollout_with_goal_switch,
)

gym.register_envs(gymnasium_robotics)

ENV_ID = "FetchReach-v4"
PROBE_SEED = 42


def _zero_action(env: gym.Env) -> np.ndarray:
    return np.zeros(env.action_space.shape, dtype=env.action_space.dtype)


def _achieved_goal_after_n_zero_steps(env: gym.Env, *, seed: int, n_steps: int) -> np.ndarray:
    """Replay `n_steps` zero actions from a fresh reset and return the resulting `achieved_goal`.

    Used to precompute a real, physically-reachable point for the
    guaranteed-success test case below, from the same seed the actual test
    rollout will reset with -- MuJoCo is deterministic given a seed and a
    fixed action sequence (confirmed via manual probe), so replaying the
    identical seed + zero actions in the real rollout reproduces this exact
    point.
    """
    obs, _info = env.reset(seed=seed)
    action = _zero_action(env)
    for _ in range(n_steps):
        obs, _reward, _terminated, _truncated, _info = env.step(action)
    return np.array(obs["achieved_goal"], copy=True)


class _RecordingStubModel:
    """Fixed-action stub that records every `obs["desired_goal"]` it's asked to predict on."""

    def __init__(self, env: gym.Env, *, action: np.ndarray | None = None) -> None:
        self.desired_goals_seen: list[np.ndarray] = []
        self._action = action if action is not None else _zero_action(env)

    def predict(self, obs: dict, *, deterministic: bool = True) -> tuple[np.ndarray, None]:
        del deterministic
        self.desired_goals_seen.append(np.array(obs["desired_goal"], copy=True))
        return self._action, None


class _EmbeddingRecordingStubModel:
    """Fixed-action stub with a real `GoalEmbeddingExtractor`, probing its output every step.

    Mirrors test_episode_recording.py's `_StubModel`, but each `predict` call
    also runs `actor.features_extractor` on a fixed dummy observation and
    records the resulting desired-goal embedding slice -- proof that the
    correct pinned embedding (initial vs. new) is active at each step,
    independent of whether the extractor happens to be patched at call time.
    """

    class _Actor:
        def __init__(self, features_extractor: GoalEmbeddingExtractor) -> None:
            self.features_extractor = features_extractor

    def __init__(self, env: gym.Env, *, embed_dim: int = 4, action: np.ndarray | None = None) -> None:
        self.actor = self._Actor(GoalEmbeddingExtractor(env.observation_space, embed_dim=embed_dim))
        self._action = action if action is not None else _zero_action(env)
        self.embed_dim = embed_dim
        self.desired_embeddings_seen: list[torch.Tensor] = []
        self._dummy_observations = {
            "observation": torch.rand(1, 10, dtype=torch.float64),
            "achieved_goal": torch.rand(1, 3, dtype=torch.float64),
            "desired_goal": torch.rand(1, 3, dtype=torch.float64),
        }

    def predict(self, obs: dict, *, deterministic: bool = True) -> tuple[np.ndarray, None]:
        del deterministic, obs
        output = self.actor.features_extractor(self._dummy_observations)
        self.desired_embeddings_seen.append(output[:, -self.embed_dim :].detach().clone())
        return self._action, None


class TestRolloutWithGoalSwitch:
    """rollout_with_goal_switch: swap the policy's goal input mid-episode, no reset."""

    def test_returns_goal_switch_result(self) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)

        result = rollout_with_goal_switch(
            model,
            env,
            initial_goal_xyz=np.array([1.3, 0.7, 0.5]),
            switch_step=2,
            new_goal_xyz=np.array([1.4, 0.8, 0.6]),
            max_steps=5,
            base_seed=PROBE_SEED,
        )

        assert isinstance(result, GoalSwitchResult)
        env.close()

    def test_runs_exactly_max_steps_when_env_never_truncates_first(self) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)

        result = rollout_with_goal_switch(
            model,
            env,
            initial_goal_xyz=np.array([1.3, 0.7, 0.5]),
            switch_step=2,
            new_goal_xyz=np.array([1.4, 0.8, 0.6]),
            max_steps=5,
            base_seed=PROBE_SEED,
        )

        assert result.n_steps == 5
        env.close()

    def test_echoes_back_switch_step(self) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)

        result = rollout_with_goal_switch(
            model,
            env,
            initial_goal_xyz=np.array([1.3, 0.7, 0.5]),
            switch_step=3,
            new_goal_xyz=np.array([1.4, 0.8, 0.6]),
            max_steps=6,
            base_seed=PROBE_SEED,
        )

        assert result.switch_step == 3
        env.close()

    def test_policy_sees_initial_goal_before_switch_step_and_new_goal_after(
        self,
    ) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)
        initial_goal = np.array([1.3, 0.7, 0.5])
        new_goal = np.array([1.4, 0.8, 0.6])

        rollout_with_goal_switch(
            model,
            env,
            initial_goal_xyz=initial_goal,
            switch_step=3,
            new_goal_xyz=new_goal,
            max_steps=6,
            base_seed=PROBE_SEED,
        )

        seen = model.desired_goals_seen
        assert len(seen) == 6
        for goal_seen in seen[:3]:
            assert np.allclose(goal_seen, initial_goal)
        for goal_seen in seen[3:]:
            assert np.allclose(goal_seen, new_goal)
        env.close()

    def test_max_steps_caps_episode_length_below_env_default(self) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)

        result = rollout_with_goal_switch(
            model,
            env,
            initial_goal_xyz=np.array([1.3, 0.7, 0.5]),
            switch_step=1,
            new_goal_xyz=np.array([1.4, 0.8, 0.6]),
            max_steps=4,
            base_seed=PROBE_SEED,
        )

        # FetchReach-v4's registered TimeLimit is 50 steps -- if our own cap
        # weren't enforced, the episode would run far longer than 4.
        assert result.n_steps == 4
        env.close()

    def test_success_reflects_synthetic_guaranteed_success_case(self) -> None:
        # Precompute the real achieved_goal 2 zero-action steps after a
        # fresh reset with the same seed the real rollout below uses --
        # MuJoCo is deterministic given seed + action sequence, so replaying
        # identically reproduces this exact point (confirmed by manual probe:
        # a zero action moves achieved_goal by ~2e-4m, far under the 0.05m
        # distance_threshold).
        probe_env = gym.make(ENV_ID)
        guaranteed_reachable_point = _achieved_goal_after_n_zero_steps(
            probe_env, seed=PROBE_SEED, n_steps=2
        )
        probe_env.close()

        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)

        result = rollout_with_goal_switch(
            model,
            env,
            initial_goal_xyz=np.array([1.3, 0.7, 0.5]),
            switch_step=2,
            new_goal_xyz=guaranteed_reachable_point,
            max_steps=3,
            base_seed=PROBE_SEED,
        )

        assert result.success is True
        env.close()

    def test_success_reflects_synthetic_guaranteed_failure_case(self) -> None:
        probe_env = gym.make(ENV_ID)
        near_point = _achieved_goal_after_n_zero_steps(probe_env, seed=PROBE_SEED, n_steps=2)
        probe_env.close()
        far_point = near_point + 10.0  # 10m away, nowhere near FetchReach's workspace

        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)

        result = rollout_with_goal_switch(
            model,
            env,
            initial_goal_xyz=np.array([1.3, 0.7, 0.5]),
            switch_step=2,
            new_goal_xyz=far_point,
            max_steps=3,
            base_seed=PROBE_SEED,
        )

        assert result.success is False
        env.close()

    def test_pre_switch_success_never_counts_toward_the_result(self) -> None:
        # initial_goal_xyz is set to the policy's actual starting position,
        # so the pre-switch phase is "successful" by construction. If that
        # leaked into the final result, this would report success=True even
        # though the post-switch goal is unreachable in the budget given.
        probe_env = gym.make(ENV_ID)
        obs, _info = probe_env.reset(seed=PROBE_SEED)
        starting_point = np.array(obs["achieved_goal"], copy=True)
        probe_env.close()
        far_point = starting_point + 10.0

        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)

        result = rollout_with_goal_switch(
            model,
            env,
            initial_goal_xyz=starting_point,
            switch_step=2,
            new_goal_xyz=far_point,
            max_steps=3,
            base_seed=PROBE_SEED,
        )

        assert result.success is False
        env.close()

    def test_literal_mode_never_touches_the_model_actor(self) -> None:
        class _ActorlessStubModel:
            def __init__(self, env: gym.Env) -> None:
                self._action = _zero_action(env)

            def predict(self, _obs: dict, *, deterministic: bool = True) -> tuple[np.ndarray, None]:
                del deterministic
                return self._action, None

        env = gym.make(ENV_ID)
        model = _ActorlessStubModel(env)

        result = rollout_with_goal_switch(
            model,
            env,
            initial_goal_xyz=np.array([1.3, 0.7, 0.5]),
            switch_step=1,
            new_goal_xyz=np.array([1.4, 0.8, 0.6]),
            max_steps=3,
            base_seed=PROBE_SEED,
        )

        assert result.n_steps == 3
        env.close()

    def test_rejects_switch_step_at_or_past_max_steps(self) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)

        with pytest.raises(ValueError, match="switch_step"):
            rollout_with_goal_switch(
                model,
                env,
                initial_goal_xyz=np.array([1.3, 0.7, 0.5]),
                switch_step=5,
                new_goal_xyz=np.array([1.4, 0.8, 0.6]),
                max_steps=5,
                base_seed=PROBE_SEED,
            )
        env.close()

    def test_rejects_switch_step_of_zero(self) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)

        with pytest.raises(ValueError, match="switch_step"):
            rollout_with_goal_switch(
                model,
                env,
                initial_goal_xyz=np.array([1.3, 0.7, 0.5]),
                switch_step=0,
                new_goal_xyz=np.array([1.4, 0.8, 0.6]),
                max_steps=5,
                base_seed=PROBE_SEED,
            )
        env.close()

    def test_rejects_max_steps_exceeding_the_envs_registered_limit(self) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)

        with pytest.raises(ValueError, match="max_episode_steps"):
            rollout_with_goal_switch(
                model,
                env,
                initial_goal_xyz=np.array([1.3, 0.7, 0.5]),
                switch_step=2,
                new_goal_xyz=np.array([1.4, 0.8, 0.6]),
                max_steps=51,
                base_seed=PROBE_SEED,
            )
        env.close()

    def test_rejects_mismatched_embedding_args(self) -> None:
        env = gym.make(ENV_ID)
        model = _EmbeddingRecordingStubModel(env)

        with pytest.raises(ValueError, match="embedding"):
            rollout_with_goal_switch(
                model,
                env,
                initial_goal_xyz=np.array([1.3, 0.7, 0.5]),
                switch_step=2,
                new_goal_xyz=np.array([1.4, 0.8, 0.6]),
                max_steps=5,
                base_seed=PROBE_SEED,
                initial_goal_embedding=torch.randn(4),
                new_goal_embedding=None,
            )
        env.close()

    def test_embedding_mode_pins_initial_then_new_embedding_at_switch_step(
        self,
    ) -> None:
        env = gym.make(ENV_ID)
        model = _EmbeddingRecordingStubModel(env, embed_dim=4)
        initial_embedding = torch.randn(4)
        new_embedding = torch.randn(4)

        rollout_with_goal_switch(
            model,
            env,
            initial_goal_xyz=np.array([1.3, 0.7, 0.5]),
            switch_step=2,
            new_goal_xyz=np.array([1.4, 0.8, 0.6]),
            max_steps=4,
            base_seed=PROBE_SEED,
            initial_goal_embedding=initial_embedding,
            new_goal_embedding=new_embedding,
        )

        seen = model.desired_embeddings_seen
        assert len(seen) == 4
        for embedding_seen in seen[:2]:
            assert torch.allclose(
                embedding_seen.to(torch.float32), initial_embedding.expand(1, -1).to(torch.float32)
            )
        for embedding_seen in seen[2:]:
            assert torch.allclose(
                embedding_seen.to(torch.float32), new_embedding.expand(1, -1).to(torch.float32)
            )
        env.close()

    def test_embedding_mode_restores_original_forward_after_rollout(self) -> None:
        env = gym.make(ENV_ID)
        model = _EmbeddingRecordingStubModel(env, embed_dim=4)
        extractor = model.actor.features_extractor
        original_forward = extractor.forward

        rollout_with_goal_switch(
            model,
            env,
            initial_goal_xyz=np.array([1.3, 0.7, 0.5]),
            switch_step=1,
            new_goal_xyz=np.array([1.4, 0.8, 0.6]),
            max_steps=2,
            base_seed=PROBE_SEED,
            initial_goal_embedding=torch.randn(4),
            new_goal_embedding=torch.randn(4),
        )

        assert extractor.forward == original_forward
        env.close()

    def test_embedding_mode_uses_the_same_pin_mechanism_as_episode_recording(
        self,
    ) -> None:
        # Not a behavioral test -- a design-intent guard. Stage 5 must not
        # grow a third, independently-maintained copy of the desired-goal
        # monkeypatch (episode_recording.py already generalized stage 3's
        # inline version once). Importing it directly here is the actual
        # mechanism reuse this test is guarding against regressing.
        import lang_goal_rl.midepisode_regoal as module

        assert module._pin_desired_goal_embedding is _pin_desired_goal_embedding


class TestRolloutFreshWithBudget:
    """rollout_fresh_with_budget: the budget-matched fresh-episode baseline."""

    def test_returns_a_bool(self) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)

        result = rollout_fresh_with_budget(
            model,
            env,
            goal_xyz=np.array([1.3, 0.7, 0.5]),
            max_steps=5,
            base_seed=PROBE_SEED,
        )

        assert isinstance(result, bool)
        env.close()

    def test_stops_at_max_steps_well_below_env_default(self) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)
        n_calls = 0
        original_predict = model.predict

        def counting_predict(obs: dict, *, deterministic: bool = True) -> tuple[np.ndarray, None]:
            nonlocal n_calls
            n_calls += 1
            return original_predict(obs, deterministic=deterministic)

        model.predict = counting_predict

        rollout_fresh_with_budget(
            model,
            env,
            goal_xyz=np.array([1.3, 0.7, 0.5]),
            max_steps=4,
            base_seed=PROBE_SEED,
        )

        assert n_calls == 4
        env.close()

    def test_policy_sees_the_given_goal_xyz_every_step(self) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)
        goal = np.array([1.35, 0.72, 0.48])

        rollout_fresh_with_budget(
            model,
            env,
            goal_xyz=goal,
            max_steps=3,
            base_seed=PROBE_SEED,
        )

        assert len(model.desired_goals_seen) == 3
        for goal_seen in model.desired_goals_seen:
            assert np.allclose(goal_seen, goal)
        env.close()

    def test_success_true_for_synthetic_guaranteed_reachable_goal(self) -> None:
        probe_env = gym.make(ENV_ID)
        guaranteed_reachable_point = _achieved_goal_after_n_zero_steps(
            probe_env, seed=PROBE_SEED, n_steps=2
        )
        probe_env.close()

        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)

        result = rollout_fresh_with_budget(
            model,
            env,
            goal_xyz=guaranteed_reachable_point,
            max_steps=2,
            base_seed=PROBE_SEED,
        )

        assert result is True
        env.close()

    def test_success_false_for_synthetic_unreachable_goal(self) -> None:
        probe_env = gym.make(ENV_ID)
        obs, _info = probe_env.reset(seed=PROBE_SEED)
        far_point = np.array(obs["achieved_goal"], copy=True) + 10.0
        probe_env.close()

        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)

        result = rollout_fresh_with_budget(
            model,
            env,
            goal_xyz=far_point,
            max_steps=2,
            base_seed=PROBE_SEED,
        )

        assert result is False
        env.close()

    def test_rejects_max_steps_exceeding_the_envs_registered_limit(self) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)

        with pytest.raises(ValueError, match="max_episode_steps"):
            rollout_fresh_with_budget(
                model,
                env,
                goal_xyz=np.array([1.3, 0.7, 0.5]),
                max_steps=51,
                base_seed=PROBE_SEED,
            )
        env.close()
