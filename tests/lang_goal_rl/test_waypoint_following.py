"""Tests for waypoint_following: stage 9's N-waypoint generalization of stage 5's goal switch.

Uses the real FetchReach-v4 env throughout (MuJoCo dynamics are deterministic
given a seed and a fixed action sequence -- the same property
`test_midepisode_regoal.py` already confirmed and relies on), with a
lightweight fixed-action stub policy standing in for a trained checkpoint.
This is a mechanism test suite, not a proof-gate RL run.

The single most important test in this file is
`TestEquivalenceWithMidepisodeRegoal` -- it proves N=2 waypoint-following
with one `steps_per_leg` value literally reduces to
`midepisode_regoal.rollout_with_goal_switch`'s own result (same success
outcome, same step count), using the identical stub model, seed, and goals
in both calls. That is the concrete evidence stage 9 generalizes stage 5's
already-proven mechanism rather than reimplementing something that merely
looks similar.
"""

from __future__ import annotations

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import pytest

from lang_goal_rl.midepisode_regoal import rollout_with_goal_switch
from lang_goal_rl.waypoint_following import WaypointResult, rollout_with_waypoints

gym.register_envs(gymnasium_robotics)

ENV_ID = "FetchReach-v4"
PROBE_SEED = 42


def _zero_action(env: gym.Env) -> np.ndarray:
    return np.zeros(env.action_space.shape, dtype=env.action_space.dtype)


def _achieved_goal_after_n_zero_steps(env: gym.Env, *, seed: int, n_steps: int) -> np.ndarray:
    """Replay `n_steps` zero actions from a fresh reset and return the resulting `achieved_goal`.

    A zero-action stub's `achieved_goal` trajectory depends only on the seed
    and the (fixed, zero) action sequence -- never on whatever goal happens
    to be active -- so this precomputes exact, physically-reachable points
    at any cumulative step index for constructing guaranteed-success /
    guaranteed-failure test fixtures, the same technique
    `test_midepisode_regoal.py` uses.
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


class TestEquivalenceWithMidepisodeRegoal:
    """N=2 waypoint-following must be numerically identical to `rollout_with_goal_switch`."""

    def test_n2_waypoints_matches_goal_switch_success_and_step_count(self) -> None:
        goal_a = np.array([1.3, 0.7, 0.5])
        goal_b = np.array([1.4, 0.8, 0.6])
        switch_step = 3
        max_steps = 6

        switch_env = gym.make(ENV_ID)
        switch_model = _RecordingStubModel(switch_env)
        switch_result = rollout_with_goal_switch(
            switch_model,
            switch_env,
            initial_goal_xyz=goal_a,
            switch_step=switch_step,
            new_goal_xyz=goal_b,
            max_steps=max_steps,
            base_seed=PROBE_SEED,
        )
        switch_env.close()

        waypoint_env = gym.make(ENV_ID)
        waypoint_model = _RecordingStubModel(waypoint_env)
        waypoint_result = rollout_with_waypoints(
            waypoint_model,
            waypoint_env,
            waypoints=[goal_a, goal_b],
            steps_per_leg=[switch_step, max_steps - switch_step],
            base_seed=PROBE_SEED,
        )
        waypoint_env.close()

        assert waypoint_result.n_steps == switch_result.n_steps
        assert waypoint_result.per_waypoint_success[-1] == switch_result.success
        assert waypoint_result.leg_boundaries == (switch_step, max_steps)

    def test_n2_waypoints_matches_goal_switch_on_guaranteed_success_case(self) -> None:
        # Mirrors test_midepisode_regoal's guaranteed-success construction:
        # the second leg's goal is set to the exact achieved_goal the
        # zero-action stub will be at after the switch step, so success is
        # forced regardless of policy quality -- an independent check that
        # the equivalence holds on the success=True branch too, not just
        # step counts.
        switch_step = 2
        max_steps = 3

        probe_env = gym.make(ENV_ID)
        guaranteed_reachable_point = _achieved_goal_after_n_zero_steps(
            probe_env, seed=PROBE_SEED, n_steps=switch_step
        )
        probe_env.close()

        goal_a = np.array([1.3, 0.7, 0.5])

        switch_env = gym.make(ENV_ID)
        switch_model = _RecordingStubModel(switch_env)
        switch_result = rollout_with_goal_switch(
            switch_model,
            switch_env,
            initial_goal_xyz=goal_a,
            switch_step=switch_step,
            new_goal_xyz=guaranteed_reachable_point,
            max_steps=max_steps,
            base_seed=PROBE_SEED,
        )
        switch_env.close()

        waypoint_env = gym.make(ENV_ID)
        waypoint_model = _RecordingStubModel(waypoint_env)
        waypoint_result = rollout_with_waypoints(
            waypoint_model,
            waypoint_env,
            waypoints=[goal_a, guaranteed_reachable_point],
            steps_per_leg=[switch_step, max_steps - switch_step],
            base_seed=PROBE_SEED,
        )
        waypoint_env.close()

        assert switch_result.success is True
        assert waypoint_result.per_waypoint_success[-1] is True
        assert waypoint_result.n_steps == switch_result.n_steps


class TestRolloutWithWaypointsBasics:
    """Basic return-shape and single-value steps_per_leg behavior."""

    def test_returns_waypoint_result(self) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)

        result = rollout_with_waypoints(
            model,
            env,
            waypoints=[np.array([1.3, 0.7, 0.5]), np.array([1.4, 0.8, 0.6])],
            steps_per_leg=2,
            base_seed=PROBE_SEED,
        )

        assert isinstance(result, WaypointResult)
        env.close()

    def test_single_int_steps_per_leg_applies_to_every_leg(self) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)

        result = rollout_with_waypoints(
            model,
            env,
            waypoints=[np.array([1.3, 0.7, 0.5]), np.array([1.4, 0.8, 0.6]), np.array([1.35, 0.75, 0.55])],
            steps_per_leg=2,
            base_seed=PROBE_SEED,
        )

        assert result.leg_boundaries == (2, 4, 6)
        assert result.n_steps == 6
        env.close()

    def test_no_reset_happens_between_legs(self) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)
        reset_calls = 0
        original_reset = env.reset

        def counting_reset(*args: object, **kwargs: object) -> tuple[dict, dict]:
            nonlocal reset_calls
            reset_calls += 1
            return original_reset(*args, **kwargs)

        env.reset = counting_reset

        rollout_with_waypoints(
            model,
            env,
            waypoints=[np.array([1.3, 0.7, 0.5]), np.array([1.4, 0.8, 0.6]), np.array([1.35, 0.75, 0.55])],
            steps_per_leg=2,
            base_seed=PROBE_SEED,
        )

        assert reset_calls == 1
        env.close()

    def test_policy_sees_each_legs_goal_for_exactly_its_own_budget(self) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)
        goal_a = np.array([1.3, 0.7, 0.5])
        goal_b = np.array([1.4, 0.8, 0.6])
        goal_c = np.array([1.35, 0.75, 0.55])

        rollout_with_waypoints(
            model,
            env,
            waypoints=[goal_a, goal_b, goal_c],
            steps_per_leg=[1, 2, 3],
            base_seed=PROBE_SEED,
        )

        seen = model.desired_goals_seen
        assert len(seen) == 6
        assert np.allclose(seen[0], goal_a)
        for goal_seen in seen[1:3]:
            assert np.allclose(goal_seen, goal_b)
        for goal_seen in seen[3:6]:
            assert np.allclose(goal_seen, goal_c)
        env.close()


class TestChainLengths:
    """N=3, 4, 5 chains: per-leg success and cumulative leg_boundaries."""

    @pytest.mark.parametrize("n_waypoints", [3, 4, 5])
    def test_leg_boundaries_are_cumulative_step_counts(self, n_waypoints: int) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)
        waypoints = [np.array([1.3 + 0.01 * i, 0.7, 0.5]) for i in range(n_waypoints)]
        steps_per_leg = 2

        result = rollout_with_waypoints(
            model,
            env,
            waypoints=waypoints,
            steps_per_leg=steps_per_leg,
            base_seed=PROBE_SEED,
        )

        expected = tuple(steps_per_leg * (i + 1) for i in range(n_waypoints))
        assert result.leg_boundaries == expected
        assert result.n_steps == expected[-1]
        assert len(result.per_waypoint_success) == n_waypoints
        env.close()

    @pytest.mark.parametrize("n_waypoints", [3, 4, 5])
    def test_all_succeeded_is_exactly_all_of_per_waypoint_success(self, n_waypoints: int) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)
        # All unreachable (10m away) -- every leg fails, so all_succeeded must be False.
        waypoints = [np.array([11.3 + i, 10.7, 10.5]) for i in range(n_waypoints)]

        result = rollout_with_waypoints(
            model,
            env,
            waypoints=waypoints,
            steps_per_leg=2,
            base_seed=PROBE_SEED,
        )

        assert result.all_succeeded == all(result.per_waypoint_success)
        assert result.all_succeeded is False
        env.close()

    def test_all_succeeded_true_when_every_leg_is_guaranteed_reachable(self) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)
        steps_per_leg = 2
        n_waypoints = 3

        probe_env = gym.make(ENV_ID)
        reachable_points = [
            _achieved_goal_after_n_zero_steps(probe_env, seed=PROBE_SEED, n_steps=steps_per_leg * (i + 1))
            for i in range(n_waypoints)
        ]
        probe_env.close()

        result = rollout_with_waypoints(
            model,
            env,
            waypoints=reachable_points,
            steps_per_leg=steps_per_leg,
            base_seed=PROBE_SEED,
        )

        assert result.per_waypoint_success == (True, True, True)
        assert result.all_succeeded is True
        env.close()


class TestEarlierLegFailureBehavior:
    """Design decision: every remaining leg is still attempted after an earlier leg fails.

    Aborting early on the first failed leg would hide information about
    whether later legs are independently recoverable -- exactly the signal
    this stage exists to measure (does error compound over a chain, or does
    each leg get a fair independent shot). So a failed leg never short-
    circuits the rest of the rollout; only the env itself ending the episode
    (terminated/truncated) can stop later legs from running.
    """

    def test_leg_failure_does_not_abort_remaining_legs(self) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)
        steps_per_leg = 2

        probe_env = gym.make(ENV_ID)
        # Leg 0 target is far away (guaranteed failure). Leg 1 target is the
        # exact achieved_goal reached after both legs' worth of zero-action
        # steps (guaranteed success), which can only be recorded if leg 1
        # actually runs its full budget.
        far_point = _achieved_goal_after_n_zero_steps(probe_env, seed=PROBE_SEED, n_steps=steps_per_leg) + 10.0
        reachable_point = _achieved_goal_after_n_zero_steps(
            probe_env, seed=PROBE_SEED, n_steps=steps_per_leg * 2
        )
        probe_env.close()

        result = rollout_with_waypoints(
            model,
            env,
            waypoints=[far_point, reachable_point],
            steps_per_leg=steps_per_leg,
            base_seed=PROBE_SEED,
        )

        assert result.per_waypoint_success == (False, True)
        assert result.n_steps == steps_per_leg * 2
        assert result.leg_boundaries == (steps_per_leg, steps_per_leg * 2)
        env.close()

    def test_leg_success_does_not_leak_into_a_later_failing_leg(self) -> None:
        # The reverse direction: a successful leg 0 must not make leg 1
        # falsely read as successful too -- each leg's success is judged
        # only from its own steps.
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)
        steps_per_leg = 2

        probe_env = gym.make(ENV_ID)
        near_point = _achieved_goal_after_n_zero_steps(probe_env, seed=PROBE_SEED, n_steps=steps_per_leg)
        probe_env.close()
        far_point = near_point + 10.0

        result = rollout_with_waypoints(
            model,
            env,
            waypoints=[near_point, far_point],
            steps_per_leg=steps_per_leg,
            base_seed=PROBE_SEED,
        )

        assert result.per_waypoint_success == (True, False)
        env.close()


class TestValidation:
    """Misconfiguration guards, mirroring midepisode_regoal's fail-loudly style."""

    def test_rejects_empty_waypoints(self) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)

        with pytest.raises(ValueError, match="waypoints"):
            rollout_with_waypoints(
                model,
                env,
                waypoints=[],
                steps_per_leg=2,
                base_seed=PROBE_SEED,
            )
        env.close()

    def test_rejects_steps_per_leg_sequence_of_wrong_length(self) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)

        with pytest.raises(ValueError, match="steps_per_leg"):
            rollout_with_waypoints(
                model,
                env,
                waypoints=[np.array([1.3, 0.7, 0.5]), np.array([1.4, 0.8, 0.6])],
                steps_per_leg=[2, 2, 2],
                base_seed=PROBE_SEED,
            )
        env.close()

    def test_rejects_non_positive_leg_budget(self) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)

        with pytest.raises(ValueError, match="steps_per_leg"):
            rollout_with_waypoints(
                model,
                env,
                waypoints=[np.array([1.3, 0.7, 0.5]), np.array([1.4, 0.8, 0.6])],
                steps_per_leg=[2, 0],
                base_seed=PROBE_SEED,
            )
        env.close()

    def test_rejects_max_steps_below_total_leg_budget(self) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)

        with pytest.raises(ValueError, match="max_steps"):
            rollout_with_waypoints(
                model,
                env,
                waypoints=[np.array([1.3, 0.7, 0.5]), np.array([1.4, 0.8, 0.6])],
                steps_per_leg=[3, 3],
                base_seed=PROBE_SEED,
                max_steps=5,
            )
        env.close()

    def test_rejects_total_budget_exceeding_envs_registered_limit(self) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)

        with pytest.raises(ValueError, match="max_episode_steps"):
            rollout_with_waypoints(
                model,
                env,
                waypoints=[np.array([1.3, 0.7, 0.5]), np.array([1.4, 0.8, 0.6])],
                steps_per_leg=30,
                base_seed=PROBE_SEED,
            )
        env.close()
