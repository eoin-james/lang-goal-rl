"""Tests for relative_move: stage 8's relative-move capability.

Mirrors `test_midepisode_regoal.py`'s testing approach: the real
FetchReach-v4 env throughout (deterministic given a seed and a fixed action
sequence), a lightweight fixed-action stub instead of a trained checkpoint,
and the "replay identical seed + zero actions to precompute a real,
physically-reachable point" trick for guaranteed-success/failure cases.

`compute_relative_goal` and `clip_to_box` are tested against an INJECTED
synthetic direction dict (never the real `AXIS_DIRECTIONS`-derived
`DIRECTION_UNIT_VECTORS`) -- this math must hold regardless of which
real-world direction labels Stage 7's still-pending human sign-off
eventually confirms or corrects.
"""

from __future__ import annotations

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import pytest

from lang_goal_rl.goal_region_vocabulary import GoalBox
from lang_goal_rl.relative_move import (
    RelativeMoveResult,
    clip_to_box,
    compute_relative_goal,
    rollout_with_relative_move,
)

gym.register_envs(gymnasium_robotics)

ENV_ID = "FetchReach-v4"
PROBE_SEED = 42


def _zero_action(env: gym.Env) -> np.ndarray:
    return np.zeros(env.action_space.shape, dtype=env.action_space.dtype)


def _achieved_goal_after_n_zero_steps(env: gym.Env, *, seed: int, n_steps: int) -> np.ndarray:
    """Replay `n_steps` zero actions from a fresh reset and return the resulting `achieved_goal`.

    Identical technique to `test_midepisode_regoal.py`'s helper of the same
    name -- MuJoCo is deterministic given a seed and a fixed action
    sequence, so replaying the same seed + zero actions in the real test
    rollout reproduces this exact point.
    """
    obs, _info = env.reset(seed=seed)
    action = _zero_action(env)
    for _ in range(n_steps):
        obs, _reward, _terminated, _truncated, _info = env.step(action)
    return np.array(obs["achieved_goal"], copy=True)


class _RecordingStubModel:
    """Fixed-action stub that records every `obs["desired_goal"]` it's asked to predict on.

    Identical pattern to `test_midepisode_regoal.py`'s stub of the same
    name -- reused here rather than inventing a second flavor, per the
    project's existing test convention of each test file defining its own
    copy of this exact shape.
    """

    def __init__(self, env: gym.Env, *, action: np.ndarray | None = None) -> None:
        self.desired_goals_seen: list[np.ndarray] = []
        self._action = action if action is not None else _zero_action(env)

    def predict(self, obs: dict, *, deterministic: bool = True) -> tuple[np.ndarray, None]:
        del deterministic
        self.desired_goals_seen.append(np.array(obs["desired_goal"], copy=True))
        return self._action, None


class TestClipToBox:
    """clip_to_box: clamp a point into a box's axis-aligned bounds, per axis independently."""

    def test_point_inside_bounds_is_unchanged(self) -> None:
        box = GoalBox(axis_min=np.array([0.0, 0.0, 0.0]), axis_max=np.array([1.0, 1.0, 1.0]))
        point = np.array([0.5, 0.5, 0.5])

        result = clip_to_box(point, box)

        assert np.allclose(result, point)

    def test_point_outside_one_axis_is_clamped_on_just_that_axis(self) -> None:
        box = GoalBox(axis_min=np.array([0.0, 0.0, 0.0]), axis_max=np.array([1.0, 1.0, 1.0]))
        point = np.array([1.5, 0.5, 0.5])

        result = clip_to_box(point, box)

        assert np.allclose(result, np.array([1.0, 0.5, 0.5]))

    def test_points_outside_multiple_axes_are_clamped_independently_per_axis(self) -> None:
        box = GoalBox(axis_min=np.array([0.0, 0.0, 0.0]), axis_max=np.array([1.0, 1.0, 1.0]))
        point = np.array([-0.5, 1.5, 0.5])

        result = clip_to_box(point, box)

        assert np.allclose(result, np.array([0.0, 1.0, 0.5]))


class TestComputeRelativeGoal:
    """compute_relative_goal: current + distance * direction_vectors[direction], clipped.

    Uses a synthetic direction dict throughout -- never the real
    `AXIS_DIRECTIONS`-derived vocabulary -- so these tests hold regardless
    of what Stage 7's human sign-off eventually confirms.
    """

    def test_math_is_exactly_current_plus_distance_times_unit_vector_when_no_clipping(
        self,
    ) -> None:
        box = GoalBox(axis_min=np.array([-10.0, -10.0, -10.0]), axis_max=np.array([10.0, 10.0, 10.0]))
        directions = {"north": np.array([0.0, 1.0, 0.0])}
        current = np.array([1.0, 2.0, 3.0])

        result = compute_relative_goal(
            current, "north", 0.5, direction_vectors=directions, box=box
        )

        assert np.allclose(result, np.array([1.0, 2.5, 3.0]))

    def test_clipping_engages_when_the_result_would_exit_the_box(self) -> None:
        box = GoalBox(axis_min=np.array([0.0, 0.0, 0.0]), axis_max=np.array([2.0, 2.0, 2.0]))
        directions = {"north": np.array([0.0, 1.0, 0.0])}
        current = np.array([1.0, 1.0, 1.0])

        result = compute_relative_goal(
            current, "north", 5.0, direction_vectors=directions, box=box
        )

        assert np.allclose(result, np.array([1.0, 2.0, 1.0]))

    def test_unknown_direction_raises_value_error(self) -> None:
        directions = {"north": np.array([0.0, 1.0, 0.0])}

        with pytest.raises(ValueError, match="north"):
            compute_relative_goal(
                np.array([1.0, 1.0, 1.0]), "south", 1.0, direction_vectors=directions
            )


class TestRolloutWithRelativeMove:
    """rollout_with_relative_move: resolve the target from the achieved position at switch_step."""

    def test_returns_a_relative_move_result(self) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)
        directions = {"east": np.array([1.0, 0.0, 0.0])}

        result = rollout_with_relative_move(
            model,
            env,
            initial_goal_xyz=np.array([1.3, 0.7, 0.5]),
            switch_step=2,
            direction="east",
            distance_m=0.01,
            max_steps=4,
            base_seed=PROBE_SEED,
            direction_vectors=directions,
        )

        assert isinstance(result, RelativeMoveResult)
        env.close()

    def test_echoes_back_switch_step(self) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)
        directions = {"east": np.array([1.0, 0.0, 0.0])}

        result = rollout_with_relative_move(
            model,
            env,
            initial_goal_xyz=np.array([1.3, 0.7, 0.5]),
            switch_step=3,
            direction="east",
            distance_m=0.01,
            max_steps=6,
            base_seed=PROBE_SEED,
            direction_vectors=directions,
        )

        assert result.switch_step == 3
        env.close()

    def test_policy_sees_initial_goal_before_switch_step_and_resolved_target_after(
        self,
    ) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)
        initial_goal = np.array([1.3, 0.7, 0.5])
        directions = {"east": np.array([1.0, 0.0, 0.0])}

        result = rollout_with_relative_move(
            model,
            env,
            initial_goal_xyz=initial_goal,
            switch_step=3,
            direction="east",
            distance_m=0.01,
            max_steps=6,
            base_seed=PROBE_SEED,
            direction_vectors=directions,
        )

        seen = model.desired_goals_seen
        assert len(seen) == 6
        for goal_seen in seen[:3]:
            assert np.allclose(goal_seen, initial_goal)
        for goal_seen in seen[3:]:
            assert np.allclose(goal_seen, result.resolved_target_xyz)
        env.close()

    def test_resolved_target_uses_achieved_position_at_switch_step_not_initial_goal(
        self,
    ) -> None:
        # If the code wrongly used initial_goal_xyz (or the reset position)
        # as "current position" instead of the achieved_goal actually
        # observed at switch_step, this would fail: initial_goal_xyz is
        # picked far from where a zero-action policy actually ends up after
        # 2 steps, so the two candidate "current position" values disagree
        # enough to distinguish which one the code used.
        probe_env = gym.make(ENV_ID)
        achieved_at_switch = _achieved_goal_after_n_zero_steps(
            probe_env, seed=PROBE_SEED, n_steps=2
        )
        probe_env.close()

        initial_goal = np.array([1.19, 0.60, 0.68])
        assert not np.allclose(initial_goal, achieved_at_switch, atol=0.05), (
            "test fixture invalid: initial_goal_xyz must differ from the real "
            "achieved position at switch_step for this test to be meaningful"
        )

        box = GoalBox(axis_min=np.array([0.0, 0.0, 0.0]), axis_max=np.array([10.0, 10.0, 10.0]))
        directions = {"east": np.array([1.0, 0.0, 0.0])}
        distance_m = 0.1

        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)

        result = rollout_with_relative_move(
            model,
            env,
            initial_goal_xyz=initial_goal,
            switch_step=2,
            direction="east",
            distance_m=distance_m,
            max_steps=3,
            base_seed=PROBE_SEED,
            box=box,
            direction_vectors=directions,
        )
        env.close()

        expected_from_achieved = achieved_at_switch + distance_m * np.array([1.0, 0.0, 0.0])
        wrong_from_initial = initial_goal + distance_m * np.array([1.0, 0.0, 0.0])

        assert np.allclose(result.resolved_target_xyz, expected_from_achieved)
        assert not np.allclose(result.resolved_target_xyz, wrong_from_initial)

    def test_success_is_judged_against_the_resolved_clipped_target_not_the_raw_request(
        self,
    ) -> None:
        # A tiny box centered exactly on the achieved position at
        # switch_step, with a huge requested offset: the *raw* target is
        # 5m away (unreachable by any real policy in 1 step), but the
        # *resolved* (clipped) target sits only 0.02m from the achieved
        # position -- well under FetchReach's 0.05m distance_threshold.
        # Success here proves the code checks the resolved target, not the
        # raw one.
        probe_env = gym.make(ENV_ID)
        achieved_at_switch = _achieved_goal_after_n_zero_steps(
            probe_env, seed=PROBE_SEED, n_steps=2
        )
        probe_env.close()

        box = GoalBox(axis_min=achieved_at_switch - 0.02, axis_max=achieved_at_switch + 0.02)
        directions = {"east": np.array([1.0, 0.0, 0.0])}

        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)

        result = rollout_with_relative_move(
            model,
            env,
            initial_goal_xyz=np.array([1.3, 0.7, 0.5]),
            switch_step=2,
            direction="east",
            distance_m=5.0,
            max_steps=3,
            base_seed=PROBE_SEED,
            box=box,
            direction_vectors=directions,
        )
        env.close()

        assert result.was_clipped is True
        assert result.success is True
        assert np.allclose(result.resolved_target_xyz, achieved_at_switch + np.array([0.02, 0.0, 0.0]))

    def test_was_clipped_is_false_when_the_requested_point_stays_in_bounds(self) -> None:
        probe_env = gym.make(ENV_ID)
        achieved_at_switch = _achieved_goal_after_n_zero_steps(
            probe_env, seed=PROBE_SEED, n_steps=2
        )
        probe_env.close()

        box = GoalBox(
            axis_min=achieved_at_switch - 1.0, axis_max=achieved_at_switch + 1.0
        )
        directions = {"east": np.array([1.0, 0.0, 0.0])}

        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)

        result = rollout_with_relative_move(
            model,
            env,
            initial_goal_xyz=np.array([1.3, 0.7, 0.5]),
            switch_step=2,
            direction="east",
            distance_m=0.01,
            max_steps=3,
            base_seed=PROBE_SEED,
            box=box,
            direction_vectors=directions,
        )
        env.close()

        assert result.was_clipped is False

    def test_success_false_for_a_resolved_target_far_from_reach(self) -> None:
        probe_env = gym.make(ENV_ID)
        achieved_at_switch = _achieved_goal_after_n_zero_steps(
            probe_env, seed=PROBE_SEED, n_steps=2
        )
        probe_env.close()

        box = GoalBox(
            axis_min=achieved_at_switch + 5.0, axis_max=achieved_at_switch + 6.0
        )
        directions = {"east": np.array([1.0, 0.0, 0.0])}

        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)

        result = rollout_with_relative_move(
            model,
            env,
            initial_goal_xyz=np.array([1.3, 0.7, 0.5]),
            switch_step=2,
            direction="east",
            distance_m=0.01,
            max_steps=3,
            base_seed=PROBE_SEED,
            box=box,
            direction_vectors=directions,
        )
        env.close()

        assert result.success is False

    def test_rejects_unknown_direction(self) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)
        directions = {"east": np.array([1.0, 0.0, 0.0])}

        with pytest.raises(ValueError, match="south"):
            rollout_with_relative_move(
                model,
                env,
                initial_goal_xyz=np.array([1.3, 0.7, 0.5]),
                switch_step=2,
                direction="south",
                distance_m=0.01,
                max_steps=4,
                base_seed=PROBE_SEED,
                direction_vectors=directions,
            )
        env.close()

    def test_rejects_switch_step_of_zero(self) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)
        directions = {"east": np.array([1.0, 0.0, 0.0])}

        with pytest.raises(ValueError, match="switch_step"):
            rollout_with_relative_move(
                model,
                env,
                initial_goal_xyz=np.array([1.3, 0.7, 0.5]),
                switch_step=0,
                direction="east",
                distance_m=0.01,
                max_steps=4,
                base_seed=PROBE_SEED,
                direction_vectors=directions,
            )
        env.close()

    def test_rejects_switch_step_at_or_past_max_steps(self) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)
        directions = {"east": np.array([1.0, 0.0, 0.0])}

        with pytest.raises(ValueError, match="switch_step"):
            rollout_with_relative_move(
                model,
                env,
                initial_goal_xyz=np.array([1.3, 0.7, 0.5]),
                switch_step=4,
                direction="east",
                distance_m=0.01,
                max_steps=4,
                base_seed=PROBE_SEED,
                direction_vectors=directions,
            )
        env.close()

    def test_rejects_max_steps_exceeding_the_envs_registered_limit(self) -> None:
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)
        directions = {"east": np.array([1.0, 0.0, 0.0])}

        with pytest.raises(ValueError, match="max_episode_steps"):
            rollout_with_relative_move(
                model,
                env,
                initial_goal_xyz=np.array([1.3, 0.7, 0.5]),
                switch_step=2,
                direction="east",
                distance_m=0.01,
                max_steps=51,
                base_seed=PROBE_SEED,
                direction_vectors=directions,
            )
        env.close()

    def test_uses_production_direction_vectors_by_default(self) -> None:
        # Not a behavioral correctness check of the labels themselves (that
        # is Stage 7's still-pending human sign-off) -- just confirms the
        # default wiring reaches the production dict without a caller
        # having to pass one explicitly.
        env = gym.make(ENV_ID)
        model = _RecordingStubModel(env)

        result = rollout_with_relative_move(
            model,
            env,
            initial_goal_xyz=np.array([1.3, 0.7, 0.5]),
            switch_step=2,
            direction="reach forward",
            distance_m=0.01,
            max_steps=4,
            base_seed=PROBE_SEED,
        )

        assert isinstance(result, RelativeMoveResult)
        env.close()
