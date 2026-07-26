"""Tests for episode_recording: real-render episode capture, encoded to GIF.

Uses the real FetchReach-v4 env and its real `render_mode="rgb_array"`
renderer throughout — the whole point of this module is proof that
rendering actually produces content, so the renderer itself is never
mocked. Only the *policy* (`model.predict`) is a lightweight stub where the
test doesn't care about actual learned behavior, e.g. frame bookkeeping —
the override tests still wire up a real `GoalEmbeddingExtractor` so the
monkeypatch mechanism under test is exercised for real.
"""

from __future__ import annotations

import gymnasium as gym
import gymnasium_robotics
import imageio.v2 as imageio
import numpy as np
import pytest
import torch

from lang_goal_rl.episode_recording import (
    EpisodeRecording,
    _pin_desired_goal_embedding,
    record_episode,
)
from lang_goal_rl.goal_embedding_extractor import GoalEmbeddingExtractor

gym.register_envs(gymnasium_robotics)

ENV_ID = "FetchReach-v4"


class _StubModel:
    """Minimal SB3-`predict`-compatible stub: random valid actions, no learned behavior.

    Used where a test only cares about `record_episode`'s frame/bookkeeping
    logic, not about actual policy behavior. `actor.features_extractor` is a
    real `GoalEmbeddingExtractor` so override-mode tests exercise the real
    monkeypatch target.
    """

    class _Actor:
        def __init__(self, features_extractor: GoalEmbeddingExtractor) -> None:
            self.features_extractor = features_extractor

    def __init__(self, env: gym.Env, *, embed_dim: int = 4) -> None:
        self.actor = self._Actor(
            GoalEmbeddingExtractor(env.observation_space, embed_dim=embed_dim)
        )
        self._action_space = env.action_space

    def predict(
        self, _obs: dict, *, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        del deterministic
        return self._action_space.sample(), None


class _NoRenderEnv:
    """Fake env whose `render()` returns `None`, to exercise the loud-failure path.

    Deliberately not a real env: this test is about `record_episode`'s own
    defensive check, not about whether MuJoCo rendering works (that's
    covered by the real-env tests below).
    """

    def reset(self, **_kwargs: object) -> tuple[dict, dict]:
        return {"observation": np.zeros(1)}, {}

    def render(self) -> None:
        return None


class TestRecordEpisode:
    """record_episode rolls out one real episode and encodes it as a GIF."""

    def test_returns_episode_recording_with_gif_written_to_out_path(
        self, tmp_path
    ) -> None:
        env = gym.make(ENV_ID, render_mode="rgb_array")
        model = _StubModel(env)
        out_path = tmp_path / "episode.gif"

        result = record_episode(env, model, out_path=out_path, max_steps=3)

        assert isinstance(result, EpisodeRecording)
        assert result.path == out_path
        assert out_path.exists()
        env.close()

    def test_frame_count_matches_n_steps_plus_initial_reset_frame(
        self, tmp_path
    ) -> None:
        env = gym.make(ENV_ID, render_mode="rgb_array")
        model = _StubModel(env)
        out_path = tmp_path / "episode.gif"

        result = record_episode(env, model, out_path=out_path, max_steps=3)

        frames = imageio.mimread(out_path)
        assert len(frames) == result.n_steps + 1
        env.close()

    def test_max_steps_caps_episode_length(self, tmp_path) -> None:
        env = gym.make(ENV_ID, render_mode="rgb_array")
        model = _StubModel(env)
        out_path = tmp_path / "episode.gif"

        result = record_episode(env, model, out_path=out_path, max_steps=3)

        assert result.n_steps == 3
        env.close()

    def test_recorded_frames_are_real_non_blank_renders(self, tmp_path) -> None:
        # The failure mode this guards against: headless rendering silently
        # returning a solid black/gray frame instead of erroring. A real
        # MuJoCo render of FetchReach has visible structure (robot arm,
        # table, background), so per-pixel variance across frames must be
        # well above zero.
        env = gym.make(ENV_ID, render_mode="rgb_array")
        model = _StubModel(env)
        out_path = tmp_path / "episode.gif"

        record_episode(env, model, out_path=out_path, max_steps=3)

        frames = np.stack(imageio.mimread(out_path))
        assert frames.std() > 5.0
        env.close()

    def test_success_field_reflects_real_info_is_success(self, tmp_path) -> None:
        env = gym.make(ENV_ID, render_mode="rgb_array")
        model = _StubModel(env)
        out_path = tmp_path / "episode.gif"

        result = record_episode(env, model, out_path=out_path, max_steps=3)

        assert isinstance(result.success, bool)
        env.close()

    def test_creates_missing_parent_directories(self, tmp_path) -> None:
        env = gym.make(ENV_ID, render_mode="rgb_array")
        model = _StubModel(env)
        out_path = tmp_path / "nested" / "dir" / "episode.gif"

        record_episode(env, model, out_path=out_path, max_steps=2)

        assert out_path.exists()
        env.close()

    def test_raises_runtime_error_when_render_returns_none(self, tmp_path) -> None:
        env = _NoRenderEnv()
        model = None  # never reached: the error fires right after reset
        out_path = tmp_path / "episode.gif"

        with pytest.raises(TypeError, match="render_mode"):
            record_episode(env, model, out_path=out_path, max_steps=1)

    def test_override_mode_runs_without_touching_env_ground_truth(
        self, tmp_path
    ) -> None:
        env = gym.make(ENV_ID, render_mode="rgb_array")
        model = _StubModel(env, embed_dim=4)
        out_path = tmp_path / "episode.gif"
        override = torch.randn(4)

        result = record_episode(
            env,
            model,
            out_path=out_path,
            goal_embedding_override=override,
            max_steps=3,
        )

        assert out_path.exists()
        assert result.n_steps == 3
        env.close()

    def test_override_mode_restores_original_forward_after_recording(
        self, tmp_path
    ) -> None:
        env = gym.make(ENV_ID, render_mode="rgb_array")
        model = _StubModel(env, embed_dim=4)
        extractor = model.actor.features_extractor
        original_forward = extractor.forward
        out_path = tmp_path / "episode.gif"

        record_episode(
            env,
            model,
            out_path=out_path,
            goal_embedding_override=torch.randn(4),
            max_steps=2,
        )

        assert extractor.forward == original_forward
        env.close()

    def test_literal_mode_never_touches_the_model_actor(self, tmp_path) -> None:
        # goal_embedding_override=None (default) must not require or patch
        # model.actor.features_extractor at all -- a model without an
        # `actor` attribute (unlike SAC) should still work.
        class _ActorlessStubModel:
            def __init__(self, env: gym.Env) -> None:
                self._action_space = env.action_space

            def predict(
                self, _obs: dict, *, deterministic: bool = True
            ) -> tuple[np.ndarray, None]:
                del deterministic
                return self._action_space.sample(), None

        env = gym.make(ENV_ID, render_mode="rgb_array")
        model = _ActorlessStubModel(env)
        out_path = tmp_path / "episode.gif"

        result = record_episode(env, model, out_path=out_path, max_steps=2)

        assert out_path.exists()
        assert result.n_steps == 2
        env.close()


class TestPinDesiredGoalEmbedding:
    """_pin_desired_goal_embedding: the reusable monkeypatch for a fixed desired-goal input."""

    def _observations(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        return {
            "observation": torch.rand(batch_size, 10, dtype=torch.float64),
            "achieved_goal": torch.rand(batch_size, 3, dtype=torch.float64),
            "desired_goal": torch.rand(batch_size, 3, dtype=torch.float64),
        }

    def _extractor(self, embed_dim: int = 4) -> GoalEmbeddingExtractor:
        from gymnasium import spaces

        obs_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    -np.inf, np.inf, shape=(10,), dtype=np.float64
                ),
                "achieved_goal": spaces.Box(
                    -np.inf, np.inf, shape=(3,), dtype=np.float64
                ),
                "desired_goal": spaces.Box(
                    -np.inf, np.inf, shape=(3,), dtype=np.float64
                ),
            },
        )
        return GoalEmbeddingExtractor(obs_space, embed_dim=embed_dim)

    def test_substitutes_the_desired_goal_slice_with_the_override(self) -> None:
        extractor = self._extractor(embed_dim=4)
        observations = self._observations()
        override = torch.randn(4)

        with _pin_desired_goal_embedding(extractor, override):
            output = extractor(observations)

        desired_slice = output[:, 10 + 4 :].to(torch.float32)
        assert torch.allclose(desired_slice, override.expand(1, -1))

    def test_leaves_the_achieved_goal_slice_computed_from_the_real_encoder(
        self,
    ) -> None:
        extractor = self._extractor(embed_dim=4)
        observations = self._observations()
        with torch.no_grad():
            expected_achieved = extractor.goal_encoder(observations["achieved_goal"])

        with _pin_desired_goal_embedding(extractor, torch.randn(4)):
            output = extractor(observations)

        achieved_slice = output[:, 10 : 10 + 4].to(torch.float32)
        assert torch.allclose(achieved_slice, expected_achieved)

    def test_restores_the_original_forward_after_the_context_exits(self) -> None:
        extractor = self._extractor(embed_dim=4)
        original_forward = extractor.forward

        with _pin_desired_goal_embedding(extractor, torch.randn(4)):
            pass

        assert extractor.forward == original_forward

    def test_restores_original_forward_even_if_the_body_raises(self) -> None:
        extractor = self._extractor(embed_dim=4)
        original_forward = extractor.forward

        with (
            pytest.raises(ValueError, match="boom"),
            _pin_desired_goal_embedding(extractor, torch.randn(4)),
        ):
            raise ValueError("boom")

        assert extractor.forward == original_forward

    def test_forward_no_longer_pinned_after_context_exits(self) -> None:
        extractor = self._extractor(embed_dim=4)
        with _pin_desired_goal_embedding(extractor, torch.randn(4)):
            pass

        observations_a = self._observations()
        observations_b = self._observations()
        output_a = extractor(observations_a)
        output_b = extractor(observations_b)
        desired_slice_a = output_a[:, 10 + 4 :]
        desired_slice_b = output_b[:, 10 + 4 :]
        # Different desired_goal inputs must produce different embeddings
        # once unpinned -- proving the override no longer applies.
        assert not torch.allclose(desired_slice_a, desired_slice_b)
