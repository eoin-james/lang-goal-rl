"""Generate the stage-1 demo GIF: literal xyz goal, no language involved at all.

Uses `lang_goal_rl.episode_recording.record_episode` against
`checkpoints/seed_0.zip` -- one of this stage's 8 healthy seeds (seeds 2 and
7 are excluded: `report.md`'s Pass 2 reviewer verdict documents both as
showing the SAC deterministic-eval-collapse signature, a known algorithm-
level fragility unrelated to goal-conditioning itself).

Literal-goal mode means `record_episode` is called with no
`goal_embedding_override` -- the policy sees exactly what `env.reset()`
samples as `desired_goal`, unmodified, matching this stage's own `evaluate()`
protocol in `train.py` (`obs, _info = env.reset(seed=1000 + episode)`,
deterministic actions). Ground truth is whatever `info["is_success"]`
reports; nothing here touches the env's goal.

Seed-selection fix (this script previously stopped at the first seed that
succeeded, e.g. seed 1000): FetchReach-v4 samples goals only a few
centimeters from a fixed reset pose, so "first success" is very often an
almost-imperceptible nudge -- a real success, just visually boring on a wide
camera framing the whole robot+table. Instead, this script tries every seed
in a small, bounded range (`EVAL_SEEDS`, matching `evaluate()`'s held-out
range), records each real success, and keeps the one with the largest
`EpisodeRecording.total_travel` (summed per-step displacement of the
gripper's `achieved_goal`, not just start-to-end distance -- see
`episode_recording.EpisodeRecording`'s docstring). This never fabricates or
cherry-picks a non-success: only real successes are compared, and if no
tried seed had meaningfully more travel than the first success, that's
reported honestly rather than forced.

This script's seed range (1000-1014) is deliberately distinct from
`02_contrastive_goal_embedding/make_demo.py`'s range (2000-2014) even though
the two scripts load different checkpoints (this stage's own seed_0 vs.
stage 3's), so the two stages' demo episodes can never coincide even by
accident.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC

from lang_goal_rl.episode_recording import EpisodeRecording, record_episode

EXPERIMENT_DIR = Path(__file__).parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
DEMOS_DIR = REPO_ROOT / "demos"

ENV_ID = "FetchReach-v4"
CHECKPOINT_PATH = EXPERIMENT_DIR / "checkpoints" / "seed_0.zip"
OUT_PATH = DEMOS_DIR / "04_stage1_literal_baseline.gif"

EVAL_SEEDS = range(1000, 1015)
"""Matches `train.py`'s `evaluate()` held-out eval seed range (`1000 +
episode`), widened from the original single-attempt range to a bounded
15-seed search for the most visually dynamic real success. Distinct from
stage 2's `EVAL_SEEDS` (2000-2014) even though the two stages don't share a
checkpoint, per the task's dedup requirement."""

MEANINGFUL_IMPROVEMENT_RATIO = 1.2
"""The selected episode's `total_travel` must be at least this multiple of
the first success's to count as "meaningfully more dynamic" -- below this,
the search result is reported honestly as a non-improvement rather than
silently presented as a win."""


def main() -> None:
    """Search the eval-seed range for the most visually dynamic real success and print the outcome."""
    DEMOS_DIR.mkdir(parents=True, exist_ok=True)
    gym.register_envs(gymnasium_robotics)
    env = gym.make(ENV_ID, render_mode="rgb_array")
    model = SAC.load(CHECKPOINT_PATH, env=env)
    print(f"loaded checkpoint from {CHECKPOINT_PATH} (no training, eval-only)")

    successes: list[tuple[int, EpisodeRecording, Path]] = []
    first_success: tuple[int, EpisodeRecording] | None = None
    last_attempt: tuple[int, EpisodeRecording, Path] | None = None

    with tempfile.TemporaryDirectory(prefix="stage1-demo-candidates-") as scratch_dir:
        scratch_root = Path(scratch_dir)
        for seed in EVAL_SEEDS:
            candidate_path = scratch_root / f"seed_{seed}.gif"
            env.reset(seed=seed)
            result = record_episode(env, model, out_path=candidate_path, max_steps=50)
            print(
                f"[stage1] seed={seed} success={result.success} n_steps={result.n_steps} "
                f"total_travel={result.total_travel:.4f}"
            )
            last_attempt = (seed, result, candidate_path)
            if result.success:
                successes.append((seed, result, candidate_path))
                if first_success is None:
                    first_success = (seed, result)

        if not successes:
            print(
                "[stage1] WARNING: no success found across tried seeds -- keeping the last recording, labeled honestly"
            )
            assert last_attempt is not None  # EVAL_SEEDS is non-empty
            _, _, last_path = last_attempt
            shutil.copyfile(last_path, OUT_PATH)
            env.close()
            return

        best_seed, best_result, best_path = max(
            successes, key=lambda item: item[1].total_travel
        )
        shutil.copyfile(best_path, OUT_PATH)

    assert first_success is not None
    first_seed, first_result = first_success
    print(
        f"[stage1] first success: seed={first_seed} total_travel={first_result.total_travel:.4f} | "
        f"selected: seed={best_seed} total_travel={best_result.total_travel:.4f} "
        f"({len(successes)}/{len(EVAL_SEEDS)} seeds tried succeeded)"
    )
    if first_result.total_travel > 0 and (
        best_result.total_travel
        < first_result.total_travel * MEANINGFUL_IMPROVEMENT_RATIO
    ):
        print(
            "[stage1] NOTE: no tried seed had meaningfully more travel than the first success -- "
            "the selected clip is a real success but not a clearly more dynamic one."
        )

    env.close()


if __name__ == "__main__":
    main()
