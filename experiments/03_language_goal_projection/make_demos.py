"""Generate the 3 stage-3 demo GIFs proving (or honestly disproving) the language-goal mechanism visually.

Uses `lang_goal_rl.episode_recording.record_episode` against
`checkpoints/seed_0.zip` (the same checkpoint every stage-3 attempt evaluates,
unchanged since attempt 1). Three clips, matching the task brief:

1. `01_baseline_success.gif` — literal-goal mode (no override). The env's own
   `desired_goal` drives the policy exactly as trained; expected to succeed
   near-certainly given the checkpoint's 1.000 literal eval success rate.
   Searches a bounded range of eval seeds (`BASELINE_EVAL_SEEDS`) and, among
   every real success found, keeps the one with the largest
   `EpisodeRecording.total_travel` (summed per-step gripper displacement) --
   see the seed-selection fix note below for why "first success" alone isn't
   good enough here.
2. `02_broken_language_failure.gif` — attempt 1's broken projection
   (`artifacts/language_goal_projection.pt`, the pre-scale-fix InfoNCE
   checkpoint) overriding the desired-goal embedding for one fixed
   instruction ("reach up high"). Not cherry-picked for outcome — records
   whatever the first eval-seed episode actually does, since attempt 1's
   near-0% success rate means failure is the expected, honest outcome here.
   Left untouched by the travel-based selection fix below: this clip already
   shows real, sustained motion (the policy visibly reaching for the wrong
   place and never converging), and the whole point of this clip is the
   actual failure mode, not a best-case draw among successes -- there's
   nothing to select among since it's expected to fail.
3. `03_language_goal_result.gif` — same checkpoint and instruction, attempt
   3's fixed-centroid-regression projection
   (`artifacts/language_goal_projection_v3.pt`), now evaluated with
   attempt 4's corrected ground truth: `train.compute_region_centroid`'s
   fixed xyz centroid, reused every attempt, instead of a freshly resampled
   random in-region point. Attempt 3's own per-instruction eval for this
   exact instruction (against a resampled random point) scored 0.120 --
   attempt 4's eval-protocol fix (aligning ground truth with what the
   projected embedding actually represents) measured 1.000 across all 3
   seeds x 50 episodes for this and every other instruction. Searches a
   bounded range of seeds (`LANGUAGE_EVAL_SEEDS`) and picks the most
   travel-heavy real success among them, same as clip 1.

Every clip's `EpisodeRecording.success` (grounded in the env's real
`info["is_success"]`, per `episode_recording.py`) is the source of truth for
`demos/README.md`'s per-clip label — nothing here overrides that field.

Seed-selection fix (clips 1 and 3 only): both previously stopped at the
first eval seed that succeeded (seed 1000 for clip 1). FetchReach-v4 samples
goals only a few centimeters from a fixed reset pose, so "first success" is
often an almost-imperceptible nudge -- a real success, just visually boring.
Clip 1 also loads the *exact same checkpoint* as
`02_contrastive_goal_embedding/make_demo.py`'s literal-baseline clip, and
that script previously started its own search at the same seed (1000),
which is exactly why `01_baseline_success.gif` and
`05_stage2_embedding_baseline.gif` came out byte-identical. `BASELINE_EVAL_SEEDS`
is now a 15-seed range disjoint from that script's `EVAL_SEEDS` (2000-2014)
so the two selected episodes can never coincide, and the selection itself now
compares real successes by `EpisodeRecording.total_travel` rather than
taking the first one. This never fabricates or cherry-picks a non-success --
only real successes are compared -- and if no tried seed had meaningfully
more travel than the first success, that's reported honestly rather than
forced.
"""

from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from stable_baselines3 import SAC
from train import compute_region_centroid

from lang_goal_rl.episode_recording import EpisodeRecording, record_episode
from lang_goal_rl.goal_region_vocabulary import MEASURED_GOAL_BOX, sample_region_goals
from lang_goal_rl.language_embedding import encode_instructions
from lang_goal_rl.language_goal_projection import LanguageGoalProjection

if TYPE_CHECKING:
    from collections.abc import Iterator

EXPERIMENT_DIR = Path(__file__).parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
DEMOS_DIR = REPO_ROOT / "demos"

ENV_ID = "FetchReach-v4"
CHECKPOINT_PATH = EXPERIMENT_DIR / "checkpoints" / "seed_0.zip"
BROKEN_PROJECTION_PATH = EXPERIMENT_DIR / "artifacts" / "language_goal_projection.pt"
FIXED_PROJECTION_PATH = EXPERIMENT_DIR / "artifacts" / "language_goal_projection_v3.pt"
DEMO_INSTRUCTION = "reach up high"
DEMO_REGION = "reach up high"

BASELINE_EVAL_SEEDS = range(6000, 6015)
"""Bounded 15-seed search range for clip 1. Deliberately disjoint from
`02_contrastive_goal_embedding/make_demo.py`'s `EVAL_SEEDS` (2000-2014) even
though both scripts load the identical checkpoint in literal-goal mode --
that collision (both previously starting their search at seed 1000) is what
produced the byte-identical `01_baseline_success.gif` /
`05_stage2_embedding_baseline.gif` pair this fix resolves. Also disjoint from
stage 1's own `EVAL_SEEDS` (1000-1014, a different checkpoint but kept
distinct per the task's dedup requirement) and from `LANGUAGE_EVAL_SEEDS`
below."""

LANGUAGE_EVAL_SEEDS = range(9000, 9015)
"""A seed range distinct from every stage-3 eval protocol's base seeds
(`LITERAL_EVAL_BASE_SEED=1000`, `LANGUAGE_EVAL_BASE_SEED=5000`) and from
`BASELINE_EVAL_SEEDS` above, widened from 10 to 15 seeds to give clip 3's
travel-based search a bounded but meaningful pool of real successes to
choose from."""

MEANINGFUL_IMPROVEMENT_RATIO = 1.2
"""The selected episode's `total_travel` must be at least this multiple of
the first success's to count as "meaningfully more dynamic" -- below this,
the search result is reported honestly as a non-improvement rather than
silently presented as a win."""


@contextmanager
def _pin_ground_truth_goal(env: gym.Env, target: np.ndarray) -> Iterator[None]:
    """Temporarily monkeypatch `env.reset` to force the env's ground-truth goal after every reset.

    `record_episode` (`episode_recording.py`) calls `env.reset()` internally
    with no seed and no goal argument -- any ground-truth goal set on `env`
    *before* calling `record_episode` gets silently discarded the moment its
    internal reset runs, since `MujocoFetchEnv.reset` always draws a fresh
    `self.goal` via its own `_sample_goal()`. This wrapper is the fix: it
    lets the real reset run first (for a normal, valid initial observation),
    then overwrites `env.unwrapped.goal` and the returned `obs["desired_goal"]`
    to `target` before handing control back to the caller -- the same
    ground-truth substitution `train.py`'s `evaluate_language_goal` performs
    directly (it controls its own reset call, so it doesn't need this
    wrapper), just applied around a caller that doesn't expose that hook.

    Only `env.unwrapped.goal` is actually load-bearing for success/reward
    (`MujocoRobotEnv.step` reads `self.goal` directly, confirmed in
    `train.py`'s `evaluate_language_goal` docstring) -- `obs["desired_goal"]`
    is overwritten too only for consistency with that existing protocol, not
    because anything downstream reads it (the policy's desired-goal input is
    fully replaced by `goal_embedding_override`, never `obs["desired_goal"]`).

    Args:
        env: The env instance to patch.
        target: The xyz ground-truth goal to force after every reset.

    Yields:
        Nothing; the patch is active for the duration of the `with` block
        and unconditionally restored on exit.

    """
    original_reset = env.reset
    fixed_target = target.copy()

    def patched_reset(*args, **kwargs):
        obs, info = original_reset(*args, **kwargs)
        env.unwrapped.goal = fixed_target.copy()
        obs["desired_goal"] = fixed_target.copy()
        return obs, info

    env.reset = patched_reset
    try:
        yield
    finally:
        env.reset = original_reset


def load_projection(path: Path) -> LanguageGoalProjection:
    """Load a `LanguageGoalProjection` checkpoint saved by `train_projection.py`."""
    checkpoint = torch.load(path, map_location="cpu")
    projection = LanguageGoalProjection(
        input_dim=checkpoint["input_dim"], embed_dim=checkpoint["embed_dim"]
    )
    projection.load_state_dict(checkpoint["state_dict"])
    projection.eval()
    return projection


def projected_embedding_for(projection_path: Path, instruction: str) -> torch.Tensor:
    """Project one instruction through a saved `LanguageGoalProjection` checkpoint."""
    projection = load_projection(projection_path)
    sentence_embedding = torch.from_numpy(encode_instructions([instruction]))
    with torch.no_grad():
        return projection(sentence_embedding).squeeze(0)


def record_literal_baseline(env: gym.Env, model: SAC, out_path: Path) -> None:
    """Record clip 1: search `BASELINE_EVAL_SEEDS`, keep the most travel-heavy real success.

    See the module docstring's "Seed-selection fix" section for why this
    searches a bounded range and compares `EpisodeRecording.total_travel`
    rather than stopping at the first seed that succeeds.
    """
    successes: list[tuple[int, EpisodeRecording, Path]] = []
    first_success: tuple[int, EpisodeRecording] | None = None
    last_attempt: tuple[int, EpisodeRecording, Path] | None = None

    with tempfile.TemporaryDirectory(
        prefix="stage3-baseline-candidates-"
    ) as scratch_dir:
        scratch_root = Path(scratch_dir)
        for seed in BASELINE_EVAL_SEEDS:
            candidate_path = scratch_root / f"seed_{seed}.gif"
            env.reset(seed=seed)
            result = record_episode(env, model, out_path=candidate_path, max_steps=50)
            print(
                f"[baseline] seed={seed} success={result.success} n_steps={result.n_steps} "
                f"total_travel={result.total_travel:.4f}"
            )
            last_attempt = (seed, result, candidate_path)
            if result.success:
                successes.append((seed, result, candidate_path))
                if first_success is None:
                    first_success = (seed, result)

        if not successes:
            print(
                "[baseline] WARNING: no success found across tried seeds -- keeping the last recording, labeled honestly"
            )
            assert last_attempt is not None  # BASELINE_EVAL_SEEDS is non-empty
            _, _, last_path = last_attempt
            shutil.copyfile(last_path, out_path)
            return

        best_seed, best_result, best_path = max(
            successes, key=lambda item: item[1].total_travel
        )
        shutil.copyfile(best_path, out_path)

    assert first_success is not None
    first_seed, first_result = first_success
    print(
        f"[baseline] first success: seed={first_seed} total_travel={first_result.total_travel:.4f} | "
        f"selected: seed={best_seed} total_travel={best_result.total_travel:.4f} "
        f"({len(successes)}/{len(BASELINE_EVAL_SEEDS)} seeds tried succeeded)"
    )
    if first_result.total_travel > 0 and (
        best_result.total_travel
        < first_result.total_travel * MEANINGFUL_IMPROVEMENT_RATIO
    ):
        print(
            "[baseline] NOTE: no tried seed had meaningfully more travel than the first success -- "
            "the selected clip is a real success but not a clearly more dynamic one."
        )


def record_language_override(
    env: gym.Env,
    model: SAC,
    *,
    override_embedding: torch.Tensor,
    label: str,
    out_path: Path,
    max_seeds_tried: int = 1,
    fixed_target: np.ndarray | None = None,
) -> tuple[bool, int, int]:
    """Record episode(s) under a fixed goal-embedding override, ground truth from a region target.

    Tries up to `max_seeds_tried` distinct seeds from `LANGUAGE_EVAL_SEEDS`.
    With `max_seeds_tried=1` (attempt 1's broken projection, expected ~0%
    success) this records exactly one episode with no cherry-picking, per
    the task brief -- there is only ever one candidate, so there is nothing
    to select among.

    With `max_seeds_tried > 1` (attempt 4's fixed projection), every
    candidate seed is tried (not just until the first success), every real
    success's recording is kept, and the one with the largest
    `EpisodeRecording.total_travel` is written to `out_path` -- the same
    travel-based selection `record_literal_baseline` uses, for the same
    reason (a "first success" is often a visually boring near-imperceptible
    nudge). This never fabricates or cherry-picks a non-success: only
    episodes that actually succeeded are compared.

    Args:
        env: The FetchReach-v4 env instance.
        model: The trained SAC model whose actor's features extractor gets patched.
        override_embedding: Fixed desired-goal embedding to substitute (shape (embed_dim,)).
        label: Short tag for log lines (e.g. "broken-v1" or "fixed-v3").
        out_path: Where to write the GIF.
        max_seeds_tried: Number of distinct seeds to try. `1` disables
            selection entirely (see above); `>1` searches for the most
            travel-heavy real success among all of them.
        fixed_target: If given, this exact xyz point is used as ground truth
            for every attempt -- attempt 4's corrected protocol
            (`train.compute_region_centroid`), matching what the fixed
            `override_embedding` actually represents. If `None` (attempts
            1-3's original, since-diagnosed-as-broken protocol, kept only for
            clip 2's historical illustration of the pre-fix failure mode), a
            *fresh* random in-region point is resampled from
            `sample_region_goals` on every attempt instead.

    Returns:
        A tuple `(selected_success, n_successes, n_tried)`: whether the
        GIF written to `out_path` is a real success, how many of the tried
        episodes succeeded in total, and how many were tried.

    """
    successes: list[tuple[int, EpisodeRecording, Path]] = []
    first_success: tuple[int, EpisodeRecording] | None = None
    last_attempt: tuple[int, EpisodeRecording, Path] | None = None
    n_tried = 0

    with tempfile.TemporaryDirectory(
        prefix=f"stage3-{label}-candidates-"
    ) as scratch_dir:
        scratch_root = Path(scratch_dir)
        for seed in LANGUAGE_EVAL_SEEDS:
            if n_tried >= max_seeds_tried:
                break
            target = (
                fixed_target
                if fixed_target is not None
                else sample_region_goals(
                    DEMO_REGION, 1, seed=seed, box=MEASURED_GOAL_BOX
                )[0]
            )
            candidate_path = scratch_root / f"seed_{seed}.gif"
            env.reset(seed=seed)

            with _pin_ground_truth_goal(env, target):
                result = record_episode(
                    env,
                    model,
                    out_path=candidate_path,
                    goal_embedding_override=override_embedding,
                    max_steps=50,
                )
            n_tried += 1
            last_attempt = (seed, result, candidate_path)
            print(
                f"[{label}] instruction={DEMO_INSTRUCTION!r} seed={seed} success={result.success} "
                f"n_steps={result.n_steps} total_travel={result.total_travel:.4f} ({n_tried}/{max_seeds_tried} tried)",
            )
            if result.success:
                successes.append((seed, result, candidate_path))
                if first_success is None:
                    first_success = (seed, result)

        if not successes:
            print(
                f"[{label}] summary: 0/{n_tried} tried episodes succeeded; "
                "keeping the last recording, labeled honestly"
            )
            assert (
                last_attempt is not None
            )  # max_seeds_tried >= 1 guarantees at least one attempt
            _, last_result, last_path = last_attempt
            shutil.copyfile(last_path, out_path)
            return last_result.success, 0, n_tried

        best_seed, best_result, best_path = max(
            successes, key=lambda item: item[1].total_travel
        )
        shutil.copyfile(best_path, out_path)

    assert first_success is not None
    first_seed, first_result = first_success
    print(
        f"[{label}] summary: {len(successes)}/{n_tried} tried episodes succeeded -- "
        f"first success: seed={first_seed} total_travel={first_result.total_travel:.4f} | "
        f"selected: seed={best_seed} total_travel={best_result.total_travel:.4f}"
    )
    if first_result.total_travel > 0 and (
        best_result.total_travel
        < first_result.total_travel * MEANINGFUL_IMPROVEMENT_RATIO
    ):
        print(
            f"[{label}] NOTE: no tried seed had meaningfully more travel than the first success -- "
            "the selected clip is a real success but not a clearly more dynamic one."
        )
    return best_result.success, len(successes), n_tried


def main() -> None:
    """Generate all 3 demo GIFs into `demos/` and print each clip's real success/failure."""
    DEMOS_DIR.mkdir(parents=True, exist_ok=True)
    gym.register_envs(gymnasium_robotics)
    env = gym.make(ENV_ID, render_mode="rgb_array")
    model = SAC.load(CHECKPOINT_PATH, env=env)
    print(f"loaded checkpoint from {CHECKPOINT_PATH} (no training, eval-only)")

    record_literal_baseline(env, model, DEMOS_DIR / "01_baseline_success.gif")

    broken_embedding = projected_embedding_for(BROKEN_PROJECTION_PATH, DEMO_INSTRUCTION)
    record_language_override(
        env,
        model,
        override_embedding=broken_embedding,
        label="broken-v1",
        out_path=DEMOS_DIR / "02_broken_language_failure.gif",
        max_seeds_tried=1,
    )

    fixed_embedding = projected_embedding_for(FIXED_PROJECTION_PATH, DEMO_INSTRUCTION)
    fixed_centroid_target = compute_region_centroid(DEMO_REGION)
    record_language_override(
        env,
        model,
        override_embedding=fixed_embedding,
        label="fixed-v3-attempt4",
        out_path=DEMOS_DIR / "03_language_goal_result.gif",
        max_seeds_tried=12,
        fixed_target=fixed_centroid_target,
    )

    env.close()


if __name__ == "__main__":
    main()
