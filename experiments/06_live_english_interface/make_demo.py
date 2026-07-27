"""Generate the stage-6 capstone demo GIF: type an instruction, the robot goes; type a
different one partway through, it redirects live -- no reset, no retraining.

This is the single clip meant to make the whole project's thesis obvious on sight. Every
other `make_demo.py` in this repo demonstrates one mechanism at a time (literal goals,
learned embeddings, open vocabulary, mid-episode switching); this one wires all of them
together exactly as `live_regoal_eval.py`'s `run_switch_suite` already does for its
measured Set A/Set B numbers, but for one hand-picked, maximally legible instruction pair
instead of the full paired sweep:

- `LiveGoalController.instruction_to_goal_embedding` (`live_goal_controller.py`) turns each
  English sentence into a live 16-dim goal embedding, exactly as `live_regoal_eval.py` does
  for its Set A/Set B evaluations -- same frozen encoder, same k=1 nearest-neighbor lookup
  over the 84-sentence combined reference.
- `train.compute_region_centroid` (`03_language_goal_projection/train.py`) supplies each
  instruction's ground-truth xyz, the same region-centroid convention every stage since
  stage 3 has used -- never a freshly resampled point.
- `episode_recording.record_episode_with_goal_switch`'s embedding-override mode (both
  `initial_goal_embedding`/`new_goal_embedding` given) pins the policy's desired-goal input
  to each instruction's live embedding per phase, the same monkeypatch
  `rollout_with_goal_switch_timed` uses -- ground truth for success is still the env's real
  `desired_goal`, written from the two centroids above.

Instruction pair: both drawn from `set_b_vocabulary.SET_B_INSTRUCTIONS` -- the 7 phrasings
stage 6 wrote fresh and verified disjoint from every vocabulary this project has ever used
(training, augmented-training, held-out, compositional) -- rather than reusing one of Set
A's 14 already-measured paraphrases, since Set B is this project's strongest "someone just
typed something ad-hoc" story. "reach up high" and "reach down low" are picked over every
other region pair: they're visually opposite ends of FetchReach's workspace (straight up
vs. straight down), so a viewer can tell the redirect happened without reading anything.
Both instructions also individually measured a clean 1.0 no-switch success rate for this
checkpoint in stage 6's own `runs/seed_0/results.json` no-switch control -- not cherry-
picked after the fact, checked before writing this script, exactly the kind of due
diligence `run_switch_suite` itself can't do per-pair since it sweeps all 7 regions at
once.

`switch_step=20`/`max_steps=50` match stage 6's own `live_regoal_eval.py` constants (`SWITCH_STEP`,
`MAX_STEPS`) -- the same budget every Set A/Set B switch episode in that experiment used.
`checkpoints/seed_0.zip` is stage 3's checkpoint (same one every stage 3-6 script reuses),
unlike stage 5's demo which had to fall back to stage 1's checkpoint. Stage 6's own
`report.md` documents no per-seed exclusion (unlike stage 1/5's seeds 2/7) -- all 3 seeds are
equally valid; seed 0 is used here only because every other stage-3-based demo in this
project already does.

Up to 3 attempts are tried, varying only the episode reset seed (not the instruction pair or
checkpoint), matching this project's other demo scripts' retry discipline: an honest report
of how many tries it took, not a silently discarded failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC

EXPERIMENT_DIR = Path(__file__).parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
STAGE3_DIR = EXPERIMENT_DIR.parent / "03_language_goal_projection"
sys.path.insert(0, str(STAGE3_DIR))

from set_b_vocabulary import SET_B_INSTRUCTIONS
from train import (
    DEFAULT_ENCODER_PATH,
    ENV_ID,
    compute_region_centroid,
    load_frozen_encoder,
)

from lang_goal_rl.episode_recording import record_episode_with_goal_switch
from lang_goal_rl.live_goal_controller import LiveGoalController

DEMOS_DIR = REPO_ROOT / "demos"
CHECKPOINT_PATH = STAGE3_DIR / "checkpoints" / "seed_0.zip"
OUT_PATH = DEMOS_DIR / "08_stage6_live_english_capstone.gif"

_SET_B_TEXT_BY_REGION = {
    instruction.region_name: instruction.text for instruction in SET_B_INSTRUCTIONS
}

INITIAL_REGION = "reach up high"
NEW_REGION = "reach down low"
INITIAL_INSTRUCTION = _SET_B_TEXT_BY_REGION[INITIAL_REGION]
NEW_INSTRUCTION = _SET_B_TEXT_BY_REGION[NEW_REGION]
"""Both instructions are Set B's brand-new phrasings for the two most visually opposite
regions in the workspace -- see the module docstring for why this pair, not one of the 7*6
other possible pairings."""

SWITCH_STEP = 20
MAX_STEPS = 50
"""Matches `live_regoal_eval.py`'s `SWITCH_STEP`/`MAX_STEPS` -- the same budget stage 6's
own Set A/Set B switch episodes used."""

SWITCH_BASE_SEED = 70_000
"""A fresh seed range, offset from every prior stage's demo/eval seed ranges (stage 5's
demo: 9000-9002; `live_regoal_eval.py`: 30000-60000) so this capstone clip's episode draws
never silently collide with an existing measurement's seed for a different purpose."""

ATTEMPT_SEEDS = [SWITCH_BASE_SEED + episode_index for episode_index in range(3)]
"""At most 3 attempts, varying only the episode reset seed -- per this project's honesty-
over-luck retry cap, matching stage 5's own `make_demo.py`."""


def main() -> None:
    """Record the capstone live-English mid-episode redirect demo and print the real, measured outcome."""
    DEMOS_DIR.mkdir(parents=True, exist_ok=True)
    gym.register_envs(gymnasium_robotics)
    env = gym.make(ENV_ID, render_mode="rgb_array")
    model = SAC.load(CHECKPOINT_PATH, env=env)
    print(f"loaded checkpoint from {CHECKPOINT_PATH} (no training, eval-only)")

    encoder = load_frozen_encoder(DEFAULT_ENCODER_PATH)
    controller = LiveGoalController(encoder)

    initial_embedding = controller.instruction_to_goal_embedding(INITIAL_INSTRUCTION)
    new_embedding = controller.instruction_to_goal_embedding(NEW_INSTRUCTION)
    initial_goal = compute_region_centroid(INITIAL_REGION)
    new_goal = compute_region_centroid(NEW_REGION)

    for attempt, seed in enumerate(ATTEMPT_SEEDS, start=1):
        env.reset(seed=seed)
        result = record_episode_with_goal_switch(
            env,
            model,
            out_path=OUT_PATH,
            initial_goal_xyz=initial_goal,
            switch_step=SWITCH_STEP,
            new_goal_xyz=new_goal,
            initial_goal_embedding=initial_embedding,
            new_goal_embedding=new_embedding,
            max_steps=MAX_STEPS,
        )
        print(
            f"[stage6-capstone] attempt={attempt} seed={seed} "
            f'initial_instruction="{INITIAL_INSTRUCTION}" ({INITIAL_REGION}) '
            f'new_instruction="{NEW_INSTRUCTION}" ({NEW_REGION}) '
            f"switch_step={SWITCH_STEP} success={result.success} n_steps={result.n_steps}"
        )
        if result.success:
            break
    else:
        print(
            "[stage6-capstone] WARNING: no success found across tried seeds -- keeping the last recording, labeled honestly"
        )

    env.close()


if __name__ == "__main__":
    main()
