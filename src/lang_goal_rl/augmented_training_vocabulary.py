"""Stage 4's data-augmentation fix: 10 diverse phrasings per region instead of 2.

Stage 4 (open vocabulary) FAILED: `LanguageGoalProjection` (384->64->16,
~25,600 parameters) was trained via direct MSE regression on
`goal_region_vocabulary.ALL_INSTRUCTIONS` -- exactly 14 fixed sentences, 2
per region -- which is enough capacity to memorize those 14 points with zero
pressure to generalize between them. A zero-training nearest-neighbor
ceiling test (blend of the nearest training targets in raw 384-dim sentence-
embedding space, bypassing the MLP entirely) scored 0.714 accuracy on
`held_out_paraphrases`'s 14 held-out phrases vs. the trained MLP's 0.286,
proving the raw embedding space already clusters by region well and the MLP
was discarding that signal by memorizing instead of learning a generalizing
rule. See `ROADMAP.md`'s "Projection-layer overfitting to a minimal
vocabulary" entry for the full diagnosis.

This module is the reviewer-recommended fix: a larger, more diverse training
set per region so a 25,600-parameter network can't just memorize its way to
zero training loss. `AUGMENTED_REGIONS` reuses `goal_region_vocabulary`'s 7
region names and `GoalRegion` shape unchanged (`goal_region_vocabulary.REGIONS`
and `ALL_INSTRUCTIONS` are stage 3's already-Done fixed vocabulary and are
never modified here) but gives each region 10 phrasings -- varied in verb
choice and sentence structure (imperative vs. descriptive, different subjects
such as "your hand"/"your arm"/"the gripper"/"the end effector") rather than
minor rewordings of the original 2.

Every phrasing here is disjoint (exact and trivial-single-word-edit, see
`tests/lang_goal_rl/test_augmented_training_vocabulary.py`) from
`held_out_paraphrases.HELD_OUT_PARAPHRASES` and `.COMPOSITIONAL_INSTRUCTIONS`
-- stage 4's fixed test set, already used to measure the MLP's 0.286 baseline
and the nearest-neighbor ceiling's 0.714. Training on this set and
re-evaluating against that same fixed held-out set keeps the retest an
apples-to-apples comparison with those two already-measured numbers.
"""

from __future__ import annotations

from lang_goal_rl.goal_region_vocabulary import GoalRegion

AUGMENTED_REGIONS: tuple[GoalRegion, ...] = (
    GoalRegion(
        "center",
        (
            "hold your gripper steady at the center of the workspace",
            "position the end effector at the midpoint of the workspace",
            "keep your arm hovering near the workspace's centroid",
            "bring the hand to rest in the workspace middle",
            "the gripper should sit at the center of its reachable area",
            "stay balanced in the middle of the reachable space",
            "guide the end effector back to a neutral middle spot",
            "align your arm with the center point of the workspace",
            "maintain a resting position near the workspace's midpoint",
            "float your hand near the middle of the robot's reach",
        ),
    ),
    GoalRegion(
        "reach forward",
        (
            "guide your arm out toward the front of the workspace",
            "advance the gripper away from your body",
            "stretch your hand out ahead of you",
            "drive the end effector forward across the workspace",
            "extend your arm toward the front edge",
            "the gripper should glide forward away from the base",
            "press onward extending your reach in front",
            "angle your hand outward toward the front",
            "propel your arm forward away from the torso",
            "slide the gripper ahead into open space",
        ),
    ),
    GoalRegion(
        "reach back",
        (
            "withdraw your arm back toward your body",
            "retract the gripper toward the base",
            "bring your hand inward back toward yourself",
            "ease the end effector backward closer to home",
            "draw the arm rearward away from the front edge",
            "the gripper should glide back toward the torso",
            "tuck your hand in behind its current spot",
            "reel your arm back toward the robot's body",
            "pull the end effector rearward closer to the base",
            "slide backward away from the workspace's front edge",
        ),
    ),
    GoalRegion(
        "reach left",
        (
            "guide your hand toward the left boundary",
            "steer the gripper across the workspace to the left",
            "angle the end effector toward the left flank",
            "drift the hand leftward across the workspace",
            "the gripper should ease over to the left side",
            "veer your arm toward the left portion of the space",
            "nudge the hand across the workspace toward the left",
            "position the end effector on the left flank",
            "sweep your arm leftward across the reachable area",
            "lean the gripper toward the left-hand boundary",
        ),
    ),
    GoalRegion(
        "reach right",
        (
            "guide your hand toward the right boundary",
            "steer the gripper across the workspace to the right",
            "angle the end effector toward the right flank",
            "drift the hand rightward across the workspace",
            "the gripper should ease over to the right side",
            "veer your arm toward the right portion of the space",
            "nudge the hand across the workspace toward the right",
            "position the end effector on the right flank",
            "sweep your arm rightward across the reachable area",
            "lean the gripper toward the right-hand boundary",
        ),
    ),
    GoalRegion(
        "reach up high",
        (
            "extend your arm upward as far as it can go",
            "raise the gripper toward the top of the workspace",
            "lift your hand up high away from the table",
            "guide the end effector skyward",
            "elevate your arm toward the ceiling",
            "push upward stretching toward the top boundary",
            "hoist the gripper up above its resting height",
            "the hand should climb toward the workspace's upper limit",
            "angle your arm toward the topmost point",
            "ascend with the gripper toward the top of the space",
        ),
    ),
    GoalRegion(
        "reach down low",
        (
            "lower your arm downward as far as it can go",
            "drop the gripper toward the bottom of the workspace",
            "sink your hand down low toward the table",
            "guide the end effector earthward",
            "bring your arm down toward the floor level",
            "push downward stretching toward the bottom boundary",
            "dip the gripper down below its resting height",
            "the hand should sink toward the workspace's lower limit",
            "angle your arm toward the bottommost point",
            "guide the gripper toward the bottom of the space",
        ),
    ),
)
"""7 regions x 10 diverse phrasings each (70 total) -- the augmented training
vocabulary. Same 7 region names as `goal_region_vocabulary.region_names()`;
`goal_region_vocabulary.REGIONS`/`ALL_INSTRUCTIONS` are untouched."""

AUGMENTED_INSTRUCTIONS: tuple[str, ...] = tuple(
    instruction for region in AUGMENTED_REGIONS for instruction in region.instructions
)
"""Flat tuple of every augmented instruction across all regions, in
`AUGMENTED_REGIONS` order -- same pattern as
`goal_region_vocabulary.ALL_INSTRUCTIONS`."""

_AUGMENTED_INSTRUCTION_TO_REGION: dict[str, str] = {
    instruction: region.name for region in AUGMENTED_REGIONS for instruction in region.instructions
}


def augmented_instruction_to_region() -> dict[str, str]:
    """Return the full augmented-instruction -> region-name lookup.

    One entry per `AUGMENTED_INSTRUCTIONS` row. The experiment-runner's most
    common use is deriving a region-name-per-row list in
    `AUGMENTED_INSTRUCTIONS` order for
    `goal_region_vocabulary.compute_region_target_embeddings` /
    `language_goal_projection.precompute_instruction_targets`, e.g.:
    `[augmented_instruction_to_region()[i] for i in AUGMENTED_INSTRUCTIONS]`.

    Returns:
        Dict mapping every instruction in `AUGMENTED_INSTRUCTIONS` to its
        region name (one of `goal_region_vocabulary.region_names()`).

    """
    return dict(_AUGMENTED_INSTRUCTION_TO_REGION)
