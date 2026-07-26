"""Fixed instruction vocabulary, grounded in FetchReach-v4's measured goal-sampling box.

Stage 3's proof gate requires a fixed set of English instructions that each
map to a distinct region of FetchReach's real `desired_goal` distribution —
not to invented coordinates. `measure_goal_box` resets the env many times and
records the actual per-axis min/max; `MEASURED_GOAL_BOX` freezes one such
measurement (`measure_goal_box(n_samples=2000, seed=0)`, run once via `uv run
python` and pasted in below) as the box every region/instruction in this
module is derived from.

Axis-to-direction convention: FetchReach's goal is a 3D xyz point with
x = depth (toward/away from the robot base), y = lateral (left/right),
z = height (up/down) — the standard Fetch robot frame used across
gymnasium-robotics' Fetch envs. The env doesn't label which sign is
"forward" or "left"; that mapping is this module's own labeling choice
(`AXIS_DIRECTIONS` below), not a measured fact.

Region partitioning: a point is classified by how far it deviates from the
box's centroid, normalized by the box's half-range per axis. If every axis's
normalized deviation is small (within `CENTER_THRESHOLD`), the point is
"center". Otherwise it's assigned to the single axis with the largest
normalized deviation, in that deviation's sign's direction. This yields 7
mutually-exclusive regions (1 center + 2 per axis x 3 axes) covering the
whole measured box — matching the roadmap's example region set ("reach
left/right/forward/up high/down low/center") while staying derived from
the measured box rather than hand-picked coordinate cutoffs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import numpy.typing as npt
import torch

from lang_goal_rl.goal_encoder import GoalEncoder

if TYPE_CHECKING:
    from collections.abc import Sequence

ENV_ID = "FetchReach-v4"


@dataclass(frozen=True)
class GoalBox:
    """Axis-aligned bounding box of a 3D goal-sampling distribution.

    Attributes:
        axis_min: Per-axis minimum, shape (3,).
        axis_max: Per-axis maximum, shape (3,).

    """

    axis_min: npt.NDArray[np.floating]
    axis_max: npt.NDArray[np.floating]

    @property
    def centroid(self) -> npt.NDArray[np.floating]:
        """Midpoint of the box on every axis."""
        return (self.axis_min + self.axis_max) / 2.0

    @property
    def half_range(self) -> npt.NDArray[np.floating]:
        """Half the box's span on every axis."""
        return (self.axis_max - self.axis_min) / 2.0


def measure_goal_box(n_samples: int = 1000, seed: int = 0) -> GoalBox:
    """Reset FetchReach-v4 many times and record `desired_goal`'s per-axis min/max.

    This is the grounding step: FetchReach's goal-sampling region is
    determined empirically here, not assumed. Resets use seeds
    `seed, seed + 1, ..., seed + n_samples - 1`, so a given `(n_samples,
    seed)` pair is fully reproducible and a smaller `n_samples` with the same
    `seed` draws a strict subset of a larger run's seeds.

    Args:
        n_samples: Number of resets to perform (1000+ recommended for a
            grounding measurement; smaller values are useful for fast tests
            that only check consistency with a larger frozen measurement).
        seed: Base seed for the reset sequence.

    Returns:
        A `GoalBox` covering every sampled `desired_goal`.

    """
    gym.register_envs(gymnasium_robotics)
    env = gym.make(ENV_ID)
    goals = np.empty((n_samples, 3), dtype=np.float64)
    for i in range(n_samples):
        obs, _info = env.reset(seed=seed + i)
        goals[i] = obs["desired_goal"]
    env.close()
    return GoalBox(axis_min=goals.min(axis=0), axis_max=goals.max(axis=0))


MEASURED_GOAL_BOX = GoalBox(
    axis_min=np.array([1.1919916950134846, 0.5991946315749674, 0.3848410890322601]),
    axis_max=np.array([1.4917979081817305, 0.8990967575021397, 0.6845422370972912]),
)
"""FetchReach-v4's real `desired_goal` sampling box, frozen from
`measure_goal_box(n_samples=2000, seed=0)`. Every region and instruction in
this module partitions this exact box. Re-measuring with the same
`(n_samples, seed)` reproduces it exactly (`measure_goal_box` is
deterministic); `test_goal_region_vocabulary.py` checks a smaller,
faster-to-run measurement stays within these bounds so drift would be
caught."""

CENTER_THRESHOLD = 0.35
"""Fraction of each axis's half-range within which a point's normalized
deviation from the centroid must fall, on *every* axis, to be classified
"center" rather than assigned to a directional region. 0.35 keeps "center"
a clearly-interior sub-region (roughly the middle 70% of each axis) while
leaving most of the box's volume to the 6 directional regions, which is
where a distinguishable instruction vocabulary needs the probability mass
for `sample_region_goals`'s rejection sampling to converge quickly."""

AXIS_DIRECTIONS: tuple[tuple[str, str], tuple[str, str], tuple[str, str]] = (
    ("reach back", "reach forward"),  # x axis: (negative, positive)
    ("reach right", "reach left"),  # y axis: (negative, positive)
    ("reach down low", "reach up high"),  # z axis: (negative, positive)
)
"""Region name for the (negative, positive) direction of each axis (x, y, z),
per this module's axis-to-direction convention documented at the top."""


@dataclass(frozen=True)
class GoalRegion:
    """One region of the goal box and its fixed instruction phrasings.

    Attributes:
        name: Region identifier, matching what `classify_region` returns.
        instructions: Fixed set of English phrasings for this region.

    """

    name: str
    instructions: tuple[str, ...]


REGIONS: tuple[GoalRegion, ...] = (
    GoalRegion(
        "center",
        ("move your hand to the center", "keep the gripper in the middle of the workspace"),
    ),
    GoalRegion("reach forward", ("move your hand forward", "reach out in front of you")),
    GoalRegion("reach back", ("pull your hand back", "reach backward toward yourself")),
    GoalRegion("reach left", ("move your hand to the left", "reach toward the left side")),
    GoalRegion("reach right", ("move your hand to the right", "reach toward the right side")),
    GoalRegion("reach up high", ("reach up high", "move your hand upward")),
    GoalRegion("reach down low", ("reach down low", "move your hand downward")),
)
"""The fixed, closed instruction vocabulary for stage 3 — 7 regions x
1-2 phrasings each (14 instructions total). Not an open-vocabulary set: this
is exactly the set the projection layer is trained and evaluated against."""

ALL_INSTRUCTIONS: tuple[str, ...] = tuple(
    instruction for region in REGIONS for instruction in region.instructions
)
"""Flat tuple of every fixed instruction across all regions, in `REGIONS` order."""

_INSTRUCTION_TO_REGION: dict[str, str] = {
    instruction: region.name for region in REGIONS for instruction in region.instructions
}


def region_names() -> tuple[str, ...]:
    """Return every defined region's name, in `REGIONS` order."""
    return tuple(region.name for region in REGIONS)


def classify_region(point: npt.NDArray[np.floating], box: GoalBox) -> str:
    """Classify an xyz point into one of the 7 fixed regions of `box`.

    Args:
        point: A single xyz point, shape (3,).
        box: The `GoalBox` whose centroid/half-range defines the partition.

    Returns:
        The matching region's name (see `REGIONS`).

    """
    deviation = (point - box.centroid) / box.half_range
    if np.all(np.abs(deviation) < CENTER_THRESHOLD):
        return "center"
    axis = int(np.argmax(np.abs(deviation)))
    direction_index = 1 if deviation[axis] > 0 else 0
    return AXIS_DIRECTIONS[axis][direction_index]


def instruction_to_region(instruction: str) -> str:
    """Look up the fixed region name an instruction belongs to.

    Args:
        instruction: One of the strings in `ALL_INSTRUCTIONS`.

    Returns:
        The instruction's region name.

    Raises:
        ValueError: If `instruction` is not in the fixed vocabulary.

    """
    if instruction not in _INSTRUCTION_TO_REGION:
        msg = f"{instruction!r} is not in the fixed instruction vocabulary (see ALL_INSTRUCTIONS)"
        raise ValueError(msg)
    return _INSTRUCTION_TO_REGION[instruction]


def sample_region_goals(
    region_name: str, n_samples: int, seed: int, box: GoalBox = MEASURED_GOAL_BOX,
) -> npt.NDArray[np.floating]:
    """Rejection-sample `n_samples` xyz points that classify into `region_name`.

    FetchReach's true goal distribution is uniform within its sampling box
    (see `pretrain_encoder.py`'s docstring in
    `experiments/02_contrastive_goal_embedding/`), so points are drawn
    uniformly from `box` and kept only if `classify_region` assigns them to
    `region_name` — a direct, unbiased sample from "the true goal
    distribution, conditioned on being in this region".

    Args:
        region_name: One of `region_names()`.
        n_samples: Number of points to return.
        seed: Seed for the rejection-sampling draws; deterministic for a
            given `(region_name, n_samples, seed, box)`.
        box: The box to sample within and classify against.

    Returns:
        Array of shape (n_samples, 3), every row classifying into `region_name`.

    Raises:
        RuntimeError: If far more draws than expected are needed (a defensive
            bound — in practice each region covers a large enough fraction
            of the box that this is never approached).

    """
    rng = np.random.default_rng(seed)
    collected: list[npt.NDArray[np.floating]] = []
    max_draws = max(n_samples * 200, 10_000)
    draws = 0
    while len(collected) < n_samples and draws < max_draws:
        batch = rng.uniform(box.axis_min, box.axis_max, size=(n_samples, 3))
        draws += n_samples
        for point in batch:
            if classify_region(point, box) == region_name:
                collected.append(point)
                if len(collected) == n_samples:
                    break
    if len(collected) < n_samples:
        msg = f"could not sample {n_samples} points for region {region_name!r} within {max_draws} draws"
        raise RuntimeError(msg)
    return np.array(collected)


def compute_region_target_embeddings(
    goal_encoder: GoalEncoder,
    names: Sequence[str],
    *,
    box: GoalBox = MEASURED_GOAL_BOX,
    n_samples: int = 200,
    seed: int = 0,
) -> torch.Tensor:
    """Compute each region's mean true goal-embedding under a frozen `GoalEncoder`.

    For every region name in `names`, samples `n_samples` xyz points from
    that region (via `sample_region_goals`) and returns the mean of their
    `goal_encoder` embeddings — an estimate of "where this region sits" in
    stage 2's embedding space. Used both to train the language projection
    layer (regression target) and to ground the collapse diagnostic's
    epsilon threshold in the real target space's scale.

    Args:
        goal_encoder: Stage 2's frozen encoder (embeddings computed under
            `torch.no_grad()`; this function never updates its weights).
        names: Region names to compute a target embedding for, in order.
        box: The goal box to sample within.
        n_samples: Number of xyz samples per region used to estimate the mean.
        seed: Base seed; region `i` in `names` samples with `seed + i`.

    Returns:
        Tensor of shape (len(names), goal_encoder.embed_dim).

    """
    rows = []
    with torch.no_grad():
        for i, name in enumerate(names):
            goals = sample_region_goals(name, n_samples, seed=seed + i, box=box)
            embeddings = goal_encoder(torch.from_numpy(goals).float())
            rows.append(embeddings.mean(dim=0))
    return torch.stack(rows)
