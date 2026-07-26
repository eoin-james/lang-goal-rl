"""LanguageGoalProjection: frozen sentence embedding -> stage 2's goal-embedding space.

Stage 3's proof gate needs a sentence to land at a point in the *same*
16-dim space stage 2's `GoalEncoder` already defines (see
`goal_encoder.py`), so stage 2's trained policies stay chainable. This
module owns two things:

1. `LanguageGoalProjection` — the small learnable layer doing the mapping.
2. `train_projection` — the procedure that fits it, given a frozen
   `GoalEncoder` and a fixed instruction vocabulary's region assignments
   (see `goal_region_vocabulary.py`).

Loss choice, attempt 1/2 (superseded, kept here for context): a batch of
true xyz goals was sampled from each instruction's region
(`sample_region_goals`) and averaged through the frozen `GoalEncoder` to
get that region's mean embedding — but this was *resampled every training
step* from a small batch, a noisy per-step estimate. `contrastive.info_nce_loss`
was used on top of it (projected embedding as anchor, resampled region mean
as positive) to add an explicit push against every other instruction's
region embedding in the same batch, targeting "distinct instructions don't
collapse". Attempt 1 found `info_nce_loss` is scale-invariant
(`F.normalize()` on both inputs), so attempt 2 added an explicit
norm-matching MSE term (`combined_projection_loss`) alongside it. Attempt 2
fixed the scale defect (fail-fast norm check passed) but RL success rate
still only reached ~7% — see
`experiments/03_language_goal_projection/report.md`'s attempt 2 reviewer
verdict.

**Attempt 3 (current): direct regression to a fixed, precomputed centroid.**
The reviewer's diagnosis: with a closed, ~14-instruction fixed vocabulary,
there is no reason to keep resampling a noisy per-step target when the
target regions' *true* mean embeddings are a fixed, known quantity that can
be computed once. The earlier "MSE alone might let two nearby regions
collapse" concern (which motivated adding InfoNCE's separation term in the
first place) is moot here: collapse is a property of whether the true
region centroids under the frozen `GoalEncoder` are separated, which is a
structural fact about `GoalEncoder`'s pretraining, not something this
projection's loss needs to enforce — and it's already independently
confirmed true (24.68x margin over the collapse threshold, attempt 2's
`instruction_collapse_diagnostic` re-check). So the fix is: precompute each
instruction's region centroid once via `precompute_instruction_targets`
(large sample, no resampling), then regress the projection's output
straight to it with plain MSE (`regression_loss`) — no InfoNCE term, no
separate norm term. Matching the target vector exactly also matches its
norm (norm is a continuous function of the vector), so the norm-matching
term attempt 2 added is now redundant by construction; this is verified
empirically, not just asserted, by
`test_trained_projection_output_norms_track_target_norms_without_a_separate_norm_term`
in `tests/lang_goal_rl/test_language_goal_projection.py`. Whether directional
accuracy against the true centroid (see
`instruction_direction_diagnostic.py`) is actually what predicts RL success
-- or whether some other confound is at play -- is an open question this
module doesn't answer; it only makes fixed-target regression available so
that question can be investigated on a clean signal instead of a noisy one.

`check_projection_norm_range` (below) is kept as an offline post-hoc sanity
check: given a trained projection's outputs and the frozen encoder's
*measured* operating-range norms (via `measure_reference_norms`), it flags
any instruction whose output norm falls outside a tolerance band around the
reference mean. It is no longer load-bearing for correctness (regression to
the encoder's own embeddings makes an in-range norm essentially automatic),
but it is cheap and catches a training bug (e.g. a bad loss edit) before
spending an RL budget rediscovering it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as f  # noqa: N812 -- mirrors contrastive.py's common torch.nn.functional alias
from torch import nn

from lang_goal_rl.goal_encoder import DEFAULT_EMBED_DIM
from lang_goal_rl.goal_region_vocabulary import MEASURED_GOAL_BOX, compute_region_target_embeddings

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lang_goal_rl.goal_encoder import GoalEncoder
    from lang_goal_rl.goal_region_vocabulary import GoalBox

DEFAULT_INPUT_DIM = 384
"""Matches `language_embedding.LANGUAGE_EMBED_DIM` (`all-MiniLM-L6-v2`'s output size)."""

DEFAULT_N_TARGET_SAMPLES = 1000
"""Number of xyz samples averaged per region when precomputing each
instruction's fixed regression target (`precompute_instruction_targets`).
1000 (the reviewer's suggested floor) trades off a tighter estimate of the
region's true mean embedding against the one-time cost of this precompute
-- run once per training call, not per step, so a larger sample here is
cheap relative to attempt 2's old per-step resampling cost."""

DEFAULT_NORM_RANGE_TOLERANCE = 2.0
"""Default multiplicative tolerance for `check_projection_norm_range`: a
projected norm passes if it falls within
`[reference_mean / tolerance, reference_mean * tolerance]`. Chosen by
measuring the frozen `GoalEncoder`'s own reference norms (500 samples over
FetchReach's real goal box): their spread is roughly 0.55x-1.87x of their
mean, so a 2x band comfortably contains genuine in-distribution scale while
still catching the kind of failure stage 3 hit (projected norms 5-10x the
reference mean)."""


class LanguageGoalProjection(nn.Module):
    """Small MLP mapping a (batch, input_dim) sentence embedding to (batch, embed_dim).

    Same Linear-ReLU-Linear shape as `GoalEncoder` (see `goal_encoder.py`),
    for the same reason: a single hidden layer gives the mapping enough
    capacity to reach a nonlinear region of the 16-dim target space without
    over-parameterizing a projection that's only ever trained against a
    closed, ~14-instruction fixed vocabulary — a deeper network would mostly
    add overfitting risk here, not useful capacity.
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        embed_dim: int = DEFAULT_EMBED_DIM,
        hidden_dim: int = 64,
    ) -> None:
        """Build the projection's layers.

        Args:
            input_dim: Dimensionality of the frozen sentence embedding
                (384 for `all-MiniLM-L6-v2`).
            embed_dim: Dimensionality of stage 2's goal-embedding space (16
                for `GoalEncoder`'s default).
            hidden_dim: Width of the single hidden layer.

        """
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, sentence_embeddings: torch.Tensor) -> torch.Tensor:
        """Map a batch of sentence embeddings to stage 2's goal-embedding space.

        Args:
            sentence_embeddings: Tensor of shape (batch, input_dim).

        Returns:
            Tensor of shape (batch, embed_dim).

        """
        return self.net(sentence_embeddings.to(torch.float32))


def precompute_instruction_targets(
    goal_encoder: GoalEncoder,
    region_names: Sequence[str],
    *,
    box: GoalBox = MEASURED_GOAL_BOX,
    n_samples: int = DEFAULT_N_TARGET_SAMPLES,
    seed: int = 0,
) -> torch.Tensor:
    """One row per name in `region_names`: that region's fixed, precomputed-once true centroid.

    Computed once per *unique* region name (via `compute_region_target_embeddings`,
    a large `n_samples`-point average through the frozen `goal_encoder`) and
    broadcast back to every row of `region_names`, so two instructions
    sharing a region (e.g. synonyms) regress toward the exact same point
    rather than two independently-sampled noisy estimates of it. This is the
    core change from attempts 1/2's per-step resampling: called once by
    `train_projection`, before its optimization loop starts, not once per
    step.

    Args:
        goal_encoder: Stage 2's pretrained `GoalEncoder`, used as a frozen
            source of ground-truth region embeddings.
        region_names: Region name for each row of the target this function
            builds (e.g. via `goal_region_vocabulary.instruction_to_region`).
            May contain duplicates.
        box: Goal box to sample regions within.
        n_samples: xyz samples averaged per *unique* region to estimate its
            true centroid. See `DEFAULT_N_TARGET_SAMPLES`'s docstring.
        seed: Base seed for region sampling; each unique region name gets a
            distinct offset seed, deterministic given `(region_names, seed)`.

    Returns:
        Tensor of shape (len(region_names), goal_encoder.embed_dim).

    """
    unique_names = list(dict.fromkeys(region_names))
    unique_targets = compute_region_target_embeddings(
        goal_encoder, unique_names, box=box, n_samples=n_samples, seed=seed,
    )
    name_to_target = dict(zip(unique_names, unique_targets, strict=True))
    return torch.stack([name_to_target[name] for name in region_names])


def regression_loss(anchor_embeddings: torch.Tensor, target_embeddings: torch.Tensor) -> torch.Tensor:
    """Plain MSE between each instruction's projected output and its fixed target.

    Chosen over Huber loss: the target here is a fixed, precomputed constant
    for the whole training run (see `precompute_instruction_targets`), not a
    noisy per-step label needing robustness to outliers -- Huber's advantage
    over MSE (a linear rather than quadratic penalty past some delta, to
    limit the influence of outlier labels) has no defect here to guard
    against. MSE also directly optimizes both direction *and* magnitude in
    one term with no separate scale-matching mechanism needed (see the
    module docstring's "Attempt 3" section and
    `test_trained_projection_output_norms_track_target_norms_without_a_separate_norm_term`
    in `tests/lang_goal_rl/test_language_goal_projection.py`), unlike attempt
    1/2's InfoNCE-based loss, which is provably scale-invariant.

    Args:
        anchor_embeddings: Tensor of shape (batch, embed_dim) — the
            projection's output for each instruction.
        target_embeddings: Tensor of shape (batch, embed_dim), row-aligned
            with `anchor_embeddings` — each instruction's fixed target.

    Returns:
        Scalar MSE loss tensor.

    """
    return f.mse_loss(anchor_embeddings, target_embeddings)


def train_projection(
    goal_encoder: GoalEncoder,
    sentence_embeddings: torch.Tensor,
    region_names: Sequence[str],
    *,
    box: GoalBox = MEASURED_GOAL_BOX,
    n_steps: int = 500,
    n_target_samples: int = DEFAULT_N_TARGET_SAMPLES,
    learning_rate: float = 1e-3,
    seed: int = 0,
    projection: LanguageGoalProjection | None = None,
) -> tuple[LanguageGoalProjection, list[float]]:
    """Train a projection via direct regression to each instruction's fixed, precomputed-once target.

    `goal_encoder` is frozen throughout (its parameters' `requires_grad` is
    left untouched, and the target-embedding computation runs under
    `torch.no_grad()`) — only `projection`'s weights are updated. The
    regression target is computed exactly once, before the optimization loop
    starts (see `precompute_instruction_targets`), not resampled per step —
    the fix for attempt 2's diagnosed defect (a noisy per-step target adding
    directional noise a closed, fixed vocabulary doesn't need to tolerate).

    Args:
        goal_encoder: Stage 2's pretrained `GoalEncoder`, used as a frozen
            source of ground-truth region embeddings.
        sentence_embeddings: Precomputed frozen sentence embeddings, shape
            (n_instructions, input_dim), one row per instruction. Computed
            once outside this function (e.g. via
            `language_embedding.encode_instructions`) since the encoder is
            frozen and the fixed vocabulary's text never changes.
        region_names: Region name for each row of `sentence_embeddings`, same
            length and order (e.g. via
            `goal_region_vocabulary.instruction_to_region`).
        box: Goal box to sample regions within.
        n_steps: Number of optimizer steps.
        n_target_samples: xyz samples averaged per unique region when
            precomputing its fixed target (see
            `precompute_instruction_targets`) — a one-time cost, not
            per-step.
        learning_rate: Adam learning rate for `projection`'s parameters.
        seed: Seed for the one-time target precompute's region sampling.
        projection: An existing `LanguageGoalProjection` to continue training.
            If `None`, a fresh one is constructed sized to
            `sentence_embeddings`' and `goal_encoder`'s dimensions.

    Returns:
        A tuple `(projection, loss_history)`: the trained module and the
        regression loss recorded at every step (useful for a caller checking
        that training actually reduced the loss).

    Raises:
        ValueError: If `sentence_embeddings` and `region_names` have
            different lengths.

    """
    if sentence_embeddings.shape[0] != len(region_names):
        msg = (
            f"row count mismatch: sentence_embeddings has {sentence_embeddings.shape[0]} rows, "
            f"region_names has {len(region_names)}"
        )
        raise ValueError(msg)

    resolved_projection = projection or LanguageGoalProjection(
        input_dim=sentence_embeddings.shape[1], embed_dim=goal_encoder.embed_dim,
    )

    goal_encoder.eval()
    for parameter in goal_encoder.parameters():
        parameter.requires_grad = False

    targets = precompute_instruction_targets(
        goal_encoder, region_names, box=box, n_samples=n_target_samples, seed=seed,
    )

    optimizer = torch.optim.Adam(resolved_projection.parameters(), lr=learning_rate)
    frozen_sentence_embeddings = sentence_embeddings.detach().to(torch.float32)

    loss_history: list[float] = []
    for _step in range(n_steps):
        anchor = resolved_projection(frozen_sentence_embeddings)

        loss = regression_loss(anchor, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    return resolved_projection, loss_history


def measure_reference_norms(
    goal_encoder: GoalEncoder,
    *,
    box: GoalBox = MEASURED_GOAL_BOX,
    n_samples: int = 500,
    seed: int = 0,
) -> torch.Tensor:
    """Sample goals uniformly from `box` and return the frozen encoder's output norms.

    This is the RL policy's *actual* training-time input distribution for
    the desired-goal embedding: FetchReach samples `desired_goal` uniformly
    over its box on every reset, and every one of those goals is fed through
    `goal_encoder` before the policy sees it. A projection whose output norm
    falls outside this distribution's range is feeding the policy something
    it never learned to interpret — exactly the failure diagnosed in
    `experiments/03_language_goal_projection/report.md`.

    Args:
        goal_encoder: The frozen encoder to measure (its own parameters are
            never touched; sampling runs under `torch.no_grad()`).
        box: The goal box to sample within — should match whatever box the
            encoder was actually trained/used against.
        n_samples: Number of goals to sample.
        seed: Seed for the uniform sampling.

    Returns:
        Tensor of shape (n_samples,): the L2 norm of each sampled goal's
        embedding.

    """
    rng = np.random.default_rng(seed)
    goals = rng.uniform(box.axis_min, box.axis_max, size=(n_samples, 3))
    with torch.no_grad():
        embeddings = goal_encoder(torch.from_numpy(goals).float())
    return embeddings.norm(dim=1)


@dataclass(frozen=True)
class ProjectionNormCheck:
    """Result of comparing a trained projection's output norms to the frozen encoder's reference range.

    Attributes:
        passed: `True` iff every instruction's projected norm falls within
            `[lower_bound, upper_bound]`.
        instructions: The instruction (or other row label) checked, in order.
        projected_norms: `projected_embeddings.norm(dim=1)`'s values, one per
            instruction, same order as `instructions`.
        reference_mean: Mean of the reference norms the check was run against
            (see `measure_reference_norms`).
        reference_std: Standard deviation of those same reference norms.
        tolerance: The multiplicative tolerance used to derive the bounds.
        lower_bound: `reference_mean / tolerance`.
        upper_bound: `reference_mean * tolerance`.
        out_of_range_instructions: The subset of `instructions` whose
            projected norm fell outside `[lower_bound, upper_bound]`.

    """

    passed: bool
    instructions: tuple[str, ...]
    projected_norms: tuple[float, ...]
    reference_mean: float
    reference_std: float
    tolerance: float
    lower_bound: float
    upper_bound: float
    out_of_range_instructions: tuple[str, ...]

    def summary(self) -> str:
        """Render a human-readable, log-friendly report of the check.

        Intended for a caller to `print(result.summary())` and redirect
        stdout to a file — the stage-3 reviewer flagged that the previous
        diagnostic script's output was never captured to a log, weakening
        the evidence trail; this method exists so a caller only has to
        redirect, not hand-format, to fix that.
        """
        lines = [
            f"ProjectionNormCheck: {'PASSED' if self.passed else 'FAILED'}",
            f"  reference: mean={self.reference_mean:.4f} std={self.reference_std:.4f} "
            f"(tolerance={self.tolerance:.1f}x -> bounds=[{self.lower_bound:.4f}, {self.upper_bound:.4f}])",
        ]
        for instruction, norm in zip(self.instructions, self.projected_norms, strict=True):
            flag = "" if self.lower_bound <= norm <= self.upper_bound else "  <-- OUT OF RANGE"
            lines.append(f"  {norm:.4f}  {instruction!r}{flag}")
        if self.out_of_range_instructions:
            lines.append(f"  {len(self.out_of_range_instructions)} instruction(s) out of range: "
                          f"{list(self.out_of_range_instructions)}")
        return "\n".join(lines)


def check_projection_norm_range(
    projected_embeddings: torch.Tensor,
    reference_norms: torch.Tensor,
    instructions: Sequence[str],
    *,
    tolerance: float = DEFAULT_NORM_RANGE_TOLERANCE,
) -> ProjectionNormCheck:
    """Fail-fast check: do a trained projection's output norms fall within the encoder's reference range.

    Cheap and offline — no RL training or evaluation required, so a caller
    (e.g. an experiment-runner script) can run this immediately after
    `train_projection` returns and catch a repeat of stage 3's scale-mismatch
    failure before spending an RL training budget rediscovering it.

    Args:
        projected_embeddings: The trained projection's output, shape
            (n_instructions, embed_dim).
        reference_norms: The frozen encoder's reference-distribution norms
            (see `measure_reference_norms`), shape (n_reference_samples,).
        instructions: Label for each row of `projected_embeddings`, same
            length and order.
        tolerance: Multiplicative bound around `reference_norms.mean()` (see
            `DEFAULT_NORM_RANGE_TOLERANCE`'s docstring for how this default
            was picked).

    Returns:
        A `ProjectionNormCheck` with `passed=True` iff every instruction's
        norm falls within the tolerance band.

    Raises:
        ValueError: If `projected_embeddings` and `instructions` have
            different lengths.

    """
    if projected_embeddings.shape[0] != len(instructions):
        msg = (
            f"row count mismatch: projected_embeddings has {projected_embeddings.shape[0]} rows, "
            f"instructions has {len(instructions)}"
        )
        raise ValueError(msg)

    reference_mean = float(reference_norms.mean().item())
    reference_std = float(reference_norms.std().item())
    lower_bound = reference_mean / tolerance
    upper_bound = reference_mean * tolerance

    with torch.no_grad():
        projected_norms = projected_embeddings.norm(dim=1).tolist()

    out_of_range = tuple(
        instruction
        for instruction, norm in zip(instructions, projected_norms, strict=True)
        if not (lower_bound <= norm <= upper_bound)
    )

    return ProjectionNormCheck(
        passed=len(out_of_range) == 0,
        instructions=tuple(instructions),
        projected_norms=tuple(projected_norms),
        reference_mean=reference_mean,
        reference_std=reference_std,
        tolerance=tolerance,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        out_of_range_instructions=out_of_range,
    )
