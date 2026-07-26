"""LanguageGoalProjection: frozen sentence embedding -> stage 2's goal-embedding space.

Stage 3's proof gate needs a sentence to land at a point in the *same*
16-dim space stage 2's `GoalEncoder` already defines (see
`goal_encoder.py`), so stage 2's trained policies stay chainable. This
module owns two things:

1. `LanguageGoalProjection` — the small learnable layer doing the mapping.
2. `train_projection` — the procedure that fits it, given a frozen
   `GoalEncoder` and a fixed instruction vocabulary's region assignments
   (see `goal_region_vocabulary.py`).

Loss choice: for each instruction, a batch of true xyz goals is sampled from
its region (`sample_region_goals`) and averaged through the frozen
`GoalEncoder` to get that region's mean embedding — a stochastic estimate of
"where this region truly sits". Regressing the projection's output straight
to that mean with MSE (no separation term) was considered and rejected:
nothing prevents two regions whose *true* mean embeddings happen to sit
close together (adjacent regions of a small, contrastively-pretrained goal
space aren't guaranteed to be far apart) from converging to near-identical
projected points — that's exactly the failure stage 3's proof gate needs
ruled out. Reusing `contrastive.info_nce_loss` (already used to pretrain
`GoalEncoder` itself, see `contrastive.py`) with each instruction's
projected embedding as the anchor and its region's mean embedding as the
positive gets the MSE-like pull *and* an explicit push against every other
instruction's region embedding in the same training batch — directly
targeting "distinct instructions don't collapse".

**Scale fix (stage 3 FAIL post-mortem, see `experiments/03_language_goal_projection/report.md`):**
`info_nce_loss` calls `F.normalize()` on both its inputs, which makes it
mathematically invariant to any positive rescaling of the projection's
output — no amount of training against that loss alone can pull the output
norm toward the frozen `GoalEncoder`'s real operating range, which is
exactly what happened (trained outputs landed 5-10x outside that range,
collapsing RL success to ~0% despite instructions staying well-separated).
`combined_projection_loss` below adds an explicit MSE penalty between the
projection's per-instruction output norm and its region's target-embedding
norm, alongside the existing InfoNCE separation term. `DEFAULT_NORM_LOSS_WEIGHT`
was picked by sweeping {1, 10, 50, 200} against both a real trained
`GoalEncoder` checkpoint (16-dim) and this module's small random-init test
fixtures (4-dim): weight 10 consistently pulled the mean output norm within
single-digit percent of the target norm in a few hundred steps, while
leaving the converged InfoNCE term within the same order of magnitude as
running without the norm term at all (i.e. separation is not measurably
degraded) — weight 1 undercorrected (~3x residual mismatch) and weight 50-200
gave no further norm-matching benefit while measurably slowing the InfoNCE
term's convergence.

`check_projection_norm_range` (also below) is the fail-fast check the
reviewer asked for: given a trained projection's outputs and the frozen
encoder's *measured* operating-range norms (via `measure_reference_norms`),
it flags any instruction whose output norm falls outside a tolerance band
around the reference mean — cheap enough to run right after
`train_projection` returns, catching a repeat of this exact failure mode
before spending an RL training budget on it again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as f  # noqa: N812 -- mirrors contrastive.py's common torch.nn.functional alias
from torch import nn

from lang_goal_rl.contrastive import info_nce_loss
from lang_goal_rl.goal_encoder import DEFAULT_EMBED_DIM
from lang_goal_rl.goal_region_vocabulary import MEASURED_GOAL_BOX, sample_region_goals

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lang_goal_rl.goal_encoder import GoalEncoder
    from lang_goal_rl.goal_region_vocabulary import GoalBox

DEFAULT_INPUT_DIM = 384
"""Matches `language_embedding.LANGUAGE_EMBED_DIM` (`all-MiniLM-L6-v2`'s output size)."""

DEFAULT_NORM_LOSS_WEIGHT = 10.0
"""Weight applied to the norm-matching MSE term in `combined_projection_loss`.
See the module docstring's "Scale fix" section for the sweep that picked this
value: it corrects the output-norm mismatch to within a few percent without
measurably degrading the InfoNCE separation term's convergence."""

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


def _region_mean_embeddings(
    goal_encoder: GoalEncoder,
    region_names: Sequence[str],
    box: GoalBox,
    n_samples: int,
    seed: int,
) -> torch.Tensor:
    """One row per name in `region_names`: the mean frozen-encoder embedding of a fresh region sample.

    Resampled (with a different `seed`) by the caller on every training
    step, so this is a noisy but unbiased per-step estimate of each region's
    true embedding centroid rather than a single fixed target — acting as a
    mild data-augmentation effect on the regression/contrastive target.
    """
    rows = []
    with torch.no_grad():
        for i, name in enumerate(region_names):
            goals = sample_region_goals(name, n_samples, seed=seed + i, box=box)
            embeddings = goal_encoder(torch.from_numpy(goals).float())
            rows.append(embeddings.mean(dim=0))
    return torch.stack(rows)


def combined_projection_loss(
    anchor_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    *,
    norm_loss_weight: float = DEFAULT_NORM_LOSS_WEIGHT,
) -> torch.Tensor:
    """InfoNCE separation plus an explicit norm-matching penalty.

    `info_nce_loss` alone cannot constrain the projection's output scale:
    it L2-normalizes both its inputs internally, so it is provably invariant
    to any positive rescaling of `anchor_embeddings` (see
    `test_plain_info_nce_loss_is_scale_invariant_to_anchor_rescaling` in
    `tests/lang_goal_rl/test_language_goal_projection.py` for a direct
    demonstration). The added term is an MSE penalty between each row's
    anchor norm and its matched positive's norm, which *does* depend on
    scale and therefore gives training an actual gradient toward the
    frozen encoder's real output magnitude.

    Args:
        anchor_embeddings: Tensor of shape (batch, embed_dim) — the
            projection's output for each instruction.
        positive_embeddings: Tensor of shape (batch, embed_dim), row-aligned
            with `anchor_embeddings` — each instruction's region target
            embedding.
        norm_loss_weight: Multiplier on the norm-matching term relative to
            the InfoNCE term. See `DEFAULT_NORM_LOSS_WEIGHT`'s docstring for
            how this default was picked.

    Returns:
        Scalar loss tensor: `info_nce_loss(...) + norm_loss_weight * norm_mse`.

    """
    separation_loss = info_nce_loss(anchor_embeddings, positive_embeddings)
    norm_loss = f.mse_loss(anchor_embeddings.norm(dim=1), positive_embeddings.norm(dim=1))
    return separation_loss + norm_loss_weight * norm_loss


def train_projection(
    goal_encoder: GoalEncoder,
    sentence_embeddings: torch.Tensor,
    region_names: Sequence[str],
    *,
    box: GoalBox = MEASURED_GOAL_BOX,
    n_steps: int = 500,
    n_goal_samples_per_step: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 0,
    projection: LanguageGoalProjection | None = None,
    norm_loss_weight: float = DEFAULT_NORM_LOSS_WEIGHT,
) -> tuple[LanguageGoalProjection, list[float]]:
    """Train a projection so each instruction's embedding lands near its region and away from others.

    `goal_encoder` is frozen throughout (its parameters' `requires_grad` is
    left untouched, and the region-embedding computation runs under
    `torch.no_grad()`) — only `projection`'s weights are updated.

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
        n_goal_samples_per_step: xyz samples averaged per region, per step,
            to estimate that step's target embedding (see
            `_region_mean_embeddings`).
        learning_rate: Adam learning rate for `projection`'s parameters.
        seed: Seed for region-sampling randomness across training steps.
        projection: An existing `LanguageGoalProjection` to continue training.
            If `None`, a fresh one is constructed sized to
            `sentence_embeddings`' and `goal_encoder`'s dimensions.
        norm_loss_weight: Weight on the norm-matching term added to the
            InfoNCE loss (see `combined_projection_loss`). Set to `0.0` to
            recover the old (scale-unconstrained) behavior.

    Returns:
        A tuple `(projection, loss_history)`: the trained module and the
        combined (InfoNCE + norm-matching) loss recorded at every step
        (useful for a caller checking that training actually reduced the
        loss).

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

    optimizer = torch.optim.Adam(resolved_projection.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)

    frozen_sentence_embeddings = sentence_embeddings.detach().to(torch.float32)

    loss_history: list[float] = []
    for _step in range(n_steps):
        step_seed = int(rng.integers(0, 2**31 - 1))
        positive = _region_mean_embeddings(
            goal_encoder, region_names, box, n_goal_samples_per_step, step_seed,
        )
        anchor = resolved_projection(frozen_sentence_embeddings)

        loss = combined_projection_loss(anchor, positive, norm_loss_weight=norm_loss_weight)
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
