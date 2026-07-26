# Experiment Status

This tracks research progress only — stages, proof gates, actual numbers.
No setup/tooling work shown here; that's implementation detail, not an
experiment result. Completed stages are kept short here — full detail
always lives in that stage's linked report.

**TL;DR:** Stages 1-2 done and proven. Stage 3 underway (first stage
introducing actual language). **Progress: 2/7 stages (29%), running
autonomously stage-by-stage — no check-ins unless something needs a real
decision.**

## Active run

**Stage 3 — starting.** Frozen language embedding → goal space: a fixed
vocabulary of instructions mapped through a frozen sentence embedding,
projected into stage 2's goal-embedding space via a learned layer. First
stage where a sentence actually determines the goal, not a coordinate.

## Completed stages

### Stage 1 — Goal-conditioned baseline (UVFA+HER) — PASS

10 seeds, median/mode 1.000, mean 0.840, 8/10 ≥0.98. Took two review
passes: mean alone looked shaky, but the reviewer traced the 2 failures to
a specific, diagnosed SAC mechanism (entropy-coefficient instability
corrupting the deterministic action for that seed) — not a UVFA/HER defect.
Now a tracked Known risk: every later stage must compare at median/mode,
same seed count, and check failed seeds against this exact signature before
blaming their own new component.
[Full report](experiments/01_uvfa_her_baseline/report.md)

### Stage 2 — Learned continuous goal embedding — PASS

10/10 seeds ≥0.98 (median/mode 1.000, better than stage 1's 8/10 — zero
seeds hit stage 1's known failure mode). Distance-in-latent check: Pearson
r=0.571 on 500 held-out goals — real and significant, not the in-sample
0.878 (that gap is contrastive-pretraining overfitting, tracked but doesn't
invalidate the result). Scoped honestly as a simplification of Eysenbach et
al.'s method (frozen encoder + feature-extractor swap, not a full
critic-loss replacement) — documented as such rather than overclaiming.
Bonus: the builder's integration test caught a real bug (a "frozen" encoder
silently shared across SAC's actor/critic/critic-target and drifting via
target-network averaging) before it could corrupt a result.
[Full report](experiments/02_contrastive_goal_embedding/report.md)

## Stage-by-stage table

| # | Stage | Proof gate | Result | Report |
|---|-------|------------|--------|--------|
| 0 | Plumbing | Env loop runs end-to-end | ✅ Pass | — |
| 1 | Goal-conditioned baseline (UVFA+HER) | Near-100% success on FetchReach | ✅ PASS (10/10 seeds, 8/10 ≥0.98) | [link](experiments/01_uvfa_her_baseline/report.md) |
| 2 | Learned continuous goal embedding | Success matches stage 1; distance-in-latent correlates | ✅ PASS (10/10 seeds, r=0.571) | [link](experiments/02_contrastive_goal_embedding/report.md) |
| 3 | Frozen language embedding → goal space | Success ≈ stage 2; no instruction collapse | In progress | — |
| 4 | Open vocabulary | Graceful degradation on unseen phrasing | Not run | — |
| 5 | Mid-episode re-goaling | Zero-shot goal-swap ≈ fresh-episode baseline | Not run | — |
| 6 | Live English interface | End-to-end demo, task success + redirect time | Not run | — |

## What's queued next

Stage 3 build starting now (fixed-vocabulary instruction set, frozen
sentence-transformer, learned projection into stage 2's goal space). Will
keep moving through 4, 5, 6 the same way — build, run, review, close out —
without stopping unless a result needs a real decision from you.
