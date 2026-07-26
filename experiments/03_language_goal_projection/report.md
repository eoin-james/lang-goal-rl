# Stage 3: Frozen language embedding -> goal space
**Date:** 2026-07-26 **Seeds run:** [0, 1, 2] **Candidates:** 1 (locked-in)

## Proof gate (verbatim from ROADMAP.md)
> Success rate on language goals ~ stage-2 baseline; projection doesn't collapse distinct instructions to one point.

## Result summary
### Half 1 — literal-goal protocol reproduction (sanity check before the language test)

| Seed | Literal success rate (50 eval episodes, stage-2 protocol) |
|------|------------------------------------------------------------|
| 0 | 1.000 |
| 1 | 1.000 |
| 2 | 1.000 |

Stage 2's 10-seed baseline: mean=1.000, median=1.000, mode=1.000 — all 3 tiered seeds reproduce it exactly.

### Half 2a — language-goal substitution success rate (the actual stage-3 test)

| Seed | Mean success rate across 14 instructions (50 episodes each) |
|------|----------------------------------------------------------------|
| 0 | 0.000 |
| 1 | 0.007 |
| 2 | 0.000 |

Aggregate across all 3 seeds x 14 instructions (42 success-rate samples): mean=0.002, median=0.000, max=0.100.

### Half 2a — per-instruction detail (seed 0)

| Instruction | Region | Success rate |
|-------------|--------|---------------|
| move your hand to the center | center | 0.000 |
| keep the gripper in the middle of the workspace | center | 0.000 |
| move your hand forward | reach forward | 0.000 |
| reach out in front of you | reach forward | 0.000 |
| pull your hand back | reach back | 0.000 |
| reach backward toward yourself | reach back | 0.000 |
| move your hand to the left | reach left | 0.000 |
| reach toward the left side | reach left | 0.000 |
| move your hand to the right | reach right | 0.000 |
| reach toward the right side | reach right | 0.000 |
| reach up high | reach up high | 0.000 |
| move your hand upward | reach up high | 0.000 |
| reach down low | reach down low | 0.000 |
| move your hand downward | reach down low | 0.000 |

### Half 2b — collapse diagnostic (re-verified independently, not cited from the builder)

`min_cross_region_pairwise_distance / collapse_epsilon` = **143.85x** (threshold is 1x; anything above 1x is "not collapsed"). `is_collapsed` = **False**. Full numeric readout: `artifacts/collapse_diagnostic_stdout.log`.


## Charts
![literal_goal_success_rate.png](/Users/eoinmca/Projects/lang-goal-rl/experiments/03_language_goal_projection/charts/literal_goal_success_rate.png)

![language_goal_success_rate.png](/Users/eoinmca/Projects/lang-goal-rl/experiments/03_language_goal_projection/charts/language_goal_success_rate.png)

![embedding_projection.png](/Users/eoinmca/Projects/lang-goal-rl/experiments/03_language_goal_projection/charts/embedding_projection.png)

## Raw output
- [stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/03_language_goal_projection/runs/seed_0/stdout.log)
- [stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/03_language_goal_projection/runs/seed_1/stdout.log)
- [stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/03_language_goal_projection/runs/seed_2/stdout.log)

## Anomalies (factual, not judged)
The language-goal substitution test failed near-uniformly: mean success rate 0.002 across all 3 seeds x 14 instructions x 50 episodes (vs. literal-goal 1.000 on the same 3 checkpoints, reproducing stage 2's baseline exactly). This is NOT seed noise -- all 3 seeds show the same near-total failure, so scaling to the full 10-seed budget was skipped per the tiered-seed strategy (a tier-1 result this uniformly bad would not change qualitatively with 7 more seeds).

Root-caused via `debug_language_eval.py`, run against the trained seed_0 checkpoint:
- Check 1: feeding the policy the *correct* `goal_encoder(literal_target)` embedding through the exact same monkeypatch substitution machinery used for the language test reproduces success_rate=1.000 over 20 episodes -- so the substitution mechanism itself (env goal override + features-extractor monkeypatch) is verified sound, not the source of the failure.
- Check 2: norm-scale mismatch. `goal_encoder(desired_goal)` outputs, for goals actually drawn from the env's real training-time distribution (uniform over the measured box), have norm mean=0.039 std=0.009 (range ~0.022-0.073) over 500 samples. The trained `LanguageGoalProjection`'s outputs for the 14 fixed instructions have norms in the ~0.25-0.41 range -- 5-10x larger than anything the policy ever saw as a goal-embedding input during training. `train_projection`'s InfoNCE-style loss pulls each instruction toward its region's mean embedding and pushes it away from other regions' mean embeddings, but nothing in that objective constrains the *overall scale* of the projection's output to match the frozen encoder's actual output range -- it converged to well-separated points (satisfying half 2b's collapse check) that sit far outside the policy's training-distribution manifold (failing half 2a's success-rate check). The embedding-projection chart shows this directly: the projected instructions and the training-distribution goal-embedding cloud occupy visually distinct regions of the PCA plot.

## Known-risks cross-check
This failure does not match ROADMAP.md's documented SAC deterministic-eval-collapse signature (good training curve -> collapsed eval, preceded by an ent_coef_loss spike): that signature is about the *literal*-goal eval collapsing after training, but here the literal-goal eval is a clean 1.000 on all 3 seeds -- training and the frozen-encoder-based policy are both fine. The failure is specific to the language-projection substitution step, which is new to this stage and not something stage 1's cross-check applies to. The 'Metric mismatch' known risk (sentence-transformer's contrastive cosine-similarity space vs. a raw-distance-based reward) is adjacent but not quite what happened here either -- this stage never trains a distance-based reward off the sentence embedding directly; `train_projection` regresses into the *frozen GoalEncoder's* space via InfoNCE, and the resulting scale mismatch is a property of that regression's loss (no scale term), not of the sentence-embedding metric per se. Recording this as a new, distinct failure mode rather than force-fitting it to an existing Known risks entry. Per the ROADMAP's scope decision, this result is FetchReach-only and says nothing about harder tasks; it is a mechanism-level finding (projection output scale vs. training-distribution scale) that would need to be re-checked on any task, not something specific to FetchReach's dynamics.

## Reviewer verdict

**Verdict: FAIL**

Independently re-verified every claim, not taken on the runner's framing.
Literal-goal control (1.000 all 3 seeds) and the 0.000/0.007/0.000
language-goal failure both confirmed directly from raw
`runs/seed_*/stdout.log` — 41/42 (seed × instruction) samples are exactly
0.000, one outlier at 0.100. Collapse diagnostic independently re-verified
against `artifacts/collapse_diagnostic_stdout.log`: ratio 143.85x, not
collapsed — that half of the gate genuinely passes. The success-rate half
does not, and the gate is conjunctive (both halves required), so the
overall verdict is FAIL.

**Root cause confirmed at the code level, not just inferred from behavior.**
`contrastive.py`'s `info_nce_loss` calls `F.normalize()` on both anchor and
positive embeddings before computing the loss — this makes the training
objective **mathematically scale-invariant** in the projection's output.
Any positive rescaling of the projection's output has exactly zero effect
on this loss. There is no mechanism by which training could have pulled
the output norm toward the frozen `GoalEncoder`'s ~0.02-0.07 operating
range — the loss simply cannot see scale at all. This is stronger evidence
than "the numbers didn't match": the architecture cannot produce a
scale-correct result no matter how long or well it trains.

One evidence gap noted for the record: `debug_language_eval.py`'s output
was never saved to a log file, so its specific cited numbers (the
Check-1 1.000 result, the exact norm ranges) aren't independently
re-checkable from a raw artifact — only corroborated indirectly (via the
loss-normalization code and a geometric cross-check against the collapse
diagnostic's own numbers, which lined up). Not disqualifying, but the fix
should include saving this diagnostic's output going forward.

3-seed sufficiency confirmed for a specific reason: the projection is
trained once with a fixed seed and shared unchanged across all 3 RL seeds,
so a scale-mismatch defect (a property of that one fixed checkpoint, not of
RL randomness) predicts near-identical failure across seeds — exactly what
was observed. This is evidence *for* the diagnosis, not just an early stop.

Known-risks cross-check confirmed directly from raw logs: `ent_coef_loss`
stayed in ±12 across all 3 seeds (nowhere near the 19-52 SAC eval-collapse
signature), and literal eval was clean 1.000 on all 3 — that known risk
correctly does not apply here. This is a new, distinct failure mode
(loss-structural scale invariance) worth tracking since it will recur in
stage 4 if the projection architecture/loss carries forward unchanged.

**Recommendation to manager:** send back to the builder — not more seeds.
Fix must constrain projection *output magnitude*, since the current loss
provably cannot do this on its own:
1. Add an explicit norm-matching term to `train_projection`'s loss (e.g.
   MSE/Huber between `anchor.norm(dim=1)` and `positive.norm(dim=1)`,
   weighted alongside the InfoNCE term) — not a fixed global rescale,
   since the correct target norm varies per region.
2. Add a fast fail-fast check before spending RL training time again:
   assert the trained projection's output norms fall within ~2x of the
   frozen encoder's measured range (mean 0.039, std 0.009) — and this time
   save the check's output to `artifacts/`.
