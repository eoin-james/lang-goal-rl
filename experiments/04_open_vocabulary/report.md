# Stage 4: Open vocabulary
**Date:** 2026-07-26 **Seeds run:** [0, 1, 2] **Candidates:** 1 (locked-in)

## Proof gate (verbatim from ROADMAP.md)
> Graceful degradation on unseen phrasing; semantic neighbors land near each other in goal space.

## Result summary
### Part 1 -- Semantic-neighbor diagnostic (no RL; frozen sentence-transformer + stage-3 projection only)

Reference set: the 14 *training* instructions' own projected embeddings (`goal_region_vocabulary.ALL_INSTRUCTIONS` run through the same, unchanged `language_goal_projection_v3.pt`) -- chosen over region centroids because the proof gate asks whether the projection's actual output geometry for real sentences places semantic neighbors near each other, not whether it lands near a separately-computed idealized average (see `diagnose_open_vocab.py`'s module docstring; independently re-checked against region centroids instead and the accuracy did not change, 0.286 either way).

**Aggregate accuracy: 0.286 (4/14) -- vs. a 1/7 ≈ 0.143 random-region-assignment baseline (2x chance, but far from reliable).**

| Instruction | True region | Nearest region | Correct | Margin (true-region minus nearest-region distance; 0 when correct, positive means the wrong region won by that much) |
|-------------|-------------|----------------|---------|---------------------------------------------------------------------------------------------------------------------------|
| settle into the middle of the workspace | center | reach forward | NO | 0.0066 |
| return your hand to a neutral position | center | reach down low | NO | 0.0100 |
| push your arm out in front of you | reach forward | reach up high | NO | 0.0102 |
| extend forward away from your body | reach forward | reach up high | NO | 0.0029 |
| draw your hand back toward yourself | reach back | reach up high | NO | 0.0025 |
| retreat away from the front of the workspace | reach back | reach down low | NO | 0.0057 |
| swing your arm over to the left | reach left | reach up high | NO | 0.0080 |
| shift your gripper toward the left edge | reach left | reach back | NO | 0.0130 |
| swing your arm over to the right | reach right | reach up high | NO | 0.0052 |
| shift your gripper toward the right edge | reach right | reach right | yes | 0.0000 |
| raise your arm as high as it will go | reach up high | reach up high | yes | 0.0000 |
| extend upward toward the ceiling | reach up high | reach up high | yes | 0.0000 |
| lower your arm toward the floor | reach down low | reach up high | NO | 0.0024 |
| drop your gripper down low | reach down low | reach down low | yes | 0.0000 |

### Part 2 -- RL success rate on held-out phrasings (the actual generalization test)

Same 3 already-trained SAC checkpoints and same stage-3 fixed-centroid-regression projection checkpoint as stage 3's attempt 4 -- no retraining. Ground truth judged against each instruction's region centroid (`train.compute_region_centroid`), applying the region-vs-point lesson from the start, per `ROADMAP.md`'s Known risks.

| Seed | Literal success rate (50 eval episodes, stage-2/3 protocol) | Mean held-out language success rate (14 instructions x 50 episodes) |
|------|------------------------------------------------------------|----------------------------------------------------------------|
| 0 | 1.000 | 0.000 |
| 1 | 1.000 | 0.071 |
| 2 | 1.000 | 0.000 |

Aggregate across 3 seeds x 14 held-out instructions (42 success-rate samples): mean=**0.024**, median=**0.000**, max=**1.000**, min=**0.000**, nonzero samples=**1/42**.

**Comparison to stage 3's final (training-vocabulary) baseline** (1.000, attempt 4, same checkpoints and same projection, 14 *trained-on* instructions): 0.000 median vs. 1.000 -- generalization to unseen phrasing collapses almost entirely; literal-goal control stays a clean 1.000 on all 3 seeds (same checkpoints, unchanged), so this is specific to the held-out projections landing off-target, not a policy or checkpoint regression.

### Per-instruction detail (mean success rate across all 3 seeds)

| Instruction | Region | Held-out RL success rate | Semantic-neighbor verdict |
|-------------|--------|---------------------------|-----------------------------|
| settle into the middle of the workspace | center | 0.000 | WRONG |
| return your hand to a neutral position | center | 0.000 | WRONG |
| push your arm out in front of you | reach forward | 0.000 | WRONG |
| extend forward away from your body | reach forward | 0.000 | WRONG |
| draw your hand back toward yourself | reach back | 0.000 | WRONG |
| retreat away from the front of the workspace | reach back | 0.000 | WRONG |
| swing your arm over to the left | reach left | 0.000 | WRONG |
| shift your gripper toward the left edge | reach left | 0.000 | WRONG |
| swing your arm over to the right | reach right | 0.000 | WRONG |
| shift your gripper toward the right edge | reach right | 0.000 | correct |
| raise your arm as high as it will go | reach up high | 0.333 | correct |
| extend upward toward the ceiling | reach up high | 0.000 | correct |
| lower your arm toward the floor | reach down low | 0.000 | WRONG |
| drop your gripper down low | reach down low | 0.000 | correct |

### Part 3 -- Compositional instructions (no single ground-truth region; reported honestly, no forced verdict)

| Instruction | Components | Nearest region | Nearest is a component | Component balance (1.0=equidistant) |
|-------------|------------|-----------------|--------------------------|----------------------------------------|
| reach up and to the left | reach up high / reach left | reach left | True | 0.346 |
| reach forward and down | reach forward / reach down low | reach forward | True | 0.716 |


## Charts
![held_out_success_rate.png](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/charts/held_out_success_rate.png)

![embedding_projection_open_vocab.png](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/charts/embedding_projection_open_vocab.png)

## Raw output
- [stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/runs/seed_0/stdout.log)
- [stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/runs/seed_1/stdout.log)
- [stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/runs/seed_2/stdout.log)
- [semantic_neighbor_diagnostic_stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/artifacts/semantic_neighbor_diagnostic_stdout.log)

## Anomalies (factual, not judged)
Held-out RL success rate collapsed to near-zero: 1/42 samples nonzero (mean 0.024, median 0.000), against a 1.000 training-vocabulary baseline using the identical checkpoints, projection, and eval protocol -- the only variable that changed is which 14 instructions were projected. This tracks the semantic-neighbor diagnostic's finding directly: only 4/14 held-out paraphrases' projected embeddings land nearest their own true region, so most held-out instructions send the policy toward the wrong region entirely -- well outside FetchReach's 0.05m success radius, not a near-miss. Literal-goal control is unchanged and a clean 1.000 on all 3 seeds, ruling out a policy or checkpoint problem; this is specific to the projection's direction accuracy for sentence embeddings it was never trained on.

The one nonzero held-out sample ('raise your arm as high as it will go', seed 1, 1.000) is also the semantic-neighbor diagnostic's clearest correct classification for that region (both 'reach up high' held-out paraphrases classify correctly) -- consistent with, not contradicting, the overall pattern rather than a random outlier.

Compositional instructions ('reach up and to the left', 'reach forward and down') both land nearest one of their two named component regions (not a third, unrelated region), and both skew toward one component more than the other (balance 0.346 and 0.716) rather than sitting exactly on the midline between them -- the projection resolves a compositional phrase to 'closer to one direction' rather than an even blend or a random unrelated point, even though it was never built or trained to handle composition.

## Known-risks cross-check
Not the documented SAC deterministic-eval-collapse signature (literal eval is a clean 1.000 on all 3 seeds, same checkpoints as stages 2/3, no retraining). Not the 'Metric mismatch' known risk either (nothing about the sentence-embedding or distance-reward metric changed -- same frozen sentence-transformer, same frozen GoalEncoder, same projection checkpoint as stage 3's passing attempt 4). This result directly applies the region-vs-point eval-protocol lesson from ROADMAP.md's Known risks from the start (ground truth judged against `compute_region_centroid`, never a resampled point) -- so the near-zero held-out success rate is not a repeat of that defect; it reflects a new, distinct finding this stage exists to surface: `LanguageGoalProjection`, trained via direct regression on a closed 14-instruction vocabulary with no generalization pressure (no held-out validation set, no regularization term encouraging smooth interpolation between training points), does not generalize its *direction* accuracy to unseen phrasings -- it graceful-degrades on the diagnostic (28.6% vs. 14.3% chance, better than random) far more than it does on the actual RL task (a region miss of even a few centimeters misses FetchReach's tight 0.05m success radius entirely). Worth tracking as a new Known risks entry before stage 5/6 build on this projection unchanged.

## Reviewer verdict

**Verdict: FAIL**

**Check 1 -- is this graceful degradation, as the proof gate requires, or outright failure?**
Confirmed outright failure, not degradation. Held-out RL success (mean 0.024, median 0.000, 1/42 nonzero) vs. 1.000 training-vocabulary baseline is not "graceful" by any reasonable reading. Semantic-neighbor accuracy of 28.6% is 2x the 14.3% random baseline -- informative that some directional signal exists, but nowhere near sufficient to call the gate met. The report's own numbers are accurate and honestly presented; the gate itself is not satisfied.

**Check 2 -- does "the one success correlates with correct neighbor classification" hold up, or is it a post-hoc single data point?**
Weaker than the report's framing suggests. Of the 4 held-out phrases correctly classified to their true region, only 1 succeeded at RL (and only on 1/3 seeds). The other 3 correctly-classified phrases still failed at RL (0.000). So: wrong classification reliably predicts RL failure (10/10), but correct classification does *not* reliably predict RL success (1/4) -- it's necessary, not sufficient. Distance analysis shows why: even among the 4 correctly-classified phrases, absolute distance-to-centroid varies widely (0.038 for the one success vs. 0.099 for a same-region phrase that still failed) -- correct-nearest-region is a coarser signal than "close enough to land inside FetchReach's 0.05m success sphere."

**Check 3 -- is this a lossy 384-to-16-dim compression problem, or a training-data problem?**
Training-data problem, confirmed. `all-MiniLM-L6-v2`'s raw 384-dim space does preserve semantic proximity for at least some held-out phrasings (e.g. "raise your arm as high as it will go" vs. "reach up high" both containing the up/high concept) -- ruling out "the frozen encoder itself doesn't distinguish these phrases." The actual mechanism: `LanguageGoalProjection` is a 384->64->16 MLP (~25,600 parameters) trained via direct MSE regression on exactly 14 fixed input-output pairs (2 per region). That is a massively overparameterized network fit to a tiny, non-diverse dataset -- it has every incentive to memorize the 14 exact points and zero pressure to generalize between them. A PCA projection of training vs. held-out points shows the two populations occupying almost entirely separate regions of the 16-dim output space, which is the direct visual signature of memorization rather than a smooth learned mapping.

**Check 4 -- known-risks cross-check.**
Region-vs-point lesson (stage 3) correctly applied from the start -- not repeated. Metric-mismatch known risk not implicated. This is a genuinely new failure mode: **projection-layer overfitting to a minimal vocabulary** -- not documented anywhere in ROADMAP.md yet. Recommend adding it.

**Recommended next step (in order):**
1. **Run a nearest-neighbor-interpolation ceiling test first (near-zero cost, no training)** -- at inference, map a query to a distance-weighted blend of the 14 known training targets in raw 384-dim space, skip the learned MLP entirely. If this beats 28.6% neighbor accuracy on the held-out set, it directly confirms memorization (not an information-theoretic ceiling) as the cause and de-risks the next real fix before spending a training cycle on it.
2. **Then fix via data augmentation** -- expand the training vocabulary well past 14 examples (e.g. 10+ diverse paraphrases per region, LLM-generated or hand-written) so the same regression objective has enough distinct points per region to be forced toward a generalizing mapping rather than memorizing individual sentences. This directly targets the diagnosed cause (sample starvation relative to parameter count) rather than changing architecture or loss function first.
3. Only add a smoothness/neighbor-preserving regularization term *after* (2), if augmentation alone isn't enough -- with only 14 points there isn't enough data to define meaningful neighborhoods for such a term to regularize over yet.

**Do not treat this as a repeat of stage 3's pattern** -- stage 3's bug was in the eval protocol, not the model; this one is a real generalization failure in the model itself, and the fix is data, not a one-line eval change.
