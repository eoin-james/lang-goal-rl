# Stage 4: Open vocabulary

Two attempts recorded below, same pattern as stage 3's multi-attempt retrofit
-- attempt 1's FAIL data is preserved verbatim, not overwritten. Attempt 2 is
the reviewer's prescribed data-augmentation fix, retraining the projection on
a 70-sentence vocabulary and reusing the same 3 already-trained SAC
checkpoints unchanged (no new RL training in attempt 2).

## Proof gate (verbatim from ROADMAP.md)
> Graceful degradation on unseen phrasing; semantic neighbors land near each other in goal space.

---

## Attempt 1 (2026-07-26) — FAIL: projection-layer overfitting to a minimal vocabulary
**Seeds run:** [0, 1, 2] **Candidates:** 1 (locked-in)

### Result summary
#### Part 1 -- Semantic-neighbor diagnostic (no RL; frozen sentence-transformer + stage-3 projection only)

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

#### Part 2 -- RL success rate on held-out phrasings (the actual generalization test)

Same 3 already-trained SAC checkpoints and same stage-3 fixed-centroid-regression projection checkpoint as stage 3's attempt 4 -- no retraining. Ground truth judged against each instruction's region centroid (`train.compute_region_centroid`), applying the region-vs-point lesson from the start, per `ROADMAP.md`'s Known risks.

| Seed | Literal success rate (50 eval episodes, stage-2/3 protocol) | Mean held-out language success rate (14 instructions x 50 episodes) |
|------|------------------------------------------------------------|----------------------------------------------------------------|
| 0 | 1.000 | 0.000 |
| 1 | 1.000 | 0.071 |
| 2 | 1.000 | 0.000 |

Aggregate across 3 seeds x 14 held-out instructions (42 success-rate samples): mean=**0.024**, median=**0.000**, max=**1.000**, min=**0.000**, nonzero samples=**1/42**.

**Comparison to stage 3's final (training-vocabulary) baseline** (1.000, attempt 4, same checkpoints and same projection, 14 *trained-on* instructions): 0.000 median vs. 1.000 -- generalization to unseen phrasing collapses almost entirely; literal-goal control stays a clean 1.000 on all 3 seeds (same checkpoints, unchanged), so this is specific to the held-out projections landing off-target, not a policy or checkpoint regression.

#### Per-instruction detail (mean success rate across all 3 seeds)

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

#### Part 3 -- Compositional instructions (no single ground-truth region; reported honestly, no forced verdict)

| Instruction | Components | Nearest region | Nearest is a component | Component balance (1.0=equidistant) |
|-------------|------------|-----------------|--------------------------|----------------------------------------|
| reach up and to the left | reach up high / reach left | reach left | True | 0.346 |
| reach forward and down | reach forward / reach down low | reach forward | True | 0.716 |


### Charts
![held_out_success_rate.png](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/charts/held_out_success_rate.png)

![embedding_projection_open_vocab.png](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/charts/embedding_projection_open_vocab.png)

### Raw output
- [stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/runs/seed_0/stdout.log)
- [stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/runs/seed_1/stdout.log)
- [stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/runs/seed_2/stdout.log)
- [semantic_neighbor_diagnostic_stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/artifacts/semantic_neighbor_diagnostic_stdout.log)

### Anomalies (factual, not judged)
Held-out RL success rate collapsed to near-zero: 1/42 samples nonzero (mean 0.024, median 0.000), against a 1.000 training-vocabulary baseline using the identical checkpoints, projection, and eval protocol -- the only variable that changed is which 14 instructions were projected. This tracks the semantic-neighbor diagnostic's finding directly: only 4/14 held-out paraphrases' projected embeddings land nearest their own true region, so most held-out instructions send the policy toward the wrong region entirely -- well outside FetchReach's 0.05m success radius, not a near-miss. Literal-goal control is unchanged and a clean 1.000 on all 3 seeds, ruling out a policy or checkpoint problem; this is specific to the projection's direction accuracy for sentence embeddings it was never trained on.

The one nonzero held-out sample ('raise your arm as high as it will go', seed 1, 1.000) is also the semantic-neighbor diagnostic's clearest correct classification for that region (both 'reach up high' held-out paraphrases classify correctly) -- consistent with, not contradicting, the overall pattern rather than a random outlier.

Compositional instructions ('reach up and to the left', 'reach forward and down') both land nearest one of their two named component regions (not a third, unrelated region), and both skew toward one component more than the other (balance 0.346 and 0.716) rather than sitting exactly on the midline between them -- the projection resolves a compositional phrase to 'closer to one direction' rather than an even blend or a random unrelated point, even though it was never built or trained to handle composition.

### Known-risks cross-check
Not the documented SAC deterministic-eval-collapse signature (literal eval is a clean 1.000 on all 3 seeds, same checkpoints as stages 2/3, no retraining). Not the 'Metric mismatch' known risk either (nothing about the sentence-embedding or distance-reward metric changed -- same frozen sentence-transformer, same frozen GoalEncoder, same projection checkpoint as stage 3's passing attempt 4). This result directly applies the region-vs-point eval-protocol lesson from ROADMAP.md's Known risks from the start (ground truth judged against `compute_region_centroid`, never a resampled point) -- so the near-zero held-out success rate is not a repeat of that defect; it reflects a new, distinct finding this stage exists to surface: `LanguageGoalProjection`, trained via direct regression on a closed 14-instruction vocabulary with no generalization pressure (no held-out validation set, no regularization term encouraging smooth interpolation between training points), does not generalize its *direction* accuracy to unseen phrasings -- it graceful-degrades on the diagnostic (28.6% vs. 14.3% chance, better than random) far more than it does on the actual RL task (a region miss of even a few centimeters misses FetchReach's tight 0.05m success radius entirely). Worth tracking as a new Known risks entry before stage 5/6 build on this projection unchanged.

### Reviewer verdict

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

#### Part 4 -- Nearest-neighbor-interpolation ceiling test (zero-training diagnostic, reviewer-requested)

Per the reviewer's "Recommended next step (1)" above: before committing to a data-augmentation fix, check whether the raw `all-MiniLM-L6-v2` embedding space itself carries the region-clustering signal that `LanguageGoalProjection`'s trained MLP is throwing away. `nearest_neighbor_projection.py` (new, shipped by the rl-builder for this test) bypasses the MLP entirely: for each held-out paraphrase, it takes a distance-weighted blend of the `k` nearest of the 14 training instructions' fixed region-centroid targets, computed directly in the frozen sentence-transformer's raw 384-dim space -- no learned weights, no training loop. The blended 16-dim output is then classified by nearest region centroid, using the identical `(n_samples=1000, seed=0)` centroid computation stage 3/4 used throughout, so this is an apples-to-apples comparison against Part 1's 28.6% (4/14) MLP figure. `k=1, 3, 5` were all tried and are all reported below -- not cherry-picked to whichever looked best.

| k | Accuracy | vs. MLP Part-1 (0.286, 4/14) |
|---|----------|-------------------------------|
| 1 | 0.714 (10/14) | +0.429 |
| 3 | 0.500 (7/14) | +0.214 |
| 5 | 0.357 (5/14) | +0.071 |

Every k tried beat the MLP's 28.6% baseline; k=1 (pure 1-nearest-neighbor, no blending) scored highest.

**Instructions that flipped vs. the MLP's Part-1 classification:**

| Instruction | MLP (Part 1) | k=1 | k=3 | k=5 |
|---|---|---|---|---|
| settle into the middle of the workspace | WRONG | correct | correct | WRONG |
| return your hand to a neutral position | WRONG | WRONG | WRONG | WRONG |
| push your arm out in front of you | WRONG | WRONG | WRONG | WRONG |
| extend forward away from your body | WRONG | WRONG | correct | WRONG |
| draw your hand back toward yourself | WRONG | correct | WRONG | WRONG |
| retreat away from the front of the workspace | WRONG | WRONG | WRONG | WRONG |
| swing your arm over to the left | WRONG | correct | correct | correct |
| shift your gripper toward the left edge | WRONG | correct | correct | correct |
| swing your arm over to the right | WRONG | correct | correct | WRONG |
| shift your gripper toward the right edge | correct | correct | WRONG | WRONG |
| raise your arm as high as it will go | correct | correct | WRONG | WRONG |
| extend upward toward the ceiling | correct | correct | WRONG | correct |
| lower your arm toward the floor | WRONG | correct | correct | correct |
| drop your gripper down low | correct | correct | correct | correct |

At k=1, 6 instructions flip WRONG (MLP) -> correct (NN); no instruction flips the other direction, so k=1's higher aggregate accuracy is a strict improvement per-instruction, not offset by new losses elsewhere. At k=3 and k=5, some previously-correct MLP classifications (e.g. "shift your gripper toward the right edge", "raise your arm as high as it will go") flip to WRONG under blending -- averaging in centroids further from the query's true region evidently hurts those specific instructions even as it helps others.

**Script:** `experiments/04_open_vocabulary/nn_ceiling_test.py`
**Raw output:** [nn_ceiling_test_stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/runs/nn_ceiling_test_stdout.log)

---

## Attempt 2 (2026-07-26) — data augmentation fix
**Seeds run:** [0, 1, 2] **Candidates:** 1 (locked-in)

Attempt 1's reviewer diagnosed `LanguageGoalProjection` (384->64->16, ~25,600
parameters) as overfit to its 14-sentence training vocabulary -- confirmed
by the NN-ceiling test above (0.714 vs. the trained MLP's 0.286) -- and
recommended data augmentation as the fix, in order, after the ceiling test.
The rl-builder built `augmented_training_vocabulary.py`: 70 sentences, 10
diverse phrasings per region, disjoint from `held_out_paraphrases`'s 14
held-out phrases and its 2 compositional instructions. This attempt: (1)
retrains `LanguageGoalProjection` on the 70-sentence vocabulary
(`train_projection_augmented.py`), reusing every hyperparameter
(`n_steps=2000`, `learning_rate=1e-3`, `n_target_samples=1000`,
`box=MEASURED_GOAL_BOX`, `seed=0`) from the run that trained
`language_goal_projection_v3.pt` unchanged, saved to a new checkpoint
(`artifacts/language_goal_projection_v5_augmented.pt`, v3 untouched); (2)
reruns Parts 1-3 against the same, unchanged 14 held-out phrases and 2
compositional instructions; (3) reuses the same 3 already-trained SAC
checkpoints from stage 3/attempt 1 -- **no new RL training happened in this
attempt**; (4) adds a sanity check not run in attempt 1: does the retrained
projection still ace the *original* 14 stage-3 training instructions.

### Result summary
#### What changed

`LanguageGoalProjection` was retrained (`train_projection_augmented.py`) on `augmented_training_vocabulary.AUGMENTED_INSTRUCTIONS` -- 70 sentences, 10 diverse phrasings per region -- instead of `goal_region_vocabulary.ALL_INSTRUCTIONS` (14 sentences, 2 per region). Every hyperparameter (`n_steps=2000`, `learning_rate=1e-3`, `n_target_samples=1000`, `box=MEASURED_GOAL_BOX`, `seed=0`) is unchanged from the run that trained `language_goal_projection_v3.pt` -- confirmed from stage 3's attempt-3 section (loss dropped to 0.0000 over 2000 steps, matching this retrain's `runs/attempt2/projection_train_stdout.log`, which shows the identical early_mean=0.0020/late_mean=0.0000 pattern). Saved as a new checkpoint (`artifacts/language_goal_projection_v5_augmented.pt`) so v3 stays available for comparison/provenance. No SAC policy was retrained -- all RL evals below reuse the same 3 stage-3 checkpoints (`03_language_goal_projection/checkpoints/seed_{0,1,2}.zip`) attempt 1 used.

#### Part 1 -- Semantic-neighbor diagnostic (no RL; frozen sentence-transformer + augmented projection only)

Reference set: the 70 *augmented-training* instructions' own projected embeddings (through the new `language_goal_projection_v5_augmented.pt`) -- same reference-set choice as attempt 1 (the training instructions' own projected output geometry, not a separately-computed centroid), just over the larger vocabulary this projection actually trained on. Query set is unchanged: the same 14 `held_out_paraphrases.HELD_OUT_PARAPHRASES`.

**Aggregate accuracy: 0.643 (9/14) -- vs. attempt 1's 0.286 (4/14) and the NN-ceiling's 0.714 (10/14, k=1).**

| Instruction | True region | Nearest region | Correct | Margin (true-region minus nearest-region distance; 0 when correct, positive means the wrong region won by that much) |
|-------------|-------------|----------------|---------|---------------------------------------------------------------------------------------------------------------------------|
| settle into the middle of the workspace | center | center | yes | 0.0000 |
| return your hand to a neutral position | center | reach back | NO | 0.0075 |
| push your arm out in front of you | reach forward | reach up high | NO | 0.0024 |
| extend forward away from your body | reach forward | reach forward | yes | 0.0000 |
| draw your hand back toward yourself | reach back | reach back | yes | 0.0000 |
| retreat away from the front of the workspace | reach back | reach forward | NO | 0.0086 |
| swing your arm over to the left | reach left | reach left | yes | 0.0000 |
| shift your gripper toward the left edge | reach left | reach right | NO | 0.0037 |
| swing your arm over to the right | reach right | reach up high | NO | 0.0066 |
| shift your gripper toward the right edge | reach right | reach right | yes | 0.0000 |
| raise your arm as high as it will go | reach up high | reach up high | yes | 0.0000 |
| extend upward toward the ceiling | reach up high | reach up high | yes | 0.0000 |
| lower your arm toward the floor | reach down low | reach down low | yes | 0.0000 |
| drop your gripper down low | reach down low | reach down low | yes | 0.0000 |

#### Part 2 -- RL success rate on held-out phrasings (the actual generalization test)

Same 3 already-trained SAC checkpoints as attempt 1 -- no retraining. Only the projection checkpoint changed (v3 -> v5_augmented). Ground truth judged against each instruction's region centroid (`train.compute_region_centroid`), unchanged from attempt 1.

| Seed | Literal success rate (50 eval episodes, stage-2/3 protocol) | Mean held-out language success rate (14 instructions x 50 episodes) |
|------|------------------------------------------------------------|----------------------------------------------------------------|
| 0 | 1.000 | 0.071 |
| 1 | 1.000 | 0.143 |
| 2 | 1.000 | 0.071 |

Aggregate across 3 seeds x 14 held-out instructions (42 success-rate samples): mean=**0.095**, median=**0.000**, max=**1.000**, min=**0.000**, nonzero samples=**4/42**.

**Comparison to attempt 1** (mean=0.024, median=0.000, nonzero=1/42): mean 0.024 -> 0.095, nonzero samples 1/42 -> 4/42. **Comparison to stage 3's training-vocabulary baseline** (1.000): 0.000 median vs. 1.000 -- still far short. Literal-goal control stays a clean 1.000 on all 3 seeds (same checkpoints, unchanged), so both this attempt's improvement and its remaining gap are specific to the projection, not a policy or checkpoint change.

#### Per-instruction detail (mean success rate across all 3 seeds)

| Instruction | Region | Held-out RL success rate | Semantic-neighbor verdict |
|-------------|--------|---------------------------|-----------------------------|
| settle into the middle of the workspace | center | 0.000 | correct |
| return your hand to a neutral position | center | 0.000 | WRONG |
| push your arm out in front of you | reach forward | 0.000 | WRONG |
| extend forward away from your body | reach forward | 0.000 | correct |
| draw your hand back toward yourself | reach back | 0.000 | correct |
| retreat away from the front of the workspace | reach back | 0.000 | WRONG |
| swing your arm over to the left | reach left | 0.000 | correct |
| shift your gripper toward the left edge | reach left | 0.000 | WRONG |
| swing your arm over to the right | reach right | 0.000 | WRONG |
| shift your gripper toward the right edge | reach right | 1.000 | correct |
| raise your arm as high as it will go | reach up high | 0.333 | correct |
| extend upward toward the ceiling | reach up high | 0.000 | correct |
| lower your arm toward the floor | reach down low | 0.000 | correct |
| drop your gripper down low | reach down low | 0.000 | correct |

#### Part 3 -- Compositional instructions (no single ground-truth region; reported honestly, no forced verdict)

| Instruction | Components | Nearest region | Nearest is a component | Component balance (1.0=equidistant) |
|-------------|------------|-----------------|--------------------------|----------------------------------------|
| reach up and to the left | reach up high / reach left | reach up high | True | 0.973 |
| reach forward and down | reach forward / reach down low | reach forward | True | 0.733 |

#### Sanity check -- does the augmented-vocabulary projection still ace the ORIGINAL stage-3 training vocabulary?

The 70-sentence `augmented_training_vocabulary` has **zero string overlap** with the original 14 `goal_region_vocabulary.ALL_INSTRUCTIONS` (independently verified: `set(AUGMENTED_INSTRUCTIONS) & set(ALL_INSTRUCTIONS) == set()`) -- it replaces, rather than extends, the original training sentences. This check evaluates the new projection on the original 14 instructions it no longer trains on directly, using the same 3 SAC checkpoints and the same fixed-centroid ground truth as Part 2.

| Seed | Literal success rate (50 eval episodes) | Mean success rate on ORIGINAL 14 training instructions (50 episodes each) |
|------|-------------------------------------------|-------------------------------------------------------------------------|
| 0 | 1.000 | 0.214 |
| 1 | 1.000 | 0.071 |
| 2 | 1.000 | 0.143 |

Aggregate across 3 seeds x 14 original instructions (42 success-rate samples): mean=**0.143**, median=**0.000**, max=**1.000**, min=**0.000**, nonzero samples=**6/42**. This is **not** a clean ~1.000 reproduction of stage 3's attempt-4 baseline -- see Anomalies below.

| Instruction | Region | Success rate on original vocabulary |
|-------------|--------|----------------------------------------|
| move your hand to the center | center | 0.333 |
| keep the gripper in the middle of the workspace | center | 0.000 |
| move your hand forward | reach forward | 0.000 |
| reach out in front of you | reach forward | 0.000 |
| pull your hand back | reach back | 0.000 |
| reach backward toward yourself | reach back | 0.000 |
| move your hand to the left | reach left | 0.000 |
| reach toward the left side | reach left | 0.000 |
| move your hand to the right | reach right | 1.000 |
| reach toward the right side | reach right | 0.667 |
| reach up high | reach up high | 0.000 |
| move your hand upward | reach up high | 0.000 |
| reach down low | reach down low | 0.000 |
| move your hand downward | reach down low | 0.000 |

#### Before/after comparison

| Metric | Attempt 1 (14-sentence vocab) | Attempt 2 (70-sentence vocab) | NN-ceiling (k=1, zero-training) |
|--------|-------------------------------|-------------------------------|-----------------------------------|
| Semantic-neighbor accuracy | 0.286 (4/14) | 0.643 (9/14) | 0.714 (10/14) |
| Held-out RL success (mean) | 0.024 | 0.095 | n/a -- geometry-only test, no RL policy involved |
| Held-out RL success (median) | 0.000 | 0.000 | n/a |
| Held-out RL nonzero samples | 1/42 | 4/42 | n/a |

### Charts
![held_out_success_rate_v2.png](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/charts/held_out_success_rate_v2.png)

![stage3_vocab_regression_check_v2.png](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/charts/stage3_vocab_regression_check_v2.png)

![embedding_projection_open_vocab_v2.png](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/charts/embedding_projection_open_vocab_v2.png)

### Raw output
- [projection_train_stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/runs/attempt2/projection_train_stdout.log)
- [stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/runs/attempt2/seed_0/stdout.log)
- [stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/runs/attempt2/seed_1/stdout.log)
- [stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/runs/attempt2/seed_2/stdout.log)
- [stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/runs/attempt2/regression_check/seed_0/stdout.log)
- [stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/runs/attempt2/regression_check/seed_1/stdout.log)
- [stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/runs/attempt2/regression_check/seed_2/stdout.log)
- [semantic_neighbor_diagnostic_v2_stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/artifacts/semantic_neighbor_diagnostic_v2_stdout.log)

### Anomalies (factual, not judged)
Semantic-neighbor accuracy more than doubled: 0.286 (4/14, attempt 1) -> 0.643 (9/14, attempt 2) -- closer to, though still below, the NN-ceiling's 0.714 (10/14). Held-out RL success improved in the same direction but by a smaller margin: mean 0.024 -> 0.095, nonzero samples 1/42 -> 4/42 -- still far short of the 1.000 training-vocabulary baseline and still a FAIL-magnitude gap by any reasonable reading, not graceful degradation. Data augmentation helped -- direction, not magnitude, is the honest summary.

**The regression check surfaced a result the task did not anticipate:** the augmented-vocabulary projection does NOT reproduce stage 3's ~1.000 success rate on the ORIGINAL 14 training instructions (mean=0.143, median=0.000, nonzero=6/42 -- roughly the same order of magnitude as this attempt's own held-out result, not a clean pass). Root cause, verified directly (not inferred): `set(AUGMENTED_INSTRUCTIONS) & set(ALL_INSTRUCTIONS) == set()` -- the 70-sentence augmented vocabulary shares no exact strings with the original 14, so it functions as a *replacement* training set, not an *extension* of it. From the retrained projection's point of view, the original 14 sentences are now just as unseen as `held_out_paraphrases` -- which is exactly why their success rate lands in the same range as the held-out set's, rather than at 1.000. This is evidence *for*, not against, the generalization diagnosis: a projection trained on 70 diverse phrasings generalizes moderately to *any* unseen phrasing (original-14 or held-out-14 alike) rather than memorizing one specific closed set perfectly and failing everywhere else, which is qualitatively the behavior change data augmentation was supposed to produce -- it just hasn't (yet) produced enough of it to clear either bar.

Per-instruction pattern: `'move your hand to the right'`/`'reach toward the right side'`/`'shift your gripper toward the right edge'` (all `reach right`) are disproportionately represented among this attempt's nonzero successes across both the held-out and regression-check evals -- 5 of the 10 total nonzero samples across both evals involve a `reach right` instruction, versus 7 regions sharing roughly equal representation in each vocabulary. Not enough samples to generalize from, but worth watching if a future attempt investigates per-region variance.

Compositional placement changed direction on both instructions: `'reach up and to the left'` now lands nearest `'reach up high'` with balance 0.973 (near-equidistant between its two components, up from attempt 1's 0.346, which was skewed hard toward one side); `'reach forward and down'` now lands nearest `'reach forward'` with balance 0.733 (up from attempt 1's 0.716). Both moved toward a more balanced placement between their two named components, consistent with a smoother, less memorized output geometry -- reported factually, not as a pass/fail signal (per `held_out_paraphrases.py`'s design, compositional instructions have no single ground-truth region).

### Known-risks cross-check
Directly extends ROADMAP.md's 'Projection-layer overfitting to a minimal vocabulary' entry (added after attempt 1): the reviewer's prescribed fix (data augmentation) produced a real, measured improvement in the predicted direction (semantic-neighbor accuracy 0.286 -> 0.643, held-out RL mean 0.024 -> 0.095) but did not close the gap to the proof gate. Not the documented SAC deterministic-eval-collapse signature (literal eval is a clean 1.000 on all 3 seeds throughout, same checkpoints as attempt 1). Not the 'Metric mismatch' known risk (same frozen sentence-transformer, same frozen GoalEncoder, only the projection's training vocabulary changed). New finding worth adding to Known risks: augmenting a fixed-vocabulary projection's training set without deliberately including the *original* training sentences turns those original sentences into held-out data for the retrained projection -- if a future stage needs both 'ace the original vocabulary' and 'generalize to new phrasing' simultaneously, the training set must include both, not just a larger disjoint replacement set.

### Reviewer verdict

**Verdict: FAIL**

**Check 1 -- number verification.** All of attempt 2's reported numbers
independently re-derived from raw logs and confirmed exact: semantic-neighbor
0.643 (9/14, counted directly from `semantic_neighbor_diagnostic_v2_stdout.log`),
held-out RL mean=0.095/median=0.000/nonzero=4/42 (re-summed per-seed:
seed0=0.071, seed1=0.143, seed2=0.071), regression-check mean=0.143/nonzero=6/42.
No inflation, no substitution -- the report's own numbers are honest.

**Check 2 -- the real diagnosis is sharper than "improved but still far below
gate."** The bottleneck has *shifted*, not just shrunk. Attempt 1's bottleneck
was misclassification (4/14 correct). Attempt 2's classification is now
mostly correct (9/14, near the NN-ceiling's 10/14) but RL success stays
near-zero regardless. Spot-checking distance-to-true-region-cluster against
RL outcome for all 9 correctly-classified instructions breaks the simple
"closer wins" pattern attempt 1 found: "lower your arm toward the floor" is
the *closest* correctly-classified instruction (distance 0.0151) and still
scores 0.000 RL success, while "shift your gripper toward the right edge"
at a *farther* distance (0.0200) scores 1.000 on all 3 seeds. RL success is
concentrated almost entirely in the "reach right" region (5 of 10 total
nonzero samples across both the held-out and regression-check evals) --
this is region-dependent policy tolerance, not projection precision alone.

**Check 3 -- does combining vocabularies (84 = 14 original + 70 augmented)
fix this?** Partially, and not the main gap. It will restore the original
14's ~1.000 (fixing the accidental replace-not-extend regression), but the
regression check already shows what an unseen-but-augmented-vocabulary-
adjacent instruction does under the current projection: ~0.143 mean, still
dominated by "reach right." Combining vocabularies doesn't change what the
*held-out* eval measures, so it won't move the 0.095 figure much on its own.
The MLP's classification (0.643) is already close to the NN-ceiling (0.714)
-- classification is nearly maxed out for this data regime. The real gap is
between "correct region" and "close enough for the policy," and that gap is
uneven across regions.

**Check 4 -- known-risks cross-check.** Region-vs-point lesson (stage 3)
correctly applied throughout (ground truth via `compute_region_centroid`,
confirmed from `eval_held_out.py`'s reuse of stage 3's `evaluate_language_goal`).
Not the SAC eval-collapse signature (literal success stays 1.000 throughout).
Not the metric-mismatch risk (same frozen encoders throughout). The
"Projection-layer overfitting to a minimal vocabulary" entry is *partially*
confirmed -- augmentation fixed classification as predicted, but did not
close the RL-success gap, because that gap turns out to be a different
mechanism. **New risk to log:** per-region policy tolerance varies --
the trained SAC policy's basin of attraction around each region's target
embedding is nonuniform (some regions, like "reach right," tolerate real
imprecision; others need near-exact centroid matches), which makes
classification accuracy a poor predictor of RL success on its own.

**Recommendation to manager -- do NOT mark Done. Run this specific two-part
experiment for attempt 3, in order:**

**Part A (diagnostic, zero training, run first):** for each of the 7
regions, take the exact target centroid used in training, inject Gaussian
noise at several L2 magnitudes (e.g. 0.005, 0.010, 0.015, 0.020, 0.030,
0.050), and re-run the existing 3 SAC checkpoints against each perturbed
centroid via the same `evaluate_language_goal` infrastructure already in
use (no projection, no sentences -- pure policy-tolerance measurement).
This produces a per-region "how much deviation from the exact centroid can
the policy tolerate" map and will explain directly why "reach right"
succeeds under imprecision while "reach down low" doesn't.

**Part B (conditional on Part A's result):**
- If most regions tolerate >0.020 deviation: the projection itself is still
  the bottleneck for those regions -- combine the original 14 + augmented
  70 into one 84-sentence training set (fixing the accidental
  replace-not-extend bug) and consider more phrasings.
- If most regions tolerate <0.015: this is a policy-robustness problem, not
  a data problem -- the fix is retraining the SAC policies with noise
  injected into the goal embedding during training (domain randomization),
  not more projection data.
- If tolerance is genuinely region-dependent (most likely given "reach
  right" vs. everything else): stage 2's `GoalEncoder` embedding space has
  nonuniform density across regions -- upstream of both the projection and
  the RL policy, and neither can fully fix it alone.

Do not spend another round on "more training sentences" alone without first
running Part A -- attempt 2 already shows classification improvements don't
translate to RL-success improvements at anywhere near the same rate, so the
next experiment needs to isolate which layer (projection precision vs.
policy tolerance vs. embedding-space geometry) actually owns the remaining
gap before spending effort fixing the wrong one.
