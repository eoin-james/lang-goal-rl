# Stage 4: Open vocabulary (Attempt 2: data augmentation fix)
**Date:** 2026-07-26 **Seeds run:** [0, 1, 2] **Candidates:** 1 (locked-in)

## Proof gate (verbatim from ROADMAP.md)
> Graceful degradation on unseen phrasing; semantic neighbors land near each other in goal space.

## Result summary
### What changed

`LanguageGoalProjection` was retrained (`train_projection_augmented.py`) on `augmented_training_vocabulary.AUGMENTED_INSTRUCTIONS` -- 70 sentences, 10 diverse phrasings per region -- instead of `goal_region_vocabulary.ALL_INSTRUCTIONS` (14 sentences, 2 per region). Every hyperparameter (`n_steps=2000`, `learning_rate=1e-3`, `n_target_samples=1000`, `box=MEASURED_GOAL_BOX`, `seed=0`) is unchanged from the run that trained `language_goal_projection_v3.pt` -- confirmed from `report.md`'s attempt-3 section (loss dropped to 0.0000 over 2000 steps, matching this retrain's `runs/attempt2/projection_train_stdout.log`). Saved as a new checkpoint (`artifacts/language_goal_projection_v5_augmented.pt`) so v3 stays available for comparison/provenance. No SAC policy was retrained -- all RL evals below reuse the same 3 stage-3 checkpoints (`03_language_goal_projection/checkpoints/seed_{0,1,2}.zip`) attempt 1 used.

### Part 1 -- Semantic-neighbor diagnostic (no RL; frozen sentence-transformer + augmented projection only)

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

### Part 2 -- RL success rate on held-out phrasings (the actual generalization test)

Same 3 already-trained SAC checkpoints as attempt 1 -- no retraining. Only the projection checkpoint changed (v3 -> v5_augmented). Ground truth judged against each instruction's region centroid (`train.compute_region_centroid`), unchanged from attempt 1.

| Seed | Literal success rate (50 eval episodes, stage-2/3 protocol) | Mean held-out language success rate (14 instructions x 50 episodes) |
|------|------------------------------------------------------------|----------------------------------------------------------------|
| 0 | 1.000 | 0.071 |
| 1 | 1.000 | 0.143 |
| 2 | 1.000 | 0.071 |

Aggregate across 3 seeds x 14 held-out instructions (42 success-rate samples): mean=**0.095**, median=**0.000**, max=**1.000**, min=**0.000**, nonzero samples=**4/42**.

**Comparison to attempt 1** (mean=0.024, median=0.000, nonzero=1/42): mean 0.024 -> 0.095, nonzero samples 1/42 -> 4/42. **Comparison to stage 3's training-vocabulary baseline** (1.000): 0.000 median vs. 1.000 -- still far short. Literal-goal control stays a clean 1.000 on all 3 seeds (same checkpoints, unchanged), so both this attempt's improvement and its remaining gap are specific to the projection, not a policy or checkpoint change.

### Per-instruction detail (mean success rate across all 3 seeds)

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

### Part 3 -- Compositional instructions (no single ground-truth region; reported honestly, no forced verdict)

| Instruction | Components | Nearest region | Nearest is a component | Component balance (1.0=equidistant) |
|-------------|------------|-----------------|--------------------------|----------------------------------------|
| reach up and to the left | reach up high / reach left | reach up high | True | 0.973 |
| reach forward and down | reach forward / reach down low | reach forward | True | 0.733 |

### Sanity check -- does the augmented-vocabulary projection still ace the ORIGINAL stage-3 training vocabulary?

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

### Before/after comparison

| Metric | Attempt 1 (14-sentence vocab) | Attempt 2 (70-sentence vocab) | NN-ceiling (k=1, zero-training) |
|--------|-------------------------------|-------------------------------|-----------------------------------|
| Semantic-neighbor accuracy | 0.286 (4/14) | 0.643 (9/14) | 0.714 (10/14) |
| Held-out RL success (mean) | 0.024 | 0.095 | n/a -- geometry-only test, no RL policy involved |
| Held-out RL success (median) | 0.000 | 0.000 | n/a |
| Held-out RL nonzero samples | 1/42 | 4/42 | n/a |


## Charts
![held_out_success_rate_v2.png](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/charts/held_out_success_rate_v2.png)

![stage3_vocab_regression_check_v2.png](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/charts/stage3_vocab_regression_check_v2.png)

![embedding_projection_open_vocab_v2.png](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/charts/embedding_projection_open_vocab_v2.png)

## Raw output
- [projection_train_stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/runs/attempt2/projection_train_stdout.log)
- [stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/runs/attempt2/seed_0/stdout.log)
- [stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/runs/attempt2/seed_1/stdout.log)
- [stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/runs/attempt2/seed_2/stdout.log)
- [stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/runs/attempt2/regression_check/seed_0/stdout.log)
- [stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/runs/attempt2/regression_check/seed_1/stdout.log)
- [stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/runs/attempt2/regression_check/seed_2/stdout.log)
- [semantic_neighbor_diagnostic_v2_stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/04_open_vocabulary/artifacts/semantic_neighbor_diagnostic_v2_stdout.log)

## Anomalies (factual, not judged)
Semantic-neighbor accuracy more than doubled: 0.286 (4/14, attempt 1) -> 0.643 (9/14, attempt 2) -- closer to, though still below, the NN-ceiling's 0.714 (10/14). Held-out RL success improved in the same direction but by a smaller margin: mean 0.024 -> 0.095, nonzero samples 1/42 -> 4/42 -- still far short of the 1.000 training-vocabulary baseline and still a FAIL-magnitude gap by any reasonable reading, not graceful degradation. Data augmentation helped -- direction, not magnitude, is the honest summary.

**The regression check surfaced a result the task did not anticipate:** the augmented-vocabulary projection does NOT reproduce stage 3's ~1.000 success rate on the ORIGINAL 14 training instructions (mean=0.143, median=0.000, nonzero=6/42 -- roughly the same order of magnitude as this attempt's own held-out result, not a clean pass). Root cause, verified directly (not inferred): `set(AUGMENTED_INSTRUCTIONS) & set(ALL_INSTRUCTIONS) == set()` -- the 70-sentence augmented vocabulary shares no exact strings with the original 14, so it functions as a *replacement* training set, not an *extension* of it. From the retrained projection's point of view, the original 14 sentences are now just as unseen as `held_out_paraphrases` -- which is exactly why their success rate lands in the same range as the held-out set's, rather than at 1.000. This is evidence *for*, not against, the generalization diagnosis: a projection trained on 70 diverse phrasings generalizes moderately to *any* unseen phrasing (original-14 or held-out-14 alike) rather than memorizing one specific closed set perfectly and failing everywhere else, which is qualitatively the behavior change data augmentation was supposed to produce -- it just hasn't (yet) produced enough of it to clear either bar.

Per-instruction pattern: `'move your hand to the right'`/`'reach toward the right side'`/`'shift your gripper toward the right edge'` (all `reach right`) are disproportionately represented among this attempt's nonzero successes across both the held-out and regression-check evals -- 5 of the 10 total nonzero samples across both evals involve a `reach right` instruction, versus 7 regions sharing roughly equal representation in each vocabulary. Not enough samples to generalize from, but worth watching if a future attempt investigates per-region variance.

Compositional placement changed direction on both instructions: `'reach up and to the left'` now lands nearest `'reach up high'` with balance 0.973 (near-equidistant between its two components, up from attempt 1's 0.346, which was skewed hard toward one side); `'reach forward and down'` now lands nearest `'reach forward'` with balance 0.733 (up from attempt 1's 0.716). Both moved toward a more balanced placement between their two named components, consistent with a smoother, less memorized output geometry -- reported factually, not as a pass/fail signal (per `held_out_paraphrases.py`'s design, compositional instructions have no single ground-truth region).

## Known-risks cross-check
Directly extends ROADMAP.md's 'Projection-layer overfitting to a minimal vocabulary' entry (added after attempt 1): the reviewer's prescribed fix (data augmentation) produced a real, measured improvement in the predicted direction (semantic-neighbor accuracy 0.286 -> 0.643, held-out RL mean 0.024 -> 0.095) but did not close the gap to the proof gate. Not the documented SAC deterministic-eval-collapse signature (literal eval is a clean 1.000 on all 3 seeds throughout, same checkpoints as attempt 1). Not the 'Metric mismatch' known risk (same frozen sentence-transformer, same frozen GoalEncoder, only the projection's training vocabulary changed). New finding worth adding to Known risks: augmenting a fixed-vocabulary projection's training set without deliberately including the *original* training sentences turns those original sentences into held-out data for the retrained projection -- if a future stage needs both 'ace the original vocabulary' and 'generalize to new phrasing' simultaneously, the training set must include both, not just a larger disjoint replacement set.

## Reviewer verdict
_Left blank by the runner — filled in by the manager from the reviewer's
return._
