# Stage 3: Frozen language embedding -> goal space (Attempt 3: fixed-centroid regression retest)
**Date:** 2026-07-26 **Seeds run:** [0, 1, 2] **Candidates:** 1 (locked-in)

## Proof gate (verbatim from ROADMAP.md)
> Success rate on language goals ~ stage-2 baseline; projection doesn't collapse distinct instructions to one point.

## Result summary
### Projection retraining (fixed-centroid regression, no InfoNCE term)

Full readout saved to `artifacts/projection_train_stdout_v3.log`. Loss (plain MSE to each instruction's precomputed-once true region centroid) dropped from mean=0.0020 (first 20 steps) to mean=0.0000 (last 20 steps) over 2000 steps -- the projection converges to match its fixed target almost exactly, as expected for a closed-form regression target (see `test_trained_projection_output_matches_fixed_target_closely` in `tests/lang_goal_rl/test_language_goal_projection.py`).

### Fail-fast norm-range check (run immediately after retraining, before any RL eval)

Full readout saved to `artifacts/norm_range_check_v3.log`. `PASSED=False` -- **8 of 14 instructions fell outside the 2x reference band** (mean=0.0393, bounds=[0.0196, 0.0786]), all 6 directional non-center regions except 'reach forward' and 'reach back', with norms as low as 0.0163 ('reach down low' / 'move your hand downward'). This is **not** the attempt-1/2 defect recurring: the projection matches its fixed target almost exactly (see above), so these low norms are the *true* per-region centroid norms under the frozen `GoalEncoder` -- some regions (the ones near the edges/corners of the measured box) genuinely have smaller-magnitude embeddings than the box-wide average this check's reference distribution is built from. The check's own module docstring flags it as no longer load-bearing for correctness once regression-to-true-target is used (an in-range norm becomes 'essentially automatic' only when the *true* target norms cluster near the box-wide mean, which turned out not to hold for every region here) -- treated as a factual finding about region geometry, not a fail-fast stop, and the RL eval below was run regardless per that reasoning.

### Collapse re-check (fixed projection, run before the RL eval)

Full readout saved to `artifacts/collapse_diagnostic_v3_stdout.log`. `min_cross_region_pairwise_distance / collapse_epsilon` = **9.70x** (threshold is 1x). `is_collapsed` = **False**. Lower than attempt 2's 24.68x (attempt 3's true-centroid targets sit closer together in absolute terms for some region pairs than attempt 2's noisy per-step-estimated targets did) but still well clear of the collapse threshold.

### Direction-alignment vs. success-rate correlation (attempt 2's open question, resolved with existing data -- no new RL runs)

Full readout saved to `artifacts/direction_alignment_correlation.log` (`correlate_direction_alignment.py`). Computed `measure_instruction_direction_alignment` against **attempt 2's** projection checkpoint (loaded, not retrained) and correlated the resulting per-instruction cosine similarities against attempt 2's already-recorded per-instruction success rates (mean across all 3 seeds x 50 episodes, parsed from `runs_v2/seed_*/stdout.log`): **Pearson r = 0.3453** (n=14 instructions). 'center' region instructions had a higher mean cosine similarity (0.9217) than non-center instructions (0.8388), consistent with attempt 2's report noting 'center' scored highest -- but r=0.345 is only a weak-to-moderate positive correlation, not the value you'd expect if direction alignment were the dominant explanation for attempt 2's per-instruction success-rate variation. This settles the open question as: **directional accuracy against the true centroid explains some, but clearly not most, of attempt 2's per-instruction variation** -- a FetchReach-geometry confound (e.g. some regions' true goals sitting closer to the arm's reset position, independent of embedding quality) is a plausible co-factor and was not ruled out by this analysis.

### Literal-goal protocol reproduction (unchanged from attempts 1/2 -- same 3 saved SAC checkpoints, no retraining)

| Seed | Literal success rate (50 eval episodes, stage-2 protocol) |
|------|------------------------------------------------------------|
| 0 | 1.000 |
| 1 | 1.000 |
| 2 | 1.000 |

Confirms the 3 checkpoints are untouched and still reproduce stage 2's baseline exactly -- this retest changes only the projection checkpoint, nothing about the trained SAC policies.

### Language-goal substitution success rate (the actual attempt-3 retest)

| Seed | Mean success rate across 14 instructions (50 episodes each) |
|------|----------------------------------------------------------------|
| 0 | 0.170 |
| 1 | 0.143 |
| 2 | 0.157 |

Aggregate across all 3 seeds x 14 instructions (42 success-rate samples): mean=**0.157**, median=**0.120**, max=**0.440**, min=**0.020**.

**Comparison to stage-2 baseline** (mean=median=mode=1.000 over 10 seeds, per ROADMAP Known risks' judge-at-median/mode guidance): 0.120 median vs. 1.000 -- the gate's "~ stage-2 baseline" bar is still not met.

**Comparison to this stage's attempt-2 result** (mean=0.069 across the identical 3-seeds x 14-instructions x 50-episodes protocol): mean improved 0.069 -> 0.157, a ~2.3x increase in absolute terms. Regressing directly to the true, precomputed-once centroid (no noisy per-step InfoNCE target) produced a real, further improvement over attempt 2 -- but the result is still well short of the proof gate's bar.

### Per-instruction detail (seed 0)

| Instruction | Region | Success rate |
|-------------|--------|---------------|
| move your hand to the center | center | 0.400 |
| keep the gripper in the middle of the workspace | center | 0.440 |
| move your hand forward | reach forward | 0.080 |
| reach out in front of you | reach forward | 0.220 |
| pull your hand back | reach back | 0.080 |
| reach backward toward yourself | reach back | 0.160 |
| move your hand to the left | reach left | 0.200 |
| reach toward the left side | reach left | 0.080 |
| move your hand to the right | reach right | 0.120 |
| reach toward the right side | reach right | 0.100 |
| reach up high | reach up high | 0.120 |
| move your hand upward | reach up high | 0.120 |
| reach down low | reach down low | 0.120 |
| move your hand downward | reach down low | 0.140 |


## Charts
![language_goal_success_rate_v3.png](/Users/eoinmca/Projects/lang-goal-rl/experiments/03_language_goal_projection/charts/language_goal_success_rate_v3.png)

![embedding_projection_v3.png](/Users/eoinmca/Projects/lang-goal-rl/experiments/03_language_goal_projection/charts/embedding_projection_v3.png)

## Raw output
- [stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/03_language_goal_projection/runs_v3/seed_0/stdout.log)
- [stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/03_language_goal_projection/runs_v3/seed_1/stdout.log)
- [stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/03_language_goal_projection/runs_v3/seed_2/stdout.log)
- [projection_train_stdout_v3.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/03_language_goal_projection/artifacts/projection_train_stdout_v3.log)
- [norm_range_check_v3.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/03_language_goal_projection/artifacts/norm_range_check_v3.log)
- [collapse_diagnostic_v3_stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/03_language_goal_projection/artifacts/collapse_diagnostic_v3_stdout.log)
- [direction_alignment_correlation.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/03_language_goal_projection/artifacts/direction_alignment_correlation.log)

## Anomalies (factual, not judged)
The fail-fast norm-range check FAILED this attempt (8/14 instructions out of the 2x reference band) even though the projection's training loss converged almost exactly to its fixed target (mean loss 0.0000 over the last 20 steps) -- this is the check's own documented limitation (module docstring: 'no longer load-bearing for correctness... an in-range norm is essentially automatic' only holds if the true per-region centroid norms cluster near the box-wide reference mean, which several directional regions' true centroids do not) firing on real region geometry, not a training defect. Verified this is not a repeat of attempt 1's bug: attempt 1's projection was 5-10x *outside* the range in the *high* direction (0.25-0.41 vs. reference mean 0.039) because the InfoNCE loss had no scale term at all; attempt 3's out-of-range instructions are all *slightly below* the lower bound (0.0163-0.0195 vs. lower bound 0.0196) because those regions' true target norms are genuinely smaller than the box-wide average -- a fundamentally different, much smaller, and structurally-explained deviation. Proceeded to the RL eval regardless, per the module docstring's own framing of this check as a heuristic proxy, superseded by direct-target-matching for correctness.

The collapse check still passes (9.70x margin, `artifacts/collapse_diagnostic_v3_stdout.log`), so distinct instructions are still not collapsing to one point.

Aggregate language-goal success rate improved again: mean 0.157 / median 0.120, up from attempt 2's 0.069 mean / 0.040 median -- a ~2.3x further improvement. Still nowhere near the required ~1.000. Literal-goal control is unchanged and still a clean 1.000 on all 3 seeds (same checkpoints, no retraining), so this remains specific to the projection, not a policy regression.

The direction-alignment correlation analysis (Pearson r=0.3453 against attempt 2's per-instruction success rates) resolves the open question from attempt 2's reviewer: directional accuracy correlates only weakly-to-moderately with success, so the 'center does better' pattern observed in both attempt 2 and attempt 3 is very likely a mix of genuine directional-accuracy effect *and* a FetchReach-geometry confound (goals near the workspace center may simply be easier for this policy to reach, independent of how accurate the goal embedding is) -- not purely explained by either factor alone. This matters for interpreting attempt 3's residual gap too: even with near-perfect target-matching (attempt 3's training loss), the substitution eval still tops out at a 0.44 max per-instruction success rate, well below the 1.000 literal baseline, which is more consistent with a remaining representational or distributional gap between 'true region centroid under GoalEncoder' and 'what the policy actually needs to see to succeed' than with any residual direction/scale defect in the projection itself.

Per the tiered-seed strategy, this 3-seed result is uniform enough (per-seed means 0.143-0.170) that scaling to the full 10-seed budget would not change the qualitative picture -- skipped for the same reason attempts 1 and 2 skipped it.

## Known-risks cross-check
Same framing as attempts 1 and 2: this result is not the documented SAC deterministic-eval-collapse signature (literal eval is still a clean 1.000 on all 3 seeds, using the same unretrained checkpoints), and it is not quite the "Metric mismatch" known risk either (the projection regresses into the frozen `GoalEncoder`'s space via direct MSE to a precomputed centroid, not a raw sentence-embedding distance reward). Attempt 2's diagnosed defect (InfoNCE's separation term having no pressure to pull direction toward the true centroid) is now directly addressed by construction (the loss *is* distance-to-true-centroid) -- and the result improved again, consistent with that diagnosis being at least partially correct. But the residual gap (max 0.44 per-instruction, mean 0.157, vs. required ~1.000) persists even with near-exact target matching, which is new evidence against 'projection accuracy' being the whole story -- something else in the true-centroid-to-policy pathway (e.g. a region's true centroid embedding not actually being what the policy needs for goals sampled elsewhere in that region, since the centroid is a single point but each eval episode samples a random point within the region) may be the next thing to investigate. Recording this as a new, distinct residual gap rather than folding it into an existing entry, for the same reason attempts 1 and 2 declined to force-fit their findings there.

## Reviewer verdict
_Left blank by the runner — filled in by the manager from the reviewer's
return._
