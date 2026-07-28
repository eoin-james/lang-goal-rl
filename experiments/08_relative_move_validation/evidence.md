# Stage 8: Relative-move validation — Full Evidence

**Date:** 2026-07-28 **Seeds run:** [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] **Candidates:** relative_move, budget_matched_baseline

### Proof gate (verbatim from ROADMAP.md)
> Reaches relative-move targets (multiple directions/magnitudes/switch-points) at a rate matching a budget-matched fresh baseline

### Result summary
#### Literal-goal sanity check (all seeds run, including known-collapse seeds 2/7 if present)

| Seed | Sanity success rate (literal control, full 50-step, no relative move) | Episodes |
|---|---|---|
| 0 | 1.000 | 50 |
| 1 | 1.000 | 50 |
| 2 (known SAC collapse seed) | 0.000 | 50 |
| 3 | 1.000 | 50 |
| 4 | 1.000 | 50 |
| 5 | 1.000 | 50 |
| 6 | 1.000 | 50 |
| 7 (known SAC collapse seed) | 0.400 | 50 |
| 8 | 1.000 | 50 |
| 9 | 1.000 | 50 |
| **Mean** | **0.840** | |
| **Median** | **1.000** | |

All breakdown tables below use only the healthy seeds ([0, 1, 3, 4, 5, 6, 8, 9]) --
seeds resembling the known SAC deterministic-eval collapse signature are
excluded from the mechanism verdict per ROADMAP.md's stage-1 lesson
("compare against baselines using the same seed count, judge at
median/mode not mean, and check whether a failed seed shows this exact
signature before attributing a regression to the new component").

#### Breakdown by direction (aggregated across switch_step, magnitude, healthy seeds)

| Direction | Relative-move mean | Relative-move median | Baseline mean | Baseline median | Clip rate | Episodes |
|---|---|---|---|---|---|---|
| reach back | 1.000 | 1.000 | 1.000 | 1.000 | 0.558 | 1440 |
| reach down low | 1.000 | 1.000 | 1.000 | 1.000 | 0.569 | 1440 |
| reach forward | 1.000 | 1.000 | 1.000 | 1.000 | 0.567 | 1440 |
| reach left | 0.999 | 1.000 | 1.000 | 1.000 | 0.581 | 1440 |
| reach right | 1.000 | 1.000 | 1.000 | 1.000 | 0.576 | 1440 |
| reach up high | 1.000 | 1.000 | 1.000 | 1.000 | 0.574 | 1440 |

#### Breakdown by magnitude (aggregated across switch_step, direction, healthy seeds)

| Magnitude | Relative-move mean | Relative-move median | Baseline mean | Baseline median | Clip rate | Episodes |
|---|---|---|---|---|---|---|
| clip_forcing_35cm | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | 2880 |
| medium_15cm | 1.000 | 1.000 | 1.000 | 1.000 | 0.541 | 2880 |
| small_5cm | 1.000 | 1.000 | 1.000 | 1.000 | 0.172 | 2880 |

#### Breakdown by switch_step (aggregated across direction, magnitude, healthy seeds)

| Switch step | Relative-move mean | Relative-move median | Baseline mean | Baseline median | Clip rate | Episodes |
|---|---|---|---|---|---|---|
| 10 | 1.000 | 1.000 | 1.000 | 1.000 | 0.565 | 2880 |
| 25 | 1.000 | 1.000 | 1.000 | 1.000 | 0.559 | 2880 |
| 40 | 0.999 | 1.000 | 1.000 | 1.000 | 0.589 | 2880 |

#### Overall aggregate (proof-gate comparison)

**Overall (all switch_steps x directions x magnitudes, 432 combos, 8640 episodes):** relative-move mean=1.000 median=1.000; budget-matched-baseline mean=1.000 median=1.000; overall clip rate=0.571

For completeness, the same aggregate including every seed run (collapse
seeds not excluded): **Overall (all switch_steps x directions x magnitudes, 540 combos, 10800 episodes):** relative-move mean=0.854 median=1.000; budget-matched-baseline mean=0.859 median=1.000; overall clip rate=0.577


### Charts
![sanity_check_success_rate.png](charts/sanity_check_success_rate.png)

![success_rate_by_direction.png](charts/success_rate_by_direction.png)

![success_rate_by_magnitude.png](charts/success_rate_by_magnitude.png)

![success_rate_by_switch_step.png](charts/success_rate_by_switch_step.png)

![clip_rate_by_magnitude.png](charts/clip_rate_by_magnitude.png)

### Raw output
- [stdout.log](runs/seed_0/stdout.log)
- [stdout.log](runs/seed_1/stdout.log)
- [stdout.log](runs/seed_2/stdout.log)
- [stdout.log](runs/seed_3/stdout.log)
- [stdout.log](runs/seed_4/stdout.log)
- [stdout.log](runs/seed_5/stdout.log)
- [stdout.log](runs/seed_6/stdout.log)
- [stdout.log](runs/seed_7/stdout.log)
- [stdout.log](runs/seed_8/stdout.log)
- [stdout.log](runs/seed_9/stdout.log)

### Anomalies (factual, not judged)
clip_forcing_35cm magnitude clip rate across healthy seeds: 1.000 (confirms was_clipped=True was actually forced, not merely assumed from the algebra in the magnitude's docstring).

seed 2's literal-goal sanity check scored 0.000 (< 0.8) -- resembles the known SAC deterministic-eval collapse signature (ROADMAP.md Known risks), not necessarily a stage-8 mechanism defect.; seed 7's literal-goal sanity check scored 0.400 (< 0.8) -- resembles the known SAC deterministic-eval collapse signature (ROADMAP.md Known risks), not necessarily a stage-8 mechanism defect.

### Known-risks cross-check
**Direction-sensitivity, not just distance (stage 4)**: the by-direction breakdown table above is the direct check this risk requires -- see report.md for whether any direction under- or over-performs its peers. **SAC deterministic-eval collapse (~20% of seeds, confirmed stage 1)**: checked via the sanity-check table above before trusting any relative-move result from that seed; healthy-seed breakdown tables exclude any seed matching the collapse signature. **Non-stationarity at stage 5**: not directly applicable here -- stage 8 tests a different mid-episode capability (relative move from an arbitrary achieved position, not a caller-supplied literal goal switch), though it shares the same budget-matched-baseline comparison methodology. **Region-vs-point / NN-lookup coverage density**: not applicable -- this stage uses exact literal xyz throughout, no embedding substitution engaged, deliberately isolating the relative-move mechanism from every embedding-layer confound stages 2-4 spent effort on.

### Reviewer verdict

**Verdict: PASS**

**Check 1 -- numbers, independently re-derived.** Healthy-8-seed aggregate
(8638/8640 = 0.99977, rounds to 1.000) and full-10-seed aggregate
(9222/10800 rm = 0.854, 9278/10800 baseline = 0.859) both re-summed
directly from every seed's raw per-episode JSON, not taken from the
report's own tables.

**Check 2 -- truly zero-shot.** Confirmed from the run script: loads
`experiments/01_uvfa_her_baseline/checkpoints/seed_<k>.zip` via `SAC.load()`,
never calls `.learn()` or any training method.

**Check 3 -- direction-lopsidedness, checked directly, not accepted on
faith.** Among the 7 seeds scoring a perfect 1.000, all 6 directions are
exactly 1.000 -- zero variation. Seed 3's single sub-1.0 point brings
"reach left" to 0.999 (2 failed episodes out of 1440 for that direction) --
not a lopsidedness pattern, a near-isolated stochastic edge case.

**Check 4 -- the one sub-1.0 data point (seed 3, switch_step=40, reach
left, clip-forcing).** Verified genuinely isolated: the identical combo
scores 1.000 on all 7 other healthy seeds; seed 3 itself scores 1.000 on
all 53 of its other combos. The two failing episodes are specific,
non-recurring instances, not a systematic gap. "One data point, not a
pattern" holds up under direct inspection.

**Check 5 -- judged against the correct (clipped) target.** Confirmed in
code: `_run_goal_phase` receives `resolved_target` (already clipped);
`rollout_fresh_with_budget` receives the identical `resolved_target_xyz`;
`info["is_success"]` compares against `env.unwrapped.goal`, set to the
resolved target in both conditions. The clip-forcing bucket's near-1.000
result is genuine, not an artifact of judging against something
unreachable.

**Check 6 -- seeds 2/7 framing.** Justified by the evidence: both
conditions (relative-move and baseline) degrade proportionally on these
two seeds (seed 7: rm=0.510 vs bl=0.552, a near-identical gap), consistent
with the SAC deterministic-eval-collapse signature documented since stage
1 and confirmed present in these exact two seeds at every prior stage this
project has checked them.

**Check 7 -- known-risks cross-check.** Direction-sensitivity (stage 4):
NOT confirmed here -- all 6 directions effectively identical. SAC collapse:
confirmed present, handled consistently with every prior stage. NN-lookup
coverage and "live" ambiguity: correctly not applicable, no embedding
layer engaged in this stage.

**Check 8 -- does the unresolved Stage 7 sign-off affect this verdict?**
No, correctly out of scope. This stage's proof gate is about whether the
mechanism reaches a relative-move target computed from an arbitrary
achieved position -- not whether "reach left" is correctly labeled in
camera-frame terms. The injectable `direction_vectors` parameter design
cleanly separates mechanism correctness (settled here) from label
correctness (still pending).

**Recommendation to manager:** Mark Done in `PHASE2_ROADMAP.md`. Flag for
downstream: this is a clean, near-ceiling result on FetchReach-v4's
literal-xyz task, consistent with every prior stage's informativeness
ceiling (an oracle solves this task in ~3-5 steps) -- a single relative
move apparently isn't hard for this policy. Stage 9's waypoint chaining is
the first real test of whether accumulated imprecision over multiple
sequential moves degrades, since one move alone clearly doesn't.
