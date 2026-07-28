# Stage 9: Waypoint following — Full Evidence

**Date:** 2026-07-28 **Seeds run:** [0] (single checkpoint, zero-shot -- see "Methodology note" below) **Candidates:** literal/tight, literal/generous, relative/tight, relative/generous

### Proof gate (verbatim from ROADMAP.md)
> N=2 reduces exactly to stage 5's `rollout_with_goal_switch` result (regression test); N=3-5 chains don't show compounding degradation.

### Regression test: N=2 reduces to stage 5's own mechanism

`uv run pytest tests/lang_goal_rl/test_waypoint_following.py -v` -- **20 passed**, including both
`TestEquivalenceWithMidepisodeRegoal` tests
(`test_n2_waypoints_matches_goal_switch_success_and_step_count`,
`test_n2_waypoints_matches_goal_switch_on_guaranteed_success_case`), which construct an
`N=2` `rollout_with_waypoints` call and its literal `rollout_with_goal_switch`
equivalent from the same stub model/seed/goals and assert identical
`n_steps`, `leg_boundaries`, and success outcome. This ran clean, not
skipped -- full output in
[runs/n2_equivalence_regression_test.log](runs/n2_equivalence_regression_test.log).
That equivalence is settled going into this experiment; everything below is
new evidence about N=3/4/5 chains, which the regression test doesn't cover.

### Methodology note: single checkpoint, tiered episode counts

Per the task brief ("same checkpoint stage 8 uses, for consistency across
Phase 2a"), this experiment reuses one literal-xyz checkpoint
(`experiments/01_uvfa_her_baseline/checkpoints/seed_0.zip`) zero-shot,
rather than the usual multi-model-seed tiering (CONTRACTS.md) -- stage 9 is
testing one mechanism's behavior on one already-validated policy, not
variance across differently-trained policies. In its place, "tiered for
speed" is applied along the episode-count axis: a 15-episodes/condition
pass ran first (`runs/tier1_results.json`), then a 50-episodes/condition
final pass (`runs/final_results.json`) once the smaller pass showed no
degenerate collapse. Both are kept as raw output; see Anomalies for the
tier1-vs-final consistency check.

### Result summary
#### Checkpoint sanity check
literal-goal control, default 50-step episode, no waypoint chain: **1.000** over 50 episodes (checkpoint: `experiments/01_uvfa_her_baseline/checkpoints/seed_0.zip`)

#### literal sequences, tight budget (50 episodes/condition)

| Chain length | Per-leg chain success rate (leg 1..N) | Per-leg baseline success rate (leg 1..N) | Whole-chain success rate | Episodes |
|---|---|---|---|---|
| N=2 | [1.000, 1.000] | [1.000, 1.000] | 1.000 | 50 |
| N=3 | [1.000, 0.980, 1.000] | [1.000, 1.000, 1.000] | 0.980 | 50 |
| N=5 | [1.000, 1.000, 1.000, 0.960, 1.000] | [1.000, 1.000, 1.000, 1.000, 1.000] | 0.960 | 50 |

#### literal sequences, generous budget (50 episodes/condition)

| Chain length | Per-leg chain success rate (leg 1..N) | Per-leg baseline success rate (leg 1..N) | Whole-chain success rate | Episodes |
|---|---|---|---|---|
| N=2 | [1.000, 1.000] | [1.000, 1.000] | 1.000 | 50 |
| N=3 | [1.000, 1.000, 1.000] | [1.000, 1.000, 1.000] | 1.000 | 50 |
| N=5 | [1.000, 1.000, 1.000, 1.000, 1.000] | [1.000, 1.000, 1.000, 1.000, 1.000] | 1.000 | 50 |

#### relative sequences, tight budget (50 episodes/condition)

| Chain length | Per-leg chain success rate (leg 1..N) | Per-leg baseline success rate (leg 1..N) | Whole-chain success rate | Episodes |
|---|---|---|---|---|
| N=2 | [1.000, 1.000] | [1.000, 1.000] | 1.000 | 50 |
| N=3 | [1.000, 1.000, 0.980] | [1.000, 1.000, 1.000] | 0.980 | 50 |
| N=5 | [1.000, 1.000, 0.980, 0.980, 0.980] | [1.000, 1.000, 1.000, 1.000, 1.000] | 0.940 | 50 |

#### relative sequences, generous budget (50 episodes/condition)

| Chain length | Per-leg chain success rate (leg 1..N) | Per-leg baseline success rate (leg 1..N) | Whole-chain success rate | Episodes |
|---|---|---|---|---|
| N=2 | [1.000, 1.000] | [1.000, 1.000] | 1.000 | 50 |
| N=3 | [1.000, 1.000, 1.000] | [1.000, 1.000, 1.000] | 1.000 | 50 |
| N=5 | [1.000, 1.000, 1.000, 1.000, 1.000] | [1.000, 1.000, 1.000, 1.000, 1.000] | 1.000 | 50 |


### Charts
![whole_chain_success_vs_length.png](charts/whole_chain_success_vs_length.png)

![per_leg_literal_tight.png](charts/per_leg_literal_tight.png)

![per_leg_literal_generous.png](charts/per_leg_literal_generous.png)

![per_leg_relative_tight.png](charts/per_leg_relative_tight.png)

![per_leg_relative_generous.png](charts/per_leg_relative_generous.png)

### Raw output
- [n2_equivalence_regression_test.log](runs/n2_equivalence_regression_test.log)
- [tier1_stdout.log](runs/tier1_stdout.log)
- [tier1_results.json](runs/tier1_results.json)
- [final_stdout.log](runs/final_stdout.log)
- [final_results.json](runs/final_results.json)

### Anomalies (factual, not judged)
Tier-1 (15 episodes/condition) and the final tier (50 episodes/condition) are numerically consistent wherever both have enough resolution to compare (e.g. relative/N=5/tight: 14/15=0.933 vs. 47/50=0.940) -- no sign of a fluke result at the smaller tier. Every generous-budget condition (both sequence kinds, all 3 chain lengths) scored a clean 1.000 on every leg, with zero baseline failures anywhere in the whole experiment -- this checkpoint sits at an oracle-solvable ceiling for this task at this budget, the same informativeness limit ROADMAP.md already documents for stages 1/3/5's 1.000 scores (not a new finding, just re-observed here). The only non-1.000 results appear at the tight budget, and only for chain lengths >= 3; see the per-leg tables and whole_chain_success_vs_length.png. Every non-1.000 condition's individual failing episodes, inspected for whether a miss at one leg drags down subsequent legs in the *same* episode (compounding) or is an isolated single-leg miss that the next leg recovers from cleanly:
- literal/N=3/tight, episode 43: failed leg(s) [2] of 3 -- isolated (1 leg)
- literal/N=5/tight, episode 34: failed leg(s) [4] of 5 -- isolated (1 leg)
- literal/N=5/tight, episode 45: failed leg(s) [4] of 5 -- isolated (1 leg)
- relative/N=3/tight, episode 33: failed leg(s) [3] of 3 -- isolated (1 leg)
- relative/N=5/tight, episode 7: failed leg(s) [4] of 5 -- isolated (1 leg)
- relative/N=5/tight, episode 31: failed leg(s) [5] of 5 -- isolated (1 leg)
- relative/N=5/tight, episode 40: failed leg(s) [3] of 5 -- isolated (1 leg)

### Known-risks cross-check
**Direction-sensitivity, not just distance (stage 4, carried into Phase 2a's known risks)**: not directly probed here -- this stage's relative-move sequences use a fixed 0.15m step in a randomly chosen direction per leg (not systematically varied per direction), so a direction-specific failure mode would not necessarily surface in this result; stage 8's own relative-move validation is the right place to check that per-direction. **Oracle-solvable ceiling (ROADMAP.md, stages 1/3/5)**: directly re-observed here -- every generous-budget condition and most tight-budget conditions hit 1.000, so this result's informativeness is concentrated in the small number of tight-budget, longer-chain conditions that show any variance at all; a harder task or a smaller tight-budget value would give this test more room to actually fail if the mechanism were going to.

### Reviewer verdict
_Left blank by the runner — filled in by the manager from the reviewer's
return._
