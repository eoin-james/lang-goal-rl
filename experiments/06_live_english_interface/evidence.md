# Stage 6: Live English interface — Full Evidence


**Date:** 2026-07-27 **Seeds run:** [0, 1, 2] **Candidates:** set_a_no_switch_control, set_a_switch, set_b_no_switch_control, set_b_switch

### Proof gate (verbatim from ROADMAP.md)
> End-to-end demo across ad-hoc live phrasings: task success + time-to-redirect.

### Result summary
#### Literal-goal sanity check (reused stage-3 checkpoints, no new training)

| Seed | Literal sanity success rate | Episodes |
|---|---|---|
| 0 | 1.000 | 50 |
| 1 | 1.000 | 50 |
| 2 | 1.000 | 50 |
| **Mean** | **1.000** | |
| **Median** | **1.000** | |

#### Set A no-switch control (stage 4's 14 held-out paraphrases -- also this experiment's sanity cross-check)

| Instruction | Region | Mean success rate (3 seeds x 50 episodes) |
|---|---|---|
| settle into the middle of the workspace | center | 1.000 |
| return your hand to a neutral position | center | 0.000 |
| push your arm out in front of you | reach forward | 0.000 |
| extend forward away from your body | reach forward | 0.000 |
| draw your hand back toward yourself | reach back | 1.000 |
| retreat away from the front of the workspace | reach back | 0.000 |
| swing your arm over to the left | reach left | 1.000 |
| shift your gripper toward the left edge | reach left | 1.000 |
| swing your arm over to the right | reach right | 1.000 |
| shift your gripper toward the right edge | reach right | 0.000 |
| raise your arm as high as it will go | reach up high | 0.000 |
| extend upward toward the ceiling | reach up high | 1.000 |
| lower your arm toward the floor | reach down low | 1.000 |
| drop your gripper down low | reach down low | 1.000 |
| **Aggregate (42 samples)** | | **mean=0.571 median=1.000** |

**Cross-check against stage 4:** Set-A no-switch-control mean=0.571,
median=1.000 vs. stage 4's already-measured 0.571 mean / 1.000 median
(identical k=1 NN-lookup mechanism, via `LiveGoalController` instead of
stage 4's script calling `nearest_neighbor_projection` directly, and with a
different eval-seed range so this is not a byte-identical rerun). The
per-instruction pattern matches stage 4's own table exactly, all 14
instructions -- **MATCHES. Harness wiring confirmed correct before trusting
Set B.**

#### Set B no-switch control (7 brand-new phrasings, never used anywhere in this project before)

| Instruction | Region | Mean success rate (3 seeds x 50 episodes) |
|---|---|---|
| keep the robotic hand hovering exactly at the workspace's midpoint | center | 1.000 |
| push the end effector forward, away from the robot's base | reach forward | 0.000 |
| bring the arm back in, closer to the robot's chassis | reach back | 1.000 |
| carry the hand across the workspace toward its left boundary | reach left | 1.000 |
| swing the arm over so the gripper reaches the right-hand side of the space | reach right | 1.000 |
| send the arm climbing toward the highest point it can reach | reach up high | 1.000 |
| let the arm descend toward the lowest point it can reach | reach down low | 1.000 |
| **Aggregate (21 samples)** | | **mean=0.857 median=1.000** |

#### Set A live mid-episode switch test (proof-gate metric)

| Pair | instr1 -> instr2 | Task success rate (3 seeds) | Redirected (n/3 seeds) | Time-to-redirect per seed that redirected |
|---|---|---|---|---|
| 0 | "settle into the middle of the workspace" -> "shift your gripper toward the left edge" | 3/3 | 3/3 | 2, 2, 4 |
| 1 | "return your hand to a neutral position" -> "swing your arm over to the right" | 3/3 | 3/3 | 3, 3, 4 |
| 2 | "push your arm out in front of you" -> "extend upward toward the ceiling" | 3/3 | 3/3 | 4, 3, 4 |
| 3 | "extend forward away from your body" -> "drop your gripper down low" | 2/3 | 2/3 | 6, 7 |
| 4 | "draw your hand back toward yourself" -> "settle into the middle of the workspace" | 3/3 | 3/3 | 3, 3, 3 |
| 5 | "retreat away from the front of the workspace" -> "return your hand to a neutral position" | 0/3 | 0/3 | did not redirect (any seed) |
| 6 | "swing your arm over to the left" -> "extend upward toward the ceiling" | 3/3 | 3/3 | 3, 4, 3 |
| 7 | "shift your gripper toward the left edge" -> "drop your gripper down low" | 3/3 | 3/3 | 3, 4, 3 |
| 8 | "swing your arm over to the right" -> "push your arm out in front of you" | 0/3 | 0/3 | did not redirect (any seed) |
| 9 | "shift your gripper toward the right edge" -> "extend forward away from your body" | 0/3 | 0/3 | did not redirect (any seed) |
| 10 | "raise your arm as high as it will go" -> "lower your arm toward the floor" | 3/3 | 3/3 | 1, 1, 1 |
| 11 | "extend upward toward the ceiling" -> "retreat away from the front of the workspace" | 0/3 | 0/3 | did not redirect (any seed) |
| 12 | "lower your arm toward the floor" -> "extend forward away from your body" | 0/3 | 0/3 | did not redirect (any seed) |
| 13 | "drop your gripper down low" -> "shift your gripper toward the right edge" | 0/3 | 0/3 | did not redirect (any seed) |

**Set A aggregate:** task_success_rate=0.548 redirect_success_rate=0.548 over 42 episodes (14 pairs x 3 seeds)

**Set A time-to-redirect distribution** (computed only over episodes that
redirected -- "did not redirect" episodes are counted separately above, not
folded into this distribution): 23/42 episodes redirected. mean=3.22
median=3.0 min=1 max=7. All values:
`[1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 6, 7]`

#### Set B live mid-episode switch test (proof-gate metric)

| Pair | instr1 -> instr2 | Task success rate (3 seeds) | Redirected (n/3 seeds) | Time-to-redirect per seed that redirected |
|---|---|---|---|---|
| 0 | "keep the robotic hand hovering exactly at the workspace's midpoint" -> "let the arm descend toward the lowest point it can reach" | 3/3 | 3/3 | 2, 3, 2 |
| 1 | "push the end effector forward, away from the robot's base" -> "bring the arm back in, closer to the robot's chassis" | 3/3 | 3/3 | 1, 1, 1 |
| 2 | "bring the arm back in, closer to the robot's chassis" -> "keep the robotic hand hovering exactly at the workspace's midpoint" | 3/3 | 3/3 | 3, 3, 3 |
| 3 | "carry the hand across the workspace toward its left boundary" -> "push the end effector forward, away from the robot's base" | 0/3 | 0/3 | did not redirect (any seed) |
| 4 | "swing the arm over so the gripper reaches the right-hand side of the space" -> "bring the arm back in, closer to the robot's chassis" | 3/3 | 3/3 | 4, 3, 4 |
| 5 | "send the arm climbing toward the highest point it can reach" -> "swing the arm over so the gripper reaches the right-hand side of the space" | 3/3 | 3/3 | 3, 4, 3 |
| 6 | "let the arm descend toward the lowest point it can reach" -> "bring the arm back in, closer to the robot's chassis" | 3/3 | 3/3 | 4, 3, 3 |

**Set B aggregate:** task_success_rate=0.857 redirect_success_rate=0.857 over 21 episodes (7 pairs x 3 seeds)

**Set B time-to-redirect distribution** (computed only over episodes that
redirected): 18/21 episodes redirected. mean=2.78 median=3.0 min=1 max=4.
All values: `[1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4]`

### Charts
![literal_sanity_success_rate.png](charts/literal_sanity_success_rate.png)

![control_vs_switch_success_rate.png](charts/control_vs_switch_success_rate.png)

![time_to_redirect_distribution.png](charts/time_to_redirect_distribution.png)

### Raw output
- [stdout.log](runs/seed_0/stdout.log)
- [stdout.log](runs/seed_1/stdout.log)
- [stdout.log](runs/seed_2/stdout.log)

### Anomalies (factual, not judged)
No literal-sanity-check collapse observed on any seed (all 3 seeds a clean
1.000, matching stage 3/4's checkpoints unchanged). This experiment's
`rollout_with_goal_switch_timed` (in `live_regoal_eval.py`) is a local,
instrumented duplicate of `midepisode_regoal.rollout_with_goal_switch` --
the reusable function only tracks the *final* step's success, not the
*first* post-switch success step, which stage 6's time-to-redirect metric
needs. Flagged as a candidate for promotion into `midepisode_regoal.py` if
a future stage needs the same timing data (also logged in
`.claude/findings.md`). Separately: in this run, every switch episode's
final-step task-success outcome matched its ever-redirected outcome
exactly -- `task_success_rate == redirect_success_rate` for both Set A
(0.548) and Set B (0.857). Once an episode reached the new target it
stayed there through episode end in every case observed here; this is a
factual observation from this specific `SWITCH_STEP=20`/50-step
configuration, not assumed to generalize to other switch points or longer
post-switch budgets.

### Known-risks cross-check
**Nearest-neighbor lookup's generalization ceiling is bounded by
reference-vocabulary coverage density**: this is exactly what Set B tests
directly (7 brand-new phrasings never in the 84-sentence reference) -- Set
B's no-switch-control aggregate (0.857 mean/1.000 median) did not show a
coverage-gap collapse on this particular set of 7 new phrasings, though 7
sentences is too small a sample to conclude the risk is resolved in
general. **Non-stationarity / embedding noise interacting with a goal-swap
(flagged explicitly in ROADMAP.md as untested going into stage 6)**: this
experiment is the first direct measurement -- the switch-vs-no-switch-control
comparison chart above shows switch success tracking the same-set
no-switch control closely for both Set A (0.548 vs. 0.571) and Set B (0.857
vs. 0.857), i.e. no measurable degradation specific to switching under the
live language pipeline, at this one switch point and episode budget.
**Region-vs-point ground truth**: applied from the start here
(`compute_region_centroid`, never a resampled point), per the lesson from
stage 3. **SAC deterministic-eval collapse**: checked via the literal
sanity table above before trusting any result -- clean 1.000 on all 3
seeds.

### Reviewer verdict

**Verdict: PASS (with mandatory scope qualifications below)**

**Check 1 -- numbers, independently re-derived.** Every Set A/B no-switch
and switch number matches the raw per-seed logs exactly, including the
full time-to-redirect distributions (re-sorted and re-averaged by hand,
not just re-read off the table). The single most important check: Set A's
no-switch control reproduces stage 4 attempt 4's per-instruction table
**exactly, on all 14 instructions, not just in aggregate** -- this
confirms `LiveGoalController` wires the same k=1 lookup mechanism
correctly, not just coincidentally similar aggregate numbers.

**Check 2 -- is `task_success == redirect_success` a real finding or a
predictable artifact?** The latter. "Task success" (final-step
`is_success`) and "redirect success" (first post-switch step
`is_success` flips true) are genuinely different definitions, but for
FetchReach's simple dynamics plus a converged, goal-holding policy with
30 post-switch steps to spare, once the policy reaches a goal it has no
reason to drift off it again. This is a predictable mechanical consequence
of this specific setup (easy task, short episode, converged policy) --
not evidence of "no overshoot" in any general sense, and not something
to carry forward as a property of the mechanism itself.

**Check 3 -- does switch success really track the no-switch control, or
is "closely" doing a lot of work?** Checked the actual statistics rather
than eyeballing "close enough." Set A's switch success is almost
perfectly predicted by whether the SECOND instruction alone (no-switch)
succeeds: of 42 episodes, exactly 1 breaks that pattern (a marginal,
late-redirecting case). A one-sided exact binomial test against "switch
equals control" gives p=0.71 -- nowhere near significant; the observed 1/42
gap is comfortably explained by noise. Set B matches its control exactly
(0/21 gap). Conclusion: the switch mechanism itself adds no detectable
cost: success is governed almost entirely by whether the NN lookup maps
each instruction to the right region, not by anything about switching
mid-episode.

**Check 4 -- does the proof gate's literal wording pass?** Yes. "End-to-end
demo across ad-hoc live phrasings: task success + time-to-redirect" --
the full pipeline runs live, both named metrics are measured and reported
for both sets, and Set B (7 genuinely new phrasings, verified disjoint
from every vocabulary used anywhere else in this project) is the real
test of "ad-hoc," with Set A serving as a calibration cross-check against
already-known numbers. The gate asks for a demo with these two metrics
measured, not a specific numeric bar -- that's what's delivered.

**Check 5 -- known-risks cross-check, the important one for a capstone
stage.** The "NN-lookup reference-coverage-density" risk (logged after
stage 4) is **NOT contradicted by this result, and should not be read as
resolved.** Set B's 6/7 success on 7 brand-new sentences is consistent
with that risk being real but not severely triggered on this particular
small draw -- 7 sentences cannot characterize an 84-sentence reference
set's coverage density at the scale "ad-hoc" implies, and the one Set-B
failure ("extend forward") fails in exactly the coverage-gap mode that
risk predicted. Stage 5's scope caveat (embedding noise interacting with
a live goal-swap was untested there) is **partially resolved**: this
result shows no measurable degradation from combining live language
embeddings with mid-episode switching, under this one switch point/budget/
environment -- a real, positive finding, but narrow in scope, not a
general robustness guarantee.

**Check 6 -- honest capstone framing.** This is genuinely the mechanism
the whole project set out to prove: type English, get a continuous-goal
robot that re-targets mid-episode, live. It works, end to end. What it
does NOT prove: that the 84-sentence reference vocabulary generalizes to
arbitrary open-ended phrasing at scale (7 test sentences is a demo, not a
statistically powered claim), or that this holds beyond FetchReach's
easiest-task, short-episode, 3-seed scope. Both should be stated plainly
in ROADMAP and STATUS, not left implicit.

**Recommendation to manager:** Mark Done in ROADMAP, with an explicit
scope statement in the Status text (mechanism proven end-to-end; coverage-
density and cross-task generalization remain open, not claimed) rather
than a bare "Done."
