# Stage 8: Relative-move validation

## In plain English
This stage tests a new capability: "move 10cm forward from wherever the
robot's hand actually is right now," not "go to this fixed coordinate."
Every earlier stage's mid-episode redirect (stage 5) handed the policy a
pre-chosen destination; stage 8 instead computes the destination live, from
the robot's real position at the moment the command lands, then clips it
into the workspace if the move would overshoot. To make sure "wherever the
robot actually is" was genuinely varied and not always near the same spot,
each trial started the robot toward a randomly-chosen destination first,
then issued the relative move at an early, middle, or late point in that
approach -- so the position the move computes from is different every time,
by design. Across 8 independently-trained agents (2 more showed a
previously-documented training glitch, called out honestly below, not
folded into the headline numbers), every one of 6 directions, 3 move
distances (including a deliberately oversized one meant to force the move
to hit the workspace's edge), and 3 timing points reached its computed
target about as reliably as a policy given that same target from a full
fresh start with the same remaining time. The forced-oversized move did
overshoot the workspace edge on every single trial, confirming the
edge-clipping safeguard was actually exercised, not just assumed to work.

## Result
Measured, not adjudicated -- see "Reviewer verdict" in Full evidence below
for the actual pass/fail call. Across the 8 healthy seeds (2 more showed
the already-known SAC training-collapse signature -- see below), the
relative-move success rate and the budget-matched fresh-start baseline
success rate were statistically indistinguishable everywhere it was
checked: by direction (all 6 near-identical, 0.999-1.000, no
direction stood out as harder), by magnitude (5cm/15cm/oversized all
0.999-1.000), and by switch timing (early/mid/late, 0.999-1.000). The
deliberately oversized move forced the edge-clipping safeguard to trigger
on 100% of trials, as intended -- confirming the safeguard is real, not
just present in the code and never exercised. The two seeds that showed
the pre-existing SAC-collapse signature (documented since stage 1, unrelated
to this stage's mechanism) scored low on both the relative-move condition
and the fresh-start baseline by a similar amount each -- consistent with
"the checkpoint itself is broken," not "the relative-move mechanism failed
where a fresh start would have succeeded."

![success_rate_by_direction.png](charts/success_rate_by_direction.png)

## How this was tested
8 previously-trained SAC policies (stage 1's plain, no-language checkpoints,
zero-shot, no new training -- excluding seeds 2 and 7, which show a
documented training-time instability unrelated to this stage) were each run
through 54 combinations of switch timing (step 10, 25, or 40 out of a
50-step episode), direction (all 6 of `reach forward/back/left/right/up
high/down low`), and move distance (5cm, matching PHASES.md's own "move
left 5cm" example; 15cm, a bigger stress test; and 35cm, deliberately larger
than the whole workspace so it always overshoots and forces the clip-to-box
safeguard). For each combination, 20 episodes were run: the robot was first
driven toward a randomly-sampled destination for the pre-switch steps
(varying the real position the move later computes from, not a fixed
starting point), then the relative move was computed live from wherever
that left the robot and clipped into the measured workspace box. Every
relative-move result was compared against a budget-matched fresh-start
baseline aimed at the *same already-clipped target* with the *same*
remaining step budget -- the fair comparison, since judging the relative
move against its raw, pre-clip request would blame the mechanism for a
deliberate safety clamp rather than for its own accuracy. In total: 8
seeds x 54 combinations x 20 episodes = 8,640 relative-move episodes (plus
an equal number of baseline episodes), plus 2 more seeds run the same way
and reported separately once their training-collapse signature was
confirmed via the same literal-goal sanity check every prior stage's report
uses.

---
## Full evidence
The complete technical record — proof gate, full result tables, charts,
raw logs, anomalies, known-risks cross-check, and the reviewer
verdict — lives in [`evidence.md`](evidence.md).
