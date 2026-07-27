# Stage 3: Frozen language embedding -> goal space

## In plain English

This stage tests whether a robot-arm policy that was trained to reach a
literal target point can instead be steered by a plain-English instruction
like "reach up high" or "move your hand to the left" — by translating the
instruction's sentence embedding into the same goal representation the
policy already understands, and checking it succeeds as often as it does
with a literal target. It took four attempts to get there. Attempt 1 failed
almost completely (succeeding on essentially none of 42 test cases) because
the training method used to build the translation had no way to control how
large its outputs were, so the translated instructions ended up 5-10x
outside the range the policy had ever seen. Attempt 2 fixed that scale
problem, which helped a lot but still only reached about a 7% success rate —
the translations were now the right size but pointing in slightly the wrong
direction. Attempt 3 fixed the direction by training the translation to aim
directly at each instruction's true target point, pushing success to about
16% — real progress, but still far short of the goal. Attempt 4 found the
actual culprit: not the translation or the policy at all, but the test
itself. The test was checking success against a randomly moving target
inside a region far larger than the policy's margin for error, while the
policy was only ever shown one fixed point — an unwinnable test no matter how
good the translation was. Fixing the test to check against the same fixed
point the policy was actually shown made success jump instantly to 100%
across every seed and every instruction, matching the original baseline
exactly.

## Result

**Passed on attempt 4 — success rate 1.000 across all 3 seeds and all 14
instructions, exactly matching stage 2's literal-goal baseline, after 3
earlier attempts each diagnosed and fixed a different bug (output scale,
then direction, then a flawed evaluation protocol).**

![language_goal_success_rate_v4.png](charts/language_goal_success_rate_v4.png)

## How this was tested

Three already-trained policies (one per seed: 0, 1, 2) were each given 14
natural-language instructions covering 7 spatial regions of the robot arm's
workspace (center, forward, back, left, right, up, down), with 50 test
episodes run per instruction per seed — 42 seed/instruction combinations and
2,100 episodes in total per attempt. Each instruction's text was converted
into the same kind of embedding the policy expects as a goal, and "success"
means the robot arm's gripper ended the episode within the same small
distance-to-target threshold used in the original literal-goal task (stage
2), which these policies already hit 100% of the time when given the real
target directly — that 100% literal-goal number was re-confirmed on the same
checkpoints in every attempt below, as a control showing the policies
themselves never changed.

---
## Full evidence
The complete technical record — proof gate, full result tables, charts,
raw logs, anomalies, known-risks cross-check, and the reviewer
verdict — lives in [`evidence.md`](evidence.md).
