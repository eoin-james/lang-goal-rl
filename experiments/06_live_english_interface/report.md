# Stage 6: Live English interface

## In plain English
This stage builds the thing the whole project has been leading up to: type
an English sentence, have it become a real-time robot goal, and have the
robot correctly re-target if you switch instructions partway through --
without any special training for that live scenario. Every earlier stage's
building block gets wired together and actually run live: turning a
sentence into a goal embedding (stage 4's nearest-neighbor lookup), feeding
that live embedding straight into the already-trained policy (stage 2/3's
embedding-based SAC), and swapping it mid-episode (stage 5's switch
mechanism). Two test sets were used: 14 previously-measured paraphrases
("Set A" -- a sanity check that this new live wiring reproduces stage 4's
already-known numbers) and 7 completely new instructions never used
anywhere in this project before ("Set B" -- the actual test of "ad-hoc" live
phrasing the way the proof gate means it). Across 3 independently-trained
agents, switching mid-episode to a live instruction reached the new target
about as often as just being given that instruction from the very start --
the switch mechanism itself does not appear to add extra difficulty on top
of whatever accuracy the underlying language-to-goal mapping already had.
When redirection did happen, it was fast: a handful of steps after the
switch, out of a much larger remaining budget.

## Result
Measured, not adjudicated -- see "Reviewer verdict" in Full evidence below
for the actual pass/fail call. Set A (14 previously-measured paraphrases)
reproduced stage 4's 0.571 mean / 1.000 median no-switch result almost
exactly (this run: 0.571 mean / 1.000 median), confirming the live pipeline
is wired correctly before trusting anything new. Its live mid-episode
switch success (0.548) tracked that same-set no-switch baseline closely --
switching cost little to nothing beyond whatever the underlying language
accuracy already was for that instruction. Set B (7 brand-new phrasings)
scored higher on both the no-switch control (0.857) and the switch test
(0.857) -- the same close switch-vs-no-switch tracking held, just at a
different accuracy level for this particular set of new phrasings. Where
redirection happened, it was fast: median 3 steps after the switch for both
sets, out of 30 steps of budget remaining post-switch.

![control_vs_switch_success_rate.png](charts/control_vs_switch_success_rate.png)

## How this was tested
Three previously-trained SAC policies (stage 3's checkpoints, no new
training) were each given a live English instruction, converted in real
time to a 16-dim goal embedding via stage 4's nearest-neighbor lookup
(`LiveGoalController`), and run for a 50-step episode. Two instruction sets
were used: Set A (14 phrasings stage 4 already measured, reused here as a
sanity cross-check) and Set B (7 phrasings written fresh for this
experiment and checked to be genuinely different wording from every
sentence this project has ever used, in training or test data). For each
set, every instruction was paired with one different-region instruction;
the episode targeted the first instruction's live-encoded goal for 20
steps, then switched live to the second instruction's for the remaining 30.
"Task success" is whether the robot's hand ended within FetchReach's 5cm
tolerance of the second instruction's target region by the end of the
episode, judged against that region's fixed centroid (never a freshly
resampled point -- the same region-vs-point lesson every stage since stage
3 has applied). "Time-to-redirect" is how many steps after the switch it
first got there, counted only for episodes that did reach it -- episodes
that never did are reported separately as "did not redirect," never
averaged in as an arbitrary penalty value. A no-switch control (same
instructions, full episode, no swap) was run alongside as the baseline for
isolating whether the switch mechanism itself costs anything beyond the
language pipeline's own accuracy.

---
## Full evidence
The complete technical record — proof gate, full result tables, charts,
raw logs, anomalies, known-risks cross-check, and the reviewer
verdict — lives in [`evidence.md`](evidence.md).
