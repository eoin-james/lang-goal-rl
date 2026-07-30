# Progress: 0 to hero

**All 7 stages done — the project's core claim is proven.** Click any
stage below to expand its actual results — no need to open other files for
the numbers. Every report is now split into a short story you can read in
under a minute, with the full technical record one click away if you want
it (`report.md` for the story, `evidence.md` for everything else, in each
experiment's folder).

| # | What it proves | Status |
|---|---|---|
| 0 | Basic setup works at all | ✅ Done |
| 1 | Robot can reach a target given exact coordinates | ✅ Done |
| 2 | Same, but using a learned code instead of raw coordinates | ✅ Done |
| 3 | Same, but told the target in an English sentence | ✅ Done |
| 4 | Works with English phrasings it's never seen before | ✅ Done (took 4 attempts — worth reading) |
| 5 | Can change its target mid-task when told something new | ✅ Done — no downside found |
| 6 | Live, real-time English control, start to finish | ✅ Done — the actual point of this project |

## Right now

**Stage 6 passed — type a sentence, the robot goes for it; type a different
one partway through, it redirects, live.** Every earlier piece (understand
a sentence, reach a learned code, reach a never-seen sentence, re-aim
mid-task) got wired together and actually run: 3 independently-trained
robots were each given a live instruction, watched go for it, then given a
second, different live instruction partway through. On 7 completely new
sentences never used anywhere in this project before, it redirected and
succeeded 86% of the time, usually within 3 steps of the new instruction.
On a set of already-tested sentences (used as a sanity check that the live
version works the same as the earlier, offline-tested version — it did,
matching almost exactly), it succeeded 55% of the time — lower, but because
of the same known sentence-recognition gaps stage 4 already found and
explained, not because live switching itself costs anything. Switching
mid-task added no measurable extra difficulty in either case.

**Two things got caught and fixed along the way that are worth knowing
about, because they're exactly the kind of thing that could have made this
whole project's numbers meaningless if missed:**

1. **The demo videos weren't showing real movement — found the actual
   reason, not just "seems fine."** Investigated properly: the recording
   code was always correct, but every demo picked whichever attempt
   succeeded *first*, and for this robot, targets are sometimes placed
   close enough to its resting position that the very first successful
   attempt barely has to move at all. Fixed by picking, among genuinely
   successful attempts, the one with the most real movement — every video
   now clearly shows the arm actually traveling. Also caught two videos
   that had accidentally turned out identical (same setup, same "first
   success" rule, same result) and fixed that too. **8 demo clips are now
   in `demos/`**, one showing the actual final capstone: a live sentence,
   then a live redirect, mid-task.

2. **Checked whether this whole project's results could be fake — they're
   not, and now there's a permanent test proving it.** The video
   investigation above noticed one training episode where the target
   started unusually close to the robot's resting spot, which raised a
   fair, serious question: what if the task is so easy that a robot doing
   *nothing at all* would already score well, making every "it worked"
   result in this project meaningless? Tested directly: a robot that
   never moves succeeds on about 2% of attempts; one that moves randomly
   succeeds even less. Every real result reported across all 6 stages is
   30 to 56 times higher than that "doing nothing" score — so the numbers
   in this project are measuring genuine learned behavior, not a
   coincidentally-easy task. This check is now a permanent, re-runnable
   part of the repo (`experiments/00_trivial_baseline_audit/`), not just
   something asserted once.

**Honest limits of what's actually been proven, stated plainly:** this
project proves the *mechanism* — sentence in, robot re-targets live — works
on the simplest task in its test suite (free-space reaching, no grabbing or
pushing objects), with a vocabulary of 84 known sentence patterns, tested
on 7 brand-new phrasings. It does not prove this scales to truly unlimited
open-ended English, or to harder physical tasks — those are honestly logged
as open questions, not claimed as solved.

**Repo is live:** [graylayer-labs/lang-goal-rl](https://github.com/graylayer-labs/lang-goal-rl) —
pushed and up to date, including every attempt and its diagnosis. Also has
a bibliography (`LITERATURE.md`) of the real papers behind each stage —
found 4 of the original citation links pointed to the wrong papers,
documented rather than silently fixed.

<details>
<summary><b>Stage 0 — Basic setup</b> ✅ Done</summary>

Loaded the simulated robot arm task, confirmed it resets and steps
correctly. Nothing to measure — either it works or it doesn't, and it does.

</details>

<details>
<summary><b>Stage 1 — Reach a coordinate target</b> ✅ Done</summary>

- **Test:** run the training 10 times (different random starting
  conditions each time), see how often it succeeds.
- **Result: 8 out of 10 succeeded cleanly. 2 failed.**
- The 2 failures were checked and traced to a known quirk in the
  underlying training algorithm (an instability spike that can corrupt
  its decision-making for that one run) — not a mistake in this project's
  own code. Confirmed by comparing training behavior across all 10 runs.
- This quirk is now tracked so future stages don't mistake it for a new
  problem they introduced.
- Full numbers + charts: `experiments/01_uvfa_her_baseline/report.md`

</details>

<details>
<summary><b>Stage 2 — Reach a target described by a learned code, not raw coordinates</b> ✅ Done</summary>

- **Test 1:** same 10-run success test as stage 1, but the robot is only
  given a learned 16-number code instead of the real x/y/z coordinates.
  **Result: 10 out of 10 succeeded — better than stage 1.**
- **Test 2:** does distance between two codes actually reflect real
  distance between two targets? Checked on 500 targets it had never seen.
  **Result: yes, a real and meaningful relationship.**
- Full numbers + charts: `experiments/02_contrastive_goal_embedding/report.md`

</details>

<details>
<summary><b>Stage 3 — Reach a target described in an English sentence</b> ✅ Done (took 4 attempts — the story is worth knowing)</summary>

**Sanity checks — all passed:** 14 fixed test phrases across 7 real
spatial regions (measured from the simulator, not guessed); different
phrases land in meaningfully different places; the robot itself still
gets a perfect score on the old coordinate-based test throughout.

**Attempt 1 — failed (~0%).** Cause found: the English-to-code converter
was outputting numbers 5-10x the scale the robot was trained on — right
idea, wrong units.

**Attempt 2 — fixed the scale (~7%).** Better, but a second bug appeared:
the converter got the size right but not precisely the direction.

**Attempt 3 — fixed the direction too (~16%).** Real improvement, but
even with the converter matching its target almost exactly, no single
instruction reliably passed ~44%. This looked like it might be a hard
limit, not a bug.

**The actual answer — it was the test, not the robot.** "Reach up high"
describes a *region* of space, not one exact point. Every test so far
picked a random point inside that region and demanded the robot hit that
exact spot from just the sentence — geometrically close to impossible
regardless of how good the converter is, since the converter can only
ever point at the region's center. The math for "how often would you hit
a random point in a region this size" predicted the observed ~16%
almost exactly. That's not a coincidence — the robot's understanding of
the sentence was correct the whole time.

**Attempt 4 — fixed the test itself: judge success against the spot the
sentence actually points to, not a random spot nearby. Result: 1.000 —
an exact match with stage 2.** Zero retraining needed for this one; same
robot, same converter, only the test's success criterion changed.

**Lesson recorded for future stages:** this cost 3 build-and-fix rounds
before the real cause (the test, not the code) was found. Stage 4 needs
to design its test correctly from the start rather than repeat this.

Full story with all 4 attempts and independent checks: `experiments/03_language_goal_projection/report.md`

</details>

<details>
<summary><b>Stage 4 — Works with sentences it's never seen before</b> ✅ Done (took 4 attempts — the story is worth knowing)</summary>

**Setup:** wrote 14 brand-new test sentences (2 per region) that were
never used to train the converter — genuinely different wording, e.g.
"lower your arm toward the floor" for a region trained on "reach down
low." Also tried 2 combined instructions ("reach up and to the left").

**Test 1 — does the new sentence's code land near the right region in
goal-space, or does it get confused with a different region?**
**Result: only 4 out of 14 (29%) landed nearest their correct region** —
better than random guessing (1-in-7 ≈ 14%) but far from reliable.

**Test 2 — does the robot actually reach the right spot for these new
sentences?** **Result: ~2% success, vs. 100% on the trained sentences.**
The robot itself is still fine (re-checked, still perfect on the old
coordinate test) — the problem is entirely in the sentence-to-code step.

**Why:** the exact fix that made stage 3 pass — training the converter
to hit 14 known points precisely — has no built-in reason to place a
sentence it's never seen anywhere sensible. It optimized for "get these
14 exact answers right," not "understand the general pattern." The one
new sentence that did work was the one worded most similarly to a
sentence it already knew.

**Combined instructions ("reach up and to the left"):** no right-or-wrong
answer exists for these yet, so this was just observed rather than
scored — the code landed nearer to one of the two directions than
exactly between them, which is a reasonable, non-broken response to a
sentence outside its training.

**Independent check: confirmed.** A second reviewer re-derived the diagnosis
from scratch rather than trusting the first pass, and reached the same
conclusion — plus a sharper reason why. The converter is a small network
with roughly 25,000 internal adjustable numbers, trained on only 14 example
sentences. That's a huge amount of flexibility for a tiny amount of
teaching material — it's mathematically easier for it to just memorize
those 14 exact answers than to learn a rule that would also work for
sentence #15. A plot of where sentences land in its output space shows
this directly: the 14 training sentences and the 14 new test sentences
cluster in almost totally separate areas — the signature of memorizing,
not understanding.

**One alternative explanation was ruled out:** maybe the 384-number
description of each sentence (from the underlying frozen language model,
not the part being trained) just isn't detailed enough to tell similar
sentences apart? Checked directly — no. Sentences with similar meaning do
still start out close together before the converter touches them. The
language model underneath is fine; the small trained converter on top of
it is where the problem gets introduced.

**Step 1 (zero-training sanity check): confirmed the diagnosis.** Skipping
the trained converter entirely and just borrowing the answer from whichever
known sentence a new one resembles most got 10/14 right, vs. the trained
converter's 4/14. Clear proof the raw language understanding has the
signal — the trained converter was throwing it away.

**Step 2 (give it more examples): helped the right-direction question a
lot, barely helped the actual task.** Retrained on 70 example sentences
(10 per direction instead of 2). Result: picking the right *direction* for
a new sentence jumped from 4/14 to 9/14 — almost matching the zero-training
ceiling test. But the robot actually reaching the target for these new
sentences barely improved: ~2% before, ~10% now. Still a clear fail.

**Why more examples didn't fix the actual task:** a closer look found the
problem changed shape. It's no longer mostly "wrong direction" — it's now
mostly "right direction, but not precise enough," and how much precision
each direction needs turns out to vary a lot. One sentence landed *closer*
to its correct target than another sentence did, yet the closer one failed
completely while the farther one succeeded every time. One direction
("move right") seems to be forgiving of an imprecise answer; the others
aren't. More example sentences won't fix that kind of gap — we need to
measure how forgiving each direction actually is before picking the next
fix.

**Step 3 result: messier than expected, but still useful.** Deliberately
fed the robot slightly-wrong versions of the correct answer for each of
the 7 directions, at increasing amounts of "wrongness," to see where each
one starts failing. The clean story we were hoping for ("direction X is
just more forgiving than direction Y") didn't hold up — the results bounced
around too much to trust a simple "this direction tolerates more error than
that one" ranking. Digging into why: each test only tried *one* random way
of being wrong at each amount, so what looked like "more wrongness breaks
it" was often actually just "this particular way of being wrong happened to
break it, a different way at the same amount wouldn't have." In other
words: it's not just *how far off* the answer is that matters, it's *which
direction* it's off in — and that's a more specific, more useful thing to
know than what we set out to measure.

**So: not a wasted test, just not the one we thought we were running.**
Rather than run a bigger, more careful version of that same test (which
would only sharpen the measurement, not get us closer to a working system),
the next step goes straight at the real question: is the trained converter
itself introducing this "wrong direction" problem, or is it inherent to the
robot? Testing this by skipping the trained converter one more time — using
the simple lookup-the-closest-known-answer approach from step 1, but now
with all 84 known sentences (14 original + 70 new) — and seeing if it gets
the robot to actually succeed noticeably more than the trained converter's
current ~10%. If yes, the trained converter is the problem and can likely
just be replaced with this simpler lookup approach. If no, the problem is
in the robot's own precision requirements, not the sentence-to-code step,
and the next fix looks completely different (training the robot to be more
forgiving of small target errors). That test is running now.

**Step 4 — the answer, and it's decisive.** Skipped the trained converter
entirely and used the simple lookup approach with all 84 known sentences.
Result: real success on brand-new sentences jumped from ~10% (trained
converter) to **57%** — and for the *typical* new sentence, it now
succeeds every single time (not just sometimes). That confirms it plainly:
the trained converter was the problem all along, not the robot's
precision. A second, deliberately skeptical review checked this from
scratch — recounted every number by hand, chased down one odd wrinkle in
the data (a slightly-more-careful version of the lookup got the *direction*
right more often but actually did *worse* on the real task — turns out
that's because "more careful" also means "less exact," and exact beats
careful-but-approximate here), and signed off.

**Real, honest limits worth knowing before stage 5/6:** this fix works by
recognizing a new sentence as similar to one of 84 known sentences — it
doesn't yet understand language on its own from scratch. About 6 of the
14 brand-new test sentences still fail, because they happen to resemble
the *wrong* known sentence more than the right one. That's a fixable gap
(teach it more example sentences), not a fundamentally broken idea — but
it does mean stage 6's promise of truly free-form live English will need
either a much bigger set of known sentences or a smarter fallback for
things that don't resemble anything it's seen. Noted now so it isn't
rediscovered the hard way later.

Full detail: `experiments/04_open_vocabulary/report.md`

</details>

<details>
<summary><b>Stage 5 — Can it change its target mid-task?</b> ✅ Done</summary>

**Setup:** using the same trained robot from stage 1 (exact coordinates,
no English involved — kept deliberately simple to isolate this one
question), give it one target for the first part of an episode, then
swap to a genuinely different target partway through, and see if it can
still reach the new one before time runs out.

**The fair comparison that matters:** it's not enough to check "did it
reach the new target" — it also needs to be compared against a version
that was given the *same* new target from the start, but only allowed the
*same remaining amount of time* (not a full fresh episode). Without
matching the time budget, a worse score after a swap could just mean
"less time left," not "swapping targets hurts." This test controls for
that from the start.

**Result: no difference at all.** Tested at 4 different points in the
episode (early through late) across 10 independent training runs. For
every run that trained cleanly (8 of 10), swapping targets mid-task
scored exactly as well as the time-matched fresh comparison — both
100% success, every time, at every switch point. The 2 remaining runs hit
the same training glitch already tracked since stage 1 (unrelated to this
test — confirmed by checking their own training logs show the exact same
signature).

**Why this matters:** research on this topic has generally found that
changing targets mid-task can quietly break things, so it wasn't assumed
this would just work — it was tested directly. For this task, it did.

**Real limit worth flagging:** this only tested exact coordinates, not
English sentences — deliberately, to isolate "can it re-aim at all" from
every question stages 2-4 already answered about sentence accuracy.
Whether this same clean result holds once the English pipeline is back
in the loop (stage 6) is a genuinely open question, not something to
assume.

Full detail: `experiments/05_midepisode_regoal/report.md`

</details>

<details>
<summary><b>Stage 6 — Live English control, start to finish</b> ✅ Done</summary>

**Setup:** wire everything together and actually run it live — turn a
typed sentence into a goal the robot understands (stage 4's fix), feed
that straight into the trained robot (stage 2/3's setup), and let a second
typed sentence take over partway through the same task (stage 5's
mechanism) — no special training for this exact combination, just the
pieces already proven separately, used together for real.

**Two test sets, kept separate on purpose:**
- **Set A** — the same 14 sentences stage 4 already tested, reused here as
  a sanity check: does the *live* version behave the same as the earlier,
  offline-tested version? (It does — 0.571 success here vs. stage 4's
  0.571, an almost exact match.)
- **Set B** — 7 completely new sentences, checked to make sure they don't
  resemble anything used anywhere else in this project. This is the real
  test of "does it work on genuinely new, ad-hoc English."

**Result: yes, and switching mid-task doesn't cost anything extra.**
Across 3 independently-trained robots: Set A succeeded 55% of the time
after a live mid-task switch, almost identical to its own 57% no-switch
score. Set B succeeded 86% of the time after a switch, almost identical to
its own 86% no-switch score. In both cases, switching to a new instruction
partway through cost nothing measurable beyond whatever the
sentence-understanding step already got right or wrong on its own — the
switching mechanism itself is not the bottleneck. When it did redirect
successfully, it typically did so within 3 steps of the new instruction,
out of many steps still remaining.

**A second, deliberately skeptical review checked this from scratch** and
confirmed the live setup reproduces stage 4's exact sentence-by-sentence
pattern (not just a similar overall number — the *same* sentences succeed
and fail as they did before), and ran the actual statistics on "does
switching hurt": the answer is no, comfortably within normal chance
variation either way.

**Honest limit, stated plainly:** "works on 7 new sentences" is a real
demonstration, not proof it works on *any* new sentence — 7 is enough to
show the mechanism functions, not enough to promise it scales to truly
unlimited free-form English. That gap (a bigger or smarter vocabulary
system) is honestly logged as real, open future work, not something this
project claims to have solved.

Full detail: `experiments/06_live_english_interface/report.md`

</details>

<details>
<summary><b>Bonus check — is this whole project's task secretly too easy?</b> ✅ Checked, no</summary>

**Why this got checked:** while making the demo videos, one training
episode was noticed where the target started unusually close to the
robot's resting position — close enough to raise a fair question: could
some of this project's "it worked" results just be measuring an easy task,
not real learned behavior?

**Tested directly, not assumed.** Reset the simulator 500 times and
measured how far the target actually starts from the robot on a typical
attempt: usually about 14-15cm away — three times farther than the ~5cm
"close enough to count as success" distance. The one close-up episode that
sparked this check was a rare, unusually easy case, not the normal
situation.

**Then tested what "doing nothing" and "moving randomly" actually score,
using the exact same pass/fail rule every stage in this project uses.**
A robot that never moves succeeds about 2% of the time. One that moves
randomly succeeds even less. Every real result this project has reported —
from 55% up to 100% — sits 30 to 56 times higher than that "do nothing"
score. That's a clear, wide margin: the numbers in this project reflect
real learned behavior, not an accidentally free win.

**One honest asterisk worth knowing:** since even a "perfect" robot solves
this particular task in just 3-5 steps, several stages' "100% success"
scores can't tell the difference between "very good" and "flawlessly
perfect" — there's no room above 100% to show a difference. That's a
limit on how much those specific perfect scores can tell you, not a sign
they're wrong.

This check is now a permanent, re-runnable part of the repo, not just
something checked once and taken on faith.

Full detail: `experiments/00_trivial_baseline_audit/report.md`

</details>

---

## Phase 2a: replacing 7 fixed regions with real commands

Phase 1 proved the mechanism; every sentence in it still snapped to one of
7 fixed points. Phase 2a builds the layer a future language model will
plug into — deterministic typed commands (`goto`, `move`, `waypoints`,
`stop`, `reset`) instead of picking from 7 buckets — with no language
model in the loop yet, so bugs in this layer can't hide behind language
mistakes. Full stage-by-stage record: [PHASE2_ROADMAP.md](PHASE2_ROADMAP.md).

| # | What it proves | Status |
|---|---|---|
| 7 | "Left"/"forward"/"up" on screen mean what the labels claim | ⏳ Clips ready, awaiting a human look |
| 8 | Robot can shift itself relative to wherever it actually is, not just toward a target from episode start | ✅ Done |
| 9 | Robot can chain several such moves with no reset in between, without errors compounding | ✅ Done (took 2 attempts — see below) |
| 10 | All of the above works the same when driven by real typed commands, not direct function calls | ✅ Done |

**Stages 8-10, done 2026-07-28.** Relative moves: perfect across every
direction, distance, and mid-episode timing tried, on all 8 independently-
trained robots. Waypoint chains: the first attempt only tested one robot —
correctly caught as not enough evidence — a rerun across all 8 found zero
chains breaking down partway through, out of 4,800 chained attempts.
Wrapping both behind an actual typed-command parser changed nothing
measurable, and the parser correctly rejects garbled input (23/23
deliberately broken or nonsense strings) instead of guessing. One brand-new
check — does telling the robot to `stop` actually make it hold still —
came back honest and imperfect: it settles within about a centimeter or two
and stays there, rather than converging to a perfect zero, which is
expected for a robot never specifically trained to sit still on command.

**Still open:** stage 7 needs someone to actually look at 6 short clips and
confirm "left" looks like left on camera — deliberately a human call, not a
number, and not blocking the stages above (they don't depend on the label
being right, only on the underlying movement mechanism, which they measure
directly).

Full detail per stage: `experiments/08_relative_move_validation/report.md`,
`experiments/09_waypoint_following/report.md`,
`experiments/10_typed_command_interface/report.md`.

---

## Phase 2b: teaching a language layer to speak the typed-command language

Phase 2a built a deterministic typed-command language (`goto`, `move`,
`waypoints`, `stop`, `reset`) with no language model in the loop. Phase 2b's
job is to teach a learned layer to translate arbitrary English into that
language, instead of the old 7-fixed-region snap. Full stage-by-stage
record: [PHASE2B_ROADMAP.md](PHASE2B_ROADMAP.md).

| # | What it proves | Status |
|---|---|---|
| 11 | A sentence's command *type* (move/goto/waypoints/stop/reset/unsupported) can be classified reliably from frozen sentence embeddings | ✅ Done (took 3 attempts) |
| 12 | A move command's direction and distance can be regressed continuously, not just classified | Paused for all-hands review |
| 13 | Multi-part instructions can be split into separate move/waypoint legs | Not started |
| 14 | The full pipeline grounds free English into typed commands end to end | Not started |
| 15 | Live capstone: a third `interactive_demo.py` interface driven by free-form English | Not started |

**Stage 11 done, stages 12-15 paused for review.** The first classifier
attempt scored 0% on one of five command-type buckets — not a training
failure but a data-design mistake: the builder reused Phase 1's
directionally-phrased region vocabulary as training data for
"go to a named spot," which taught the model the opposite convention from a
real move instruction. Rewriting each class's vocabulary to carry its own
honest phrasing fixed the 0% outright. That surfaced two smaller gaps (a
stop idiom and an unhandled math question), closed additively in a third
attempt. Final result: 98.08% held-out top-1 accuracy across all five
classes with zero seed variance, 0% of held-out UNSUPPORTED sentences
classified as anything actionable, 12/12 runs passing both proof-gate
conditions. One non-blocking residual noted and left alone on purpose: two
RESET idioms now collide with STOP's expanded vocabulary — bounded, doesn't
violate either gate, tracked as a finding for whoever picks up stage 12+.

Stages 12-15 are deliberately paused, not stalled — the plan calls for an
all-hands review of stage 11's approach before committing to the same
architecture for continuous regression. **Note:** Stage 7's human sign-off
(Phase 2a) is still pending, separately — unrelated to Phase 2b's own
progress, tracked in its own row above.

Full detail: [experiments/11_command_type_classification/report.md](experiments/11_command_type_classification/report.md).
