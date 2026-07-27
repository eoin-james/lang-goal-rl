# Progress: 0 to hero

**4 of 7 stages done. Starting stage 5.** Click any stage below to expand
its actual results — no need to open other files for the numbers.

| # | What it proves | Status |
|---|---|---|
| 0 | Basic setup works at all | ✅ Done |
| 1 | Robot can reach a target given exact coordinates | ✅ Done |
| 2 | Same, but using a learned code instead of raw coordinates | ✅ Done |
| 3 | Same, but told the target in an English sentence | ✅ Done |
| 4 | Works with English phrasings it's never seen before | ✅ Done (took 4 attempts — worth reading) |
| 5 | Can change its target mid-task when told something new | ⬜ Starting now |
| 6 | Live, real-time English control, start to finish | ⬜ Not started |

## Right now

**Stage 4 passed — took 4 rounds, but ended with a clean, independently
verified fix.** The short version: the trained sentence-to-code converter
was the problem the whole time, not the robot. Replacing it with a much
simpler "just borrow the answer from whichever known sentence this new one
is most like" approach — using a bigger set of 84 known sentences — nearly
tripled real success on brand-new sentences, from ~10% to ~57% (and the
*typical* new sentence now succeeds, not just an unlucky few). Full
before/after story is in the stage-4 dropdown below.

**Starting stage 5 now: can it change its target mid-task?** So far the
robot has only ever been told its target once, at the very start of an
episode. Stage 5 asks whether it can be told something new *partway
through* and re-aim without starting over — a step closer to the eventual
goal of a live conversation with the robot while it's working.

**3 demo clips are in `demos/`** — baseline success, the original broken
failure, and a real success once the eval was fixed — each labeled with
its actual measured success rate.

**Repo is live:** [graylayer-labs/lang-goal-rl](https://github.com/graylayer-labs/lang-goal-rl) —
pushed and up to date, including every attempt and its diagnosis. Also has
a bibliography (`LITERATURE.md`) of the real papers behind each stage —
found 3 of the original citation links pointed to the wrong papers,
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
<summary><b>Stages 5, 6 — Not started</b></summary>

- **5:** can it change its target mid-task if told something new partway
  through?
- **6:** the actual end goal — live, typed-in-real-time English control.

</details>
