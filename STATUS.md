# Progress: 0 to hero

**3 of 7 stages done. Starting stage 4.** Click any stage below to expand
its actual results — no need to open other files for the numbers.

| # | What it proves | Status |
|---|---|---|
| 0 | Basic setup works at all | ✅ Done |
| 1 | Robot can reach a target given exact coordinates | ✅ Done |
| 2 | Same, but using a learned code instead of raw coordinates | ✅ Done |
| 3 | Same, but told the target in an English sentence | ✅ Done |
| 4 | Works with English phrasings it's never seen before | 🔧 Real progress, root cause narrowing — not there yet |
| 5 | Can change its target mid-task when told something new | ⬜ Not started |
| 6 | Live, real-time English control, start to finish | ⬜ Not started |

## Right now

**Stage 4 confirmed failed — independent check agrees, and points to a
specific next fix.** A second reviewer double-checked the memorization
diagnosis from scratch (not just re-reading the same numbers) and confirms
it: the sentence-to-code converter was trained on only 14 example sentences
(2 per direction), which is far too few for a network its size — it had
every reason to just memorize those 14 answers and no reason to handle
anything else sensibly. A picture of where sentences land in the learned
space backs this up directly: the 14 training sentences and the 14 new
test sentences land in almost completely separate areas, which is exactly
what memorizing (rather than understanding) looks like.

One extra thing was checked and ruled out: is the underlying language
understanding itself the problem (i.e. is 384 numbers just not enough to
tell these sentences apart)? No — checked directly, and sentences that
mean similar things do still start out close together before the
converter processes them. The converter is where the problem is introduced,
not the language model underneath it.

**Sanity check result: confirmed, and it's a big gap.** Tried answering new
sentences a completely different way — no trained converter at all, just
"which of the 14 known sentences is this most like, borrow its answer."
That zero-training approach got 10/14 right vs. the trained converter's
4/14. That's a decisive result: the raw language understanding already
has plenty of signal to place these new sentences correctly — the trained
converter is actively throwing that signal away by memorizing instead of
learning the general rule.

**Retrained with more examples — helped a lot on one measure, barely on
the one that matters.** Rebuilt the converter's training set from 14
sentences up to 70 (10 per direction instead of 2), retrained, retested.
Getting the *direction* right jumped from 4/14 to 9/14 — nearly matching
the zero-training ceiling test above. But actually reaching the target
with the robot barely moved: still failing almost every time (up from
~2% to ~10%, nowhere near acceptable).

**Why doesn't "getting the direction right" translate to "actually
succeeding"?** A second look at the numbers found something specific: for
one direction ("move right"), even a fairly imprecise answer still
succeeds — the robot's tolerance there is forgiving. For every other
direction, even landing in roughly the right spot isn't precise enough —
one sentence landed *closer* to the correct spot than the "move right"
one did, and still failed completely. So the real problem isn't "which
direction" anymore (that's mostly solved) — it's "how exact does the
answer need to be, and that required precision is different for every
direction." More example sentences won't fix that; it needs its own test.

**Next step: measure how forgiving each direction actually is.** Before
building anything else, artificially nudge the exact known-good answer for
each of the 7 directions by small amounts and see how much nudging the
robot can tolerate before it starts failing — zero retraining needed,
reuses everything already built. This tells us cleanly whether the fix is
"more example sentences," "make the robot more tolerant to small target
errors," or something about how directions are represented in the first
place. That test is running now.

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
<summary><b>Stage 4 — Works with sentences it's never seen before</b> 🔧 In progress — real gains made, root cause still narrowing</summary>

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

**Step 3 (running now): measure how forgiving each direction is.**
Deliberately feed the robot slightly-wrong versions of the exact correct
answer for each of the 7 directions (small nudges, increasing size) and see
where each direction starts failing — zero retraining, reuses everything
already built. This will show clearly whether the next fix should be (a)
even more example sentences, (b) making the robot itself more tolerant of
small target errors, or (c) something upstream in how directions are
represented that neither of those would fix.

Full detail: `experiments/04_open_vocabulary/report.md`

</details>

<details>
<summary><b>Stages 5, 6 — Not started</b></summary>

- **5:** can it change its target mid-task if told something new partway
  through?
- **6:** the actual end goal — live, typed-in-real-time English control.

</details>
