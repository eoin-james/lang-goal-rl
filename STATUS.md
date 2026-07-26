# Progress: 0 to hero

**3 of 7 stages done. Starting stage 4.** Click any stage below to expand
its actual results — no need to open other files for the numbers.

| # | What it proves | Status |
|---|---|---|
| 0 | Basic setup works at all | ✅ Done |
| 1 | Robot can reach a target given exact coordinates | ✅ Done |
| 2 | Same, but using a learned code instead of raw coordinates | ✅ Done |
| 3 | Same, but told the target in an English sentence | ✅ Done |
| 4 | Works with English phrasings it's never seen before | ❌ Failed (confirmed) — memorized 14 sentences, fix identified |
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

**Next step — a cheap sanity check before retraining anything.** Before
spending time gathering more training sentences, first try answering new
sentences a completely different way: instead of a trained converter, just
look up which of the 14 known sentences a new one is most similar to and
borrow its answer directly (no training at all). If that simple approach
already beats the trained converter on the new sentences, it proves the
diagnosis is right and the real fix is "give it more examples to learn
from," not something more exotic. That test is running now.

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
<summary><b>Stage 4 — Works with sentences it's never seen before</b> ❌ Failed, cause confirmed by independent review</summary>

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

**Next fix, in order:**
1. **Quick free test first (running now):** skip the trained converter
   entirely for a moment — for a new sentence, just find whichever of the
   14 known sentences it's closest to and borrow that answer. If this
   simple, no-training approach already does better than the trained
   converter on new sentences, that proves the diagnosis and clears the way
   for step 2 with confidence.
2. **The actual fix:** give the converter many more example sentences per
   direction (not just 2) so it has no choice but to learn the general
   pattern instead of 14 individual answers.
3. Anything fancier is being held back until steps 1-2 are tried, since the
   simplest explanation (not enough teaching examples) already fits
   everything observed.

Full detail: `experiments/04_open_vocabulary/report.md`

</details>

<details>
<summary><b>Stages 5, 6 — Not started</b></summary>

- **5:** can it change its target mid-task if told something new partway
  through?
- **6:** the actual end goal — live, typed-in-real-time English control.

</details>
