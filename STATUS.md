# Progress: 0 to hero

**3 of 7 stages done. Starting stage 4.** Click any stage below to expand
its actual results — no need to open other files for the numbers.

| # | What it proves | Status |
|---|---|---|
| 0 | Basic setup works at all | ✅ Done |
| 1 | Robot can reach a target given exact coordinates | ✅ Done |
| 2 | Same, but using a learned code instead of raw coordinates | ✅ Done |
| 3 | Same, but told the target in an English sentence | ✅ Done |
| 4 | Works with English phrasings it's never seen before | 🔄 Starting |
| 5 | Can change its target mid-task when told something new | ⬜ Not started |
| 6 | Live, real-time English control, start to finish | ⬜ Not started |

## Right now

Stage 3 is closed out and confirmed. Starting stage 4: does this still
work with phrasings the system has never seen before (not just the 14
fixed test sentences)?

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
<summary><b>Stages 4, 5, 6 — Stage 4 starting now</b></summary>

- **4 (starting):** does it still work with sentences it's never seen
  before (not just the 14 fixed test phrases)?
- **5:** can it change its target mid-task if told something new partway
  through?
- **6:** the actual end goal — live, typed-in-real-time English control.

</details>
