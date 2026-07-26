# Progress: 0 to hero

**2 of 7 stages done. Currently on stage 3.** Click any stage below to
expand its actual results — no need to open other files for the numbers.

| # | What it proves | Status |
|---|---|---|
| 0 | Basic setup works at all | ✅ Done |
| 1 | Robot can reach a target given exact coordinates | ✅ Done |
| 2 | Same, but using a learned code instead of raw coordinates | ✅ Done |
| 3 | Same, but told the target in an English sentence | 🔄 Built, not tested yet |
| 4 | Works with English phrasings it's never seen before | ⬜ Not started |
| 5 | Can change its target mid-task when told something new | ⬜ Not started |
| 6 | Live, real-time English control, start to finish | ⬜ Not started |

## Right now

Stage 3's pieces are built. **Not yet tested** — that's the next step.
Nothing else happens until that test runs and comes back with a real
pass/fail.

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
  **Result: yes, a real and meaningful relationship — not a coincidence,
  and not the code just ignoring the target entirely (that would have
  scored zero).**
- Full numbers + charts: `experiments/02_contrastive_goal_embedding/report.md`

</details>

<details>
<summary><b>Stage 3 — Reach a target described in an English sentence</b> 🔄 Built, not tested yet</summary>

**Built so far:**
- 14 fixed test phrases (e.g. "reach up high", "move to the left"),
  grouped into 7 real spatial regions — measured directly from the
  simulator, not guessed.
- A converter that turns any of those sentences into the same 16-number
  code stage 2 uses.
- Sanity check: do different phrases end up in meaningfully different
  places, and do similar phrases end up close together? **Yes, checked
  and confirmed.**

**Not yet done — this is the actual test:** take the robot trained in
stage 2, and see if it can still reach the right spot when told the goal
as an English sentence instead of a coordinate or a code. No result yet.

Code: `src/lang_goal_rl/language_goal_projection.py`,
`src/lang_goal_rl/goal_region_vocabulary.py`

</details>

<details>
<summary><b>Stages 4, 5, 6 — Not started</b></summary>

- **4:** does it still work with sentences it's never seen before (not
  just the 14 fixed test phrases)?
- **5:** can it change its target mid-task if told something new partway
  through?
- **6:** the actual end goal — live, typed-in-real-time English control.

</details>
