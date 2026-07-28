# Research Log

Private research-reasoning log. Not a bibliography (`LITERATURE.md`) and not
a stage-outcome tracker (`ROADMAP.md`'s "Known risks"/status column) —
this is the *why*: methodology decisions, dead ends, and the shape of each
debugging saga, written for this project's own future reference. Dense and
technical is fine here; this is not portfolio material (`BLOG.md` is).

One entry per major decision point, dated, newest at the bottom. Keep each
entry a few sentences — link to the relevant `experiments/NN_slug/evidence.md`
section for the full numbers rather than re-deriving them here.

---

## 2026-07-26 — Stage 3 attempt 1: scale-invariant loss can't fix an embedding-scale mismatch

Language-goal substitution collapsed to ~0.002 success despite a clean
1.000 literal-goal control and a passing collapse check (143.85x margin).
Root-caused at the code level, not just from behavior: `info_nce_loss`
L2-normalizes both anchor and positive before computing the loss, making
the training objective mathematically blind to output *scale* — no amount
of training could have pulled projection norms (0.25–0.41) toward the
frozen `GoalEncoder`'s true range (0.02–0.07). Decision: fix the loss
(explicit norm-matching term) and add a fail-fast norm-range check as a
second gate before spending RL eval time again — a passing collapse check
alone is not sufficient evidence the projection is usable.
See `experiments/03_language_goal_projection/evidence.md`, Attempt 1.

## 2026-07-26 — Stage 3 attempt 2: norm fix necessary, not sufficient — direction is the next gap

Added the norm-matching MSE term; the fail-fast check now passes (all 14
projected norms within 2x of the reference band). Success rate improved
0.002 → 0.069 mean but stayed far below the ~1.0 gate. Diagnosis: InfoNCE's
separation term pulls each point toward a *noisy per-step-resampled*
positive target, not the region's true centroid — scale was fixed,
direction wasn't. Decision: stop tuning InfoNCE, switch to direct
regression against a fixed, precomputed-once centroid.
See `experiments/03_language_goal_projection/evidence.md`, Attempt 2.

## 2026-07-26 — Stage 3 attempt 3: near-perfect target matching, RL success still only 0.157 — pushed the question up a level

Regressing to a fixed centroid (plain MSE, no InfoNCE) drove training loss
to ~0.0000 — the projection matches its target almost exactly. RL success
only reached 0.157 mean / 0.120 median. Correlated direction-alignment
(cosine similarity) against attempt 2's per-instruction success: Pearson
r=0.345 — weak-to-moderate, not the dominant explanation. This ruled out
"direction accuracy is the whole story" and reframed the open question as:
if the projection matches its target near-perfectly and RL still fails,
maybe the target itself — or what's being compared against it — is wrong,
not the projection. That reframe is what led directly to attempt 4.
See `experiments/03_language_goal_projection/evidence.md`, Attempt 3.

## 2026-07-26 — Stage 3 attempt 4: the defect was the eval's ground truth, not the model — cost 3 rounds to find

Root cause: `evaluate_language_goal` judged success against a *freshly
resampled random point* in the instruction's region on every episode,
while the policy only ever saw one *fixed* embedding for that instruction
(the region centroid). FetchReach's success radius (0.05m) is 2-6x smaller
than the measured region widths — judging a fixed-embedding policy against
a random point elsewhere in a region that size is close to a geometric
impossibility regardless of embedding quality. Changing only the eval
script (judge against the same fixed centroid the projection targets) took
success from 0.157 to 1.000 with zero retraining. **Methodology lesson,
now load-bearing for every later stage:** before spending a training cycle
on a suspiciously-capped success rate, ask whether ground truth is a fixed
representative point or a resampled region member — this is cheaper to
check than to rediscover through three "fix the model" rounds.
See `experiments/03_language_goal_projection/evidence.md`, Attempt 4.

## 2026-07-26 — Stage 4 attempt 1: fixed 14-sentence vocabulary + ~25.6k-param MLP = memorization

Held-out paraphrase classification: 28.6% (barely above the 14.3% chance
floor). RL success: 0.024 mean, 1/42 nonzero. PCA of training vs. held-out
projected points showed near-total separation — the visual signature of
memorization, not a smooth mapping. Ruled out "the frozen encoder can't
tell these apart" directly (raw 384-dim space does preserve some proximity
for held-out phrasings). Decision, per reviewer's prescribed ordering: run
a zero-training NN-ceiling test *before* committing to a data-augmentation
fix, to confirm this is a training-data problem, not an information-
theoretic ceiling on what the frozen encoder can represent.
See `experiments/04_open_vocabulary/evidence.md`, Attempt 1.

## 2026-07-26 — Stage 4 NN-ceiling test: confirmed memorization, not an information ceiling — cleared the fix to proceed

Bypassing the learned MLP entirely (k=1 nearest-neighbor blend in raw
384-dim space, over the same 14 training targets) scored 0.714
classification vs. the MLP's 0.286 — 6 instructions flip wrong→correct,
zero reverse flips. This is the evidence that made "retrain with more
data" the justified next step rather than an architecture change: the raw
embedding space already carries the region-clustering signal, the trained
MLP was actively discarding it.
See `experiments/04_open_vocabulary/evidence.md`, Attempt 1 Part 4.

## 2026-07-26 — Stage 4 attempt 2: data augmentation fixes classification, barely moves RL success — bottleneck shifted

70-sentence vocabulary took classification 0.286 → 0.643 (near the 0.714
ceiling) but RL success only reached 0.095 mean / 0.000 median. Also
surfaced an unplanned regression: the augmented vocabulary shares zero
strings with the original 14, so it *replaced* rather than *extended* the
training set — the original 14 now score in the same range as the
held-out set (0.143 mean). Read as confirming evidence, not a new failure:
the projection generalizes moderately to *any* unseen phrasing rather than
memorizing one closed set — the right qualitative shape, just not enough
magnitude. New finding that reframed the problem: RL success is
concentrated in one region ("reach right") even where classification is
already correct, and a *farther* correctly-classified instruction beat a
*closer* one — classification accuracy stopped predicting RL success once
it got "good enough." This is what motivated the region-tolerance
diagnostic in attempt 3.
See `experiments/04_open_vocabulary/evidence.md`, Attempt 2.

## 2026-07-26 — Stage 4 attempt 3: region-tolerance diagnostic ran clean, its own framing didn't survive review

Injected controlled-magnitude noise directly into each region's true
target embedding (no projection, no sentences) to map per-region tolerance
radii. Result: 4/7 regions showed non-monotonic success-vs-magnitude
curves, 2 with full recovery after an apparent collapse. Root cause: each
(region, magnitude) cell used one unrelated random perturbation direction —
all 3 SAC seeds agreed deterministically per cell, proving *direction*, not
magnitude, decided pass/fail. Corrected takeaway: tolerance is
direction-sensitive within a region, not just magnitude-sensitive between
regions — a real, useful finding, just not the "region X is more forgiving"
story the diagnostic was designed to produce. Decision: don't run a
cleaner, multi-direction-averaged version of the same diagnostic — it would
sharpen characterization but not move the pass/fail needle. Go straight at
whether the MLP's own learned distortion, not policy tolerance, is the
actual bottleneck.
See `experiments/04_open_vocabulary/evidence.md`, Attempt 3.

## 2026-07-26 — Stage 4 attempt 4: swapping the MLP for zero-training k=1 NN lookup resolves the stage

Removing the learned projection entirely and using k=1 nearest-neighbor
lookup over the combined 84-sentence vocabulary (14 original + 70
augmented) took held-out RL success from 0.095 mean/0.000 median to 0.571
mean/1.000 median — confirming the MLP's own learned directional
distortion, not policy tolerance, was the real bottleneck across attempts
1-3. k=1 beat k=3 specifically because k=1 always returns an *exact*
known-good centroid (zero directional deviation) while k=3's blend never
does — a direct, concrete confirmation of attempt 3's direction-sensitivity
finding, now well enough understood to design around (pick k=1) rather than
needing further characterization. The PASS was conditional: the reviewer
required logging that k-NN's ceiling is bounded by reference-vocabulary
coverage density before treating 0.571 as risk-free heading into stage 6.
See `experiments/04_open_vocabulary/evidence.md`, Attempt 4.

## 2026-07-27 — Demo-video pixel-diff audit surfaces a "first success" selection bug

While generating visual proof for stages that had none, a pixel-diff pass
found clips 1/4/5 nearly static after the first few frames, and clips 1 and
5 byte-identical despite depicting different stages. Root cause: every
`make_demo*.py` script searched eval seeds in increasing order and stopped
at the first success. FetchReach samples goals only centimeters from a
fixed reset pose, so the first success is often an imperceptible nudge —
real, just visually useless. Fix: added `total_travel` (summed per-step
gripper displacement, not just start-to-end distance) to
`EpisodeRecording`, searched a small bounded seed range per script, kept
the most-travel real success, and made each script's seed range disjoint
so two scripts can't land on the same episode again. Methodology note:
this is also the investigation that noticed one suspiciously-close
start-to-goal episode, which is what triggered the trivial-baseline audit
below — a visualization QA pass surfaced a validity question that no
numeric result had flagged.
See `demos/README.md`, "2026-07-27 seed-selection fix."

## 2026-07-27 — Trivial-baseline audit: is FetchReach's goal distribution just too easy?

Triggered directly by the demo-video finding above, not assumed away.
Measured rather than argued: 500 resets show a median start-to-goal
distance of 0.145m (3x the 0.05m success threshold); only 2.2% of resets
start already inside it. No-op policy succeeds 1.8% of episodes, random
0.4%, oracle 100% (median 3 steps). Every real stage result — even the
weakest (0.548, stage 6 Set A no-switch) — clears the no-op floor by 30x+,
up to ~56x for the ceiling-scoring stages. Decision: keep this as a
permanent, re-runnable script (`experiments/00_trivial_baseline_audit/`)
rather than a one-off assertion, precisely because it was one visual
observation away from never being checked at all. One caveat carried
forward, not resolved: an oracle solving in a median of 3 steps means
several stages' 1.000 scores sit at an informativeness ceiling — they can't
distinguish "very good" from "perfect," which is a measurement limit, not
a validity problem.
See `experiments/00_trivial_baseline_audit/evidence.md`.

## 2026-07-28 — Phase 2a stages 8-10: relative moves, waypoint chains, and typed commands all generalize cleanly

Ran the three remaining Phase 2a stages back to back. Stage 8 (relative
move from an arbitrary achieved position) hit a clean 1.000/1.000 across
6 directions x 3 magnitudes x 3 switch-points on all 8 healthy seeds —
zero-shot, no direction-lopsidedness. Stage 9 (waypoint chaining) initially
ran on only `seed_0` and was correctly returned INCONCLUSIVE by review;
rerun across all 8 healthy seeds resolved it — zero multi-leg failures
across 4,800 chain episodes, one geometric-difficulty artifact (a hard
fixed-seed leg 4) correctly distinguished from compounding since leg 5
always recovers. Stage 10 (the typed-command grammar/executor wrapping
both) confirmed the wiring itself introduces no divergence: goto/move/
waypoint success rates through the real parser+executor pipeline match
stages 8/9's own directly-measured numbers to within 0.001-0.010,
comfortably inside each stage's own seed-to-seed noise. The one genuinely
new measurement — `stop`'s hold-in-place drift — settles to ~0.7-2.4cm and
plateaus rather than growing, a sensible result given the policy was never
trained on a self-referential goal, reported as-is rather than asserted to
"just work." Methodology note: stage 9's single-seed misstep is the second
time this project's own multi-seed discipline (`CONTRACTS.md`) caught a
result that would have looked clean on a single lucky checkpoint — worth
treating as a standing default, not a one-off correction.
See `experiments/08_relative_move_validation/evidence.md`,
`experiments/09_waypoint_following/evidence.md`,
`experiments/10_typed_command_interface/evidence.md`.
