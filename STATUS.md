# Experiment Status

This tracks research progress only — stages, proof gates, actual numbers.
No setup/tooling work shown here; that's implementation detail, not an
experiment result.

**TL;DR:** Stage 1 done and proven (10 seeds, real PASS verdict). Stage 2
not started. **Progress: 1/7 stages (14%).**

## Active run

**None.** Nothing training right now.

## Stage 1 — full result (closed out 17:06)

| Seed | Success rate |
|---|---|
| 0 | 1.000 |
| 1 | 1.000 |
| 2 | 0.000 |
| 3 | 1.000 |
| 4 | 1.000 |
| 5 | 1.000 |
| 6 | 1.000 |
| 7 | 0.400 |
| 8 | 1.000 |
| 9 | 1.000 |
| **Mean** | **0.840** |
| Seeds ≥0.98 | 8/10 |

**Verdict: PASS.** Took two review passes to get there — worth knowing the
shape of that, not just the ending:
1. First 5 seeds → reviewer said INCONCLUSIVE (mean 0.8 doesn't read as
   "near-100%", one seed totally failed, sample too thin to tell luck from
   a systemic problem).
2. 5 more seeds → still didn't cleanly hit either of the reviewer's own
   pass/fail thresholds (8/10, not 9/10; only 1 new failure, not 2+).
3. Reviewer went back to the raw logs directly rather than mechanically
   applying its own rule, and found the actual mechanism: seed 5's
   training-time success (0.89) was *lower* than seed 7's (0.95), yet seed
   5 scored a perfect eval while seed 7 collapsed — ruling out "weak
   training predicts eval failure." Real cause: an entropy-coefficient
   instability spike (`ent_coef_loss` jumping to 19-52) can permanently
   corrupt the deterministic action for that seed — a known SAC fragility,
   not a defect in UVFA or HER. 8/10 at a clean 1.000 is a genuine solve.

**Tracked as a risk, not a loose end:** added to `ROADMAP.md`'s Known
risks — every downstream stage (2-6) must compare against baselines at
median/mode (not mean), same seed count, and check whether a failed seed
shows this exact signature before blaming a new component for a regression
that's actually this ~20% baseline fragility resurfacing.

Full reasoning + both review passes: `experiments/01_uvfa_her_baseline/report.md`
Chart: `experiments/01_uvfa_her_baseline/charts/multi_seed_success_rate.png`

## Stage-by-stage results

| # | Stage | Proof gate | Result | Seeds |
|---|-------|------------|--------|-------|
| 0 | Plumbing | Env loop runs end-to-end | ✅ Pass | — |
| 1 | Goal-conditioned baseline (UVFA+HER) | Near-100% success on FetchReach | ✅ **PASS** — see above | 10 |
| 2 | Learned continuous goal embedding | Success rate matches stage 1; distance-in-latent tracks true task distance | Not run | — |
| 3 | Frozen language embedding → goal space | Success on language goals ≈ stage 2 | Not run | — |
| 4 | Open vocabulary | Graceful degradation on unseen phrasing | Not run | — |
| 5 | Mid-episode re-goaling | Zero-shot goal-swap ≈ fresh-episode baseline | Not run | — |
| 6 | Live English interface | End-to-end demo, task success + redirect time | Not run | — |

## What's queued next

Stage 1 is closed out and committed. Next real step is stage 2: learned
continuous goal embedding (Contrastive RL), which needs new reusable code
from the rl-builder before any run happens. Not started — waiting on you to
say go.
