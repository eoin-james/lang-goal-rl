# Stage 5: Mid-episode re-goaling
**Date:** 2026-07-27 **Seeds run:** [0, 1, 2] **Candidates:** swap, budget_matched_baseline, full_budget_reference

## Proof gate (verbatim from ROADMAP.md)
> Zero-shot goal-swap success rate vs. fresh-episode baseline; if it degrades, fine-tune with injected switches and re-measure.

## Result summary
### Checkpoint-provisioning sanity check
(literal-goal control, reused checkpoints only -- see "Checkpoint provisioning" below)

| Seed | Sanity success rate (literal control, full 50-step, no swap) | Episodes |
|---|---|---|
| 0 | 1.000 | 50 |
| 1 | 1.000 | 50 |
| 2 | 0.000 | 50 |
| **Mean** | **0.667** | |
| **Median** | **1.000** | |

### Proof-gate comparison: swap vs. budget-matched baseline (per seed, per switch_step)

| Seed | switch_step | Swap success rate | Budget-matched baseline success rate | Full-budget reference success rate | Episodes |
|---|---|---|---|---|---|
| 0 | 10 | 1.000 | 1.000 | 1.000 | 40 |
| 0 | 20 | 1.000 | 1.000 | 1.000 | 40 |
| 0 | 30 | 1.000 | 1.000 | 1.000 | 40 |
| 0 | 40 | 1.000 | 1.000 | 1.000 | 40 |
| 1 | 10 | 1.000 | 1.000 | 1.000 | 40 |
| 1 | 20 | 1.000 | 1.000 | 1.000 | 40 |
| 1 | 30 | 1.000 | 1.000 | 1.000 | 40 |
| 1 | 40 | 1.000 | 1.000 | 1.000 | 40 |
| 2 | 10 | 0.000 | 0.000 | 0.000 | 40 |
| 2 | 20 | 0.000 | 0.000 | 0.000 | 40 |
| 2 | 30 | 0.000 | 0.000 | 0.000 | 40 |
| 2 | 40 | 0.050 | 0.025 | 0.000 | 40 |

### Proof-gate comparison: cross-seed aggregate per switch_step

| switch_step | Swap mean | Swap median | Baseline mean | Baseline median | Full-budget mean | Full-budget median |
|---|---|---|---|---|---|---|
| 10 | 0.667 | 1.000 | 0.667 | 1.000 | 0.667 | 1.000 |
| 20 | 0.667 | 1.000 | 0.667 | 1.000 | 0.667 | 1.000 |
| 30 | 0.667 | 1.000 | 0.667 | 1.000 | 0.667 | 1.000 |
| 40 | 0.683 | 1.000 | 0.675 | 1.000 | 0.667 | 1.000 |


## Charts
![sanity_check_success_rate.png](/Users/eoinmca/Projects/lang-goal-rl/experiments/05_midepisode_regoal/charts/sanity_check_success_rate.png)

![switch_step_10_comparison.png](/Users/eoinmca/Projects/lang-goal-rl/experiments/05_midepisode_regoal/charts/switch_step_10_comparison.png)

![switch_step_20_comparison.png](/Users/eoinmca/Projects/lang-goal-rl/experiments/05_midepisode_regoal/charts/switch_step_20_comparison.png)

![switch_step_30_comparison.png](/Users/eoinmca/Projects/lang-goal-rl/experiments/05_midepisode_regoal/charts/switch_step_30_comparison.png)

![switch_step_40_comparison.png](/Users/eoinmca/Projects/lang-goal-rl/experiments/05_midepisode_regoal/charts/switch_step_40_comparison.png)

## Raw output
- [stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/05_midepisode_regoal/runs/seed_0/stdout.log)
- [stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/05_midepisode_regoal/runs/seed_1/stdout.log)
- [stdout.log](/Users/eoinmca/Projects/lang-goal-rl/experiments/05_midepisode_regoal/runs/seed_2/stdout.log)

## Anomalies (factual, not judged)
**Checkpoint provisioning (side task, not stage 5's own proof gate):** stage 1's `experiments/01_uvfa_her_baseline/train.py` never called `model.save(...)` despite being marked Done in ROADMAP.md, so no checkpoint existed on disk. `experiments/01_uvfa_her_baseline/provision_checkpoints.py` retrained 3 seeds using the exact same `build_model`/`evaluate` helpers and hyperparameters as `train.py` (imported directly, not copied) and added the one missing step -- `model.save(...)` -- persisting them to `experiments/01_uvfa_her_baseline/checkpoints/seed_<k>.zip`, with training logs under `experiments/01_uvfa_her_baseline/runs/seed_<k>/stdout.log`. This does not touch or supersede stage 1's own report.md/ROADMAP status -- it is purely a checkpoint-provisioning step in service of stage 5 (and any future stage needing a literal-goal policy). The sanity-check table above re-runs stage 1's own literal-goal eval protocol against these freshly-provisioned checkpoints, to confirm they still perform the base task before any swap result is trusted.

seed 2's literal-goal sanity check scored 0.000 (< 0.8) -- resembles the known SAC deterministic-eval collapse signature (ROADMAP.md Known risks), not necessarily a stage-5 mechanism defect.

## Known-risks cross-check
**Non-stationarity at stage 5**: this is exactly the risk this experiment measures -- see the swap-vs-baseline comparison above for whether the zero-shot goal-swap degrades relative to a fresh episode. **SAC deterministic-eval collapse (~20% of seeds, confirmed stage 1)**: checked via the sanity-check table above before trusting any swap result; see Anomalies for any seed that resembles the collapse signature. **Region-vs-point ground truth** and **NN-lookup coverage density**: not applicable here -- this stage uses exact literal xyz goals throughout (no embedding substitution engaged), deliberately isolating the re-goaling mechanism from every embedding-layer confound stages 2-4 spent effort on.

## Reviewer verdict
_Left blank by the runner — filled in by the manager from the reviewer's
return._
