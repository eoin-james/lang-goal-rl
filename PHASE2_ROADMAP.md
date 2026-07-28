# Phase 2a Roadmap

This is Phase 2a of the research program — see [PHASES.md](PHASES.md) for
where it fits. Phase 1's record (stages 0-6) is frozen at [ROADMAP.md](ROADMAP.md)
and not rewritten here; Phase 2a continues the stage numbering from 7 so
"stage N" means one consistent thing across the whole project's history.

Phase 2a's end state: replace the current 7-fixed-region sentence
classification with deterministic typed commands (absolute goals, relative
moves, waypoint sequences, stop/reset) — no learned language grounding yet,
that's Phase 2b. Each stage gates on a specific, falsifiable proof gate
(stage 7 is a deliberate exception — see its row).

| # | Stage | Reuse | New build | Proof gate | Status | Report |
|---|-------|-------|-----------|------------|--------|--------|
| 7 | Frame-of-reference check | `AXIS_DIRECTIONS` (`goal_region_vocabulary.py`), `episode_recording.record_episode`'s render path | Before/after frame pairs per direction | Human visual sign-off that each direction looks correct on camera (deliberately not a numeric gate — see report) | Not started | — |
| 8 | Relative-move validation | Literal-xyz SAC+HER checkpoint (stage 1/5), `midepisode_regoal.py`'s step-loop pattern | `relative_move.py` — `compute_relative_goal`, `rollout_with_relative_move` | Reaches relative-move targets (multiple directions/magnitudes/switch-points) at a rate matching a budget-matched fresh baseline | **Done (10 seeds) — 1.000/1.000 (rm/baseline) on 8 healthy seeds across all 6 directions, 3 magnitudes, 3 switch-points; no direction-lopsidedness. Seeds 2/7 show the known SAC collapse, degrading both conditions proportionally.** | [report](experiments/08_relative_move_validation/report.md) |
| 9 | Waypoint following | Same checkpoint, `rollout_with_goal_switch` | `waypoint_following.py` — `rollout_with_waypoints` | N=2 reduces exactly to stage 5's `rollout_with_goal_switch` result (regression test); N=3-5 chains don't show compounding degradation | **Done (8 healthy seeds) — N=2 equivalence settled (inherits stage 5's 10-seed validation). Zero multi-leg failures across 4,800 chain episodes on any of 8 checkpoints; no seed scores worse than the original result or shows a monotonic-with-position pattern. One geometric-difficulty artifact found and correctly ruled non-compounding (leg recovers every time).** | [report](experiments/09_waypoint_following/report.md) |
| 10 | Typed-command interface | Stage 8/9 modules, `interactive_demo.py`'s live loop | `command_grammar.py`, `command_executor.py`, `--interface commands` flag | Scripted harness: goto/move/waypoint success rates match stages 8-9; malformed input rejected with a clear error, not a silent guess | **Done (8 healthy seeds) — goto 1.000/1.000 (pipeline/baseline, 800 episodes); move matches stage 8 to within 0.001 in every bucket (8,640 episodes); waypoints match stage 9 to within 0.010, inside its own per-seed spread. 23/23 malformed inputs correctly rejected with specific messages; 80/80 out-of-bounds gotos clipped, 0 crashes. New first-ever stop-hold-drift measurement: settles to ~0.7-2.4cm and plateaus, doesn't converge to zero (expected, reported honestly). One non-blocking gap: mid-waypoint-queue preemption unit-tested but not exercised end-to-end.** | [report](experiments/10_typed_command_interface/report.md) |

_Status tags follow Phase 1's convention; full per-seed numbers and charts
live in each stage's linked report._

## Known risks carried forward from Phase 1 (relevant to Phase 2a)

- **Direction-sensitivity, not just distance (stage 4)**: classification/
  region-correctness was not a reliable predictor of RL success once
  reasonably accurate — direction of error mattered as much as magnitude.
  Stage 8 must check for direction-lopsided success, not just an aggregate
  rate.
- **NN-lookup reference-coverage density (stage 4)**: not directly
  applicable to 2a (no language grounding yet), but load-bearing for 2b.
- **"Live" needs a precise, stated meaning (stage 6)**: don't repeat the
  ambiguity between a scripted harness and an actual human-typed session —
  stage 10 must label every number by which one produced it.

## New risks (Phase 2a, tracked as they're found)

- **Stage 7 sign-off still pending, non-blocking**: `DIRECTION_UNIT_VECTORS`
  in `relative_move.py` mirrors `AXIS_DIRECTIONS`'s labels, unconfirmed
  against what a viewer actually sees on camera. Stage 8's PASS is about
  the relative-move *mechanism*, not the *labels* — a human visual sign-off
  on stage 7's clips (`experiments/07_frame_of_reference_check/`) is still
  needed before treating "reach left" as verified to mean visual-left.
- **`Stop`'s hold-in-place drift is real but non-zero (stage 10)**: the
  gripper settles to roughly 0.7-2.4cm of residual drift after a `stop`
  command and stays there (doesn't keep drifting) — expected, since the
  policy was never trained on a goal equal to its own current position.
  Adequate for this milestone; worth knowing before leaning on `stop` for
  anything precision-sensitive later.
- **Mid-waypoint-queue preemption untested end-to-end (stage 10)**: the
  behavior is implemented and unit-tested (`test_command_executor.py`),
  but the scripted harness never issued a new command while a waypoint
  chain was mid-execution through the real env/policy. Low risk (simple
  mechanism, already unit-tested) but flagged for a future robustness pass
  rather than silently assumed covered.
