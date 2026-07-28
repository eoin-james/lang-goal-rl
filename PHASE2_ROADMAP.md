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
| 8 | Relative-move validation | Literal-xyz SAC+HER checkpoint (stage 1/5), `midepisode_regoal.py`'s step-loop pattern | `relative_move.py` — `compute_relative_goal`, `rollout_with_relative_move` | Reaches relative-move targets (multiple directions/magnitudes/switch-points) at a rate matching a budget-matched fresh baseline | Not started | — |
| 9 | Waypoint following | Same checkpoint, `rollout_with_goal_switch` | `waypoint_following.py` — `rollout_with_waypoints` | N=2 reduces exactly to stage 5's `rollout_with_goal_switch` result (regression test); N=3-5 chains don't show compounding degradation | Not started | — |
| 10 | Typed-command interface | Stage 8/9 modules, `interactive_demo.py`'s live loop | `command_grammar.py`, `command_executor.py`, `--interface commands` flag | Scripted harness: goto/move/waypoint success rates match stages 8-9; malformed input rejected with a clear error, not a silent guess | Not started | — |

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

_(none yet — stage 7 not started)_
