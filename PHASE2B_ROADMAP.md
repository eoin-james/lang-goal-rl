# Phase 2b Roadmap

> **Pivot note (2026-07-31).** After stage 11, the project paused for an
> all-hands review and pivoted to an explicit research program — see
> [RESEARCH.md](RESEARCH.md) for the standing direction and
> [docs/research/2026-07-31-thesis-viability-assessment.md](docs/research/2026-07-31-thesis-viability-assessment.md)
> for why. Stages 12–14 are now the probe stages defined in RESEARCH.md's
> "The now" section (collision probe, negation probe, second encoder). The
> previously-planned regression-head stages 12–15 are **deferred, not
> deleted** — they remain valid as Thread 1 data points ("continuous
> parameters ground in embedding space") and will be renumbered when
> picked up. Their original rows are kept below, marked deferred, so the
> plan's history stays visible.

This is Phase 2b of the research program — see [PHASES.md](PHASES.md) for
where it fits. Phase 1's record (stages 0-6) is frozen at
[ROADMAP.md](ROADMAP.md) and Phase 2a's record (stages 7-10) is frozen at
[PHASE2_ROADMAP.md](PHASE2_ROADMAP.md); Phase 2b continues the stage
numbering from 11 so "stage N" means one consistent thing across the
whole project's history.

Phase 2b's end state: teach a learned language layer to translate
arbitrary English into Phase 2a's deterministic typed-command language
(`goto`/`move`/`waypoints`/`stop`/`reset`), instead of the old 7-fixed-
region snap. Chosen architecture: small trained heads on top of the
existing frozen sentence-transformer embeddings (`language_embedding.py`)
— no LLM API, no new dependencies.

| # | Stage | Reuse | New build | Proof gate | Status | Report |
|---|-------|-------|-----------|------------|--------|--------|
| 11 | Command-type classification | `language_embedding.encode_instructions`, `language_goal_projection.LanguageGoalProjection`'s architecture template | `command_type_vocabulary.py`, `command_type_held_out_vocabulary.py`, `command_type_classifier.py` | Held-out: (a) >=90% overall top-1 accuracy across 5 classes, (b) 0% of held-out UNSUPPORTED sentences classified as anything actionable | **Done (3 attempts) — 98.08% held-out accuracy (best config, zero seed variance), 0% UNSUPPORTED-as-actionable, 12/12 runs pass both gates. Attempt 1 FAILED on a MOVE/GOTO_NAMED_REGION vocabulary-convention collision (0% MOVE accuracy) traced to reusing Phase 1's directionally-phrased region vocabulary as GOTO_NAMED_REGION's training data. Attempt 2 fixed the collision (both classes 100%) but surfaced two narrow coverage gaps (STOP idioms, one UNSUPPORTED math question). Attempt 3 closed both additively. One non-blocking residual: two RESET idioms now collide with STOP's new vocabulary (bounded, doesn't violate either gate).** | [report](experiments/11_command_type_classification/report.md) |
| 12 | Collision probe, formalized (per [RESEARCH.md](RESEARCH.md) "The now") | Stage 11's vocabularies + trained-classifier confusion data, `language_embedding.encode_instructions` | Encoder-agnostic pre-training collision predictor (embedding-space cross-class overlap → predicted confusion) | Probe's pre-training prediction ranks the known MOVE/GOTO collision first; its score correlates with realized confusion across deliberately-constructed vocabularies | Not started | — |
| 13 | Negation probe (per [RESEARCH.md](RESEARCH.md) "The now") | Stage 12's probe machinery | Negation vocabulary + probe run ("go left" vs "don't go left"), measured as a *control-interface* failure (what does the policy do?) | Reproducible diagnostic signature for negation separability (or its absence) in frozen-embedding space | Not started | — |
| 14 | Second encoder (per [RESEARCH.md](RESEARCH.md) "The now") | Stages 12-13's probes | Probe runs against one more encoder (mpnet or GTE-small) | Both probes produce comparable results on a second encoder — the taxonomy becomes a method, not a MiniLM autopsy | Not started | — |

**Deferred (pre-pivot plan, kept for the record — renumber when picked up).**
These were the planned stages 12–15 before the 2026-07-31 pivot; they remain
valid as Thread 1 data points, currently paused behind the probe stages above.

| # (old) | Stage | Reuse | New build | Proof gate | Status | Report |
|---|-------|-------|-----------|------------|--------|--------|
| 12 | Continuous move-parameter regression | `relative_move.compute_relative_goal`/`clip_to_box`/`DIRECTION_UNIT_VECTORS`, stage 1/5's literal-xyz checkpoint family | `move_command_vocabulary.py`, `move_parameter_regression.py` | Held-out direction top-1 >=0.90 + distance MAE <=2cm (necessary); downstream RL success matches stage 8's ground-truth baseline within 0.05 absolute (actual gate) | Deferred (2026-07-31 pivot) | — |
| 13 | Compound-utterance segmentation and waypoint chaining | Old stage 12's `predict_move_command`, `relative_move.compute_relative_goal`, `waypoint_following.rollout_with_waypoints` | `compound_move_segmentation.py` | 100% correct segment boundaries (deterministic); downstream RL success matches stage 9's chain-length-matched numbers within its own per-seed spread | Deferred (2026-07-31 pivot) | — |
| 14 | End-to-end free-text grounding (integration) | Stage 11 + old stages 12-13, `interactive_demo`'s region-centroid lookup | `language_command_grounding.py` | Whole-pipeline success per command type within that type's own established tolerance; 100% of UNSUPPORTED held-out set raises `UngroundedInstructionError` | Deferred (2026-07-31 pivot) | — |
| 15 | Live capstone: third `interactive_demo.py` interface | `interactive_demo.py`'s renderer/queue plumbing, old stage 14's `ground_free_text_command` | `--interface free-language`, `run_free_language` | Human-typed live session (qualitative) + numeric harness cross-check against old stage 14's numbers | Deferred (2026-07-31 pivot) | — |

_Status tags follow Phase 1/2a's convention; full per-attempt numbers and
charts live in each stage's linked report._

## Known risks carried forward from Phase 1/2a

- **Direction-sensitivity, not just distance (stage 4)**: relevant again
  once stage 12 regresses a continuous distance per direction — check for
  direction-lopsided regression error, not just an aggregate MAE.
- **NN-lookup reference-coverage density (stage 4)**: the original
  motivation for moving to a trained regression head instead of another
  fixed-vocabulary lookup — directly load-bearing for stages 12-14.
- **Stage 7 sign-off still pending, non-blocking (Phase 2a)**: unrelated to
  Phase 2b's own work, still open.

## New risks (Phase 2b, tracked as they're found)

- **Vocabulary-convention collisions between command-type classes are a
  real, recurring failure mode, not a one-off (stage 11)**: attempt 1's
  MOVE/GOTO_NAMED_REGION collision and attempt 3's smaller RESET/STOP
  collision are the same underlying risk at two different scales — when
  two classes' training phrasings share a surface-form convention (a
  verb, an idiom, a sentence structure), a frozen sentence embedding can
  fail to separate them regardless of how much data or tuning is thrown
  at it. `command_type_vocabulary.check_cross_class_embedding_overlap`
  is the mitigation built so far (catches MOVE/GOTO-style collisions
  before a full train+eval cycle) — it does not yet cover STOP/RESET;
  extending it there is recorded as a finding, not yet built.
- **A trained classifier's "0% on one class across every hyperparameter
  config" signature reliably indicates a data-design problem, not a
  tuning problem (stage 11)**: training loss dropping to near-zero while
  held-out accuracy on a specific class stays flat at 0% across 4 very
  different configs is the diagnostic signature to watch for in stages
  12-14 too — it means look at the training data's linguistic
  conventions before spending more compute on hyperparameters.
