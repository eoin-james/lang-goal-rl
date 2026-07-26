# Roadmap

End state: an RL agent that accepts ad-hoc English instructions live during an
episode and re-targets a continuous goal accordingly.

Each stage gates on a specific, falsifiable proof. Don't start the next stage
until the current one's proof passes — update the Status column as you go.

| # | Stage | Reuse | New build | Proof gate | Status | Report |
|---|-------|-------|-----------|------------|--------|--------|
| 0 | Plumbing | gymnasium-robotics `FetchReach-v4` (MuJoCo-backed), env reset/step loop | — | Loop runs end-to-end: reset, inspect Dict obs (`observation`/`achieved_goal`/`desired_goal`), step | **Done** | — |
| 1 | Goal-conditioned baseline (UVFA + HER) | SB3 `SAC` + `HerReplayBuffer`, same env | multi-goal conditioning on literal xyz goal (`experiments/01_uvfa_her_baseline/train.py`) | Near-100% success rate over held-out eval episodes on FetchReach | **Done (8/10 seeds, 2 show known SAC eval-collapse — see Known risks)** | [report](experiments/01_uvfa_her_baseline/report.md) |
| 2 | Learned continuous goal embedding | Eysenbach et al. Contrastive RL architecture (scoped adaptation — frozen encoder pretrained via InfoNCE, not a full critic-loss replacement) | swap literal goal for a learned latent (`experiments/02_contrastive_goal_embedding/`) | Success rate matches stage-1 baseline within tolerance; distance-in-latent correlates with true task distance | **Done (10/10 seeds ≥0.98, r=0.571 held-out distance correlation)** | [report](experiments/02_contrastive_goal_embedding/report.md) |
| 3 | Frozen language embedding → goal space | LIV / VLM-RM reward pipeline (frozen CLIP-text or sentence-transformer) | learned projection layer, fixed instruction vocabulary | Success rate on language goals ≈ stage-2 baseline; projection doesn't collapse distinct instructions to one point | Not started | — |
| 4 | Open vocabulary | same pipeline | held-out paraphrases / compositional instructions | Graceful degradation on unseen phrasing; semantic neighbors land near each other in goal space | Not started | — |
| 5 | Mid-episode re-goaling | HIRO / Hi-Robot / Hindsight Instruction Relabeling as literature reference (no direct codebase) | env wrapper injecting a new instruction mid-episode | Zero-shot goal-swap success rate vs. fresh-episode baseline; if it degrades, fine-tune with injected switches and re-measure | Not started | — |
| 6 | Live English interface | everything above + live embedding inference | real-time text → embedding → goal loop | End-to-end demo across ad-hoc live phrasings: task success + time-to-redirect | Not started | — |

_Status tags like "Done (5 seeds)" reflect the primary result; full per-seed
numbers, charts, and any candidate comparison live in the linked report._

## Known risks (from literature research, not yet resolved)

- **Metric mismatch**: sentence-transformer/CLIP-text embeddings have a
  contrastive-trained, cosine-similarity-friendly space — raw LLM hidden
  states don't. If stage 3+ moves off sentence-transformers, the
  distance-based reward needs its own justification first.
- **Non-stationarity at stage 5**: HIRO exists because naively re-targeting
  broke training in an adjacent setting (changing low-level policy, not
  changing goal — but close enough to warrant caution). Don't assume the
  zero-shot goal-swap works until measured.
- No verified prior work does stage 5 or 6 as described — that's the actual
  thesis contribution, not a reproduction.
- **SAC deterministic-eval collapse (~20% of seeds, confirmed stage 1)**:
  across 10 seeds of the stage-1 baseline, 2 (seeds 2 and 7) showed a good
  training-time success curve followed by a collapsed or degraded
  deterministic eval score, preceded by an entropy-coefficient instability
  spike (`ent_coef_loss` jumping to 19-52). This is a SAC
  exploration/exploitation fragility, not a UVFA/HER defect — ruled out
  "low training success predicts eval failure" directly from raw logs
  (a lower-training-success seed scored a perfect eval). **Every downstream
  stage must compare against baselines using the same seed count (10),
  judge at median/mode not mean, and check whether a failed seed shows this
  exact signature before attributing a regression to the new component
  (embedding, language projection, etc.) being tested.** See
  `experiments/01_uvfa_her_baseline/report.md`'s Pass 2 reviewer verdict for
  full reasoning.
