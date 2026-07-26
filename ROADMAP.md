# Roadmap

End state: an RL agent that accepts ad-hoc English instructions live during an
episode and re-targets a continuous goal accordingly.

Each stage gates on a specific, falsifiable proof. Don't start the next stage
until the current one's proof passes — update the Status column as you go.

| # | Stage | Reuse | New build | Proof gate | Status |
|---|-------|-------|-----------|------------|--------|
| 0 | Plumbing | gymnasium-robotics `FetchReach-v4` (MuJoCo-backed), env reset/step loop | — | Loop runs end-to-end: reset, inspect Dict obs (`observation`/`achieved_goal`/`desired_goal`), step | **Done** |
| 1 | Goal-conditioned baseline (UVFA + HER) | SB3 `SAC` + `HerReplayBuffer`, same env | multi-goal conditioning on literal xyz goal (`experiments/01_uvfa_her_baseline/train.py`) | Near-100% success rate over held-out eval episodes on FetchReach | **Done** — 100% over 50 eval episodes (20k timesteps, seed 0) |
| 2 | Learned continuous goal embedding | Eysenbach et al. Contrastive RL architecture | swap literal goal for a learned latent | Success rate matches stage-1 baseline within tolerance; distance-in-latent correlates with true task distance | Not started |
| 3 | Frozen language embedding → goal space | LIV / VLM-RM reward pipeline (frozen CLIP-text or sentence-transformer) | learned projection layer, fixed instruction vocabulary | Success rate on language goals ≈ stage-2 baseline; projection doesn't collapse distinct instructions to one point | Not started |
| 4 | Open vocabulary | same pipeline | held-out paraphrases / compositional instructions | Graceful degradation on unseen phrasing; semantic neighbors land near each other in goal space | Not started |
| 5 | Mid-episode re-goaling | HIRO / Hi-Robot / Hindsight Instruction Relabeling as literature reference (no direct codebase) | env wrapper injecting a new instruction mid-episode | Zero-shot goal-swap success rate vs. fresh-episode baseline; if it degrades, fine-tune with injected switches and re-measure | Not started |
| 6 | Live English interface | everything above + live embedding inference | real-time text → embedding → goal loop | End-to-end demo across ad-hoc live phrasings: task success + time-to-redirect | Not started |

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
