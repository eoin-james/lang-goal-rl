# Roadmap

End state: an RL agent that accepts ad-hoc English instructions live during an
episode and re-targets a continuous goal accordingly.

Each stage gates on a specific, falsifiable proof. Don't start the next stage
until the current one's proof passes — update the Status column as you go.

**Scope decision (locked in):** every stage runs on `FetchReach-v4` only —
the easiest task in the Fetch suite (free-space reach, no contact/object
dynamics). This is deliberate: the thesis contribution is the
language→continuous-goal→live-re-goaling mechanism, not manipulation
difficulty. A PASS on FetchReach proves the mechanism works; it does **not**
prove it survives a harder task (Push, PickAndPlace, Slide — real contact
dynamics, typically 10-50x the training budget in the literature).
Generalization to harder tasks is explicitly out of scope here and belongs
in the final writeup as future work, not as an unproven claim.

| # | Stage | Reuse | New build | Proof gate | Status | Report |
|---|-------|-------|-----------|------------|--------|--------|
| 0 | Plumbing | gymnasium-robotics `FetchReach-v4` (MuJoCo-backed), env reset/step loop | — | Loop runs end-to-end: reset, inspect Dict obs (`observation`/`achieved_goal`/`desired_goal`), step | **Done** | — |
| 1 | Goal-conditioned baseline (UVFA + HER) | SB3 `SAC` + `HerReplayBuffer`, same env | multi-goal conditioning on literal xyz goal (`experiments/01_uvfa_her_baseline/train.py`) | Near-100% success rate over held-out eval episodes on FetchReach | **Done (8/10 seeds, 2 show known SAC eval-collapse — see Known risks)** | [report](experiments/01_uvfa_her_baseline/report.md) |
| 2 | Learned continuous goal embedding | Eysenbach et al. Contrastive RL architecture (scoped adaptation — frozen encoder pretrained via InfoNCE, not a full critic-loss replacement) | swap literal goal for a learned latent (`experiments/02_contrastive_goal_embedding/`) | Success rate matches stage-1 baseline within tolerance; distance-in-latent correlates with true task distance | **Done (10/10 seeds ≥0.98, r=0.571 held-out distance correlation)** | [report](experiments/02_contrastive_goal_embedding/report.md) |
| 3 | Frozen language embedding → goal space | LIV / VLM-RM reward pipeline (frozen CLIP-text or sentence-transformer) | learned projection layer, fixed instruction vocabulary | Success rate on language goals ≈ stage-2 baseline; projection doesn't collapse distinct instructions to one point | **Done (4 attempts, 3 seeds) — 1.000 success matching stage-2 baseline exactly, collapse margin 9.70x. See Known risks for the eval-protocol lesson.** | [report](experiments/03_language_goal_projection/report.md) |
| 4 | Open vocabulary | same pipeline | k=1 zero-training nearest-neighbor lookup over an 84-sentence combined vocabulary (`experiments/04_open_vocabulary/`) — resolution changed from the originally-planned learned projection layer after 3 failed attempts; see Known risks | Graceful degradation on unseen phrasing; semantic neighbors land near each other in goal space | **Done (4 attempts, 3 seeds) — k=1 NN lookup over an 84-sentence combined vocabulary; 0.571 mean / 1.000 median RL success on 14 held-out paraphrases, zero-shot, no retraining. See Known risks for the reference-coverage scalability condition before stage 6.** | [report](experiments/04_open_vocabulary/report.md) |
| 5 | Mid-episode re-goaling | HIRO / Hi-Robot / Hindsight Instruction Relabeling as literature reference (no direct codebase) | env wrapper injecting a new instruction mid-episode (literal xyz goals — isolates the re-goaling mechanism from stages 2-4's embedding confounds) | Zero-shot goal-swap success rate vs. fresh-episode baseline; if it degrades, fine-tune with injected switches and re-measure | **Done (8/10 seeds, 2 show known SAC eval-collapse — see Known risks)** — swap success == budget-matched baseline == full-budget reference == 1.000 for every healthy seed at every switch point tested. No fine-tuning needed; non-stationarity risk did not materialize for this scope (literal goals, 50-step FetchReach). | [report](experiments/05_midepisode_regoal/report.md) |
| 6 | Live English interface | everything above + live embedding inference | real-time text → embedding → goal loop (`LiveGoalController`) | End-to-end demo across ad-hoc live phrasings: task success + time-to-redirect | **Done (3 seeds) — mechanism proven end-to-end: type an English sentence, the robot goes for it; type a different one mid-episode, it redirects with no measurable extra cost (switch success matches no-switch control within noise, p=0.71). 7 genuinely brand-new phrasings: 0.857 task success, median 3-step redirect. Coverage-density and cross-task generalization remain open, not claimed — see Known risks.** | [report](experiments/06_live_english_interface/report.md) |

_Status tags like "Done (5 seeds)" reflect the primary result; full per-seed
numbers, charts, and any candidate comparison live in the linked report._

## Known risks (from literature research, not yet resolved)

- **Metric mismatch**: sentence-transformer/CLIP-text embeddings have a
  contrastive-trained, cosine-similarity-friendly space — raw LLM hidden
  states don't. If stage 3+ moves off sentence-transformers, the
  distance-based reward needs its own justification first.
- **Non-stationarity at stage 5 (measured, did not materialize for this
  scope)**: HIRO exists because naively re-targeting broke training in an
  adjacent setting (changing low-level policy, not changing goal — but
  close enough to warrant caution). Measured directly across 10 seeds,
  4 switch points, literal xyz goals on FetchReach-v4 (50-step episodes):
  zero-shot goal-swap success matched a fair, budget-matched fresh-episode
  baseline exactly (1.000 vs 1.000) for every healthy seed — no
  degradation, no fine-tuning needed. **Scope this narrowly, don't
  over-generalize**: only tested with literal coordinates (no
  embedding/projection/language pipeline engaged) on the easiest task in
  the Fetch suite with short episodes. Stage 6 re-engages the full
  language pipeline live — embedding noise interacting with a goal-swap
  has not been tested and should not be assumed safe just because the
  literal-goal case was.
  **Stage 6 status: partially resolved.** The live embedding + mid-episode
  switch combination was tested directly (not assumed) — switch success
  tracked each test set's own no-switch control within noise (p=0.71 for
  the larger set, an exact match for the smaller one), no measurable
  extra cost from combining them. Scoped narrowly: one switch point/step
  budget, one environment, 3 seeds — a real positive result, not a general
  robustness guarantee across configurations.
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
- **Region-vs-point ground truth (confirmed stage 3, cost 3 build/fix
  rounds — apply this lesson before stage 4 starts, don't rediscover it)**:
  a language instruction ("reach up high") describes a *region* of goal
  space, but a fixed embedding per instruction can only ever represent one
  point in that region (its centroid). Judging eval success against a
  freshly-sampled random point elsewhere in the region — rather than
  against the centroid the embedding actually represents — is close to a
  geometric impossibility regardless of embedding quality: stage 3's
  measured regions were 2-6x wider than FetchReach's 0.05m success radius,
  and the resulting ~15% success ceiling matched pure geometry almost
  exactly (independently re-derived and confirmed by the reviewer), not
  embedding inaccuracy. **Fix was a one-line eval change (judge against
  the region centroid, the same point the projection was trained toward),
  not further model training** — three earlier attempts spent real effort
  improving the projection's scale and direction before this was caught.
  **Stage 4 (open vocabulary) must design its eval the same way from the
  start**: know whether an instruction's target is being judged against a
  fixed representative point or a sampled region member, and pick
  deliberately — don't let this recur by default. See
  `experiments/03_language_goal_projection/report.md`'s Attempt 4 section
  and its reviewer verdict for the full geometric analysis.
- **Projection-layer overfitting to a minimal vocabulary (confirmed stage 4)**:
  `LanguageGoalProjection` is a ~25,600-parameter MLP (384→64→16) trained via
  direct MSE regression on exactly 14 fixed input/output pairs (2 sentences
  per region). That is enough capacity to memorize the 14 points exactly with
  zero pressure to generalize between them. Evidence: a PCA of training vs.
  held-out projected points shows the two populations occupying almost
  entirely separate regions of the 16-dim output space (the visual signature
  of memorization, not a smooth mapping); held-out semantic-neighbor accuracy
  is only 2x random chance (28.6% vs. 14.3%) and RL success collapses to
  ~2% even though the frozen sentence-transformer's raw 384-dim space does
  preserve some semantic proximity for held-out phrasings (ruling out "the
  encoder itself can't tell these apart" as the cause). Being classified to
  the nearest-correct region is *necessary but not sufficient* for RL
  success — the projected point must also land within FetchReach's tight
  0.05m radius of the true centroid, and even correctly-classified held-out
  phrases often don't. **Any stage that trains a mapping (sentence embedding
  → goal space, or similar) on a small, fixed, closed vocabulary must budget
  for enough diverse examples per class to force generalization — a handful
  of points per class is a memorization risk, not a generalization
  guarantee, regardless of how good the underlying frozen embeddings are.**
  See `experiments/04_open_vocabulary/report.md`'s reviewer verdict for the
  full diagnosis and recommended fix ordering (NN-interpolation ceiling test
  → data augmentation → smoothness regularization only if still needed).
  **Ceiling test result (confirms diagnosis):** a zero-training
  nearest-neighbor baseline (blend the k nearest of the 14 training
  sentences' targets in raw 384-dim space, bypassing the learned MLP
  entirely) scores 0.714 (k=1) vs. the trained MLP's 0.286 on the identical
  14 held-out phrases — 6 instructions flip from wrong to correct with zero
  reverse flips at k=1. This confirms the raw sentence-embedding space
  already carries plenty of region-clustering signal; the learned MLP is
  actively discarding it by memorizing 14 points instead of learning a
  generalizing rule. **Data-augmentation result (attempt 2, partial fix):**
  retraining on a 70-sentence vocabulary raised semantic-neighbor accuracy
  to 0.643 (near the 0.714 ceiling, confirming augmentation fixes
  classification as predicted) but held-out RL success only reached 0.095
  mean/0.000 median — the bottleneck shifted from "wrong region" to
  "region is correct but not precise enough for the policy," and that
  imprecision is concentrated unevenly by region (see the new risk entry
  below). Data augmentation alone will not close this remaining gap.
- **Policy tolerance to goal-embedding imprecision is direction-sensitive,
  not purely region-level (confirmed stage 4, attempts 2-3; corrected from
  an earlier, stronger region-level claim)**: classification accuracy (is
  the projected point nearest the correct region?) is not a reliable proxy
  for RL success (will the policy actually reach the goal?) once
  classification is reasonably good — attempt 2 showed a case where a
  *closer*-to-centroid held-out instruction (distance 0.0151) scored 0.000
  RL success while a *farther* one (0.0200) scored 1.000. Attempt 3 ran a
  direct diagnostic (inject controlled-magnitude noise straight into each
  region's exact target embedding, no projection/sentences involved, same 3
  SAC checkpoints) to map this precisely — **the diagnostic's own reviewer
  found the clean "some regions are just more forgiving" story does not
  hold up**: all 3 SAC checkpoints agree deterministically on each
  perturbed-embedding cell, and different magnitudes at the same region
  used different random directions, so pass/fail was tracking which
  *direction* got sampled, not the magnitude — 4 of 7 regions showed
  non-monotonic success-vs-magnitude curves, 2 with a full recovery after
  an apparent "collapse." **Corrected takeaway: tolerance is
  direction-sensitive within a region, at least as much as it is
  magnitude/region-sensitive** — "closer to centroid" alone is an
  incomplete predictor of success; which direction the error points in
  matters too. See `experiments/04_open_vocabulary/report.md`'s attempt-3
  reviewer verdict for the full analysis and the resulting decisive next
  experiment (swap the trained MLP for a zero-training nearest-neighbor
  lookup on the combined vocabulary, to isolate whether the MLP's own
  learned directional distortion — not policy tolerance — is the real
  bottleneck).
  **Resolution (attempt 4, stage 4 now Done):** the MLP's own learned
  directional distortion was confirmed as the actual bottleneck, not policy
  tolerance. Replacing it entirely with a zero-training k=1
  nearest-neighbor lookup (return the nearest of 84 reference sentences'
  *exact* known-good target, never an approximation) took held-out RL
  success from 0.095 mean/0.000 median to 0.571 mean/1.000 median. k=1
  beat k=3 specifically because k=1 always lands exactly on a region
  centroid (zero directional deviation), while k=3's blended output never
  does — direct confirmation that direction-sensitivity, not distance
  alone, drives success once classification is reasonably accurate.
- **Nearest-neighbor lookup's generalization ceiling is bounded by
  reference-vocabulary coverage density (confirmed stage 4, attempt 4 —
  read before stage 6 design starts)**: stage 4 passed by replacing the
  learned projection with a k=1 nearest-neighbor lookup over 84 fixed
  reference sentences, achieving 0.571 mean/1.000 median RL success on 14
  held-out paraphrases. This works because 84 sentences already cover the
  7-region vocabulary densely enough that most held-out phrasings land
  closer (in raw sentence-embedding space) to a same-region reference than
  a wrong-region one — but the 6/14 failures are exactly the phrasings
  where that isn't true, i.e. coverage gaps, not a fundamental flaw. Stage
  6's proof gate is "ad-hoc live phrasings," implying open-ended input
  diversity that a fixed 84-sentence reference set will not densely cover.
  **Stage 5/6 must not assume this mechanism scales for free** — either
  grow the reference set substantially and validate coverage density
  empirically before relying on it, or design a hybrid mechanism (e.g.
  NN lookup for in-coverage inputs, a fallback for out-of-distribution
  ones) rather than rediscovering this gap live in stage 6, the way stage
  3's eval bug and stage 4's own MLP-overfit bug were each found the hard
  way. See `experiments/04_open_vocabulary/report.md`'s attempt-4 reviewer
  verdict for the full mechanism analysis.
  **Stage 6 status: NOT contradicted, NOT resolved.** Stage 6 tested 7
  genuinely brand-new phrasings (0.857 success, 1 failure in exactly this
  risk's predicted coverage-gap mode) — too small a sample to characterize
  an 84-sentence reference set's coverage density at the scale "ad-hoc"
  implies. Treat this mechanism as demonstrated, not as proven to
  generalize to arbitrary open-ended input — growing/validating the
  reference set remains real, unstarted future work, not a solved problem.
