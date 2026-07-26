# Literature

Bibliography for `lang-goal-rl`, organized by the `ROADMAP.md` stage(s) each
source grounds. Every entry was link-checked via WebFetch on 2026-07-26.

Legend:
- **Confirmed** — adversarially verified (3-vote) by a prior deep-research
  pass on this exact project's research question, before this repo existed.
  Link-checked again here.
- **Unverified** — surfaced by that research pass but not independently
  checked for correctness at the time. Link-checked here; content match
  noted where it could be confirmed or refuted.
- **Link status** — outcome of the WebFetch check performed while building
  this file (live / dead / redirects / **content mismatch**).

---

## Stage 0 — Plumbing

No literature citations — this stage reuses `gymnasium-robotics`
`FetchReach-v4` env mechanics directly, no paper claim to trace.

- Farama Gymnasium-Robotics, Fetch environments.
  https://robotics.farama.org/envs/fetch/reach/ — **Confirmed, link live.**
  Verified the page documents `FetchReach-v4` (version history: v4 is a bug
  fix on top of v3, "initial state did not match initial state description
  in documentation") and the Dict observation space
  (`observation`/`achieved_goal`/`desired_goal`) that stage 0's proof gate
  checks for. Note: the top-level `/envs/fetch/` index page only lists `v3`
  in its example code — the version detail lives on the per-env page
  (`/envs/fetch/reach/`), not the index.

---

## Stage 1 — Goal-conditioned baseline (UVFA + HER)

**Reuse target:** SB3 `SAC` + `HerReplayBuffer`, same env.

- Schaul, Horgan, Gregor, Silver. "Universal Value Function Approximators."
  ICML 2015 (PMLR vol. 37, pp. 1312–1320).
  https://proceedings.mlr.press/v37/schaul15.html — **Confirmed, link live.**
  Introduces V(s,g;θ) generalizing over states *and* goals — the framework
  this project's UVFA+HER baseline (stage 1) directly implements.

- Andrychowicz et al. "Hindsight Experience Replay." arXiv:1707.01495,
  2017 (last revised 2018). https://arxiv.org/abs/1707.01495 —
  **Confirmed, link live.** No formal venue listed on the arXiv page itself
  (commonly cited as NeurIPS 2017 in secondary sources — not verified
  against the arXiv metadata directly). Relabels achieved states as goals
  for sparse-reward learning; this is the HER mechanism stage 1 uses via
  SB3's `HerReplayBuffer`.

- Stable-Baselines3, `HerReplayBuffer` docs.
  https://stable-baselines3.readthedocs.io/en/master/modules/her.html —
  **Confirmed, link live.** Verified it documents the exact mechanism
  stage 1's `train.py` depends on: Dict-obs env contract
  (`observation`/`achieved_goal`/`desired_goal`), `env.compute_reward()`
  requirement, and the `future`/`final`/`episode` goal-sampling strategies.

- Ren, Dong, Zhou, Liu, Peng. "Exploration via Hindsight Goal Generation"
  (HGG). NeurIPS 2019. https://arxiv.org/abs/1906.04279 —
  **Confirmed, link live.** Not used in this project's code — cited as an
  improvement on HER's heuristic imaginary-goal construction, relevant if
  stage 1's goal-sampling strategy is ever revisited.

- Pong, Gu, Dalal, Levine. "Temporal Difference Models: Model-Free Deep RL
  for Model-Based Control" (TDM). ICLR 2018.
  https://arxiv.org/abs/1802.09081 — **Confirmed, link live.** UVFA-derived
  family extending to model-based control — background context, not used
  directly.

- Liu, Zhu, Zhang. "Goal-Conditioned Reinforcement Learning: Problems and
  Solutions." IJCAI-ECAI 2022 Survey Track. https://arxiv.org/abs/2201.08299
  — **Unverified claim about content, link live.** The abstract-page fetch
  confirmed the paper exists and is the GCRL survey in question, but could
  *not* confirm from the abstract alone the specific claim attributed to it
  (that it documents UVFA/HER as the backbone for successors and states
  language goals require embedding via a pretrained model or an RNN) — that
  claim would need the full PDF, not just the abstract page, to verify. Flag
  as plausible-but-not-independently-confirmed; relevant to stage 3's
  design rationale either way.

---

## Stage 2 — Learned continuous goal embedding

**Reuse target:** Eysenbach et al. Contrastive RL architecture (scoped
adaptation).

- Eysenbach, Zhang, Salakhutdinov, Levine. "Contrastive Learning as
  Goal-Conditioned Reinforcement Learning." NeurIPS 2022.
  https://arxiv.org/abs/2206.07568 — **Confirmed, link live.** Proves
  contrastive-learned representations' inner product equals a
  goal-conditioned value function.

  **Scope note (important — don't let the citation overclaim):** this
  project's stage 2 (`experiments/02_contrastive_goal_embedding/`) is a
  *scoped adaptation* of this paper's idea — a frozen encoder pretrained via
  InfoNCE, swapped in as a feature extractor — **not** the paper's actual
  contribution, which is replacing the critic loss itself with a contrastive
  objective. `ROADMAP.md`'s own Reuse column already states this
  explicitly; repeating it here so `LITERATURE.md` doesn't imply full
  reproduction if read on its own.

- Nair, Pong, Dalal, Bahl, Lin, Levine. "Visual Reinforcement Learning with
  Imagined Goals" (RIG). NeurIPS 2018. https://arxiv.org/abs/1807.04742 —
  **Confirmed, link live.** VAE-based continuous latent goal space serving
  goal-sampling, observation-transform, and reward simultaneously. This is
  the architectural alternative stage 2 chose *not* to use — kept here for
  contrast, not as a reuse source.

- Pong, Dalal, Lin, Nair, Bahl, Levine. "Skew-Fit: State-Covering
  Self-Supervised Reinforcement Learning." ICML 2020.
  https://arxiv.org/abs/1903.03698 — **Confirmed, link live.**
  Information-theoretic exploration objective for continuous goal spaces —
  the second alternative to RIG that stage 2 also chose not to use.

---

## Stage 3 — Frozen language embedding → goal space

**Reuse target:** LIV / VLM-RM reward pipeline (frozen CLIP-text or
sentence-transformer).

- Ma, Liang, Som, Kumar, Zhang, Bastani, Jayaraman. "LIV: Language-Image
  Representations and Rewards for Robotic Control." ICML 2023 (extended
  version). https://arxiv.org/abs/2306.00958 — **Confirmed, link live.**
  Unified vision-language representation from action-free video —
  implicitly a universal value function for language-or-image goals.

- Rocamonde, Montesinos, Nava, Perez, Lindner. "Vision-Language Models are
  Zero-Shot Reward Models for Reinforcement Learning" (VLM-RM). ICLR 2024.
  https://arxiv.org/abs/2310.12921 — **Confirmed, link live.** Frozen CLIP
  similarity as RL reward, no manual reward engineering — the direct
  reference for stage 3's frozen-embedding approach.

- Adeniji, Xie, Sferrazza, Seo, James, Abbeel. "Language Reward Modulation
  for Pretraining Reinforcement Learning" (LAMP). 2023.
  https://arxiv.org/abs/2308.12270 — **Confirmed, link live.** Frozen VLM
  contrastive alignment used as an *exploration* reward (pretraining phase),
  not a task reward directly — relevant as a nearby but distinct use of
  frozen VLM embeddings.

- Rana, Melnik, Sünderhauf. "CLASP: Contrastive Language, Action, and State
  Pre-training for Robot Learning." 2023. https://arxiv.org/abs/2304.10782
  — **Confirmed, link live.** Extends CLIP-style contrastive learning to
  *distributional* embeddings (not point embeddings) for one-to-many
  language-behavior relationships. Relevant primarily to stage 4 (see
  below) but grounded here because it's a variant of stage 3's core
  approach.

**Known risk carried into stage 3+ (from `ROADMAP.md`'s Known risks
section):** sentence-transformer/CLIP-text embeddings have a
contrastive-trained, cosine-similarity-friendly space — raw LLM hidden
states don't. Stage 3's actual failure mode (collapse check passes,
success rate ~0%, diagnosed root cause per its report) is a live instance
of this exact concern; see `experiments/03_language_goal_projection/report.md`
rather than this file for the diagnosis itself — out of this file's scope.

---

## Stage 4 — Open vocabulary

**Reuse target:** same pipeline as stage 3 (frozen embedding + projection),
applied to held-out paraphrases / compositional instructions.

- Rana, Melnik, Sünderhauf. "CLASP." https://arxiv.org/abs/2304.10782 —
  **Confirmed, link live** (same entry as stage 3 above). Distributional
  embeddings are the direct reference if stage 4's open-vocabulary work
  needs to handle ambiguous instructions that map to more than one valid
  goal region rather than a single point.

No stage-4-specific new sources beyond what's already cited in stage 3 —
the research pass treated stage 4 as an extension of stage 3's pipeline,
not a new architecture.

---

## Stage 5 — Mid-episode re-goaling

**Reuse target:** HIRO / Hi-Robot / Hindsight Instruction Relabeling, cited
in `ROADMAP.md` as *literature reference only* — no direct codebase reuse.
This stage is the project's actual novel contribution; the sources below
are context/caution, not implementation targets.

- Nachum, Gu, Lee, Levine. "Data-Efficient Hierarchical Reinforcement
  Learning" (HIRO). NeurIPS 2018. https://arxiv.org/abs/1805.08296 —
  **Confirmed, link live.** Introduces an off-policy correction for
  non-stationarity when a *lower-level policy* changes during training.

  **Verify the ROADMAP's characterization:** `ROADMAP.md`'s Known risks
  section cites HIRO as the reason stage 5 needs caution, describing the
  non-stationarity as arising when "a low-level policy changes during
  training." That is accurate to what HIRO actually addresses — but note
  the analogy is inexact: HIRO's non-stationarity comes from the *lower*
  policy shifting under a fixed higher-level goal-proposal scheme; stage
  5's setting is the *goal itself* changing under a fixed policy hierarchy
  (no hierarchy at all, in fact — single flat policy, external instruction
  swap). `ROADMAP.md` already hedges this with "close enough to warrant
  caution" — that hedge is doing real work; don't let a future reader
  round it up to "HIRO proves this is a problem here."

- "Hi Robot" (Physical Intelligence). https://www.pi.website/download/hirobot.pdf
  — **Link check failed — file too large for WebFetch's content-length cap
  (>10MB), could not verify content directly.** Could not confirm title/
  authors/claims independently in this pass. Treat as **unverified** until
  someone fetches it directly (browser download, not WebFetch) or finds an
  arXiv mirror with a normal abstract page.

- Hindsight Instruction Relabeling (HIR) — **citation problem found,
  flagging rather than silently fixing:**
  - https://www.emergentmind.com/topics/hindsight-instruction-relabeling-hir
    — **Link live**, but this is a topic-aggregation page, not a paper. It
    does not name one canonical "Hindsight Instruction Relabeling" paper —
    it lists a family of related-but-distinct works (HIGhER — Cideron et
    al. 2019, arXiv:1910.09451; ETHER — Denamganaï et al. 2023,
    arXiv:2307.15494; SPRINT — Zhang et al. 2023, arXiv:2306.11886;
    Röder et al. 2022, arXiv:2204.04308, "Grounding Hindsight Instructions
    in Multi-Goal RL for Robotics" — this last one is the closest literal
    match to "HER-style relabeling extended to language-instruction
    goals").
  - https://arxiv.org/pdf/2406.05881v4 — **Content mismatch.** This
    resolves to "LGR2: Language Guided Reward Relabeling for Accelerating
    Hierarchical Reinforcement Learning" by Singh, Bhattacharyya, Namboodiri
    (2024), **not** a paper titled or framed as "Hindsight Instruction
    Relabeling." It is a real, relevant paper (language-guided reward
    relabeling for HRL) but was mis-paired with the "HIR" label in the
    source list handed to this task.
  - **Recommendation:** if "HIR" needs a single citable anchor for stage 5,
    use Röder et al. 2022 (arXiv:2204.04308) — it's the one entry on the
    emergentmind list that literally does HER-style relabeling for language
    instructions in a robotics multi-goal RL setting. Not yet independently
    link-checked here — do that before citing it as confirmed.

- "LGR2" — **two different papers share this abbreviation; source list
  conflated them:**
  - https://rlg.mlanctot.info/papers/AAAI22-RLG_paper_20.pdf — **Dead link.**
    TLS certificate mismatch (`rlg.mlanctot.info` presents a cert for
    `mlanctot.info` only); retried against `mlanctot.info` directly at the
    same path — 404. Could not verify this source's existence, title, or
    claims at all. The "LLM-generated reward functions for hierarchical RL"
    description in the source list is **unverified** and currently
    unconfirmable via this link.
  - arXiv:2406.05881v4, fetched above under the "HIR" entry, is titled
    "LGR2: Language Guided Reward Relabeling for Accelerating Hierarchical
    Reinforcement Learning" (Singh et al., 2024) — a *different* paper that
    happens to share the same "LGR2" abbreviation. **Confirmed to exist and
    be about language-guided reward relabeling for HRL** (matches the
    "decoupling high-level reward from low-level policy changes" framing in
    the source list reasonably well), but it is not the AAAI'22 workshop
    paper the dead link pointed at. If stage 5 needs an "LGR2" citation,
    use arXiv:2406.05881v4 (Singh et al. 2024) and drop the dead
    mlanctot.info link entirely rather than trying to resurrect it.

- Qiu, Mao, Zhu — LTL-specification goal-conditioned RL, NeurIPS 2023 —
  **citation problem found, link does not match the claimed paper.**
  https://arxiv.org/abs/2205.13044 resolves to "Near-Optimal Goal-Oriented
  Reinforcement Learning in Non-Stationary Environments" by Liyu Chen and
  Haipeng Luo (2022) — a theoretical regret-bounds paper, not a
  goal-conditioned RL / LTL-specification paper, and not by Qiu/Mao/Zhu.
  This project has no WebSearch tool available in this session to locate
  the actual Qiu/Mao/Zhu NeurIPS 2023 LTL paper — **do not cite
  arXiv:2205.13044 for this claim.** Left as an open gap; whoever picks
  this up next should search arXiv/NeurIPS 2023 proceedings directly for
  "goal-conditioned reinforcement learning" + "linear temporal logic" +
  "zero-shot" rather than trusting this URL.

- Bridging Language and Action survey. https://arxiv.org/abs/2312.10807 —
  **Confirmed, link live** (used the `/abs/` page instead of `/pdf/`,
  which exceeded WebFetch's size cap). "Bridging Language and Action: A
  Survey of Language-Conditioned Robot Manipulation," Yao, Zhou, Mees,
  Meng, Xiao, Bisk, Oh, Johns, Shridhar, Shah, Thomason, Huang, Chai, Bing,
  Knoll — first submitted Dec 2023, revised since. Categorizes
  language-conditioned manipulation methods (language for state
  evaluation, as policy condition, for planning/reasoning, and unified
  VLA models) — useful framing for where this project's approach
  (language → learned goal embedding → policy, not language-as-input-to-a
  single VLA) sits in the landscape.

- He, Myers et al. "Goal Representations for Instruction Following"
  (GRIF). CoRL 2023. Underlying paper: https://arxiv.org/abs/2307.00117
  (found via the BAIR blog post, https://bair.berkeley.edu/blog/2023/10/17/grif/
  — both **confirmed, links live**). Jointly trains language- and
  goal-conditioned policies with an aligned representation space (InfoNCE
  alignment between instruction and goal-image embeddings) — directly
  relevant to stage 5/6's need to reconcile a language-derived goal
  representation with the same continuous goal space stage 2 built.

- HiRL — **citation problem found, link does not match the claimed
  paper.** https://arxiv.org/abs/2106.13687 resolves to "panda-gym:
  Open-source goal-conditioned environments for robotic learning" by
  Gallouédec, Cazin, Dellandréa, Chen (NeurIPS 2021 Workshop on Robot
  Learning), **not** a paper about hierarchical RL over a continuous goal
  space. Note this is the same panda-gym the project's own history
  mentions switching away from (pybullet build failure) — the arXiv ID in
  the source list appears to have been transcribed incorrectly and landed
  on an unrelated-but-topically-adjacent paper from this project's own
  past. Could not identify the actual "HiRL" paper in this pass (no
  WebSearch available) — flagging as an open gap rather than guessing.

---

## Stage 6 — Live English interface

**Reuse target:** everything above + live embedding inference.

No new sources beyond stages 3–5. The relevant combination is: LIV/VLM-RM
(stage 3, frozen embedding), GRIF (stage 5, aligned language/goal
representations), Hi-Robot (stage 5, situated live instruction-following —
unverified per above). This stage's proof gate (end-to-end demo, live
phrasings) is an integration test of the prior stages' components, not a
new literature claim.

---

## Benchmarks / codebases actually used by this project

- Gymnasium-Robotics Fetch envs, `FetchReach-v4`.
  https://robotics.farama.org/envs/fetch/reach/ — **Confirmed, link live**
  (see stage 0 above; verified v4 exists and the Dict obs contract).

- Stable-Baselines3 `HerReplayBuffer`.
  https://stable-baselines3.readthedocs.io/en/master/modules/her.html —
  **Confirmed, link live** (see stage 1 above).

Context note (not independently verified against a project decision log —
stated in the task brief, not re-checked): the project reportedly switched
from `panda-gym` to Gymnasium-Robotics Fetch envs after a `pybullet` build
failure. Consistent with panda-gym's dependency on PyBullet
(confirmed via the panda-gym paper fetched by accident above, arXiv:2106.13687).

---

## Currency gap — open question, not resolved in this pass

All language/VLM-embedding papers cited for stage 3+ (LIV, VLM-RM, LAMP,
CLASP) date to 2023. The original deep-research pass found no confirmed
2024–2026 work using **raw LLM hidden states** (as opposed to
CLIP-text/sentence-transformer embeddings) as a goal-conditioning
representation — which matters directly for `ROADMAP.md`'s Known risks
"Metric mismatch" entry (raw LLM hidden states aren't contrastive-trained
into a cosine-similarity-friendly space the way sentence-transformer/CLIP
embeddings are).

**This pass could not re-check that gap**: no WebSearch tool was available
in this session (only WebFetch, which requires a URL up front and can't
discover new papers). Whoever next has WebSearch access should run one
fresh query — e.g. "goal-conditioned reinforcement learning LLM hidden
state embedding 2025" / "raw LLM representations as RL reward 2025 2026" —
before stage 3's redesign (post-FAIL fix) locks in sentence-transformer
embeddings as the permanent choice. If nothing has changed, that's worth
recording as a confirmed absence, not just an assumed one.

---

## Summary of citation problems found in this pass

Four entries in the original source list do not check out as given. Listed
here together so a future pass doesn't have to re-derive them from the
stage sections above:

1. **HIR** (`arxiv.org/pdf/2406.05881v4`) — resolves to LGR2 (Singh et al.
   2024), not a paper called "Hindsight Instruction Relabeling." No single
   canonical "HIR" paper exists; closest real match is Röder et al. 2022
   (arXiv:2204.04308), not yet independently verified.
2. **LGR2** (`rlg.mlanctot.info/...`) — dead link (TLS cert mismatch, then
   404 on the base domain). A different, real paper with the same LGR2
   abbreviation exists at arXiv:2406.05881v4 (Singh et al. 2024) — use that
   one, drop the dead link.
3. **Qiu, Mao, Zhu LTL paper** (`arxiv.org/abs/2205.13044`) — resolves to
   an unrelated non-stationary-regret paper by Chen & Luo. Actual
   Qiu/Mao/Zhu NeurIPS 2023 paper not located in this pass (no WebSearch
   available).
4. **HiRL** (`arxiv.org/abs/2106.13687`) — resolves to the panda-gym paper,
   not a hierarchical-RL-over-continuous-goal-space paper. Actual HiRL
   paper not located in this pass.

None of these four are used as citations elsewhere in this file — they're
recorded here as gaps for the next research pass, not silently dropped and
not silently "corrected" with a guessed substitute (except where a
plausible substitute was found and explicitly flagged as unverified above).
