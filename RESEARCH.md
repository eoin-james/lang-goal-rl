# Research Direction

## Abstract

Large language models are the default interface for commanding embodied
agents, but they are a poor fit for the platforms that need language
most: drones and small robots with hard reaction-time limits, no
connectivity guarantees, no onboard GPU, and behavior that must be
certifiable — the same command interpreted the same way, every time.
This work asks how far a language-commanded agent can get **without any
generative model**: a frozen 22M-parameter sentence encoder as the
entire language-understanding stack, trained RL policies as the skills,
and a typed command layer as the contract between them.

Three questions structure the research. **The ceiling:** where exactly
does frozen-embedding grounding break, and can each failure mode be
given a reproducible diagnostic signature — a taxonomy with a probe
suite, not anecdotes? **Repair:** when the interface misunderstands,
can a structured correction dialogue — the agent exposes its
interpretation, the human corrects it — warp the grounding map online,
per operator, recovering capability past the frozen ceiling?
**Manner:** commands that change *how*, not *where* ("stay low",
"carefully"), as the hardest case for both.

The intended contribution is a diagnostic method plus evidence that
**conversation can substitute for scale**: a commandable embodied agent
at ~1/300th the parameters of an LLM interface, whose failure modes are
mapped in advance rather than discovered in the field.

---

*Established 2026-07-31, after the Phase 2b stage-11 pause. This is the
standing frame the roadmaps answer to. It replaces the earlier implicit
framing ("staged portfolio project working toward live English
instructions") with an explicit research program. No deadlines — this is
an ongoing set of research run at whatever pace curiosity sets.*

## North star

Give commands to a drone. Concretely: talk to a drone like a competent
teammate — *"go check the top of that hill for a flat camping spot, stay
low on the way up"* — have it fly the mission, surface what it understood
when unsure, and accept corrections mid-flight ("no, lower") that improve
its understanding of *this* operator for the rest of the session.

The drone is the deployment story, not the current testbed. Everything
below happens in small, seeded, reproducible simulated worlds first.

## The research bet

The default 2026 answer to language-commanded agents is an LLM in the
loop. Our bet is deliberately different:

> **A tiny frozen language encoder (all-MiniLM-L6-v2, 22M params,
> revision-pinned, deterministic) plus trained RL skills covers most of a
> conversational command interface — and where it can't, a structured
> repair conversation covers the rest. No generative model anywhere.**

Why this matters for the drone specifically: a drone is the worst place
for an LLM — reaction-time limits, no connectivity guarantees, no room
for a GPU, and behavior must be certifiable (same command → same
interpretation, every time). A frozen encoder gives all of that for
free. The open question is whether it gives enough *understanding*.
Measuring exactly that is the research.

The constraint is principled **as a measurement program** (the output is
the quantified capability frontier), not as an identity. Small frozen
encoders are live engineering practice in 2025–26 (e.g. SVLR, arXiv
2025; MiniLM ensembles beating 120B-param LLMs at fleet task routing,
arXiv 2026) — they're just not yet a mapped scientific frontier.

## Central question

> **What is the capability ceiling of a frozen small-encoder language
> interface for embodied control, where exactly does it break
> (diagnosable how), and how much of the gap can online conversational
> repair recover — without a generative model?**

## The three standing threads

Threads, not milestones. Each pulls on the others; work on whichever is
interesting. Ordered by defensibility, not sequence.

### Thread 1 — The ceiling (the spine)

A taxonomy of frozen-embedding grounding failures *as a control
interface*, each with a reproducible diagnostic signature, packaged as a
probe suite runnable against any encoder (target 3–4: MiniLM, mpnet,
GTE-small, a CLIP text tower — a method, not a MiniLM autopsy).

Seed specimen (ours, stage 11): **vocabulary-convention collision** —
two intent classes sharing a surface phrasing convention are unseparable
by frozen embeddings regardless of tuning. Diagnostic fingerprint: 0% on
one class across all hyperparameter configs while training loss → 0
(a data-design failure, not a tuning failure). Candidate categories to
probe: negation ("don't cross the ridge"), manner–content entanglement,
referent ambiguity ("the *other* hill"), compositional novelty.

No published robotics-facing taxonomy of this kind exists (searched
2026-07-31, twice). For a drone, this map is the safety case.

### Thread 2 — Repair (the headline)

Conversational repair as the patch for what Thread 1 shows is broken.
The agent's side of the dialogue is **structured transparency** (nearest
reference sentence, inferred region/target, confidence) — never
generated text. The human corrects ("no, the other hill", "no, more to
the left"); the correction **locally warps the language→goal grounding
map online, per-operator, within-session**.

Claim if it works: *repair is the cheap substitute for a generative
model* — a quantified fraction of the frozen-ceiling gap recovered at
~1/300th the parameters of an LLM interface.

Nearest prior work to differentiate against explicitly: LILAC (Cui et
al., HRI 2023 — corrections adapt a shared-autonomy control space, not
the grounding map itself), Co-Reyes et al. (ICLR 2019), DROC (2023),
GSA-VLN (2025). The novel combination: warping the grounding map itself
+ the structured-transparency mechanism making repair sample-efficient.
Scripted repair distributions are fine for development; a small human
study (n≈10–15) is eventually needed for the claim to stick.

### Thread 3 — Manner (the stress test)

"Stay low", "be quick", "carefully" — commands that change *how*, not
*where*. Most drone-relevant, and the literature says hardest for frozen
embeddings (manner entangles with content). Deliberately **demoted from
a system-building arc to the hard case**: it's the toughest category in
Thread 1's taxonomy and the toughest target for Thread 2's repair. Not a
standalone style-conditioning system — PADL / CALM (SIGGRAPH 2022/23)
already own that ground for physics characters.

## Testbed direction

FetchReach-v4 is exhausted as a place for language to fail
interestingly: no referents, no observers, no cost to carelessness —
style collapses to speed, corrections collapse to 3D vector offsets.
The move (when taken) is a two-environment portfolio, both cheap and
seeded:

1. **FetchPush / PickAndPlace** — same MuJoCo/SAC+HER stack, harness
   largely transfers. Buys referent-bearing goals ("the red one", "the
   one on the left").
2. **A navigation env with observers and noise costs** (MiniGrid-family
   or custom) — semantically richer, physically simpler. "Stealthy" and
   "carefully" become measurable (detection events, noise budgets).
   This is the drone-recon scenario, abstracted to what's measurable —
   and where Threads 2–3 live.

Principle: **embodied enough that language can fail in every way the
taxonomy needs, and not one joint more.** Real drones/perception/
sim-to-real would swamp the language question and break the
reproducibility discipline; they come after, as an application.

## What carries forward from Phases 1–2b

- All reproducibility infrastructure: revision-pinned encoder, CI,
  multi-seed protocol, adversarial reviewer process, per-stage
  reproduce commands.
- The typed command layer (Phase 2a) as the contract between language
  and control — the boundary that keeps misunderstanding from becoming
  arbitrary motion.
- The stage-11 intent classifier and, above all, its failure analysis
  (Thread 1's seed).
- Stages 12–15 (regression heads for continuous parameters) remain
  valid as "continuous parameters ground in embedding space" — a
  Thread 1 data point — currently paused.

## The now — first tasks (2026-07-31)

Stage-sized, runnable on the existing stack, no big decisions required.
Each is a normal stage: gate, multi-seed where applicable, builder →
runner → reviewer.

1. **Stage 12 (replaces the old regression-heads stage 12): the
   collision probe, formalized.** Turn the stage-11 vocabulary-collision
   finding into a standalone, encoder-agnostic tool: given any labeled
   command vocabulary, predict *before training* which class pairs will
   collide (embedding-space cross-class overlap → predicted confusion),
   validated against actual trained-classifier confusion on the stage-11
   data. Gate: the probe's pre-training prediction ranks the known
   MOVE/GOTO collision first, and its score correlates with realized
   confusion across deliberately-constructed vocabularies. This is
   Thread 1's first brick and needs zero new environment.
2. **Stage 13: the negation probe.** Same probe machinery, second
   failure category: are "go left" / "don't go left" separable in
   frozen-embedding space? NLP literature says encoders are
   negation-blind; nobody has measured it as a *control interface*
   failure (what does the policy actually do?). Cheap, and the first
   result that speaks directly to drone safety ("don't cross the
   ridge").
3. **Stage 14: second encoder.** Run both probes against one more
   encoder (mpnet or GTE-small). The moment results exist for two
   encoders, this is a *method*, not a MiniLM autopsy.

Deliberately deferred: the testbed change (PickAndPlace / observer
gridworld) — it becomes urgent only when Thread 2 (repair) starts;
Thread 1 runs fine on vocabularies alone. The old stages 12–15
(regression heads) move behind these; renumber when they're picked up.

## Honest positioning (from the 2026-07-31 viability assessment)

An adversarial literature review (full report:
`docs/research/2026-07-31-thesis-viability-assessment.md`) concluded
**pivot, not proceed**: Phase 1's headline capability is a small-scale
instance of Interactive Language (Lynch et al., 2022); style-as-a-system
is owned by PADL/CALM; repair-as-signal is crowded (LILAC et al.). What
survived scrutiny — and what this program is therefore built on — is
the diagnostic-signature/taxonomy idea (no direct competitor found) and
the repair-recovers-the-ceiling combination. The risk to keep in view:
any thread drifting back toward "rebuild a 2021–2023 result with a
smaller encoder" is rediscovery, not research.
