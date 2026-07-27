# Stage 2: Learned continuous goal embedding (contrastive pretraining)

## In plain English

This experiment tested whether a robot-control agent can learn to recognize
goals from a compact learned "fingerprint" instead of being handed the
goal's exact xyz coordinates directly. That matters because later stages
need the agent to work from natural-language goal descriptions rather than
coordinates, so it first has to be proven that a *learned* representation
of a goal is just as usable as the raw coordinates. The result: the
fingerprint-based agent matched — and on this run slightly beat — the
coordinate-based agent on task success, and the fingerprint also reliably
reflected true physical distance between goals (goals that are close in
real space end up close in fingerprint space too). Both conditions this
stage needed to prove were met, so the experiment passed and it's safe to
build the next stage on top of this component.

## Result

**Passed — 10/10 seeds scored 0.98+ success (vs. 8/10 for the
coordinate-based baseline), and the learned goal fingerprint tracks true
physical distance between goals (correlation r=0.57 on 500 held-out
goals).**

![stage1_vs_stage2_comparison.png](charts/stage1_vs_stage2_comparison.png)

## How this was tested

Ten independent training runs (one per random seed) were completed for
each of two setups: the original stage-1 setup, where the
reinforcement-learning agent is told a goal's exact xyz coordinates, and
this stage's setup, where the agent instead sees a 16-number "fingerprint"
of the goal produced by a small neural network trained separately ahead of
time. That network was trained using a technique called contrastive
pretraining — plain-language version: it's shown pairs of goals and
learns to make similar goals produce similar fingerprints and different
goals produce different fingerprints. Each of the 10 trained agents (per
setup) was then evaluated over 50 episodes, where "success" means the
agent completed the manipulation task within the episode. Separately, to
check the fingerprint wasn't meaningless noise, 500 goals it had never
seen before were converted to fingerprints, and the distances between
those fingerprints were statistically compared to the real physical
distances between the same goals — a high correlation means the
fingerprint faithfully preserves real-world closeness.

---
## Full evidence
The complete technical record — proof gate, full result tables, charts,
raw logs, anomalies, known-risks cross-check, and the reviewer
verdict — lives in [`evidence.md`](evidence.md).
