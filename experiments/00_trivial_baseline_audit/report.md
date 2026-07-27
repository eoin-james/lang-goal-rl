# Trivial Baseline Audit: Is FetchReach-v4's Goal Distribution Too Easy?

## In plain English
While reviewing a demo GIF from stage 6, one episode's starting position
looked unusually close to its goal already — close enough to raise a real
question: what if FetchReach-v4's goals are placed close enough to the
robot's starting position that an agent barely has to learn anything, and
this project's reported 0.548-1.000 success rates across stages 1-6 are
mostly measuring that geometric freebie rather than real learned behavior?

This audit answers that directly, from scratch, with a script anyone can
re-run. It measures two things: (1) how far a freshly-reset episode's start
position actually is from its goal, before the agent takes a single action,
and (2) how often policies with zero learned behavior — doing nothing, or
moving randomly — succeed under the exact same pass/fail rule (`info["is_success"]`)
and 50-step episode length used by every other stage in this project.

## Result
**Not inflated.** Trivial policies fail almost every time: doing nothing
succeeds 1.8% of episodes, acting randomly succeeds 0.4%. A straight-line
"oracle" policy that always moves directly toward the goal — the honest
upper bound for this task, since it uses perfect information and zero
learning — succeeds 100% of the time, reaching the goal in a median of 3
steps out of a 50-step episode. Every stage's real, reported result sits far
above the do-nothing floor: roughly 30-56x it, depending on the stage (see
the full table below). Only 2.2% of episodes start already inside the 0.05m
success radius — and that number lines up almost exactly with the no-op
success rate (1.8%), which is itself a good internal consistency check: a
policy that does nothing can only succeed on episodes that started inside
the threshold already, and that's basically all it does succeed on.

Stage 5's headline finding (goal-swap success == fresh-episode baseline,
1.000 == 1.000) is a **relative** comparison — it stays valid regardless of
how easy or hard the absolute task is, because the same floor applies to
both sides of that comparison.

**Honest caveat carried forward:** because the oracle solves this task in a
median of 3 steps, stages 1-3 and 5's 1.000 scores are at ceiling — they
can't distinguish "very good" from "perfect." That's an **informativeness
limit** on those specific numbers, not a validity problem with the
project's results as a whole.

![reset_distance_histogram.png](charts/reset_distance_histogram.png)

## How this was tested
`trivial_baseline_audit.py` resets `FetchReach-v4` 500 times and records the
distance between `desired_goal` and `achieved_goal` immediately after each
reset, before any action is taken — this is the "how easy is the starting
position" measurement. It then runs 500 full episodes each for three
policies that have learned nothing: no-op (all-zero action every step),
random (uniform sample from the real action space), and a straight-line
oracle (always moves directly toward the goal, using ground-truth position —
the honest "if you could see everything and never make a mistake" upper
bound). All three use the same `info["is_success"]` criterion and the same
50-step episode length (`TimeLimit`-enforced) as every training run in this
project. Everything is seeded (distinct, non-overlapping seed blocks per
measurement) so the numbers reproduce exactly on re-run:

```
uv run python experiments/00_trivial_baseline_audit/trivial_baseline_audit.py
```

---
## Full evidence
The complete technical record — proof gate, full result tables, charts,
raw logs, anomalies, known-risks cross-check, and the reviewer
verdict — lives in [`evidence.md`](evidence.md).
