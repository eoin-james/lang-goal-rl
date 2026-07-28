# Stage 9: Waypoint following

## In plain English
Stage 5 proved a robot could accept a brand-new goal mid-episode -- one
switch, no reset. This stage asks the next question: does that same trick
still work if you string several goals together in a row, one continuous
run with no resets in between? That's "waypoint following" -- give the
robot a short list of targets and have it visit them in order. The concern
going in was compounding error: a policy that handles one switch fine
might still drift further and further off course the more times in a row
it gets re-targeted, especially if each leg only gets a few steps to get
there. The first pass through this test used only one trained checkpoint
and came back with a real methodology gap flagged by review: this
project's history shows differently-trained checkpoints from the identical
training recipe can behave very differently, and a single checkpoint whose
fresh-start baseline never once failed had no room to reveal whether
chaining costs more than a fresh start for a *different* checkpoint. This
version reruns the identical test across all 8 checkpoints known to be
healthy, and the answer holds up: reaching each waypoint in a longer chain
is essentially as reliable as reaching that same waypoint fresh from a
random start with the same budget, and not one of 4,800 chain episodes
across 8 checkpoints ever showed two waypoints failing in the same run --
the small number of misses that did happen never dragged down the next
waypoint, on any checkpoint.

## Result
Measured, not adjudicated -- see "Reviewer verdict" in Full evidence below
for the actual pass/fail call. First, the mechanism check: the regression
test proving a 2-waypoint chain is numerically identical to stage 5's own
mid-episode-switch function still passes cleanly (20/20 tests). Second, the
scaled-up evidence this rerun exists to produce: across all 8 healthy
checkpoints (seeds 0, 1, 3, 4, 5, 6, 8, 9 -- seeds 2 and 7 are excluded, the
documented SAC training-collapse signature, unrelated to this mechanism),
every checkpoint's own literal-goal sanity check scored a clean 1.000, so
all 8 are trustworthy going into the waypoint results. At the generous step
budget, every chain length, both waypoint-list styles, and every checkpoint
scored a clean 1.000 -- unchanged from the first pass, still not a very
demanding test at this budget. At the tight budget, pooling all 8
checkpoints together (400 episodes per condition instead of 50), the
picture is the same shape as the first pass, just measured at 8x the
checkpoint coverage: literal chains stay at 0.998 (N=2/N=3) and 0.978
(N=5); relative-move chains stay at 1.000 (N=2), 0.998 (N=3), and 0.990
(N=5). Individually, whole-chain success rate across the 8 checkpoints
ranges from 0.960-1.000 (literal, N=5, tight) and 0.940-1.000 (relative,
N=5, tight) -- some checkpoints score a perfect 1.000 where seed_0 didn't,
none score worse than seed_0's own first-pass numbers. The two checks this
rerun exists to answer, checked directly across all 4,800 chain episodes:
zero episodes anywhere had two or more waypoints fail in the same run, and
no checkpoint's per-leg failure rate rises with leg position in a
consistent, ongoing way -- the handful of tight-budget conditions with any
failures at all show either a flat rate or a single one-off bump, never a
climbing trend from leg 1 through leg 5.

![whole_chain_success_vs_length.png](charts/whole_chain_success_vs_length.png)

## How this was tested
8 previously-trained SAC+HER checkpoints (`experiments/
01_uvfa_her_baseline/checkpoints/seed_{0,1,3,4,5,6,8,9}.zip`, zero-shot --
no new training) each ran the identical 12 conditions the first pass used:
2 waypoint-list styles (goals from distinct regions of the workspace, vs.
each goal computed as a relative move off the previous goal) x 3 chain
lengths (2, 3, 5 waypoints) x 2 per-leg step budgets (tight = 9 steps,
generous = 18 steps), 50 episodes per condition per checkpoint (600
episodes/checkpoint, 4,800 chain episodes total). For every waypoint in
every chain, "did this leg succeed" is judged only on that leg's own steps,
and every leg is compared against a budget-matched fresh baseline (a
completely fresh episode targeting that exact same goal with that exact
same step budget, from the same random start). Every checkpoint's
literal-goal sanity check (the plain single-goal task, no waypoint chain)
was confirmed clean before trusting any of its waypoint results, per this
project's established convention for catching the known SAC
deterministic-eval collapse signature -- seeds 2 and 7 show that signature
and were excluded from this run by design. Results are kept broken out by
(checkpoint, condition), not collapsed into one grand mean, specifically so
an individual checkpoint diverging from seed_0's pattern would be visible
rather than averaged away; every failing episode across all 8 checkpoints
was scanned directly for whether it failed 2+ legs (a multi-leg failure)
and every condition's per-leg failure rate was checked for whether it rises
monotonically with leg position (a compounding signature) -- both checks
came back negative, on every checkpoint.

---
## Full evidence
The complete technical record — proof gate, full result tables, charts,
raw logs, anomalies, known-risks cross-check, and the reviewer
verdict — lives in [`evidence.md`](evidence.md).
