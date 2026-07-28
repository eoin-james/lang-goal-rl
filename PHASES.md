# Phases

This project is a longer-running research program, not a single finished
result. This file is the short overview of where it's going; for the
detailed stage-by-stage record of what's actually been done, see
[ROADMAP.md](ROADMAP.md).

## Phase 1 (complete): prove the mechanism

Language can select a continuous goal and redirect a reaching policy live,
mid-episode. Full 7-stage breakdown, results, and known risks: see
[ROADMAP.md](ROADMAP.md).

## Phase 2 (planned): continuous live command agent

Replace the current seven-region sentence classification with a continuously
running agent that accepts, interrupts, and executes parameterized motion
commands — absolute goals, relative moves, waypoint sequences, stop/reset —
instead of snapping every sentence to one of 7 fixed centroids. The trained
policy and goal encoder already operate on continuous XYZ goals; the
bottleneck is the language layer.

Key steps: verify the existing policy over densely-sampled continuous XYZ
targets (including small relative displacements); separate the live system
into input / language grounding / command queue+preemption / goal generation
/ policy execution / feedback stages; ship deterministic typed commands
before adding learned language grounding, so the controller's own
correctness isn't confounded with language-model error; support commands
arriving mid-motion under a documented preempt-or-queue rule (`stop` always
takes effect immediately); detect ambiguous or unsupported instructions
instead of silently mapping them to the nearest known behavior.

Decisions to make first: coordinate-frame conventions (what "forward" is
relative to), initial command units and safe workspace bounds, whether
trajectory-following reuses the current goal-conditioned policy with moving
waypoints or needs its own trajectory-conditioned policy, and how
unsupported commands are surfaced to the user.

## Phase 3 (planned): push goals and object interaction

Move from free-space reaching to language-directed object pushing
(`FetchPush`) — a real difficulty increase: the goal describes the object's
desired position, and the policy must reason about contact, approach
direction, and the gripper-object relationship.

Key steps: a literal-goal `FetchPush` baseline first, no language; a
trivial/scripted-baseline audit before trusting any success number (the same
discipline as Phase 1's `experiments/00_trivial_baseline_audit`); extend the
typed command representation with object identity + target
(`Push(object, target_position, speed)`); add language grounding only once
the literal push baseline is reliable; test mid-task re-goaling (change the
object's destination while it's being approached or pushed); record failure
modes separately (missed approach, bad contact, wrong push direction,
overshoot, language misinterpretation) rather than collapsing them into one
pass/fail number. Multiple objects, vision-based object reference, grasping,
and real-world transfer are explicitly later work, not part of this phase.

---

Both later phases keep Phase 1's evidence frozen as the research record —
they extend the system, they don't rewrite earlier results to look more
complete than they actually are.
