# Next Phases

The original roadmap is complete. It proves that language can select a goal
and redirect a reaching policy while an episode is running. The next work
should preserve that result, make it easy for others to inspect, and then
expand the system without overstating what it can do.

## Phase 1: Finish and present the reaching project

**Goal:** Turn the completed research record into a clear, reproducible
portfolio project.

### Work

- Rewrite the main README around the problem, approach, results, limitations,
  demos, and a copy-paste quick start.
- Standardize the reported numbers:
  - Stage 4: 57.1% mean success on 14 held-out paraphrases.
  - Stage 6: 85.7% success on a separate set of 7 novel phrasings.
  - Do not combine these into one headline result.
- Present the interactive demo as a demonstration of seven-region live
  re-goaling, not unrestricted language control.
- Add a supported experiment command that selects a stage and seed and writes
  results without requiring knowledge of internal script paths.
- Put the command, seed, checkpoint, instruction sequence, and outcome beside
  each featured demo.
- Document model downloads, runtime expectations, checkpoint requirements, and
  the tested Python/platform setup.
- Add repository hygiene:
  - Ignore editor, OS, cache, generated-run, and local environment files.
  - Add focused lint, formatting, type-checking, test, and CI configuration.
  - Avoid adding new quality-tool dependencies without deciding on them first.
- Add a changelog and create a versioned release only after reproduction and
  documentation checks pass.
- Keep `OPEN_QUESTIONS.md` visible and resolve or explicitly defer each
  portfolio concern.

### Exit gate

A new reader can understand the claim in 60 seconds, run the interactive demo
from a clean checkout with one documented command, reproduce one representative
evaluation, and trace every headline result to committed evidence.

## Phase 2: Continuous live command agent

**Goal:** Replace seven-region sentence classification with a continuously
running agent that accepts, interrupts, and executes parameterized motion
commands.

The current RL policy and goal encoder already operate on continuous XYZ
goals. The main bottleneck is the language layer, which maps every sentence to
one of seven fixed centroids.

### Command representation

Introduce a small typed command interface before training another language
model:

```text
AbsoluteGoal(x, y, z)
RelativeMove(dx, dy, dz, speed)
WaypointSequence(points, speed)
Stop
Reset
```

Examples:

```text
"move forward 5 cm"       -> RelativeMove(+0.05, 0.00, 0.00)
"go slightly left"        -> RelativeMove(0.00, +0.02, 0.00)
"move forward and back"   -> WaypointSequence([forward, start])
"stop"                    -> Stop
```

Coordinate-frame conventions must be explicit: “forward” may mean relative to
the robot base, camera, workspace, or end effector. Start with the robot-base
frame and print the interpretation in the terminal.

### Work

1. Verify the existing policy over densely sampled continuous XYZ targets,
   including small relative displacements from the current gripper position.
2. Separate the live system into:
   - command input;
   - language grounding;
   - command queue and preemption;
   - goal or trajectory generation;
   - low-level policy execution;
   - feedback and status reporting.
3. Implement deterministic continuous commands first. This establishes whether
   the controller works before language-model errors are introduced.
4. Support commands arriving during motion. A new command should preempt or
   queue according to a documented rule; `stop` must take effect immediately.
5. Add magnitudes and modifiers such as “slightly,” “10 cm,” “slowly,” and
   “twice as far.”
6. Add waypoint and trajectory tracking for commands that cannot be represented
   by one static goal, including “forward and back” and simple circles.
7. Train or integrate learned language grounding only after the typed command
   interface is stable. Evaluate held-out wording, directions, magnitudes, and
   compositions separately.
8. Detect ambiguous or unsupported instructions instead of silently mapping
   all input to the nearest known behavior.

### Measurements

- Final position error for absolute and relative goals.
- Path error for trajectories.
- Command-to-motion latency.
- Interruption latency.
- Completion rate across long command sequences.
- Language-grounding accuracy on held-out wording and held-out command
  parameters.
- Unsupported-command rejection accuracy.

### Exit gate

The agent remains live for a long session, executes at least 20 mixed
continuous commands without restarting, handles interruption and stopping,
and completes held-out relative-goal and waypoint tests at a predefined
success threshold.

## Phase 3: Push goals and object interaction

**Goal:** Move from free-space arm reaching to language-directed object
pushing, initially in `FetchPush`.

This is a real increase in difficulty. The goal now describes the object's
desired position, while the policy must reason about contact, approach
direction, and the relationship between gripper and object.

### Work

1. Establish a literal-goal `FetchPush` baseline without language.
2. Audit trivial and scripted baselines so success cannot be attributed to
   favorable object placement.
3. Train and evaluate across enough seeds to characterize instability and the
   larger training budget.
4. Define continuous object goals and relative commands:
   - “push the block left”;
   - “move it forward 10 cm”;
   - “push it to the centre”;
   - “move it back toward where it started.”
5. Extend the typed command representation with object identity and target:

   ```text
   Push(object, target_position, speed)
   ```

6. Add language grounding only after the literal continuous push baseline is
   reliable.
7. Test mid-task re-goaling: change the object's destination while the policy
   is approaching or pushing it.
8. Record failure modes separately: failure to reach the object, bad contact,
   wrong push direction, overshoot, and language misinterpretation.
9. Treat multiple objects, vision-based object references, grasping, and
   real-world transfer as later work rather than silently expanding this
   phase.

### Exit gate

The agent can accept live language commands that specify continuous
single-object push goals, redirect the target during execution, and
demonstrate performance above literal, trivial, and language-free comparison
baselines across multiple seeds.

## Decisions before Phase 2

- Define axis and coordinate-frame language precisely.
- Choose initial command units and safe workspace bounds.
- Decide whether trajectory following should reuse the current goal-conditioned
  policy with moving waypoints or train a trajectory-conditioned policy.
- Set numerical success thresholds before running experiments.
- Decide how unsupported commands are surfaced to the user.
- Keep the original roadmap and results frozen as the Phase 1 research record;
  do not rewrite earlier evidence to make later capabilities appear complete.
