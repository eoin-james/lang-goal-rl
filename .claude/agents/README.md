# Agent roster — lang-goal-rl

Four project-specific specialists. Dispatched by the main-loop manager —
never invoked by name; read the file and paste its body into a
`general-purpose` Agent call, since a fresh agent has no memory of this
repo's history.

| Agent | Owns | Role |
|---|---|---|
| `rl-builder.md` | `src/lang_goal_rl/` | Implements reusable components (encoders, wrappers, metrics), TDD |
| `experiment-runner.md` | `experiments/` | Writes/runs stage scripts, reports raw metrics, makes no judgment calls |
| `results-reviewer.md` | read-only | Independently, skeptically verifies the runner's result against `ROADMAP.md`'s proof gate before anything gets marked Done |
| `thesis-research-agent.md` | `LITERATURE.md` | Maintains the project's bibliography — cites the real papers behind each stage's "Reuse" target, does fresh research when a stage needs grounding |

## Flow per stage

1. Manager reads `ROADMAP.md`, identifies the active stage's reuse target
   and proof gate.
2. Dispatch `rl-builder` if new reusable code is needed for the stage.
3. Dispatch `experiment-runner` to execute against what the builder shipped.
4. Dispatch `results-reviewer` to verify the runner's result independently.
5. Manager updates `ROADMAP.md` and commits — only after a PASS verdict.
   FAIL or INCONCLUSIVE loops back to step 2 or 3, not straight to a retry.

No stage advances on the manager's own read of a number. The reviewer's
verdict is the gate.
