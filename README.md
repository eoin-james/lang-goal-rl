# lang-goal-rl

Research repo for goal-conditioned RL where the goal space is continuous and
derived from frozen language embeddings, working toward an agent that takes
ad-hoc English instructions live during an episode.

Staged plan and current status: see [ROADMAP.md](ROADMAP.md). This is Phase 1
of a longer-running research program — see [PHASES.md](PHASES.md) for the
phase structure and what's planned next. For the casual, first-person story
of building this — including the debugging sagas — see [BLOG.md](BLOG.md).

## Layout

- `src/lang_goal_rl/` — reusable code (env wrappers, goal encoders/projections,
  training utilities). Every experiment imports from here.
- `experiments/` — one directory per stage, run scripts + configs + result
  notes. See `experiments/README.md`.

## Interactive demo

Open a live window and control the robot from the terminal:

```bash
HF_HUB_OFFLINE=1 uv run python -m lang_goal_rl.interactive_demo --seed 0
```

Type an English instruction and press Return. Type another instruction at any
time to redirect the robot without resetting. The terminal prints the nearest
known reference sentence, inferred region, and whether the typed instruction
is a confident match or an extrapolation against the 84-sentence reference
vocabulary. Each episode runs for the real 50-step limit and reports whether
the robot actually reached the target before auto-resetting. Other commands
are `status`, `reset`, and `quit`. If no display is available, the demo falls
back to a live matplotlib window automatically.
