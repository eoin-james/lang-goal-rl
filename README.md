# lang-goal-rl

Research repo for goal-conditioned RL where the goal space is continuous and
derived from frozen language embeddings, working toward an agent that takes
ad-hoc English instructions live during an episode.

Staged plan and current status: see [ROADMAP.md](ROADMAP.md).

## Layout

- `src/lang_goal_rl/` — reusable code (env wrappers, goal encoders/projections,
  training utilities). Every experiment imports from here.
- `experiments/` — one directory per stage, run scripts + configs + result
  notes. See `experiments/README.md`.

## Interactive demo

Open a live MuJoCo window and control the robot from the terminal:

```bash
HF_HUB_OFFLINE=1 uv run python -m lang_goal_rl.interactive_demo --seed 0
```

Type an English instruction and press Return. Type another instruction at any
time to redirect the robot without resetting. The terminal prints the nearest
known reference sentence and inferred region. Other commands are `status`,
`reset`, and `quit`.
