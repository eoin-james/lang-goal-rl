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
