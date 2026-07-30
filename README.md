# lang-goal-rl

Research repo for goal-conditioned RL where the goal space is continuous and
derived from frozen language embeddings, working toward an agent that takes
ad-hoc English instructions live during an episode.

Staged plan and current status: see [ROADMAP.md](ROADMAP.md). The project is
now in Phase 2b (teaching a learned language layer to speak Phase 2a's typed
commands) — stage 11 is done, stages 12-15 are paused for an all-hands review
before continuing. See [PHASES.md](PHASES.md) for the overall phase structure.
For the casual, first-person story of building this — including the debugging
sagas — see [BLOG.md](BLOG.md).

## Status & roadmaps

- [STATUS.md](STATUS.md) — current state, stage-by-stage, across every phase
- [ROADMAP.md](ROADMAP.md) — Phase 1 staged plan (stages 0-6, frozen)
- [PHASES.md](PHASES.md) — overall phase structure
- [PHASE2_ROADMAP.md](PHASE2_ROADMAP.md) — Phase 2a staged plan (stages 7-10)
- [PHASE2B_ROADMAP.md](PHASE2B_ROADMAP.md) — Phase 2b staged plan (stages 11-15, in progress)

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

## Environment contract

- **OS**: developed on macOS/Darwin. CI (see below) runs on `ubuntu-latest`
  headless — every test that touches the env uses `render_mode="rgb_array"`,
  never `render_mode="human"`, so no display is required.
- **Python**: `>=3.12` (per `pyproject.toml`). Managed via `uv`; `uv sync`
  resolves the venv from `uv.lock`.
- **`gymnasium-robotics` / MuJoCo**: pulls in the `mujoco` Python package,
  which ships its own bundled MuJoCo binary — no separate system MuJoCo
  install needed on macOS or Linux. One known quirk from this project's
  history: the originally-planned dependency, `pybullet` (via `panda-gym`),
  failed to build from source on macOS arm64 and was dropped before Stage 0
  in favor of `gymnasium-robotics`'s MuJoCo-backed `FetchReach-v4` — see
  [BLOG.md](BLOG.md)'s day-one entry. That dependency isn't in this repo
  today, noted only so the same trap isn't rediscovered.
- **`sentence-transformers`**: auto-downloads `all-MiniLM-L6-v2` from
  huggingface.co on first use — network access is required unless the model
  is already cached locally. Set `HF_HUB_OFFLINE=1` to force offline mode
  once cached (already used in the interactive-demo command above).
- **Test suite runtime**: `uv run pytest tests/lang_goal_rl -q` (415 tests)
  runs in ~15-20s locally with a cached HF model.
- **Disk use**: see [experiments/README.md](experiments/README.md)'s
  artifact-policy section for current repo/experiments sizes and the
  canonical-vs-regenerable artifact split.
