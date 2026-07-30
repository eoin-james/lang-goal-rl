# Experiments

One directory per stage: `NN_slug/` (e.g. `01_uvfa_her_baseline/`), containing
run scripts, configs, and result notes for that stage's proof gate.

Nothing reusable lives here. If two experiments need the same code, promote it
to `src/lang_goal_rl/` and import it from both.

Every stage's `evidence.md` has a `### Reproduce` section with the exact
command(s) to regenerate its results from committed artifacts — no stage's
proof-gate numbers should require guessing at script arguments.

## Artifact policy

Measured 2026-07-30: `.git` is 73M, `experiments/` is 88M, `demos/` is 5.4M.
79 committed binary artifacts under `experiments/`: 13 `.zip` checkpoints, 5
`.pt`, 6 `.gif`, 55 `.png`. (Repo checkout total is ~1.4G, but that's almost
entirely the gitignored `.venv` — not tracked, not part of this policy.)

**Canonical — never casually regenerate:** trained-policy checkpoints
(`*.zip`, `*.pt`). Every later stage's zero-shot-reuse convention (see each
stage's `### Reproduce` section) depends on these exact files existing —
regenerating one from scratch would retrain a new policy with a different
random seed trajectory, silently invalidating every downstream stage that
reused it. Treat these as append-only history, not scratch output.

**Regenerable — safe to delete and rebuild:** charts (`*.png`), demo clips
(`*.gif`), and raw run logs. Each is produced by a `### Reproduce` command
against the canonical checkpoints and committed `results.json`/
`final_results.json` files; regenerating them should reproduce the same
numbers (deterministic evals) or fall within the same reported per-seed
spread (non-deterministic ones).

**No Git LFS yet.** At 88M for all of `experiments/`, the current size
doesn't justify LFS's setup and CI-caching overhead — plain git handles it
fine. Revisit if checkpoint count or resolution grows materially (e.g. a
harder task needing bigger policies, or video demos at higher resolution).
