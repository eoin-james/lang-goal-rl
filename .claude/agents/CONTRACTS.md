# Shared contracts

Technical specs referenced by more than one agent file. Kept here once so
they don't drift between `rl-builder.md`, `experiment-runner.md`, and
`results-reviewer.md`.

## `src/lang_goal_rl/reporting.py` signatures

```python
def plot_training_curve(timesteps, values, *, ylabel, out_path, seed=None) -> Path: ...
def plot_multi_seed_success_rate(results: dict[str, list[float]], *, out_path, proof_gate_threshold=None) -> Path: ...
def plot_embedding_projection(embeddings: np.ndarray, labels, *, out_path, n_components=2) -> Path: ...
def plot_candidate_comparison(results: dict[str, list[float]], *, out_path) -> Path: ...
def write_report(
    *, stage, title, seeds: list[int], candidates: list[str] | None = None,
    proof_gate_text, metrics_table, chart_paths, raw_output_paths, anomalies,
    known_risks_note: str = "none applicable", out_dir,
) -> Path: ...
```

`plot_embedding_projection` uses a numpy-only PCA (via SVD) — no
scikit-learn, it isn't in `uv.lock` and the project stays lightweight.

`write_report`'s `seeds` renders verbatim into the header's `Seeds run`
field. `candidates` defaults to `None`, rendered as `"1 (locked-in)"`; pass
a list of candidate names when a stage escalates to comparing candidates.
`known_risks_note` defaults to `"none applicable"` and renders into the
"Known-risks cross-check" section — pass text naming which ROADMAP.md
"Known risks" entries this result touches.

## `report.md` structure (fixed section order — every stage, no variation)

```
# Stage <N>: <title>
**Date:** ... **Seeds run:** [...] **Candidates:** <name(s) — "1 (locked-in)" unless escalated>

## Proof gate (verbatim from ROADMAP.md)
> <exact text>

## Result summary
<per-seed and aggregate metrics table>

## Charts
<embedded PNGs>

## Raw output
<links to experiments/NN_slug/runs/seed_<k>/stdout.log>

## Anomalies (factual, not judged)
<runner-reported, or "none observed">

## Known-risks cross-check
<which ROADMAP "Known risks" entries this result touches, if any>

## Reviewer verdict
<left blank by the runner — filled in by the manager from the reviewer's return>
```

Storage: `experiments/NN_slug/report.md`,
`experiments/NN_slug/charts/*.png`,
`experiments/NN_slug/runs/seed_<k>/stdout.log` (or
`runs/<candidate>/seed_<k>/` if escalated).

## Concurrency

Cap = `min(pending_runs, cores - 2)`. Check core count at runtime
(`sysctl -n hw.ncpu` on macOS, `nproc` on Linux) — don't hardcode it. Pin
each background process's math-library threading to 1
(`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, `torch.set_num_threads(1)` inside
the script) so concurrent runs don't oversubscribe cores.
