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

## Tiered seed strategy (speed — locked in after stage-3 scope discussion)

Don't spend the full 10-seed budget on a first look. Run **3 seeds first**
(seeds 0-2). If all 3 clear the proof gate's threshold cleanly, scale up to
the full 10 (seeds 0-9, reusing the 3 already run) for the actual gate
decision the reviewer checks. If any of the first 3 fail or looks marginal,
report that honestly as a 3-seed result and let the manager decide whether
to debug before spending the full budget — don't silently burn 10 seeds on
something that was going to fail obviously anyway. The **final report and
reviewer verdict always need the full 10** — the 3-seed pass is a cheap
early signal, never a substitute for the real gate.

## Reuse trained policies across stages, don't retrain by default

If a later stage's zero-shot test only needs an already-trained policy
(not new RL training), and a checkpoint from an earlier stage exists, load
it — don't retrain from scratch. Always `model.save(...)` per seed under
`experiments/NN_slug/checkpoints/seed_<k>.zip` (or `.pt` for
non-SB3 artifacts) so the next stage can reuse them. Retraining from
scratch is the single biggest cost in this pipeline — only do it when the
stage's proof gate genuinely requires new training (e.g. a new architecture
component), not for reusing an existing policy's already-learned behavior.
