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

## Milestone close-out: update the log + consider a blog entry

After any stage/phase reaches a real milestone (a PASS, a major debugging
saga resolved, a significant pivot), the manager updates
`.claude/RESEARCH_LOG.md` (dated, terse entry — the actual research
reasoning: what was tried, why, what the dead end taught) and considers a
`BLOG.md` entry (casual, portfolio-facing narrative) — alongside, not
instead of, the existing `STATUS.md`/`ROADMAP.md` update. `RESEARCH_LOG.md`
maintenance is a manager responsibility, not `thesis-research-agent`'s —
that persona is scoped to `LITERATURE.md` (citations) and explicitly does
not judge experimental results; research-reasoning entries need synthesis
across the builder/runner/reviewer roles the manager already does when
closing out a stage.

## Portfolio phase close-out

A stage can pass its proof gate without making its containing phase
portfolio-ready. A phase is complete only when a reader with no project
context can discover, understand, inspect, and reproduce it from the
repository's front page.

Before marking a phase complete, all of the following must be true:

1. **Front-page route:** `README.md` states the current phase and links
   directly to its roadmap, result narrative, demos, and reproduction
   instructions. Status documents must agree.
2. **Stage record:** every stage has a short `report.md`, full `evidence.md`,
   raw machine-readable results, and a completed independent verdict. Reports
   separate measured facts from interpretation and state limitations beside
   the headline result.
3. **Phase narrative:** `BLOG.md` has a dated, blog-style close-out covering
   the question, experimental progression, failures and corrections, final
   evidence, demo, and honest claim boundary. It links to stage reports
   instead of duplicating their tables.
4. **Demo:** at least one representative demo shows the phase's final
   capability. `demos/README.md` records what is shown, checkpoint, seed,
   success verification, visual QA, and a regeneration command. Intermediate
   or failed demos must be labeled clearly.
5. **Reproduction:** one documented top-level command reproduces each stage's
   reported evaluation from committed artifacts, and one regenerates its
   demo. Training stages separately document full retraining. Commands must
   work from a clean checkout without absolute local paths.
6. **Environment contract:** supported platforms and Python version,
   dependency setup, model-download/offline behavior, hardware expectations,
   approximate runtime, disk use, and external artifact retrieval are
   documented.
7. **Verification:** tests and quality gates pass in a clean environment.
   Intentionally excluded integration or hardware-dependent checks are named,
   justified, and given separate commands.
8. **Consistency and claim audit:** roadmaps, status, reports, verdicts,
   links, seeds, checkpoints, demos, and portfolio claims agree. Exceptions
   to shared experiment contracts require reviewer approval; disclosure alone
   is not approval.
9. **Release snapshot:** the phase has a versioned tag or release linking its
   narrative, demos, reproduction commands, and known limitations.

The manager checks `.claude/findings.md` before close-out. A phase is not
portfolio-ready while an open finding materially blocks an item above.

## Oversight agent — shared files, do not delete or revert

`.claude/findings.md` and the "Portfolio phase close-out" section above are
maintained by a separate oversight agent that may run concurrently with any
builder/runner/reviewer session.

- Do not delete, replace, clean up, or revert `.claude/findings.md` or the
  Portfolio phase close-out section in this file.
- Check `.claude/findings.md` before starting and before closing a stage or
  phase. Resolve applicable findings with evidence, then mark them complete
  rather than removing them.
- A passed experiment does not make a phase portfolio-ready — follow every
  requirement in the Portfolio phase close-out contract above.
- Other agents may edit these files concurrently. Re-read them immediately
  before writing and preserve changes you did not make. If a finding seems
  obsolete or incorrect, leave it in place and flag it for oversight review
  rather than removing it.

## Reuse trained policies across stages, don't retrain by default

If a later stage's zero-shot test only needs an already-trained policy
(not new RL training), and a checkpoint from an earlier stage exists, load
it — don't retrain from scratch. Always `model.save(...)` per seed under
`experiments/NN_slug/checkpoints/seed_<k>.zip` (or `.pt` for
non-SB3 artifacts) so the next stage can reuse them. Retraining from
scratch is the single biggest cost in this pipeline — only do it when the
stage's proof gate genuinely requires new training (e.g. a new architecture
component), not for reusing an existing policy's already-learned behavior.
