# Stage 7: Frame-of-reference check — Full Evidence

**Date:** 2026-07-28 **Seed range used:** 7000-7055 (base 7000, disjoint from
every other stage's ranges) **Checkpoint:** `experiments/01_uvfa_her_baseline/checkpoints/seed_0.zip`
(unchanged, no training — this stage is eval-only)

### Proof gate (verbatim from PHASE2_ROADMAP.md)
> Human visual sign-off that each direction looks correct on camera
> (deliberately not a numeric gate — see report)

**This gate cannot be closed by this script or by this agent.** Every prior
stage's proof gate is a number this repo's tooling can compute and compare.
This one is explicitly not: "correct on camera" is a judgment a human makes
by watching the clip, not something `record_episode`'s `info["is_success"]`
or any other measurement in this codebase can stand in for. Everything below
is the factual record of what was produced and measured — it is offered as
the evidence a human needs to render that verdict, not as a substitute for
rendering it.

### What was measured (mechanism-level, not labeling-level)
For all 6 directions, the recorded episode's `info["is_success"]` was `True`
by the episode's end — i.e., the robot's hand did end up within FetchReach's
5cm tolerance of the region's sampled xyz target. This confirms the
region-targeting mechanism (policy + `sample_region_goals` + the
`AXIS_DIRECTIONS`-derived region classification) is working correctly end to
end. It says nothing about whether the resulting on-screen motion matches
the English word used to name that region — that is exactly the open
question this stage exists to raise, not resolve.

### Per-direction results

| Direction | Axis | Sign | Attempts used | Seed (kept clip) | Target xyz (kept clip) | n_steps | total_travel |
|---|---|---|---|---|---|---|---|
| reach forward | x | positive | 1/3 | 7000 | [1.4667, 0.8153, 0.4887] | 50 | 0.1907 |
| reach back | x | negative | 1/3 | 7010 | [1.1950, 0.7678, 0.5149] | 50 | 0.2011 |
| reach left | y | positive | 1/3 | 7020 | [1.3344, 0.8947, 0.6008] | 50 | 0.2033 |
| reach right | y | negative | 1/3 | 7030 | [1.4201, 0.6589, 0.4763] | 50 | 0.1625 |
| reach up high | z | positive | 1/3 | 7040 | [1.4754, 0.6487, 0.6819] | 50 | 0.2590 |
| reach down low | z | negative | 1/3 | 7050 | [1.2717, 0.6351, 0.3970] | 50 | 0.2109 |

All 6 directions succeeded on the first of up to 3 allowed attempts — no
direction needed a retry, and no non-success was discarded (there were none
to discard). `n_steps=50` for every clip means every episode ran the full
FetchReach-v4 length (`max_steps=50`), not a truncated one — the initial
centroid-hold phase (steps 0-5) and the region-approach phase (steps 5-50)
are both fully captured in every GIF's 51 frames.

`total_travel` (summed per-step gripper displacement, from
`episode_recording.EpisodeRecording`) ranges 0.163-0.259 m across the 6
directions — comparable magnitude on every axis, no direction is a much
smaller or much larger motion than the others. This tracks a structural fact
about `MEASURED_GOAL_BOX`: all three axes span almost exactly the same range
(x: 0.2998 m, y: 0.2999 m, z: 0.2997 m total range), so no axis is
structurally biased toward a bigger or smaller directional displacement than
the others.

### Charts
Six GIFs, one per direction, in `charts/`:
- [reach_forward.gif](charts/reach_forward.gif)
- [reach_back.gif](charts/reach_back.gif)
- [reach_left.gif](charts/reach_left.gif)
- [reach_right.gif](charts/reach_right.gif)
- [reach_up_high.gif](charts/reach_up_high.gif)
- [reach_down_low.gif](charts/reach_down_low.gif)

Each GIF has 51 frames (1 initial post-reset frame + 50 step frames),
confirmed by reading each file's frame count directly (`imageio.get_reader(...).get_length()`),
not assumed from `n_steps`.

### Raw output
- [make_demos.log](runs/make_demos.log) — full stdout of `make_demos.py`,
  including every attempt's seed, target, success, step count, and total
  travel, plus the sanity check (run separately, see below) confirming
  FetchReach-v4's fixed initial gripper pose is `8.67e-5` m from
  `MEASURED_GOAL_BOX.centroid`.

### Method note: why the "before" phase is trivial here
The task brief asked to start each episode "at or very near" the measured
centroid. A quick check before writing `make_demos.py` (`uv run python -c
"..."`, output captured in `runs/make_demos.log`'s header) found
FetchReach-v4's initial gripper pose is fixed across every seed and already
sits `8.67e-5` m from `MEASURED_GOAL_BOX.centroid` — i.e., the env's default
starting position already *is* the centroid, well inside the 5cm success
tolerance. `SWITCH_STEP=5` (steps 0-5 nominally "targeting centroid") is
therefore just enough frames to show the hand visibly sitting at center
before it moves, not a real travel requirement — the policy has nothing to
correct for in that phase. This is stated here so a reviewer doesn't
mistake the clips' short "before" phase for a shortcut: it reflects a real
property of the env, not a shortened test.

### Anomalies (factual, not judged)
None observed in the mechanism itself: no seed needed a retry, no episode
truncated early, no clip is a rendering artifact (frame counts and
`info["is_success"]` were checked directly, not assumed). The one thing
worth flagging is not an anomaly in this run but a property of the setup
worth a human's attention before judging the clips — see "Early red flag
from the data" below.

### Early red flag from the data (not a rendering observation — this agent has not viewed the rendered frames)
`AXIS_DIRECTIONS` maps FetchReach's x axis (its docstring: "depth,
toward/away from the robot base") to `reach back`/`reach forward`, y
("lateral, left/right") to `reach right`/`reach left`, and z ("height,
up/down") to `reach down low`/`reach up high`. The per-direction
`total_travel` table above shows all three axes produce comparable-magnitude
motion (0.16-0.26 m) — so if forward/back looks harder to judge on camera
than left/right or up/down, it will not be because the robot moved less far
in that direction. The more likely source of ambiguity, if any exists, is
camera geometry, not trajectory magnitude: a "depth" axis (toward/away from
the camera) typically foreshortens on a 2D render in a way lateral or
vertical motion does not, for whatever camera angle FetchReach-v4's default
renderer uses. This is a plausible risk based on the geometry of the
axis-to-direction mapping and the measured (comparable) displacement
magnitudes — not a finding from having watched the rendered frames, which
this agent has not done and cannot substitute for the human check this
stage's proof gate requires.

### Reviewer verdict
**Not applicable — see proof gate above.** This stage's gate is a human
visual sign-off; there is no numeric threshold for a `results-reviewer` to
adjudicate, and no "Done" call should be made in `PHASE2_ROADMAP.md` until
a human has actually watched all 6 clips and confirmed or corrected each
direction's label.
