# Stage 7: Frame-of-reference check

## In plain English
`goal_region_vocabulary.py`'s `AXIS_DIRECTIONS` is this project's own
labeling choice for what "left", "forward", "up", etc. mean in xyz terms —
its own docstring says so directly: the env itself doesn't label which sign
is "forward" or "left". Nobody has ever watched the robot move and checked
whether that labeling actually matches what a person watching the camera
feed would call "left" or "forward". Before Stage 8 builds relative-move
commands ("move left 5cm") on top of that convention, this stage produces
the visual evidence needed to check it: one short clip per direction,
showing the robot's hand sitting at the workspace's center, then moving
toward that direction's real target.

**This is a human-judgment gate, not a measured one.** Every other stage in
this project (0-6) gates on a number — a success rate, a correlation, a
collapse margin. This stage cannot: "does this look like left" is not
something a script can measure, only something a person watching the clip
can confirm. This report lays out the 6 clips clearly enough for that
person to make the call; it does not make the call itself.

## Result
**Not adjudicated — awaiting human sign-off on each clip.** All 6 clips
were produced successfully (first attempt, no retries needed for any
direction) and each one is a real recorded episode that reached its labeled
region's target (`info["is_success"]` was true by the end of every clip).
That confirms the *mechanism* worked — the robot moved to the correct
region of xyz space for every direction. It does **not** confirm the
*labeling* is correct — only a human watching the clips can say whether
"reach left" actually looks like the hand moving left on camera, versus
right, or toward/away from the camera, etc.

| Direction | Claimed axis / sign | Clip | Success (attempt) |
|---|---|---|---|
| reach forward | x axis, positive (+x) | [reach_forward.gif](charts/reach_forward.gif) | succeeded, 1st attempt |
| reach back | x axis, negative (-x) | [reach_back.gif](charts/reach_back.gif) | succeeded, 1st attempt |
| reach left | y axis, positive (+y) | [reach_left.gif](charts/reach_left.gif) | succeeded, 1st attempt |
| reach right | y axis, negative (-y) | [reach_right.gif](charts/reach_right.gif) | succeeded, 1st attempt |
| reach up high | z axis, positive (+z) | [reach_up_high.gif](charts/reach_up_high.gif) | succeeded, 1st attempt |
| reach down low | z axis, negative (-z) | [reach_down_low.gif](charts/reach_down_low.gif) | succeeded, 1st attempt |

**What to look for in each clip:** the hand starts near the workspace's
measured center (roughly the first 5 of 51 frames), then moves toward that
direction's real target for the rest of the clip. Judge each clip on its
own — does the resulting motion look like what an English speaker watching
this camera angle would call that direction's name? Flag any clip where it
doesn't, or where two clips (e.g. forward vs. back) look ambiguous relative
to each other.

## How this was tested
For each of the 6 directional regions in `AXIS_DIRECTIONS` (`center` has no
direction to check, so it's excluded), one episode was recorded via
`lang_goal_rl.episode_recording.record_episode_with_goal_switch`, reusing
Stage 1's checkpoint (`experiments/01_uvfa_her_baseline/checkpoints/seed_0.zip`
— a healthy seed; seeds 2 and 7 are excluded project-wide for the documented
SAC deterministic-eval-collapse signature). The episode's first 5 steps
target `MEASURED_GOAL_BOX.centroid` (the workspace's measured center); the
remaining 45 steps (50-step FetchReach-v4 episode) target one real in-region
point sampled via `sample_region_goals(region_name, 1, seed=...)`. Literal-xyz
mode throughout — no language embedding involved, since this stage checks
the coordinate convention itself, not the language layer built on top of it.
Up to 3 seed attempts per direction were allowed (matching every other
stage's `make_demo.py` retry discipline); all 6 directions succeeded on the
first attempt, so no direction needed a retry.

---
## Full evidence
The complete technical record — per-direction targets, seeds, step counts,
travel distances, raw logs, and the anomaly notes worth flagging before a
human reviews the clips — lives in [`evidence.md`](evidence.md).
