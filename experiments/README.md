# Experiments

One directory per stage: `NN_slug/` (e.g. `01_uvfa_her_baseline/`), containing
run scripts, configs, and result notes for that stage's proof gate.

Nothing reusable lives here. If two experiments need the same code, promote it
to `src/lang_goal_rl/` and import it from both.
