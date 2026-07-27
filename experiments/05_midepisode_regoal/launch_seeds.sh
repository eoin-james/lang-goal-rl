#!/usr/bin/env bash
# Launch stage 5's zero-shot goal-swap eval for 3 seeds as capped concurrent
# background processes. Eval-only (no RL training) -- reuses checkpoints
# from experiments/01_uvfa_her_baseline/checkpoints/seed_<k>.zip, provisioned
# by that experiment's launch_checkpoint_seeds.sh side task (see report.md's
# "Checkpoint provisioning" section). Same concurrency convention as stage
# 3/4's launch_seeds.sh: cap = min(pending_runs, cores - 2), single-threaded
# math libs per process.
set -euo pipefail

cd "$(dirname "$0")"

FIRST_SEED="${1:-0}"
LAST_SEED="${2:-2}"
PENDING=$((LAST_SEED - FIRST_SEED + 1))

CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc)
CAP=$((CORES - 2))
if [ "$PENDING" -lt "$CAP" ]; then
  CAP=$PENDING
fi
echo "cores=$CORES pending=$PENDING cap=$CAP seeds=${FIRST_SEED}..${LAST_SEED}"

for seed in $(seq "$FIRST_SEED" "$LAST_SEED"); do
  while [ "$(jobs -rp | wc -l)" -ge "$CAP" ]; do
    wait -n
  done
  mkdir -p "runs/seed_${seed}"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python run_regoal_eval.py \
    --seed "${seed}" --episodes-per-combo 40 --sanity-episodes 50 \
    >"runs/seed_${seed}/stdout.log" 2>&1 &
  echo "launched seed ${seed} (pid $!)"
done

wait
echo "all seeds complete (${FIRST_SEED}..${LAST_SEED})"
