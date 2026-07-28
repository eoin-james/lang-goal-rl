#!/usr/bin/env bash
# Launch stage 9's zero-shot waypoint-following eval across an explicit list
# of model-checkpoint seeds, as capped concurrent background processes.
# Eval-only (no RL training) -- reuses checkpoints from
# experiments/01_uvfa_her_baseline/checkpoints/seed_<k>.zip.
#
# Seed list is explicit (not a contiguous range like stage 5/8's
# launch_seeds.sh) because the healthy-seed set skips 2 and 7 (documented
# SAC deterministic-eval collapse, ROADMAP.md Known risks) -- default is the
# 7 healthy seeds not already covered by seed_0's original tier1/final run.
#
# Same concurrency convention as stage 5/8's launch_seeds.sh: cap =
# min(pending_runs, cores - 2), single-threaded math libs per process.
set -euo pipefail

cd "$(dirname "$0")"

SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
  SEEDS=(1 3 4 5 6 8 9)
fi

PENDING=${#SEEDS[@]}
CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc)
CAP=$((CORES - 2))
if [ "$PENDING" -lt "$CAP" ]; then
  CAP=$PENDING
fi
echo "cores=$CORES pending=$PENDING cap=$CAP seeds=${SEEDS[*]}"

for seed in "${SEEDS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$CAP" ]; do
    wait -n
  done
  mkdir -p "runs/seed_${seed}"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python run_waypoint_eval.py \
    --seed "${seed}" --episodes 50 --sanity-episodes 50 --tag final \
    >"runs/seed_${seed}/final_stdout.log" 2>&1 &
  echo "launched seed ${seed} (pid $!)"
done

wait
echo "all seeds complete (${SEEDS[*]})"
