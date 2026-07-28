#!/usr/bin/env bash
# Launch stage 10's zero-shot typed-command-pipeline eval across an explicit list of
# model-checkpoint seeds, as capped concurrent background processes.
# Eval-only (no RL training) -- reuses checkpoints from
# experiments/01_uvfa_her_baseline/checkpoints/seed_<k>.zip, the same 8 healthy seeds
# stages 8/9 already validated (seeds 2 and 7 are the documented SAC deterministic-eval
# collapse seeds, ROADMAP.md Known risks -- never passed here).
#
# Same concurrency convention as stage 8/9's launch_seeds.sh: cap =
# min(pending_runs, cores - 2), single-threaded math libs per process.
set -euo pipefail

cd "$(dirname "$0")"

SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
  SEEDS=(0 1 3 4 5 6 8 9)
fi

PENDING=${#SEEDS[@]}
CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc)
CAP=$((CORES - 2))
if [ "$PENDING" -lt "$CAP" ]; then
  CAP=$PENDING
fi
if [ "$CAP" -lt 1 ]; then
  CAP=1
fi
echo "cores=$CORES pending=$PENDING cap=$CAP seeds=${SEEDS[*]}"

for seed in "${SEEDS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$CAP" ]; do
    wait -n
  done
  mkdir -p "runs/seed_${seed}"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python run_command_eval.py \
    --seed "${seed}" --sanity-episodes 50 \
    >"runs/seed_${seed}/stdout.log" 2>&1 &
  echo "launched seed ${seed} (pid $!)"
done

wait
echo "all seeds complete (${SEEDS[*]})"
