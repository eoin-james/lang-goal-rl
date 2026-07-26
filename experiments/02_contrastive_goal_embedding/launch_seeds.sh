#!/usr/bin/env bash
# Launch stage 2's 10 RL training seeds as capped concurrent background
# processes. Cap = min(pending_runs, cores - 2), each process pinned to
# single-threaded math libraries to avoid oversubscription (per the
# runner's hard rules).
set -euo pipefail

cd "$(dirname "$0")"

CORES=$(sysctl -n hw.ncpu)
PENDING=10
CAP=$((CORES - 2))
if [ "$PENDING" -lt "$CAP" ]; then
  CAP=$PENDING
fi
echo "cores=$CORES pending=$PENDING cap=$CAP"

for seed in $(seq 0 9); do
  while [ "$(jobs -rp | wc -l)" -ge "$CAP" ]; do
    wait -n
  done
  mkdir -p "runs/seed_${seed}"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python train.py \
    --seed "${seed}" --total-timesteps 20000 --eval-episodes 50 \
    >"runs/seed_${seed}/stdout.log" 2>&1 &
  echo "launched seed ${seed} (pid $!)"
done

wait
echo "all seeds complete"
