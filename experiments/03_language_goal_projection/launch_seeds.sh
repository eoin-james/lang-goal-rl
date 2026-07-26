#!/usr/bin/env bash
# Launch stage 3's RL training+eval seeds as capped concurrent background
# processes. Cap = min(pending_runs, cores - 2), each process pinned to
# single-threaded math libraries to avoid oversubscription (per the
# runner's hard rules). Seed range is passed as two args: first and last
# (inclusive), so this same script serves both the 3-seed tier (0 2) and
# the scale-up to the full budget (3 9).
set -euo pipefail

cd "$(dirname "$0")"

FIRST_SEED="${1:-0}"
LAST_SEED="${2:-2}"
PENDING=$((LAST_SEED - FIRST_SEED + 1))

CORES=$(sysctl -n hw.ncpu)
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
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python train.py \
    --seed "${seed}" --total-timesteps 20000 --eval-episodes 50 --language-eval-episodes 50 \
    >"runs/seed_${seed}/stdout.log" 2>&1 &
  echo "launched seed ${seed} (pid $!)"
done

wait
echo "all seeds complete (${FIRST_SEED}..${LAST_SEED})"
