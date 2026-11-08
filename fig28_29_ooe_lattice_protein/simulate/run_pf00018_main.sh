#!/usr/bin/env bash
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Main PF00018 OOE sweep at N_chains=1 per HANDOFF_LESSONS.md #1.
# 3 seeds × 6 k values; primary observable is max|λ_J| (not σ²_a — too noisy at N_chains=1).

set -euo pipefail
: "${PF00018_FASTA:?Set PF00018_FASTA to an absolute path to the original alignment}"
mkdir -p ../data/pf00018/n1 logs/pf00018/n1

T_AGE=${T_AGE:-10000}
GAMMA=${GAMMA:-0.01}
LR=${LR:-0.01}
N_CHAINS=1

for SEED in 0 1 2; do
  for K in 1 3 10 48 144 480; do
    OUT="../data/pf00018/n1/k${K}_s${SEED}_t${T_AGE}.h5"
    LOG="logs/pf00018/n1/k${K}_s${SEED}_t${T_AGE}.log"
    if [[ -f "$OUT" ]]; then
      echo "[skip] $OUT"
      continue
    fi
    echo "[run] k=$K seed=$SEED t_age=$T_AGE -> $OUT"
    uv run --quiet 02_ooe_train_pf00018.py --fasta "$PF00018_FASTA" \
      --k "$K" --gamma "$GAMMA" --lr "$LR" --n-chains "$N_CHAINS" \
      --t-age "$T_AGE" --seed "$SEED" \
      --log-every 50 --eig-every 200 \
      --out "$OUT" >"$LOG" 2>&1
  done
done
echo "[n1 sweep done]"
