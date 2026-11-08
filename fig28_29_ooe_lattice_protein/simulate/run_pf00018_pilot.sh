#!/usr/bin/env bash
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Pilot OOE k-sweep on PF00018 at t_age=3000.
# Run from lattice_protein_ooe/ directory.

set -euo pipefail
: "${PF00018_FASTA:?Set PF00018_FASTA to an absolute path to the original alignment}"
mkdir -p ../data/pf00018/n256 logs/pf00018

T_AGE=${T_AGE:-3000}
GAMMA=${GAMMA:-0.01}
LR=${LR:-0.01}
N_CHAINS=${N_CHAINS:-256}
SEED=${SEED:-0}

for K in 1 3 10 48 144 480 1440; do
  OUT="../data/pf00018/n256/k${K}_t${T_AGE}.h5"
  LOG="logs/pf00018/k${K}_t${T_AGE}.log"
  if [[ -f "$OUT" ]]; then
    echo "[skip] $OUT already exists"
    continue
  fi
  echo "[run] k=$K  t_age=$T_AGE  -> $OUT  (log: $LOG)"
  uv run --quiet 02_ooe_train_pf00018.py --fasta "$PF00018_FASTA" \
    --k "$K" --gamma "$GAMMA" --lr "$LR" --n-chains "$N_CHAINS" \
    --t-age "$T_AGE" --seed "$SEED" \
    --log-every 50 --eig-every 200 \
    --out "$OUT" >"$LOG" 2>&1
  echo "[done] k=$K"
done
echo "[all done]"
