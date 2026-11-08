#!/usr/bin/env bash
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Phase 1 GPU k-sweep: k ∈ {1, 10, 100} × {β_sel=1000, β_sel=100} teachers.
# Each run: t_age=5000, M=3000, seed=0, γ=0.01, M_chains=256.
set -eu
: "${LATTICE_TEACHER_DIR:?Set LATTICE_TEACHER_DIR to an absolute directory containing teacher_beta*_n10k.h5}"
mkdir -p ../data/phase1 logs

for BETA in 1000 100; do
    TEACHER="${LATTICE_TEACHER_DIR}/teacher_beta${BETA}_n10k.h5"
    if [ "$BETA" = "1000" ]; then
        SPIKE="../data/spike_diagnostic.h5"
    else
        SPIKE="../data/spike_diagnostic_b${BETA}.h5"
    fi
    for K in 1 10 100; do
        OUT="../data/phase1/b${BETA}_k${K}_g01_m3k.h5"
        LOG="logs/phase1_b${BETA}_k${K}.log"
        echo "=== β=${BETA}, k=${K} ===" | tee -a logs/phase1.summary
        uv run 02_ooe_train_gpu.py \
            --teacher "$TEACHER" --spike "$SPIKE" \
            --k "$K" --gamma 0.01 --m-train 3000 --t-age 5000 \
            --log-every 50 --eig-every 200 --seed 0 \
            --out "$OUT" 2>&1 | tee "$LOG"
        echo "=== done β=${BETA}, k=${K} (out=$OUT) ===" | tee -a logs/phase1.summary
    done
done
echo "All Phase 1 runs completed." | tee -a logs/phase1.summary
