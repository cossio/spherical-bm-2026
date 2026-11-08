#!/usr/bin/env bash
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# γ × M_train sweep with FIXED data subsample (mtrain_seed=42), 8 RBM-init seeds.
# Designed to give tighter error bars in the paper figure by isolating
# training + AIS noise from data-subsample noise.

set -euo pipefail

TAG=${1:-"I_gauss_fixed"}
M=${2:-20}
NPAR=${3:-8}
SEEDS_STR=${4:-"0,1,2,3,4,5,6,7"}
IFS=',' read -ra SEEDS <<< "$SEEDS_STR"

OUTDIR="../data/${TAG}"
mkdir -p "${OUTDIR}"

MTRAINS=(200 500 1000 2000 11428)
GAMMAS=(1e-3 3e-3 1e-2 3e-2 1e-1 3e-1 1.0 3.0 10.0)
MTRAIN_SEED=42

JOBS=()
for seed in "${SEEDS[@]}"; do
    for mtrain in "${MTRAINS[@]}"; do
        for gamma in "${GAMMAS[@]}"; do
            out="${OUTDIR}/g${gamma}_m${mtrain}_M${M}_s${seed}.h5"
            [[ -f "${out}" ]] && continue
            BS=256
            if (( mtrain < 256 )); then BS=$mtrain; fi
            JOBS+=("${gamma}|${mtrain}|${seed}|${BS}|${out}")
        done
    done
done

echo "=== I_gauss_fixed: ${#JOBS[@]} jobs, ${NPAR} parallel ==="
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

printf '%s\n' "${JOBS[@]}" | xargs -P "${NPAR}" -I {} bash -c '
    spec="{}"
    gamma="$(echo "$spec" | cut -d"|" -f1)"
    mtrain="$(echo "$spec" | cut -d"|" -f2)"
    seed="$(echo "$spec" | cut -d"|" -f3)"
    bs="$(echo "$spec" | cut -d"|" -f4)"
    out="$(echo "$spec" | cut -d"|" -f5)"
    julia --project=. -t 1 02_train_rbm_gauss.jl \
        --gamma "${gamma}" --M "'"${M}"'" --seed "${seed}" \
        --mtrain "${mtrain}" --mtrain_seed "'"${MTRAIN_SEED}"'" --batch "${bs}" \
        --iters 8000 --swag true --swag_start 6000 --swag_every 100 --swag_lr 5e-3 \
        --nbetas 10000 --nais 15 --swag_nbetas 5000 --swag_nais 10 \
        --out "${out}" > "${out%.h5}.log" 2>&1
'
echo "Done."
