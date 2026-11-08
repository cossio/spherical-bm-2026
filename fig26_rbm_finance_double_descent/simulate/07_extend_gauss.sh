#!/usr/bin/env bash
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Extend H_gauss with higher-γ runs for the paper figure.

set -euo pipefail

TAG=${1:-"H_gauss"}
M=${2:-20}
NPAR=${3:-8}
SEEDS_STR=${4:-"0,1,2"}
IFS=',' read -ra SEEDS <<< "$SEEDS_STR"

OUTDIR="../data/${TAG}"
mkdir -p "${OUTDIR}"

MTRAINS=(200 500 1000 2000 11428)
GAMMAS=(1.0 3.0 10.0)

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

echo "=== Extend H_gauss: ${#JOBS[@]} jobs, ${NPAR} parallel ==="
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
        --mtrain "${mtrain}" --batch "${bs}" \
        --iters 8000 --swag true --swag_start 6000 --swag_every 100 --swag_lr 5e-3 \
        --nbetas 10000 --nais 15 --swag_nbetas 5000 --swag_nais 10 \
        --out "${out}" > "${out%.h5}.log" 2>&1
'
echo "Done."
