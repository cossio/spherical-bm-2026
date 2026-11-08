#!/usr/bin/env bash
# Run the same TSV used by Slurm. Choose a NEW output root for reproduction.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
manifest="${1:-manifests/production.tsv}"
output_root="${2:-results/reproduction}"
workers="${3:-4}"
julia_bin="${JULIA_BIN:-julia}"
[[ "$workers" =~ ^[1-9][0-9]*$ ]] || { echo "workers must be positive" >&2; exit 2; }
mkdir -p "$output_root/logs"
pids=()
wait_batch() {
    local failure=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failure=1
    done
    pids=()
    [[ "$failure" == 0 ]]
}
while IFS=$'\t' read -r case_name n chain warmup draws thin original_output seed; do
    [[ "$case_name" == case ]] && continue
    [[ -n "$case_name" ]] || continue
    seed_args=()
    [[ -z "${seed:-}" ]] || seed_args=(--seed "$seed")
    "$julia_bin" --startup-file=no --history-file=no --project=@stdlib scripts/run_chain.jl \
        --case "$case_name" --n "$n" --chain "$chain" --warmup "$warmup" \
        --draws "$draws" --thin "$thin" --out "$output_root" "${seed_args[@]}" \
        > "$output_root/logs/${case_name}_N${n}_chain${chain}.log" 2>&1 &
    pids+=("$!")
    if [[ "${#pids[@]}" -ge "$workers" ]]; then
        wait_batch
    fi
done < "$manifest"
wait_batch
