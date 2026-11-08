# Figure 8: finite-size equilibrium simulations

Run commands from this directory. Figure rendering requires `uv` and Python
3.14+. Sampling and saddle calculations require Julia 1.12; the Julia code
uses standard libraries only.

## Render saved results

```bash
uv run --locked scripts/verify_saved.py
uv run --locked scripts/plot_comparison.py --output figures/reproduced
```

Inputs are `validation/production/summary.csv`,
`validation/n256/summary.csv`, and `results/saddle_predictions.toml`, and `results/saddle_curve.csv`.
The command writes PDF and PNG files; the reference PDF is
`figures/equilibrium_comparison.pdf`.

## Run simulations

The manifests specify all parameters, chain lengths, and seeds. Figure 8 uses
`K=2`, `c=(1.25,0.75)`, `eta=4`, sizes `32,64,128,256`, and
`gamma=0.25,0.4,0.5625,0.7,1.25,1.5`.

The last argument to `run_local.sh` sets the number of concurrent Julia
processes, each using one BLAS thread. Set `JULIA_BIN` to select an executable.
Use new output directories; completed chains are not overwritten.

```bash
julia --startup-file=no --history-file=no --project=@stdlib scripts/saddle_predictions.jl
bash scripts/run_local.sh manifests/production.tsv results/reproduction 4
uv run --locked scripts/analyze.py --input results/reproduction --output validation/reproduction --require-pass
bash scripts/run_local.sh manifests/n256.tsv results/n256_reproduction 4
uv run --locked scripts/analyze.py --input results/n256_reproduction --output validation/n256_reproduction --require-pass
uv run --locked scripts/plot_comparison.py --summary validation/reproduction/summary.csv --n256-summary validation/n256_reproduction/summary.csv --input results/reproduction --n256-input results/n256_reproduction --output figures/reproduced
```

## Controls and replication

```bash
julia --startup-file=no --history-file=no --project=@stdlib scripts/test_bingham.jl
uv run --locked scripts/analyze.py --self-test
uv run --locked scripts/exact_reference.py --output results/reproduction_n2_reference.json
bash scripts/run_local.sh manifests/controls.tsv results/controls_reproduction 4
uv run --locked scripts/validate_controls.py --controls results/controls_reproduction
uv run --locked scripts/analyze.py --input results/controls_reproduction --output validation/controls_reproduction --require-pass
bash scripts/run_local.sh manifests/replication.tsv results/replication_reproduction 4
uv run --locked scripts/compare_replication.py --original results/reproduction/deep_coalesced/N128 --replication results/replication_reproduction/deep_coalesced/N128 --output validation/replication_reproduction/comparison.json --require-pass
```

Compact numerical summaries are in `validation/`; plotted coordinates are in
`data/plot_points.json`. Regenerating chain diagnostics requires the simulation
commands above. File hashes are in `provenance/release.sha256`.
