# Figures 19–22: binary RBM simulations

Run commands from this directory with `uv` and Python 3.14+. Each script has
inline dependencies and an adjacent lockfile.

The model has 16 visible spins, 3 hidden spins, no biases, and 16 training
examples. The teacher uses two orthogonal Hadamard patterns with amplitudes
1 and 0.5. Data generation and fitting use joint inverse temperature 1.
The log posterior is `eta*(sum_data log P(v|W) - gamma*||W||_F^2/2)`.

| Figure | Reference PDF | Numerical inputs |
|---|---|---|
| 19 | `figures/small_rbm_eta.pdf` | `data/eta.json` |
| 20 | `figures/small_rbm_gamma.pdf` | `data/gamma.json`, `map/curve.json` |
| 21 | `figures/small_rbm_beta.pdf` | `data/beta.npz`, `data/beta_weights.npz` |
| 22 | `figures/small_rbm_nu.pdf` | `data/nu.npz` |

## Render saved results

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked scripts/verify.py
uv run --locked scripts/plot.py --outdir generated
```

The renderer writes PDFs to `generated/`. Sampling configurations are in
`data/posterior_configs.json`; file hashes are in `data/checksums.json`.

## Refit the MAP curve

Use a new output directory. Run each seed from 0 through 7, then collect:

```bash
for seed in 0 1 2 3 4 5 6 7; do
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked map/run_map.py --seed "$seed" --outdir /tmp/rbm-map-new
done
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked map/collect_results.py --results /tmp/rbm-map-new
```

Initialization arrays are in `map/inputs/`. The collector requires the full
candidate outputs produced by these runs; `map/results/` contains the saved
selected fits used by the figure.

## Posterior and sampling-rate simulations

`scripts/17_eta_validation.py` uses replica-exchange HMC. Supply the settings
in `data/posterior_configs.json`; use `--help` for its command-line options.
`scripts/30_nu_slow.py` runs the persistent-Gibbs and refresh controls.
The displayed rates use `nu_eff=p/0.02` and dataset seeds 8–15.

```bash
uv run --locked scripts/17_eta_validation.py --help
uv run --locked scripts/30_nu_slow.py --help
OPENBLAS_NUM_THREADS=1 uv run --locked map/validate_runner.py
uv run --locked scripts/test_eta_validation.py
uv run --locked scripts/test_nu_slow.py
uv run --locked scripts/test_nu_controls.py
```
