# Figures 15–18: fully visible binary BM simulations

Run commands from this directory with `uv` and Python 3.14+. Each script has
inline dependencies and an adjacent lockfile.

| Figure | Reference PDF | Numerical inputs |
|---|---|---|
| 15 | `figures/posterior_tempering.pdf` | `data/posterior.npz` |
| 16 | `figures/regularization_double_descent.pdf` | `data/gamma_final.npz`, `data/bm_map_sweep.npz`, `data/curve.json` |
| 17 | `figures/generation_temperature.pdf` | `data/beta.npz` |
| 18 | `figures/persistent_training.pdf` | `data/dynamics.npz` |

## Render saved results

```bash
uv run --frozen scripts/plot_experiments.py all
uv run --frozen scripts/plot_beta.py
uv run --frozen scripts/verify_release.py --figure-dir reproduced
```

Plots are written to `reproduced/`; both plotting commands accept `--outdir`.
To recompute the MAP curve and compare its arrays:

```bash
uv run --frozen scripts/map_sweep.py --output validation/map_replay --old-grid data/gamma_final.npz --threads 1
uv run --frozen scripts/verify_release.py --figure-dir reproduced --map-replay validation/map_replay
```

## Run simulations

Couplings are symmetric with zero diagonal and no fields. Teacher couplings
are `omega/N`. The posterior uses `exp(-eta*U)`, with
`U = K*logZ - sum_data sum_(i<j) J_ij*s_i*s_j + N*gamma/2*sum_(i<j) J_ij^2`.

| Experiment | N | K | omega | gamma |
|---|---:|---:|---:|---|
| Posterior temperature | 8 | 4 | 1.5 | 0.1, 1.5 |
| Regularization and MAP | 12 | 1 | 3 | Grid in `data/gamma_final.npz` |
| Generation temperature | 8 | 4 | 1.5 | 6 |
| Persistent training | 12 | 48 | 1.5 | 3.2 |

The dynamics command uses `mean-loss-gamma=0.8`, equivalent to `gamma=3.2`
in the summed convention. Learning rate is `0.003`, with
`Delta_t=2*N*0.003/K=0.0015`. The five settings use `m=1,3,6,12,120`
single-spin Gibbs updates and `nu_eff=m/(N*Delta_t)`.

```bash
uv run --frozen scripts/check_beta.py
uv run --frozen scripts/beta.py
uv run --frozen scripts/posterior.py --out results/posterior --datasets 100 --chains 4 --draws 4000 --warmup 1000 --workers 4
uv run --frozen scripts/posterior.py --out results/posterior-warm-refinement --datasets 100 --chains 4 --draws 16000 --warmup 1000 --workers 4 --gammas .1 --etas .05 .075 .1 .15
uv run --frozen scripts/posterior_merge_refinement.py
uv run --frozen scripts/dynamics.py --datasets 50 --replicas 3 --particles 1 --regularization-convention mean-loss-gamma --threads 2 --output results/dynamics
uv run --frozen scripts/gamma.py --draws 4096 --etas .2 .4 .6 1 --blocks 32 --threads 4 --output results/gamma --runtime-limit 0
uv run --frozen scripts/gamma.py --draws 4096 --etas 10 --blocks 32 --threads 4 --output results/gamma_cold_orbit --runtime-limit 0
uv run --frozen scripts/gamma_cold.py --draws 8192 --blocks 32 --threads 4 --output results/gamma_cold_laplace
uv run --frozen scripts/gamma_cold_hmc.py --draws 1024 --warmup 512 --chains 8 --threads 4 --output results/gamma_cold_hmc
uv run --frozen scripts/gamma_finalize.py --results results
```

Use `gamma_finalize.py --results data` to rebuild from the released source
summaries. `data/coverage.json` lists the stored arrays, and `data/sha256.json`
contains their hashes. After intentionally replacing data or reference figures,
run `uv run --frozen scripts/refresh_checksums.py`.
