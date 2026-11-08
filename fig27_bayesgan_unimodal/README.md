# Figure 27 — Tempered BayesGAN

Render from `data/sigma_eta_unimodal.npz` and
`data/sigma_eta_unimodal_extra.npz`:

```bash
uv run fig_bayesgan_unimodal.py
```

Outputs go to `figures/`.

For training, use `simulate/run_sigma_eta_sweep.py` with `--target gaussian`,
`--n_chains_g 16`, and explicit `--sigmas`, `--etas`, `--seeds`, and
`--out_name`. See `--help` and the saved NPZ grids for parameter values.
