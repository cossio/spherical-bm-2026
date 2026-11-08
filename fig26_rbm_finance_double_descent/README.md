# Figure 26 — Gaussian-visible RBM on Ken French data

Render from the saved likelihoods, singular values, and covariance eigenvalues:

```bash
uv run fig_rbm_finance_double_descent.py I_gauss_fixed
```

To prepare the data and run the training sweeps:

```bash
uv run simulate/01_prepare_kenfrench.py
julia --project=simulate -e "using Pkg; Pkg.instantiate()"
bash simulate/06_sweep_gauss.sh
bash simulate/07_extend_gauss.sh
bash simulate/08_sweep_gauss_fixedsubset.sh
uv run fig_rbm_finance_double_descent.py I_gauss_fixed
```

The preparation script writes `data/kenfrench49_daily.h5`; the fixed-subset
sweep writes `data/I_gauss_fixed/*.h5`.
