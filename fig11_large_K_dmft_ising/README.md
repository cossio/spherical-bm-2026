# Fig. 11 — Large-K DMFT validation on 2D Ising data

Self-contained: the script samples 2D Ising configurations (L = 32) via
Wolff, builds the empirical covariance, runs the large-K DMFT solver and
a finite-N reference, and assembles the figure. No external data.

```bash
uv run fig_large_K_dmft_ising.py
```

Output: `fig_large_K_dmft_ising.{pdf,png}` next to the script.
Parameters (in the script): T_low = 2.2, T_high = 3.2, K_dyn = 512,
γ = 0.4, η = 10, ν = 0.85.
