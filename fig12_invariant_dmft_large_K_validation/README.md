# Fig. 12 — Invariant large-K DMFT validation on Ising data

Self-contained: the script samples 2D Ising configurations (L = 45) via
Wolff, runs the DMFT solver and a finite-N reference over a ν grid, and
assembles the figure. No external data.

```bash
uv run fig_invariant_dmft_large_K_validation.py
```

Output: `fig_invariant_dmft_large_K_validation.{pdf,png}` next to the
script. Parameters (in the script): T = 1.8, K = 512, γ = 0.4, η = 10,
ν ∈ {0.50, 0.85, 1.20}.
