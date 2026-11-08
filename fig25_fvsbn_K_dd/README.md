# Fig. 25 — FVSBN double descent on Curie–Weiss and 2D Ising

The saved base and extended gamma-grid JSON files for all four K values are included in `data/cw` and `data/ising`. Run `uv run fig_fvsbn_K_dd.py` directly to regenerate the figure without training.

## Simulation utilities

K ∈ {4, 8, 16, 32}, N = 16, 10 seeds per (K, γ).

```bash
# Curie–Weiss
cd simulate/cw   && uv run run_K_sweep.py --outdir ../../data/cw

# 2D Ising
cd ../ising      && uv run run_K_sweep.py --outdir ../../data/ising

cd ../..
uv run fig_fvsbn_K_dd.py
```
