# Fig. 10 — Finite-η phase diagrams and dynamics

4 rows × 3 columns (η ∈ {1, 3, 10}), β=1, K=1, c=1. Self-contained:
the script implements the PM-bath stationary solver and the time-marched
MSR solver inline and caches intermediate results into `data/`.

```bash
julia --project=../_julia_env ../_julia_env/setup.jl    # one-time
julia -t auto fig_finite_eta_phase_diagrams.jl
```
