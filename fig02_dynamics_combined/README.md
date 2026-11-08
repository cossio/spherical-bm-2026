# Fig. 2 — Training dynamics: MSR/DMFT vs finite-N Langevin

```bash
julia --project=../_julia_env ../_julia_env/setup.jl    # one-time

# 1. K=2 FM trajectory (MSR + finite-N) for panels A, B
julia simulate/run_FM_trajectories.jl

# 2. stationary s_∞(ν) sweeps for panel C (γ ∈ {0.8, 0.85, 0.9, 0.95, 0.99})
julia simulate/s_vs_nu_clean.jl
julia simulate/s_vs_nu_g0p95_clean.jl
julia simulate/s_vs_nu_g0p99_clean.jl

# 3. assemble the figure
julia fig_dynamics_combined_traceK_with_hu1.jl
```

Parameters: γ=0.4, η=10, β=1, ν=0.7, c=(1.7, 0.3), s₀=0.1 in A/B; η=3,
β=1, c=1 in C.
