# Figure 9 — MSR/DMFT and finite-N Langevin training

Run from this directory. The simulation driver reuses existing files in
`data/validate_msr_K2/`.

```bash
julia --project=../_julia_env ../_julia_env/setup.jl
julia simulate/run_validate_K2.jl
julia fig_msr_langevin_validation.jl
```

Parameters: gamma=0.5, nu=0.3, beta=1, eta=3, N=4000, time horizon 30,
five seeds. Panels A/B use K=2 and c=(1.5,0.5)/(1.8,0.2); panels C/D use
K=1 and c=1/0.3.

For the separate discrete-SGD simulation, see
[`experiments/sgd_dmft`](../experiments/sgd_dmft/README.md).
