# Fig. 6 — Out-of-equilibrium training trajectories

The repository includes the exact finite-$N$ checkpoints and large-$N$ DMFT
trajectory tables used for the manuscript figure. To render it:

```bash
julia --project=../_julia_env ../_julia_env/setup.jl    # one-time
julia fig_ooe_wrap.jl
```

This writes `fig_ooe_wrap.pdf` and `fig_ooe_wrap.png` in this directory.

The finite-$N$ data can optionally be regenerated from scratch:

```bash
julia --threads=auto simulate/grokking_note_kl_nu_sweep_long.jl
julia fig_ooe_wrap.jl
```

The simulation runs $15$ sampling rates with $4$ seeds at $N=1500$,
$dt=0.02$, and $T_{\max}=1000$, so it is substantially more expensive than
rendering from the bundled checkpoints.

Parameters: $\omega^*=2.5$, $c=(1.6,0.4)$, $\gamma=0.4$, $\eta=10$,
$\beta=1$, and
$\nu\in\{0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.55,0.70,0.85,1.00,
1.30,1.70,2.20,3.00\}$.

Data layout:

- `data/grokking_note_kl_nu_sweep_long.jld2`: finite-$N$ trajectories.
- `data/theory/{reverse_KL,lambda1,u1sq}_merged.csv`: DMFT trajectories.
