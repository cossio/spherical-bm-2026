# Figures 5 and 13 — Reverse and forward KL

From this directory, use `uv` and Python 3.14+:

```bash
uv run --locked plot_reverse.py
uv run --locked plot_forward.py
uv run --locked verify.py
```

Outputs go to `reproduced/`; both renderers accept `--outdir`.

| Figure | Renderer | PDF | Input table |
|---|---|---|---|
| 5 | `plot_reverse.py` | `phase_diagram_gamma_omega.pdf` | `data/reverse/pp_rev_kl_2gamma.csv` |
| 13 | `plot_forward.py` | `fig_fwd_pp_kl_phase_diagram.pdf` | `data/forward/pp_fwd_kl_4gamma.csv` |

To recompute the tables, use Julia 1.12 with its standard libraries:

```bash
julia --startup-file=no --history-file=no --project=@stdlib generate_reverse.jl reproduced/reverse
julia --startup-file=no --history-file=no --project=@stdlib generate_forward.jl reproduced/forward
uv run --locked verify.py \
  --reverse-curves reproduced/reverse/pp_rev_kl_2gamma.csv \
  --forward-curves reproduced/forward/pp_fwd_kl_4gamma.csv
```

Without an output argument, the generators overwrite their tables in `data/`.
