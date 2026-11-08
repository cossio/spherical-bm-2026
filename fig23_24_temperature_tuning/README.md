# Figures 23–24 — Temperature tuning on PF00072 and PF00018

Run from this directory using the saved arrays:

```bash
uv run fig_publication_temperature_tuning.py PF00072
uv run fig_publication_temperature_tuning.py PF00018
```

Inputs: `data/PF00072_pt_merged/` and `data/PF00018_pt/`, including the
refined-grid and seed-1 arrays. Outputs: `figures/fig_temperature_tuning_<FAMILY>.{pdf,png}`.

Training utilities are in `simulate/` and require the original alignments and
model checkpoints, which are not included. Set the family and input paths in
those utilities before running; some default to PF00014.
