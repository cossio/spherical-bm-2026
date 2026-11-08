# Figures 28–29 — Lattice-protein and PF00018 dynamics

Render from the saved trajectories and spike eigenvalues in `data/`:

```bash
uv run fig_ooe_potts.py
uv run fig_ooe_pf00018.py
```

Training requires original sequences or alignments, which are not included,
and full spike diagnostics including eigenvectors. For PF00018:

```bash
uv run simulate/01_spike_diagnostic_pf00018.py --fasta /absolute/path/PF00018.fasta
PF00018_FASTA=/absolute/path/PF00018.fasta bash simulate/run_pf00018_main.sh
PF00018_FASTA=/absolute/path/PF00018.fasta T_AGE=10000 bash simulate/run_pf00018_pilot.sh
```

The pilot wrapper writes the 256-chain run to `data/pf00018/n256`; set
`T_AGE=10000` as above for the figure's duration. Wrappers skip existing output
files; use a separate checkout or move the saved plotting data before training.

For lattice-protein runs, use `simulate/02_ooe_train_gpu.py` with explicit
`--teacher`, `--spike`, `--k`, `--n-chains`, `--seed`, and `--t-age` arguments.
The exploratory `run_phase1_sweep.sh` also requires `LATTICE_TEACHER_DIR`
containing `teacher_beta1000_n10k.h5` and `teacher_beta100_n10k.h5`.
