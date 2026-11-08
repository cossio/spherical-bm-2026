# Discrete SGD validation

Code and saved data for the discrete-SGD check accompanying Figure 9.
The Figure 9 plot is reproduced separately in
[`../../fig09_msr_langevin_validation`](../../fig09_msr_langevin_validation).

## Verify saved data

Requires `uv` and Python 3.14+:

```sh
uv run verify.py
```

Checks all file hashes, 24 production settings, metadata, array invariants,
288 independently reconstructed summary fields, and both theory refinement
tables. Outputs are written to ignored `generated/`.

Optional simulation and theory kernel checks:

```sh
uv run scripts/run_sgd.py --selftest
uv run validation/independent_oracle.py
uv run validation/check_theory.py
```

The first two commands resolve PyTorch; the third requires Julia (standard
library only). Release checks used Python 3.14, NumPy 2.4.4, PyTorch 2.11.0 on
CPU, and Julia 1.12.7. Saved production runs used PyTorch 2.7.0 on V100 GPUs.

## Parameters and reproduction

Each setting has eight paired trajectories, `N=1000,4000`, learning step
`.02,.01,.005`, horizon `30`, saved interval `.1`, `nu=.3`, `beta=1`, and seed
`20261107`. One arm samples one datum per update; the paired arm uses the full
dataset. Both use one persistent spherical particle and share sampler noise.
Weights start at zero; independent weight noise is disabled (`eta=Inf`).

| Case | Covariance eigenvalues (`c`) | `gamma` | Initial overlaps (`s0`) |
|---|---|---|---|
| A | 1.5, 0.5 | 0.5 | .30, .25 |
| B | 1.8, 0.2 | 0.5 | .30, .15 |
| C | 1 | 0.5 | .30 |
| D | 1 | 1.5 | .10 |

Reproduce one simulation (set `--device cpu` for CPU execution):

```sh
uv run scripts/run_sgd.py --case A --N 4000 --replicas 8 --dt .005 \
  --Tmax 30 --save-dt .1 --nu .3 --seed 20261107 --device cuda \
  --output generated/results/case_A_N4000_dt0p005.npz
```

The `N=4000` setting stores 16 dense float64 weight matrices, about 2 GB before
intermediates. For a small run use `--N 32 --replicas 2 --Tmax .1 --dt .01
--save-dt .01`.

Reproduce a DMFT reference; choose parameters from the table and time step
`.05,.025,.0125`:

```sh
julia --startup-file=no theory/msr_reference.jl --eta=Inf --beta=1 --nu=.3 \
  --gamma=.5 --c=1.5,.5 --s0=.30,.25 --dt=.0125 --Tmax=30 \
  --outfile=generated/theory/A_dt0p0125_optimized.csv
julia --startup-file=no theory/check_refinement.jl 0p025 0p0125
```

## Files

- `results/`: 24 NPZ trajectories and matching parameter/software JSON files.
  Overlap axes are `(arm, replica, time, mode)`; remaining arrays contain
  multipliers, traces, sphere norms, signal weight blocks, data counts and data.
- `theory/`: solvers, twelve reference CSVs, and refinement records.
- `validation/summary.json`: all saved numerical summaries.
- `validation/release_checks.json`: executed numerical checks.
- `provenance/export.json`: original hashes and export adaptations.
- `checksums.json`: hashes of all package files.
