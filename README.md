# Spherical Boltzmann machines

Thomas Tulinski, Simona Cocco, Rémi Monasson, and Jorge Fernandez-de-Cossio-Diaz.
[Spherical Boltzmann machines: a solvable theory of learning and generation in energy-based models](https://arxiv.org/abs/2605.09031).
arXiv:2605.09031 (2026). Accepted in NeurIPS'2026.

Code and saved data for reproducing the paper's figures and simulations.

Use `uv` for Python scripts and Julia 1.12 for Julia scripts. Set up the shared
Julia environment once:

```bash
julia --project=_julia_env _julia_env/setup.jl
```

Follow the commands in each directory's README to render saved results or rerun
simulations.

| Figures | Instructions |
|---|---|
| 1 | [fig01_K1_phase_diagram_wide](fig01_K1_phase_diagram_wide/README.md) |
| 2 | [fig02_dynamics_combined](fig02_dynamics_combined/README.md) |
| 3 | [fig03_temperature_tuning_pmo](fig03_temperature_tuning_pmo/README.md) |
| 4 | [fig04_double_descent](fig04_double_descent/README.md) |
| 5, 13 | [fig05_13_phase_diagram_gamma_omega](fig05_13_phase_diagram_gamma_omega/README.md) |
| 6 | [fig06_ooe_wrap](fig06_ooe_wrap/README.md) |
| 7 | [fig07_phase_diagram_k2_appendix](fig07_phase_diagram_k2_appendix/README.md) |
| 8 | [experiments/equilibrium](experiments/equilibrium/README.md) |
| 9 | [fig09_msr_langevin_validation](fig09_msr_langevin_validation/README.md) |
| 10 | [fig10_finite_eta_phase_diagrams](fig10_finite_eta_phase_diagrams/README.md) |
| 11 | [fig11_large_K_dmft_ising](fig11_large_K_dmft_ising/README.md) |
| 12 | [fig12_invariant_dmft_large_K_validation](fig12_invariant_dmft_large_K_validation/README.md) |
| 14 | [fig14_sbm_vs_gaussian](fig14_sbm_vs_gaussian/README.md) |
| 15–18 | [experiments/small_bm](experiments/small_bm/README.md) |
| 19–22 | [experiments/small_rbm](experiments/small_rbm/README.md) |
| 23–24 | [fig23_24_temperature_tuning](fig23_24_temperature_tuning/README.md) |
| 25 | [fig25_fvsbn_K_dd](fig25_fvsbn_K_dd/README.md) |
| 26 | [fig26_rbm_finance_double_descent](fig26_rbm_finance_double_descent/README.md) |
| 27 | [fig27_bayesgan_unimodal](fig27_bayesgan_unimodal/README.md) |
| 28–29 | [fig28_29_ooe_lattice_protein](fig28_29_ooe_lattice_protein/README.md) |

Additional simulation: [discrete SGD / DMFT](experiments/sgd_dmft/README.md).

The paper's PDF assets are in `reference_figures/`; `figure_manifest.json`
maps figure numbers to files. Check the downloaded code and data with:

```bash
uv run verify_release.py
```
