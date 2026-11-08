#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "torch>=2.6",
#     "numpy>=2.4",
# ]
# ///
"""K-sweep on the rank-1 Curie-Weiss / Hopfield teacher.

Mirror of ../13_ising2d_van/scripts/run_K_sweep.py but on the
rank-1-by-construction CW teacher: the population covariance C* is
exactly rank-1 (xi xi^T / N times the magnetization^2), so this is the
cleanest test of the "K large, C low effective rank" extension of the
paper's K-fixed theory (nips.tex line 155, citing tulinski2026).

For each K in K_VALUES the script
  1. samples K configurations from the CW teacher,
  2. logs the eigenvalues of C = (1/N) sum_k x^k x^{k T},
  3. runs the existing MAP+SWAG gamma-sweep at that K and writes per-K
     JSON results into ../results/.

Expectation: at beta well above the CW critical point, lambda_1(C) grows
linearly with K (k samples nearly aligned with xi -> top eig ~ K * m^2)
while lambda_{2..N}(C) stay bounded by thermal sample-to-sample
fluctuation. Effective rank stays O(1). DD peak should persist.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from teacher import sample_teacher  # noqa: E402
from run_swag import (  # noqa: E402
    BURN_IN, COLLECT_EVERY, LR_ADAM, LR_SGD,
    MC_SAMPLES, N_SWAG_SAMPLES, SGD_STEPS,
    _aggregate, enumerate_configs, run_one_seed,
)


# Paper-γ grid; see Ising counterpart for the wd budget reasoning.
# implicit_l2=True is enabled below so SGD is unconditionally stable in
# wd, but we keep paper-γ_max=25 for direct cross-comparison with the
# Ising K-sweep.
GAMMA_GRID_PAPER = np.concatenate([
    np.array([0.0]),
    np.geomspace(1e-4, 1.0, 24),
    np.geomspace(1.2, 25.0, 12),
])


def c_spectrum(s_data: torch.Tensor) -> np.ndarray:
    """Eigenvalues of C = (1/N) X^T X, sorted descending."""
    K, N = s_data.shape
    X = s_data.detach().cpu().numpy().astype(np.float64)
    C = (X.T @ X) / N
    w = np.linalg.eigvalsh(C)
    return np.sort(w)[::-1]


def paper_to_code_gammas(gammas_paper: np.ndarray, N: int, K: int) -> np.ndarray:
    """Convert paper-γ to optimizer-wd via wd = N*γ/(2K).

    See feedback_paper_gamma_scaling.md.
    """
    return gammas_paper * N / (2.0 * K)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--K-values", type=int, nargs="+",
                   default=[4, 8, 16, 32])
    p.add_argument("--seeds", type=int, nargs="+",
                   default=list(range(10)))
    p.add_argument("--langevin-temp", type=float, default=0.3)
    p.add_argument("--swag-scale", type=float, default=1.0)
    p.add_argument("--burn-in", type=int, default=BURN_IN)
    p.add_argument("--sgd-steps", type=int, default=SGD_STEPS)
    p.add_argument("--exact-kl", action="store_true",
                   help="enumerate 2^N configs for noise-free reverse KL")
    p.add_argument("--tag", type=str, default="Ksweep_paperwd")
    p.add_argument("--gammas-paper", type=float, nargs="+", default=None,
                   help="Override the default paper-γ grid with this list.")
    args = p.parse_args()

    N = args.N
    gamma_grid = (np.array(args.gammas_paper, dtype=float)
                  if args.gammas_paper is not None
                  else GAMMA_GRID_PAPER)
    out_dir = _HERE.parent / "results"
    out_dir.mkdir(exist_ok=True)
    configs = enumerate_configs(N) if args.exact_kl else None

    print(
        f"K-sweep (paper-γ convention, wd=N*γ/(2K), implicit_l2=True): "
        f"N={N}, beta={args.beta}, K_values={args.K_values}, "
        f"seeds={len(args.seeds)}, T_lang={args.langevin_temp}, "
        f"sgd-steps={args.sgd_steps}, exact_kl={args.exact_kl}"
    )
    print(f"  γ_paper grid: 0..{gamma_grid.max():.1f}, {len(gamma_grid)} pts")

    for K in args.K_values:
        gammas_code = paper_to_code_gammas(gamma_grid, N, K)
        print(
            f"\n=== K = {K} (K/N = {K/N:.2f})  "
            f"wd_max = {gammas_code.max():.3f}  "
            f"lr*wd_max = {LR_SGD * gammas_code.max():.3f} ==="
        )
        runs, spectra = [], []
        for seed in args.seeds:
            t0 = time.time()
            # Replicate run_one_seed's data sampling so we can log C
            # spectrum with the same teacher draw.
            rng = np.random.default_rng(seed)
            xi_np = rng.choice([-1, 1], size=N).astype(np.float32)
            xi = torch.tensor(xi_np)
            s_data = sample_teacher(args.beta, xi, K, rng=rng)
            spec = c_spectrum(s_data)
            spectra.append(spec.tolist())
            top = " ".join(f"{w:.3f}" for w in spec[:5])
            print(f"  seed={seed} C-spec top5: {top}  rank={len(spec)}")

            r = run_one_seed(
                N=N, beta=args.beta, K=K, gammas=gammas_code, seed=seed,
                burn_steps=args.burn_in, sgd_steps=args.sgd_steps,
                lr_adam=LR_ADAM, lr_sgd=LR_SGD,
                collect_every=COLLECT_EVERY, n_swag=N_SWAG_SAMPLES,
                langevin_temp=args.langevin_temp,
                swag_scale=args.swag_scale, n_mc=MC_SAMPLES,
                configs=configs, common_rng=False, implicit_l2=True,
            )
            # Re-attach paper-γ for downstream plotting.
            r["gammas"] = gamma_grid.tolist()
            r["gammas_code"] = gammas_code.tolist()
            runs.append(r)
            print(f"  seed={seed} done ({time.time() - t0:.1f}s)")

        agg = _aggregate(runs)
        out = out_dir / (
            f"{args.tag}_K{K}_N{N}_beta{args.beta}"
            f"_T{args.langevin_temp}_seeds{len(args.seeds)}.json"
        )
        with open(out, "w") as f:
            json.dump({
                "K": K, "N": N, "beta": args.beta,
                "langevin_temp": args.langevin_temp,
                "swag_scale": args.swag_scale,
                "burn_in": args.burn_in, "sgd_steps": args.sgd_steps,
                "exact_kl": args.exact_kl,
                "seeds": args.seeds,
                "gamma_convention": "paper",
                "wd_formula": "N*gamma_paper/(2*K)",
                "gammas_paper": gamma_grid.tolist(),
                "gammas_code": gammas_code.tolist(),
                "C_spectra": spectra,
                "C_spec_mean": np.mean(spectra, axis=0).tolist(),
                "agg": agg,
            }, f, indent=2)
        print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
