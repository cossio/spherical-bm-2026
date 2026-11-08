#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "torch>=2.6",
#     "numpy>=2.4",
# ]
# ///
"""K-sweep: how does the SWAG-DD signature evolve as K grows from K << N
to K ~ N to K > N, while the *effective* rank of C stays O(1)?

For each K in K_VALUES the script
  1. samples K configurations from the 2D-Ising teacher (exact for N<=20),
  2. logs the eigenvalues of the empirical covariance
        C = (1/N) sum_{k=1..K} x^k x^{k T},     x^k in {-1,+1}^N
     so we can see whether C keeps a clean rank-1 outlier at beta>beta_c
     even as K grows past N,
  3. runs the existing MAP+SWAG gamma-sweep at that K and writes per-K
     JSON results into ../results/.

Probes the regime-(b) extension of the paper's "K fixed, N->inf" theory
(nips.tex line 155, citing tulinski2026): if the FM-phase Ising
covariance keeps a few O(1) outliers while the rest stays at the
thermal-fluctuation scale, the DD peak should *persist* as K grows.
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

from run import exact_sample, metropolis_sample  # noqa: E402
from run_swag import (  # noqa: E402
    BURN_IN, COLLECT_EVERY, LR_ADAM, LR_SGD,
    MC_SAMPLES, N_SWAG_SAMPLES, SGD_STEPS,
    _aggregate, run_one_seed,
)


# Paper-γ grid. Capped at 25 so that the largest *code* weight-decay at
# K=4, N=16 (wd = N*γ/(2K) = 2γ → wd_max = 50) stays at the existing
# 13_ising2d_van γ_max=50 limit and matches its stability budget
# lr*wd ≲ 0.3 with lr=0.005.  See feedback_paper_gamma_scaling.md.
GAMMA_GRID_PAPER = np.concatenate([
    np.array([0.0]),
    np.geomspace(1e-4, 1.0, 24),
    np.geomspace(1.2, 25.0, 12),
])


def c_spectrum(s_data: torch.Tensor) -> np.ndarray:
    """Eigenvalues of C = (1/N) X^T X, sorted descending. X has rows x^k."""
    K, N = s_data.shape
    X = s_data.detach().cpu().numpy().astype(np.float64)
    C = (X.T @ X) / N
    w = np.linalg.eigvalsh(C)
    return np.sort(w)[::-1]


def paper_to_code_gammas(gammas_paper: np.ndarray, N: int, K: int) -> np.ndarray:
    """Convert paper-γ to optimizer-wd via wd = N*γ/(2K).

    The paper writes the prior as exp(-Nγ/4 · ||W||²) and uses per-sample
    averaged log-likelihood, so the optimizer's weight_decay must scale as
    N/(2K) relative to the paper's γ.  See memory
    feedback_paper_gamma_scaling.md for the derivation.
    """
    return gammas_paper * N / (2.0 * K)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.50)
    p.add_argument("--K-values", type=int, nargs="+",
                   default=[4, 8, 16, 32])
    p.add_argument("--seeds", type=int, nargs="+",
                   default=list(range(10)))
    p.add_argument("--langevin-temp", type=float, default=0.3)
    p.add_argument("--swag-scale", type=float, default=1.0)
    p.add_argument("--burn-in", type=int, default=BURN_IN)
    p.add_argument("--sgd-steps", type=int, default=SGD_STEPS)
    p.add_argument("--n-mh-sweeps", type=int, default=200)
    p.add_argument("--tag", type=str, default="Ksweep_paperwd")
    p.add_argument("--gammas-paper", type=float, nargs="+", default=None,
                   help="Override the default paper-γ grid with this list.")
    p.add_argument("--implicit-l2", action="store_true",
                   help="Use Strang-split exact L2 (Adam + SGD), required when "
                        "lr*wd_code >= 1 — i.e. paper-γ * N / (2K) is large.")
    args = p.parse_args()

    L, N = args.L, args.L * args.L
    gamma_grid = (np.array(args.gammas_paper, dtype=float)
                  if args.gammas_paper is not None
                  else GAMMA_GRID_PAPER)
    out_dir = _HERE.parent / "results"
    out_dir.mkdir(exist_ok=True)

    print(
        f"K-sweep (paper-γ convention, wd=N*γ/(2K)): "
        f"L={L}, N={N}, beta={args.beta}, "
        f"K_values={args.K_values}, seeds={len(args.seeds)}, "
        f"T_lang={args.langevin_temp}, sgd-steps={args.sgd_steps}"
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
            # Match the rng/seed convention used inside run_one_seed.
            rng = np.random.default_rng(seed)
            if N <= 20:
                s_data = exact_sample(args.beta, L, K, rng=rng)
            else:
                s_data = metropolis_sample(
                    args.beta, L, K, n_sweeps=args.n_mh_sweeps, rng=rng,
                )
            spec = c_spectrum(s_data)
            spectra.append(spec.tolist())
            top = " ".join(f"{w:.3f}" for w in spec[:5])
            print(f"  seed={seed} C-spec top5: {top}  rank={len(spec)}")

            r = run_one_seed(
                L=L, beta=args.beta, K=K, gammas=gammas_code, kind="fvsbn",
                H=32, seed=seed,
                burn_steps=args.burn_in, sgd_steps=args.sgd_steps,
                lr_adam=LR_ADAM, lr_sgd=LR_SGD,
                collect_every=COLLECT_EVERY, n_swag=N_SWAG_SAMPLES,
                langevin_temp=args.langevin_temp,
                swag_scale=args.swag_scale,
                n_mc=MC_SAMPLES, n_mh_sweeps=args.n_mh_sweeps,
                implicit_l2=args.implicit_l2,
            )
            # Re-attach paper-γ for downstream plotting.
            r["gammas"] = gamma_grid.tolist()
            r["gammas_code"] = gammas_code.tolist()
            runs.append(r)
            print(f"  seed={seed} done ({time.time() - t0:.1f}s)")

        agg = _aggregate(runs)
        out = out_dir / (
            f"{args.tag}_K{K}_L{L}_beta{args.beta}"
            f"_T{args.langevin_temp}_seeds{len(args.seeds)}.json"
        )
        with open(out, "w") as f:
            json.dump({
                "K": K, "L": L, "N": N, "beta": args.beta,
                "langevin_temp": args.langevin_temp,
                "swag_scale": args.swag_scale,
                "burn_in": args.burn_in, "sgd_steps": args.sgd_steps,
                "seeds": args.seeds,
                "gamma_convention": "paper",
                "wd_formula": "N*gamma_paper/(2*K)",
                "implicit_l2": bool(args.implicit_l2),
                "gammas_paper": gamma_grid.tolist(),
                "gammas_code": gammas_code.tolist(),
                "C_spectra": spectra,
                "C_spec_mean": np.mean(spectra, axis=0).tolist(),
                "agg": agg,
            }, f, indent=2)
        print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
