"""Data-spike diagnostic for the PF00018 MSA (SH3 domain).

Loads the PF00018 FASTA via adabmDCA's DatasetDCA (applies the standard
0.8-identity reweighting) and checks whether the WEIGHTED centered one-hot
covariance has a detached top eigenvalue — prerequisite for defining the
c_a order-parameter direction in the OOE experiment.

Outputs:
    results/spike_diagnostic_pf00018.h5  — top-K eigenvectors + eigenvalues
    figures/spike_spectrum_pf00018.pdf   — eigenvalue spectrum

Usage:
    uv run scripts/01_spike_diagnostic_pf00018.py
"""

# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "numpy>=2.4",
#     "h5py>=3.12",
#     "matplotlib>=3.10",
#     "torch>=2.5",
#     "adabmDCA>=0.5",
# ]
# ///

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt

from adabmDCA.dataset import DatasetDCA


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent.parent
    repo_root = here.parent.parent
    default_fasta = repo_root / "temperature_tuning" / "data" / "PF00018_full.fasta"
    p = argparse.ArgumentParser()
    p.add_argument("--fasta", default=str(default_fasta))
    p.add_argument("--weights-cache",
                   default=str(here / ".." / "data" / "PF00018_weights.pt"),
                   help="torch-saved weights tensor; computed on first run")
    p.add_argument("--out-h5", default=str(here / ".." / "data" / "spike_diagnostic_pf00018.h5"))
    p.add_argument("--out-pdf", default=str(here / "figures" / "spike_spectrum_pf00018.pdf"))
    p.add_argument("--k-save", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    dev = torch.device(args.device)
    print(f"[spike-pf18] device={dev}")

    weights_cache = Path(args.weights_cache)
    if weights_cache.exists():
        print(f"[spike-pf18] loading cached weights {weights_cache}")
        ds = DatasetDCA(args.fasta, alphabet="protein", no_reweighting=True,
                        device=dev, dtype=torch.float32)
        ds.weights = torch.load(weights_cache, map_location=dev).to(
            device=dev, dtype=torch.float32)
    else:
        ds = DatasetDCA(args.fasta, alphabet="protein", clustering_th=0.8,
                        device=dev, dtype=torch.float32)
        weights_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ds.weights.cpu(), weights_cache)
        print(f"[spike-pf18] cached weights -> {weights_cache}")

    L = ds.get_num_residues()
    Q = ds.get_num_states()
    M = len(ds)
    M_eff = float(ds.get_effective_size())
    print(f"[spike-pf18] L={L} Q={Q} M={M} M_eff={M_eff:.1f} "
          f"alphabet='-ACDEFGHIKLMNPQRSTVWY' [t={time.time()-t0:.1f}s]")

    X_int = ds.data                                     # (M, L) int64
    w = ds.weights.to(torch.float64)                    # (M,)
    w_norm = w / w.sum()

    # One-hot on GPU, float64 for covariance precision
    oh = torch.zeros((M, L, Q), dtype=torch.float64, device=dev)
    oh.scatter_(2, X_int.unsqueeze(-1).to(torch.int64), 1.0)

    # Weighted mean
    f1 = (w_norm.view(M, 1, 1) * oh).sum(dim=0)         # (L, Q)
    oh_c = oh - f1.unsqueeze(0)                         # (M, L, Q)
    # Weighted covariance: C = sum_m w_m * oh_c[m] oh_c[m]^T (flattened)
    oh_flat = oh_c.reshape(M, L * Q)                    # (M, L*Q)
    sqrtw = torch.sqrt(w_norm).view(M, 1)
    X_w = oh_flat * sqrtw                               # (M, L*Q)
    C = X_w.T @ X_w                                     # (L*Q, L*Q)
    print(f"[spike-pf18] C shape {tuple(C.shape)} tr={float(C.trace()):.4f} "
          f"[t={time.time()-t0:.1f}s]")

    C_np = C.cpu().numpy()
    C_np = 0.5 * (C_np + C_np.T)
    evals, evecs = np.linalg.eigh(C_np)                 # ascending
    evals = evals[::-1]
    evecs = evecs[:, ::-1]

    N = L * Q
    zero_tol = 1e-10 * np.trace(C_np) / N
    n_zero = int((np.abs(evals) < zero_tol).sum())
    meaningful = evals[evals > zero_tol]
    print(f"[spike-pf18] eigenvalues: max={evals[0]:.4e} "
          f"min_pos={meaningful.min():.4e} near-zero count (~L={L}): {n_zero}")

    lam_1 = float(meaningful[0])
    lam_2 = float(meaningful[1])
    lam_3 = float(meaningful[2])
    bulk_start = max(1, len(meaningful) // 10)
    bulk_median = float(np.median(meaningful[bulk_start:]))
    ratio_12 = lam_1 / lam_2
    ratio_23 = lam_2 / lam_3
    ratio_to_bulk = lam_1 / bulk_median
    print(f"[spike-pf18] λ₁={lam_1:.4e} λ₂={lam_2:.4e} λ₃={lam_3:.4e}")
    print(f"[spike-pf18] λ₁/λ₂ = {ratio_12:.3f}, λ₂/λ₃ = {ratio_23:.3f}, "
          f"λ₁/bulk_median = {ratio_to_bulk:.1f}")

    K = args.k_save
    top_evecs = evecs[:, :K]
    top_evals = evals[:K]

    Path(args.out_h5).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out_h5, "w") as f:
        f["eigenvalues_all"] = evals
        f["top_eigenvectors"] = top_evecs
        f["top_eigenvalues"] = top_evals
        f["f1"] = f1.cpu().numpy()
        f.attrs["L"] = L
        f.attrs["Q"] = Q
        f.attrs["M"] = M
        f.attrs["M_eff"] = M_eff
        f.attrs["lam_1"] = lam_1
        f.attrs["lam_2"] = lam_2
        f.attrs["lam_bulk_median"] = bulk_median
        f.attrs["ratio_lam1_lam2"] = ratio_12
        f.attrs["ratio_lam1_bulk_median"] = ratio_to_bulk
        f.attrs["n_zero_modes"] = n_zero
        f.attrs["fasta"] = str(args.fasta)
        f.attrs["alphabet"] = "-ACDEFGHIKLMNPQRSTVWY"
    print(f"[spike-pf18] saved {args.out_h5}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.4))
    ranks = np.arange(1, len(meaningful) + 1)
    ax1.semilogy(ranks, meaningful, "o-", ms=3, lw=0.6)
    ax1.set_xlabel("rank $k$")
    ax1.set_ylabel(r"$\lambda_k$")
    ax1.set_title(f"PF00018 centered one-hot covariance, $M_{{\\rm eff}}={M_eff:.0f}$",
                  fontsize=10)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.axhline(bulk_median, color="C1", ls="--", lw=0.8,
                label=f"bulk median {bulk_median:.2e}")
    ax1.axhline(lam_1, color="C3", ls=":", lw=0.8,
                label=f"$\\lambda_1={lam_1:.2e}$")
    ax1.legend(fontsize=7, loc="upper right")

    n_zoom = min(30, len(meaningful))
    ax2.plot(np.arange(1, n_zoom + 1), meaningful[:n_zoom], "o-", ms=4, lw=0.8)
    ax2.set_xlabel("rank $k$")
    ax2.set_ylabel(r"$\lambda_k$")
    ax2.set_title(
        f"top {n_zoom}: $\\lambda_1/\\lambda_2={ratio_12:.2f}$, "
        f"$\\lambda_1/$bulk$={ratio_to_bulk:.0f}$",
        fontsize=10,
    )
    ax2.grid(True, alpha=0.3)
    ax2.axhline(bulk_median, color="C1", ls="--", lw=0.8)

    fig.tight_layout(pad=1.2, w_pad=2.0)
    Path(args.out_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_pdf, bbox_inches="tight")
    print(f"[spike-pf18] saved {args.out_pdf} [t={time.time()-t0:.1f}s]")

    print()
    print("=" * 60)
    verdict_ok = ratio_12 >= 1.5 and ratio_to_bulk >= 5.0
    if verdict_ok:
        print("VERDICT: clean spike. Top-1 eigenvector usable as c_1.")
    else:
        print("VERDICT: marginal spike. Consider using top 2-3 eigenvectors jointly.")
    print(f"  λ₁/λ₂          = {ratio_12:.3f}  (rule of thumb: > 1.5)")
    print(f"  λ₁/bulk_median = {ratio_to_bulk:.1f}   (rule of thumb: > 5)")
    print("=" * 60)


if __name__ == "__main__":
    main()
