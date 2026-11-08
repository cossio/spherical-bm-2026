"""Data-spike diagnostic for the lattice-protein teacher distribution.

Decides whether the fixed teacher MSA at β_sel=1000 has a detached top
eigenvalue in its centered one-hot covariance — the prerequisite for
defining c_a as in our Fig. 6C theory. If no clear detachment, we need to
fall back (plant a rank-1 signal, or lower β_sel).

Outputs:
    results/spike_diagnostic.h5  — top-K eigenvectors + eigenvalues
    figures/spike_spectrum.pdf   — eigenvalue spectrum (validate-figure)

Usage:
    uv run --project ../lattice_proteins python scripts/01_spike_diagnostic.py
"""

# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy>=2.4", "scipy>=1.17", "h5py>=3.12", "matplotlib>=3.10"]
# ///

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from matplotlib import pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent.parent
    p.add_argument(
        "--teacher",
        default=str((here / ".." / "lattice_proteins" / "data" / "teacher_beta1000_n10k.h5").resolve()),
    )
    p.add_argument("--out-h5", default=str(here / ".." / "data" / "spike_diagnostic.h5"))
    p.add_argument("--out-pdf", default=str(here / "figures" / "spike_spectrum.pdf"))
    p.add_argument("--k-save", type=int, default=5, help="number of top eigenvectors to save")
    return p.parse_args()


def one_hot(msa: np.ndarray, Q: int) -> np.ndarray:
    """(M, L) integer MSA in {0,..,Q-1} -> (M, L, Q) one-hot float."""
    M, L = msa.shape
    oh = np.zeros((M, L, Q), dtype=np.float64)
    oh[np.arange(M)[:, None], np.arange(L)[None, :], msa] = 1.0
    return oh


def main() -> None:
    args = parse_args()

    with h5py.File(args.teacher, "r") as f:
        msa = f["msa"][:]  # Julia writes (L, M); h5py reads column-major as (M, L)
        L = int(f.attrs["L"])
        Q = int(f.attrs["Q"])
        beta_sel = float(f.attrs["beta_sel"])
        H_T = float(f.attrs["H_T"])

    # Julia is 1-based, convert to 0-based
    msa = msa - 1
    assert msa.shape[1] == L
    M = msa.shape[0]
    print(f"[spike] teacher: M={M}, L={L}, Q={Q}, β_sel={beta_sel}, H_T={H_T:.3f} nats")

    # Centered one-hot
    X = one_hot(msa, Q)                              # (M, L, Q)
    f1 = X.mean(axis=0)                              # (L, Q)
    Xc = (X - f1[None, :, :]).reshape(M, L * Q)      # (M, L*Q)
    print(f"[spike] one-hot: X shape {X.shape}, centered flat shape {Xc.shape}")
    print(f"[spike] f1 entropy per site (nats) mean={(-(f1 * np.log(f1 + 1e-12)).sum(axis=1)).mean():.3f}")

    # Empirical covariance (symmetric, 540x540)
    N = L * Q
    C = (Xc.T @ Xc) / M                              # (N, N)
    print(f"[spike] C shape {C.shape}, tr={np.trace(C):.4f}, ‖C‖_F={np.linalg.norm(C):.4f}")

    # Diagonalize (symmetric)
    evals, evecs = np.linalg.eigh(C)                 # ascending order
    # Flip to descending
    evals = evals[::-1]
    evecs = evecs[:, ::-1]

    # Drop the L structurally-zero modes from the one-letter-per-site constraint
    zero_tol = 1e-10 * np.trace(C) / N
    n_zero = int((np.abs(evals) < zero_tol).sum())
    n_meaningful = N - n_zero
    print(f"[spike] eigenvalues: max={evals[0]:.4e}, min_pos={evals[evals > zero_tol].min():.4e}, "
          f"near-zero count (expected ≈ {L}): {n_zero}")

    # Clean bulk metrics: ignore the zero modes
    meaningful_evals = evals[evals > zero_tol]
    lam_1 = meaningful_evals[0]
    lam_2 = meaningful_evals[1] if len(meaningful_evals) > 1 else np.nan
    lam_3 = meaningful_evals[2] if len(meaningful_evals) > 2 else np.nan
    bulk_start = max(1, len(meaningful_evals) // 10)  # approximate bulk starts at 10th percentile
    lam_bulk_edge = meaningful_evals[bulk_start]
    bulk_median = np.median(meaningful_evals[bulk_start:])

    ratio_12 = lam_1 / lam_2 if lam_2 > 0 else np.inf
    ratio_23 = lam_2 / lam_3 if lam_3 > 0 else np.inf
    ratio_to_bulk = lam_1 / bulk_median

    print(f"[spike] λ₁={lam_1:.4e}, λ₂={lam_2:.4e}, λ₃={lam_3:.4e}")
    print(f"[spike] λ_bulk_edge (idx {bulk_start})={lam_bulk_edge:.4e}, bulk_median={bulk_median:.4e}")
    print(f"[spike] λ₁/λ₂ = {ratio_12:.3f}, λ₂/λ₃ = {ratio_23:.3f}, λ₁/bulk_median = {ratio_to_bulk:.1f}")

    # Save top-K eigenvectors (for later use as c_a in the OOE experiment)
    K = args.k_save
    top_evecs = evecs[:, :K]                         # (N, K)
    top_evals = evals[:K]

    Path(args.out_h5).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out_h5, "w") as f:
        f["eigenvalues_all"] = evals
        f["top_eigenvectors"] = top_evecs            # (L*Q, K), stored in flat basis
        f["top_eigenvalues"] = top_evals
        f["f1"] = f1
        f.attrs["L"] = L
        f.attrs["Q"] = Q
        f.attrs["M"] = M
        f.attrs["beta_sel"] = beta_sel
        f.attrs["lam_1"] = lam_1
        f.attrs["lam_2"] = lam_2
        f.attrs["lam_bulk_median"] = bulk_median
        f.attrs["ratio_lam1_bulk_median"] = ratio_to_bulk
        f.attrs["n_zero_modes"] = n_zero
        f.attrs["teacher_file"] = str(args.teacher)
    print(f"[spike] saved {args.out_h5}")

    # --- Plot spectrum ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.4))

    ax1.semilogy(np.arange(1, len(meaningful_evals) + 1), meaningful_evals, "o-", ms=3, lw=0.6)
    ax1.set_xlabel("rank $k$")
    ax1.set_ylabel(r"$\lambda_k$")
    ax1.set_title(f"Centered one-hot covariance spectrum, $M={M}$", fontsize=10)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.axhline(bulk_median, color="C1", ls="--", lw=0.8, label=f"bulk median {bulk_median:.2e}")
    ax1.axhline(lam_1, color="C3", ls=":", lw=0.8, label=f"$\\lambda_1={lam_1:.2e}$")
    ax1.legend(fontsize=7, loc="upper right")

    n_zoom = min(30, len(meaningful_evals))
    ax2.plot(np.arange(1, n_zoom + 1), meaningful_evals[:n_zoom], "o-", ms=4, lw=0.8)
    ax2.set_xlabel("rank $k$")
    ax2.set_ylabel(r"$\lambda_k$")
    ax2.set_title(
        f"top {n_zoom}: $\\lambda_1/\\lambda_2={ratio_12:.2f}$, $\\lambda_1/$bulk$={ratio_to_bulk:.0f}$",
        fontsize=10,
    )
    ax2.grid(True, alpha=0.3)
    ax2.axhline(bulk_median, color="C1", ls="--", lw=0.8)

    fig.tight_layout(pad=1.2, w_pad=2.0)
    Path(args.out_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_pdf, bbox_inches="tight")
    print(f"[spike] saved {args.out_pdf}")

    # --- Decision hint ---
    verdict_ok = ratio_12 >= 1.5 and ratio_to_bulk >= 5.0
    print()
    print("=" * 60)
    if verdict_ok:
        print("VERDICT: looks like a clean spike. Top-1 eigenvector usable as c_1.")
    else:
        print("VERDICT: marginal spike. Consider using top 2-3 eigenvectors jointly, or planting rank-1 signal.")
    print(f"  λ₁/λ₂          = {ratio_12:.3f}  (rule of thumb: > 1.5)")
    print(f"  λ₁/bulk_median = {ratio_to_bulk:.1f}   (rule of thumb: > 5)")
    print("=" * 60)


if __name__ == "__main__":
    main()
