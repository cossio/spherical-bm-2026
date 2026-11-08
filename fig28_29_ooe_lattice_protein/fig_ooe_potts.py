"""OOE on lattice-protein Potts BM. A) max|λ_J| vs k (weight overshoot,
N_chains=1, 5 seeds). B) Σσ²_a vs k (chain variance in data-spike dirs).
C) max|λ_J| vs N_chains at k=1. D) max_k U²[v_k(J), c_a] vs k for a=1,2,3.
"""

# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "numpy>=2.4",
#     "h5py>=3.12",
#     "matplotlib==3.10.9",
# ]
# ///

from __future__ import annotations

import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent / "data"
L = 27


def collect_seeds(k_vals: list[int]) -> dict:
    """Collect all seeds for each k from phase5 + phase5b + phase6."""
    data = {k: {"lam": [], "sig2_sum": [], "u_max_sq": []} for k in k_vals}

    def _add(fp, k):
        with h5py.File(fp, "r") as h:
            data[k]["lam"].append(float(np.abs(h["lam_eig"][-1]).max()))
            sig2 = h["sigma2_a"][:]
            n_last = max(1, len(sig2) // 5)
            data[k]["sig2_sum"].append(float(sig2[-n_last:].sum(axis=1).mean()))
            U2 = h["U_ka"][:] ** 2                            # (T, n_eig, n_top)
            U2_max = U2.max(axis=1)                           # (T, n_top)
            data[k]["u_max_sq"].append(U2_max[-n_last:].mean(axis=0).tolist())

    seen = set()
    # seed 0 from phase5
    for k in k_vals:
        fp = HERE / f"phase5_single_chain/k{k}.h5"
        if fp.exists():
            _add(str(fp), k)
            seen.add((k, 0))

    # phase5b
    for fp in sorted((HERE / "phase5b_seeds").glob("*.h5")):
        with h5py.File(fp, "r") as h:
            k, seed = int(h.attrs["k"]), int(h.attrs["seed"])
        if (k, seed) in seen:
            continue
        _add(str(fp), k)
        seen.add((k, seed))

    # phase6
    for fp in sorted((HERE / "phase6_seeds").glob("*.h5")):
        with h5py.File(fp, "r") as h:
            k, seed = int(h.attrs["k"]), int(h.attrs["seed"])
        if (k, seed) in seen:
            continue
        _add(str(fp), k)
        seen.add((k, seed))

    return data


def main() -> None:
    k_vals = [1, 3, 9, 27, 81, 270]
    data = collect_seeds(k_vals)

    with h5py.File(str(HERE / "spike_diagnostic.h5"), "r") as f:
        evals_data = f["top_eigenvalues"][:3]
    sig2_data = float(evals_data.sum())

    # N_chains scaling
    nc_files = sorted(
        (HERE / "phase5c_nchains").glob("nc*.h5"),
        key=lambda f: int(re.search(r"nc(\d+)", f.name).group(1)),
    )
    nc_arr, lam_nc = [], []
    for fp in nc_files:
        with h5py.File(fp, "r") as h:
            nc_arr.append(int(h.attrs["n_chains"]))
            lam_nc.append(float(np.abs(h["lam_eig"][-1]).max()))

    # Equilibrium reference
    lam_eq = float(np.mean(data[270]["lam"]))

    # Arrays for k-sweep
    k_arr = np.array(k_vals, dtype=float)
    lam_m = np.array([np.mean(data[k]["lam"]) for k in k_vals])
    lam_s = np.array([np.std(data[k]["lam"]) for k in k_vals])
    sig2_m = np.array([np.mean(data[k]["sig2_sum"]) for k in k_vals])
    sig2_s = np.array([np.std(data[k]["sig2_sum"]) for k in k_vals])

    # Per-mode max-overlap squared (a=1,2,3) for the k-sweep
    n_top = len(data[k_vals[0]]["u_max_sq"][0])
    u_max_m = np.array([
        [np.mean([row[a] for row in data[k]["u_max_sq"]]) for a in range(n_top)]
        for k in k_vals
    ])
    u_max_s = np.array([
        [np.std([row[a] for row in data[k]["u_max_sq"]]) for a in range(n_top)]
        for k in k_vals
    ])

    # --- Plot ---
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(14.5, 3.3))

    # A: λ vs k
    ax1.errorbar(k_arr / L, lam_m, yerr=lam_s, fmt="o-", color="C0", ms=5, lw=1.3, capsize=3)
    ax1.axhline(lam_eq, color="C2", ls="--", lw=1)
    ax1.text(0.97, 0.10, rf"equil. $\lambda\!={lam_eq:.1f}$",
             transform=ax1.transAxes, ha="right", va="bottom",
             fontsize=7.5, color="C2")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$k / L$  (sweeps per step)")
    ax1.set_ylabel(r"$\max_k |\lambda_k(J)|$")
    ax1.set_title("A)  Weight amplitude", fontsize=10, loc="left")
    ax1.grid(True, alpha=0.3)

    # B: Σσ² vs k
    ax2.errorbar(k_arr / L, sig2_m, yerr=sig2_s, fmt="s-", color="C3", ms=5, lw=1.3, capsize=3)
    ax2.axhline(sig2_data, color="C2", ls="--", lw=1, label=rf"data ($\Sigma\lambda_a\!={sig2_data:.2f}$)")
    ax2.axhline(0.19, color="gray", ls=":", lw=0.8, label="random")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$k / L$  (sweeps per step)")
    ax2.set_ylabel(r"$\sum_a \sigma^2_a$  (chain variance)")
    ax2.set_title("B)  Chain–data overlap", fontsize=10, loc="left")
    ax2.legend(fontsize=7.5)
    ax2.grid(True, alpha=0.3)

    # C: λ vs N_chains
    ax3.plot(nc_arr, lam_nc, "D-", color="C1", ms=5, lw=1.3)
    ax3.axhline(lam_eq, color="C2", ls="--", lw=1)
    ax3.text(0.97, 0.10, rf"equil. $\lambda\!={lam_eq:.1f}$",
             transform=ax3.transAxes, ha="right", va="bottom",
             fontsize=7.5, color="C2")
    ax3.set_xscale("log")
    ax3.set_xlabel(r"$N_{\rm chains}$")
    ax3.set_ylabel(r"$\max_k |\lambda_k(J)|$")
    ax3.set_title(r"C)  $N_{\rm chains}$ scaling ($k\!=\!1$)", fontsize=10, loc="left")
    ax3.grid(True, alpha=0.3)

    # D: per-data-mode best-overlap-squared vs k (index-invariant analog of u_1²)
    mode_colors = ["C0", "C1", "C2"]
    mode_markers = ["o", "s", "^"]
    for a in range(n_top):
        ax4.errorbar(
            k_arr / L, u_max_m[:, a], yerr=u_max_s[:, a],
            fmt=f"{mode_markers[a]}-", color=mode_colors[a], ms=5, lw=1.3, capsize=3,
            label=rf"$a={a+1}$",
        )
    ax4.set_xscale("log")
    ax4.set_xlabel(r"$k / L$  (sweeps per step)")
    ax4.set_ylabel(r"$u^2_a$")
    ax4.set_title(r"D)  Top-eigvec alignment $u^2_a$", fontsize=10, loc="left")
    ax4.set_ylim(0.0, 1.0)
    ax4.legend(fontsize=7.5, ncol=3, loc="lower right")
    ax4.grid(True, alpha=0.3)

    fig.suptitle(
        "Out-of-equilibrium training: Potts BM on lattice proteins"
        r" (1 chain, $\gamma{=}0.01$, lr${=}0.01$, 5 seeds)",
        fontsize=9.5, y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90], w_pad=2.5)

    out = Path(__file__).resolve().parent / "fig_ooe_potts.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")

    # Print table
    print(f"\n{'k':>5} {'k/L':>6} {'n':>3} {'λ':>8}±{'':>5} {'Σσ²':>8}±{'':>5}  "
          f"{'u_max1²':>7}  {'u_max2²':>7}  {'u_max3²':>7}")
    for ki, k in enumerate(k_vals):
        n = len(data[k]["lam"])
        print(f"{k:>5} {k/L:>6.2f} {n:>3} {np.mean(data[k]['lam']):>8.2f}±{np.std(data[k]['lam']):>5.2f}"
              f" {np.mean(data[k]['sig2_sum']):>8.3f}±{np.std(data[k]['sig2_sum']):>5.3f}  "
              f"{u_max_m[ki,0]:>5.3f}  {u_max_m[ki,1]:>5.3f}  {u_max_m[ki,2]:>5.3f}")


if __name__ == "__main__":
    main()
