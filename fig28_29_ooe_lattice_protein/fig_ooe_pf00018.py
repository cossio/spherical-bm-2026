"""PF00018 OOE sweep analysis (N_chains=1 main + N_chains=256 baseline).
Produces fig_ooe_pf00018_trajectories.pdf (per-k σ²_a(t), Σ‖P_J c_a‖²,
max|u_k·c_a|, |λ_k(J)|) and fig_ooe_pf00018.pdf (σ²_a vs k, ‖P_J c‖² vs k,
max|λ_J(t_end)| vs k, with the 256-chain baseline overlaid).
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

import argparse
import glob as globmod
import re
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
K_RE_N1 = re.compile(r"k(\d+)_s(\d+)_t(\d+)\.h5$")
K_RE_N256 = re.compile(r"k(\d+)_t(\d+)\.h5$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n1-glob",  default=str(DATA / "pf00018" / "n1"  / "k*_s*_t*.h5"))
    p.add_argument("--n256-glob", default=str(DATA / "pf00018" / "n256" / "k*_t*.h5"))
    p.add_argument("--spike",
                   default=str(DATA / "spike_diagnostic_pf00018.h5"))
    p.add_argument("--out-traj",
                   default=str(HERE / "fig_ooe_pf00018_trajectories.pdf"))
    p.add_argument("--out-bifurcation",
                   default=str(HERE / "fig_ooe_pf00018.pdf"))
    return p.parse_args()


def load_run(path: str) -> dict:
    with h5py.File(path, "r") as f:
        return dict(
            path=path,
            k=int(f.attrs["k"]),
            gamma=float(f.attrs["gamma"]),
            lr=float(f.attrs["lr"]),
            seed=int(f.attrs["seed"]),
            n_chains=int(f.attrs["n_chains"]),
            t_age=int(f.attrs["t_age"]),
            t=f["t"][:],
            sigma2=f["sigma2_a"][:],                 # (T, K)
            U=f["U_ka"][:],                          # (T, K_eig, K)
            lam=f["lam_eig"][:],                     # (T, K_eig)
            u_sub_sq=f["u_subspace_sq"][:],
            u_max=f["u_max_per_a"][:],
            wall=float(f.attrs["wall_time_s"]),
        )


def group_by_k(runs: list[dict]) -> dict[int, list[dict]]:
    g: dict[int, list[dict]] = defaultdict(list)
    for r in runs:
        g[r["k"]].append(r)
    return dict(sorted(g.items()))


def time_avg_last_half(y: np.ndarray) -> np.ndarray:
    """y: (T, ...) → average over last 50% of time."""
    T = y.shape[0]
    return y[T // 2:].mean(axis=0)


def main() -> None:
    args = parse_args()
    with h5py.File(args.spike, "r") as f:
        lam_data = f["top_eigenvalues"][:3]

    n1_runs = [load_run(p) for p in sorted(globmod.glob(args.n1_glob))]
    n256_runs = [load_run(p) for p in sorted(globmod.glob(args.n256_glob))]
    print(f"Loaded {len(n1_runs)} N=1 runs, {len(n256_runs)} N=256 runs")
    if not n1_runs:
        raise SystemExit("No N_chains=1 runs yet")

    grp_n1 = group_by_k(n1_runs)
    grp_n256 = group_by_k(n256_runs)
    ks_n1 = sorted(grp_n1.keys())
    K = n1_runs[0]["sigma2"].shape[1]

    cmap = plt.cm.viridis(np.linspace(0.0, 0.88, len(ks_n1)))

    # --- Trajectory figure: mean over seeds (solid) + individual seeds (faint) ---
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4))
    for k_val, c in zip(ks_n1, cmap):
        runs = grp_n1[k_val]
        t = runs[0]["t"]
        sigma_all = np.stack([r["sigma2"] for r in runs])             # (S, T, K)
        sigma_mean = sigma_all.mean(axis=0)
        u_sub_all = np.stack([r["u_sub_sq"].sum(axis=1) for r in runs])
        u_max_all = np.stack([r["u_max"].mean(axis=1) for r in runs])
        lam_max_all = np.stack([np.abs(r["lam"]).max(axis=1) for r in runs])
        label = f"k={k_val}"
        for a in range(K):
            for r in runs:
                axes[0, a].plot(r["t"], r["sigma2"][:, a], color=c, lw=0.45, alpha=0.35)
            axes[0, a].plot(t, sigma_mean[:, a], color=c, lw=1.5, label=label)
            axes[0, a].axhline(lam_data[a], color="grey", ls="--", lw=0.6)
            axes[0, a].set_title(
                rf"$\sigma^2_{{{a+1}}}(t)$  (data $\lambda_{{{a+1}}}={lam_data[a]:.2f}$)",
                fontsize=10,
            )
            axes[0, a].set_xlabel("gradient step")
            axes[0, a].grid(True, alpha=0.3)
        for arr, ax in zip([u_sub_all, u_max_all, lam_max_all], axes[1]):
            for y in arr:
                ax.plot(t, y, color=c, lw=0.45, alpha=0.35)
            ax.plot(t, arr.mean(axis=0), color=c, lw=1.5, label=label)
    axes[0, 0].legend(fontsize=6, ncol=2, loc="best")
    axes[1, 0].set_title(r"$J\to$data subspace overlap $\sum_a\|P_J c_a\|^2$", fontsize=10)
    axes[1, 0].axhline(K, color="grey", ls="--", lw=0.6)
    axes[1, 1].set_title(r"$\overline{\max_k |v_k(J)\cdot c_a|}$", fontsize=10)
    axes[1, 2].set_title(r"top $|\lambda|$ of $J$", fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("gradient step")
        ax.grid(True, alpha=0.3)

    r0 = n1_runs[0]
    fig.suptitle(
        f"PF00018 OOE sweep | N=1, 3 seeds | gamma={r0['gamma']}, "
        f"lr={r0['lr']}, t_age={r0['t_age']}",
        fontsize=11,
    )
    fig.tight_layout(pad=1.1)
    Path(args.out_traj).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_traj, bbox_inches="tight")
    print(f"Saved {args.out_traj}")

    # --- Bifurcation figure ---
    def summarize(group: dict[int, list[dict]]):
        ks = np.array(sorted(group.keys()))
        sigma2 = np.zeros((len(ks), K)); sigma2_std = np.zeros_like(sigma2)
        usub = np.zeros(len(ks));       usub_std = np.zeros_like(usub)
        lammax = np.zeros(len(ks));     lammax_std = np.zeros_like(lammax)
        umax_sq = np.zeros((len(ks), K)); umax_sq_std = np.zeros_like(umax_sq)
        for i, k_val in enumerate(ks):
            rs = group[k_val]
            s_rs = np.stack([time_avg_last_half(r["sigma2"]) for r in rs])
            u_rs = np.stack([time_avg_last_half(r["u_sub_sq"].sum(axis=1)) for r in rs])
            l_rs = np.stack([time_avg_last_half(np.abs(r["lam"]).max(axis=1)) for r in rs])
            um_rs = np.stack([
                time_avg_last_half((r["U"] ** 2).max(axis=1)) for r in rs
            ])
            sigma2[i] = s_rs.mean(axis=0); sigma2_std[i] = s_rs.std(axis=0)
            usub[i] = u_rs.mean();         usub_std[i] = u_rs.std()
            lammax[i] = l_rs.mean();       lammax_std[i] = l_rs.std()
            umax_sq[i] = um_rs.mean(axis=0); umax_sq_std[i] = um_rs.std(axis=0)
        return ks, sigma2, sigma2_std, usub, usub_std, lammax, lammax_std, umax_sq, umax_sq_std

    ks1, s2_1, s2_1std, usub_1, usub_1std, lam_1, lam_1std, um_1, um_1std = summarize(grp_n1)
    have_256 = bool(grp_n256)
    if have_256:
        ks_b, s2_b, s2_bstd, usub_b, usub_bstd, lam_b, lam_bstd, um_b, um_bstd = summarize(grp_n256)

    fig2, axes2 = plt.subplots(1, 5, figsize=(17.5, 3.7))

    markers = ["o", "s", "^"]
    colors_a = ["C0", "C1", "C2"]

    # Panel 0: σ²_a (absolute) — primary user ask
    for a in range(K):
        axes2[0].errorbar(ks1, s2_1[:, a], yerr=s2_1std[:, a],
                          marker=markers[a], color=colors_a[a], lw=1.3, capsize=2,
                          label=rf"$a={a+1}$ ($\lambda_{{{a+1}}}={lam_data[a]:.2f}$)")
        axes2[0].axhline(lam_data[a], color=colors_a[a], ls=":", lw=0.7, alpha=0.6)
    if have_256:
        for a in range(K):
            axes2[0].plot(ks_b, s2_b[:, a], color=colors_a[a], marker=markers[a],
                          mfc="none", ls="--", lw=0.9, alpha=0.6)
    axes2[0].set_xscale("log")
    axes2[0].set_xlabel(r"$k$")
    axes2[0].set_ylabel(r"$\langle\sigma^2_a\rangle$")
    axes2[0].set_title(r"chain overlap  $\sigma^2_a = \langle(x{-}f_1)\cdot c_a\rangle^2$",
                       fontsize=9)
    axes2[0].grid(True, which="both", alpha=0.3)
    axes2[0].legend(fontsize=7, loc="lower right", framealpha=0.9)

    # Panel 1: σ²_a / λ_data_a — dimensionless; equilibrium = 1
    for a in range(K):
        axes2[1].errorbar(ks1, s2_1[:, a] / lam_data[a],
                          yerr=s2_1std[:, a] / lam_data[a],
                          marker=markers[a], color=colors_a[a], lw=1.3, capsize=2,
                          label=rf"$a={a+1}$")
    if have_256:
        for a in range(K):
            axes2[1].plot(ks_b, s2_b[:, a] / lam_data[a],
                          color=colors_a[a], marker=markers[a],
                          mfc="none", ls="--", lw=0.9, alpha=0.6)
    axes2[1].axhline(1.0, color="k", ls="-", lw=0.7, alpha=0.5)
    axes2[1].set_xscale("log")
    axes2[1].set_xlabel(r"$k$")
    axes2[1].set_ylabel(r"$\sigma^2_a / \lambda_a^{\rm data}$")
    axes2[1].set_title(r"normalized chain overlap (=1: equilibrated)", fontsize=9)
    axes2[1].grid(True, which="both", alpha=0.3)
    axes2[1].legend(fontsize=7, loc="lower right", framealpha=0.9)

    # Panel 2: J subspace overlap — k-independent
    axes2[2].errorbar(ks1, usub_1, yerr=usub_1std, marker="o", color="C4", lw=1.3,
                      capsize=2, label=r"N=1 (main)")
    if have_256:
        axes2[2].plot(ks_b, usub_b, color="C4", marker="o", mfc="none", ls="--",
                      lw=0.9, alpha=0.7, label=r"N=256")
    axes2[2].axhline(K, color="grey", ls="-", lw=0.6, alpha=0.5)
    axes2[2].set_xscale("log")
    axes2[2].set_xlabel(r"$k$")
    axes2[2].set_ylabel(r"$\sum_a\|P_J c_a\|^2$")
    axes2[2].set_title(r"J subspace overlap (representation)", fontsize=9)
    axes2[2].set_ylim(0, K + 0.3)
    axes2[2].grid(True, which="both", alpha=0.3)
    axes2[2].legend(fontsize=7, loc="lower right", framealpha=0.9)

    # Panel 3: max|λ_J| — OOE overshoot diagnostic
    axes2[3].errorbar(ks1, lam_1, yerr=lam_1std, marker="o", color="C3", lw=1.3,
                      capsize=2, label=r"N=1 (main)")
    if have_256:
        axes2[3].plot(ks_b, lam_b, color="C3", marker="o", mfc="none", ls="--",
                      lw=0.9, alpha=0.7, label=r"N=256")
    axes2[3].set_xscale("log")
    axes2[3].set_xlabel(r"$k$")
    axes2[3].set_ylabel(r"$\max_k|\lambda_k(J)|$")
    axes2[3].set_title(r"coupling top-$|\lambda|$ (overshoot)", fontsize=9)
    axes2[3].grid(True, which="both", alpha=0.3)
    axes2[3].legend(fontsize=7, loc="upper right", framealpha=0.9)

    # Panel 4: per-mode best-overlap squared (in-subspace eigenvector alignment)
    for a in range(K):
        axes2[4].errorbar(ks1, um_1[:, a], yerr=um_1std[:, a],
                          marker=markers[a], color=colors_a[a], lw=1.3, capsize=2,
                          label=rf"$a={a+1}$")
    if have_256:
        for a in range(K):
            axes2[4].plot(ks_b, um_b[:, a], color=colors_a[a], marker=markers[a],
                          mfc="none", ls="--", lw=0.9, alpha=0.6)
    axes2[4].set_xscale("log")
    axes2[4].set_xlabel(r"$k$")
    axes2[4].set_ylabel(r"$u^2_a$")
    axes2[4].set_title(r"top-eigvec alignment $u^2_a$ (in-subspace)", fontsize=9)
    axes2[4].set_ylim(0.0, 1.0)
    axes2[4].grid(True, which="both", alpha=0.3)
    axes2[4].legend(fontsize=7, loc="lower right", framealpha=0.9)

    fig2.suptitle(
        f"PF00018 OOE bifurcation | "
        f"gamma={r0['gamma']}, lr={r0['lr']}, t_age={r0['t_age']} | "
        f"3 seeds at N=1; N=256 baseline dashed",
        fontsize=10,
    )
    fig2.tight_layout(pad=1.1, rect=(0, 0, 1, 0.94))

    label_kw = dict(fontsize=12, fontweight="bold", va="bottom", ha="right")
    for ax, letter in zip(axes2, "ABCDE"):
        ax.text(-0.20, 1.02, letter, transform=ax.transAxes, **label_kw)
    fig2.savefig(args.out_bifurcation, bbox_inches="tight")
    print(f"Saved {args.out_bifurcation}")

    print("\nN_chains=1 time-averaged (second half) by k:")
    print(f"{'k':>5} {'σ²_1':>10} {'σ²_2':>10} {'σ²_3':>10}  "
          f"{'Σ‖P_J c‖²':>10}  {'max|λ|':>10}  {'u²_1':>5} {'u²_2':>5} {'u²_3':>5}")
    for i, k_val in enumerate(ks1):
        print(f"{k_val:>5d} {s2_1[i,0]:>5.3f}±{s2_1std[i,0]:<4.2f} "
              f"{s2_1[i,1]:>5.3f}±{s2_1std[i,1]:<4.2f} "
              f"{s2_1[i,2]:>5.3f}±{s2_1std[i,2]:<4.2f}  "
              f"{usub_1[i]:>5.3f}±{usub_1std[i]:<4.2f}  "
              f"{lam_1[i]:>5.2f}±{lam_1std[i]:<4.1f}  "
              f"{um_1[i,0]:>5.3f} {um_1[i,1]:>5.3f} {um_1[i,2]:>5.3f}")

    if have_256:
        print("\nN_chains=256 baseline:")
        print(f"{'k':>5} {'σ²_1':>6} {'σ²_2':>6} {'σ²_3':>6}  "
              f"{'Σ‖P_J c‖²':>10}  {'max|λ|':>8}  {'u²_1':>5} {'u²_2':>5} {'u²_3':>5}")
        for i, k_val in enumerate(ks_b):
            print(f"{k_val:>5d} {s2_b[i,0]:>6.3f} {s2_b[i,1]:>6.3f} "
                  f"{s2_b[i,2]:>6.3f}  {usub_b[i]:>10.3f}  {lam_b[i]:>8.2f}  "
                  f"{um_b[i,0]:>5.3f} {um_b[i,1]:>5.3f} {um_b[i,2]:>5.3f}")


if __name__ == "__main__":
    main()
