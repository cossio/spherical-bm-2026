# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "numpy>=2.4",
#     "matplotlib==3.10.9",
#     "h5py>=3.10",
# ]
# ///
"""Paper figure: Gaussian-visible RBM on Ken French — DD peak at BBP.

Three-panel stacked plot:
  A) empirical covariance spectrum (σ-edge + leading spike)
  B) test NLL (MAP) vs γ for several M_train values
  C) σ_1/σ_2 of trained W (BBP order parameter)
"""

import argparse
import glob
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def load_run(p):
    with h5py.File(p, "r") as h:
        sv = np.asarray(h["map/W_svdvals"][()])
        return {
            "gamma": float(h["gamma"][()]),
            "mtrain": int(h["n_train"][()]),
            "seed": int(h["seed"][()]),
            "LL_map_te": float(h["map/LL_test"][()]),
            "LL_pp_te": float(h["swag/LL_test_pp"][()]),
            "sigma1": float(sv[0]),
            "sigma2": float(sv[1] if len(sv) > 1 else 1e-8),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", default="I_gauss_fixed", nargs="?")
    ap.add_argument("--outname", default="fig_rbm_finance_double_descent.pdf")
    ap.add_argument("--gamma-min", type=float, default=1e-6)
    ap.add_argument("--gamma-max", type=float, default=10.0)
    ap.add_argument("--mtrains", type=int, nargs="+",
                    default=[1000, 2000, 11428])
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    indir = root / "data" / args.tag
    runs = [load_run(p) for p in sorted(glob.glob(str(indir / "*.h5")))]
    runs = [r for r in runs
            if r["mtrain"] in args.mtrains
            and args.gamma_min <= r["gamma"] <= args.gamma_max]

    by = defaultdict(list)
    for r in runs:
        by[(r["gamma"], r["mtrain"])].append(r)

    gammas = np.array(sorted({r["gamma"] for r in runs}))
    mtrains = sorted({r["mtrain"] for r in runs})

    def stat(key):
        Mm = np.full((len(mtrains), len(gammas)), np.nan)
        Ss = np.full((len(mtrains), len(gammas)), np.nan)
        for i, m in enumerate(mtrains):
            for j, g in enumerate(gammas):
                rs = by.get((g, m), [])
                if rs:
                    vs = [r[key] for r in rs]
                    Mm[i, j] = np.mean(vs)
                    Ss[i, j] = np.std(vs)
        return Mm, Ss

    NLL_map_m, NLL_map_s = stat("LL_map_te"); NLL_map_m = -NLL_map_m
    NLL_pp_m,  _         = stat("LL_pp_te");  NLL_pp_m  = -NLL_pp_m
    sigma1_m, _ = stat("sigma1")
    sigma2_m, _ = stat("sigma2")
    ratio = sigma1_m / np.maximum(sigma2_m, 1e-8)

    # ── Plot ─────────────────────────────────────────────────────
    mpl.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    fig = plt.figure(figsize=(3.6, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(4, 1, height_ratios=[1.6, 1.0, 2.2, 1.0], hspace=0.08)
    ax_spec = fig.add_subplot(gs[0])  # Panel A: spectral density of train covariance
    ax_hi   = fig.add_subplot(gs[1])  # Panel B (top half): tail of NLL collapse (γ ≥ 1)
    ax_lo   = fig.add_subplot(gs[2], sharex=ax_hi)  # Panel B (bottom half): zoomed DD peak
    ax_bot  = fig.add_subplot(gs[3], sharex=ax_hi)  # Panel C: σ₁/σ₂ BBP diagnostic

    cmap = plt.get_cmap("viridis")

    for i, m in enumerate(mtrains):
        c = cmap(i / max(1, len(mtrains) - 1))
        for ax in (ax_hi, ax_lo):
            ax.errorbar(
                gammas, NLL_map_m[i], yerr=NLL_map_s[i],
                marker="o", color=c, ms=3, capsize=2, lw=1.2,
                label=fr"$M_{{train}}={m}$",
            )
        ax_bot.plot(gammas, sigma1_m[i], "-o", color=c, ms=3, lw=1.2)
        ax_bot.plot(gammas, sigma2_m[i], "--s", color=c, ms=2.5, lw=1.0, alpha=0.8)

    # Broken y-axis. Pick lower panel limits from the actual data (accounting
    # for error bars) so nothing gets clipped.
    NLL_map_err_lo = NLL_map_m - NLL_map_s
    NLL_map_err_hi = NLL_map_m + NLL_map_s
    mask_lo = gammas <= 1.0
    lo_min = np.nanmin(NLL_map_err_lo[:, mask_lo]) - 0.05
    lo_max = np.nanmax(NLL_map_err_hi[:, mask_lo]) + 0.05
    ax_lo.set_ylim(lo_min, lo_max)
    hi_min = max(65.0, np.nanmin(NLL_map_m[:, gammas >= 3.0]) - 0.5)
    hi_max = np.nanmax(NLL_map_m) + 0.5
    ax_hi.set_ylim(hi_min, hi_max)

    # Hide the spines between the two NLL axes and add break marks
    ax_hi.spines.bottom.set_visible(False)
    ax_lo.spines.top.set_visible(False)
    ax_hi.tick_params(bottom=False, labelbottom=False)
    ax_lo.tick_params(top=False)
    d = .015
    kwargs = dict(transform=ax_hi.transAxes, color='k', clip_on=False, lw=0.8)
    ax_hi.plot((-d, +d), (-d, +d), **kwargs)
    ax_hi.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_lo.transAxes)
    ax_lo.plot((-d, +d), (1 - d * 2.2, 1 + d * 2.2), **kwargs)
    ax_lo.plot((1 - d, 1 + d), (1 - d * 2.2, 1 + d * 2.2), **kwargs)

    # Shared y-label spanning both NLL panels — use fig.supylabel? Or
    # put it on each axis. Cleanest: set_ylabel on ax_lo only, but pad it
    # so it visually sits between the two panels. We let constrained_layout
    # handle x-position; just use a longer ylabel pad on lo.
    ax_lo.set_ylabel("test NLL  (nats / day)")
    ax_hi.set_ylabel(" ")  # placeholder so layout reserves space
    for ax in (ax_hi, ax_lo):
        ax.grid(alpha=0.3, lw=0.5)
    ax_hi.legend(ncol=2, loc="upper left", handlelength=1.4,
                 columnspacing=0.7, borderpad=0.3)

    ax_hi.tick_params(labelbottom=False)
    ax_lo.tick_params(labelbottom=False)
    ax_bot.set_xscale("log")
    ax_bot.set_xlabel(r"$\gamma$  (L2 weight decay)")
    ax_bot.set_ylabel(r"$\sigma_k(W)$")
    ax_bot.grid(alpha=0.3, which="both", lw=0.5)
    # Annotate σ₁ vs σ₂ line styles
    from matplotlib.lines import Line2D
    ax_bot.legend(
        [Line2D([], [], ls="-", color="0.3"), Line2D([], [], ls="--", color="0.3")],
        [r"$\sigma_1$", r"$\sigma_2$"],
        fontsize=8, loc="lower left", framealpha=0.9,
    )

    # ── Spectral density panel ─────────────────────────────────────
    import h5py as _h5
    data_path = root / "data" / "kenfrench49_daily.h5"  # written by simulate/01_prepare_kenfrench.py
    spectrum_path = root / "data" / "train_covariance_eigenvalues.npy"
    if spectrum_path.exists():
        eigs = np.load(spectrum_path)
    else:
        with _h5.File(data_path, "r") as h:
            Xtr = np.asarray(h["train_x"][()], dtype=np.float64)
        Ntr, Nvis = Xtr.shape
        Cmat = (Xtr.T @ Xtr) / Ntr
        eigs = np.linalg.eigvalsh(Cmat)[::-1]
    # Histogram on log scale so bulk (λ ~ 0.1–1.7) and outlier (λ_1 ≈ 25) coexist
    bins = np.logspace(np.log10(0.05), np.log10(40.0), 28)
    ax_spec.hist(eigs, bins=bins, color="0.55",
                 edgecolor="0.2", lw=0.5, zorder=2)
    # Mark the rank-1 outlier
    ax_spec.axvline(eigs[0], color="C0", lw=1.0, ls="-", zorder=4)
    ax_spec.set_xscale("log")
    ax_spec.set_xlabel(r"$\lambda(C)$  (train covariance)")
    ax_spec.set_ylabel("count")
    ax_spec.set_xlim(bins[0], bins[-1])
    ax_spec.grid(alpha=0.3, which="both", lw=0.5)
    ax_spec.legend(
        [Line2D([], [], color="0.55", lw=4),
         Line2D([], [], color="C0", lw=1.0)],
        ["bulk", rf"$\lambda_1{{\approx}}{eigs[0]:.1f}$"],
        fontsize=7, loc="upper left", framealpha=0.9,
        handlelength=1.4, borderpad=0.3,
    )

    # ── Panel labels A / B / C ─────────────────────────────────────
    label_kw = dict(fontsize=11, fontweight="bold", va="bottom", ha="right")
    ax_spec.text(-0.18, 1.02, "A", transform=ax_spec.transAxes, **label_kw)
    ax_hi.text  (-0.18, 1.02, "B", transform=ax_hi.transAxes,   **label_kw)
    ax_bot.text (-0.18, 1.02, "C", transform=ax_bot.transAxes,  **label_kw)

    out_path = indir / args.outname
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.18)
    print("wrote", out_path)
    print(f"  λ_1 = {eigs[0]:.3f}, λ_2 = {eigs[1]:.3f}")


if __name__ == "__main__":
    main()
