# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib==3.10.8"]
# ///
"""Publication figure v2 — uses validation runs.

Differences from v1:
- Panel A Pearson curves merge the original coarse β grid with the refined
  grid (γ ≥ 0.2) for smoother curves and cleaner β_opt picks.
- Panel C: β_opt is taken from the merged (original + refined) grid.
  Error bars on β_opt show the range spanned by seed=0 and seed=1 scans
  (Pearson SE is tiny, ≈0.005, so these are typically within one grid step).

Usage: uv run fig_publication_temperature_tuning.py [PF00072|PF00018]
Output: figures/fig_temperature_tuning_{FAMILY}.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)

FAMILY = sys.argv[1] if len(sys.argv) > 1 else "PF00072"
PT_DIR_NAME = {"PF00072": "PF00072_pt_merged", "PF00018": "PF00018_pt"}[FAMILY]
OUT_PDF = OUT_DIR / f"fig_temperature_tuning_{FAMILY}.pdf"
RESULTS = ROOT / "data"
GAMMA_PICK = 1.0
K_MODES = 6


def load_data():
    ev = np.load(RESULTS / FAMILY / f"{FAMILY}_all_eigvals.npz")
    c_data = np.load(RESULTS / FAMILY / f"{FAMILY}_cov_eigvals_data.npy")

    ts = np.load(RESULTS / PT_DIR_NAME / f"{FAMILY}_temperature_scan.npz")
    cov = np.load(RESULTS / PT_DIR_NAME / f"{FAMILY}_cov_eigvals_vs_beta.npz")
    tsr = np.load(RESULTS / PT_DIR_NAME /
                  f"{FAMILY}_temperature_scan_refined_full.npz")
    covr = np.load(RESULTS / PT_DIR_NAME /
                   f"{FAMILY}_cov_eigvals_vs_beta_refined_full.npz")
    ts1 = np.load(RESULTS / PT_DIR_NAME /
                  f"{FAMILY}_temperature_scan_seed1_full.npz")

    gammas = sorted({
        float(k.replace("l2_", "").split("_")[0])
        for k in ts.files if k.endswith("_pearson")
    })
    return ev, c_data, ts, cov, tsr, covr, ts1, gammas


def merge_beta_curves(ts, tsr, prefix):
    b0 = ts[f"{prefix}_beta"]
    p0 = ts[f"{prefix}_pearson"]
    if f"{prefix}_beta" in tsr.files:
        br = tsr[f"{prefix}_beta"]
        pr = tsr[f"{prefix}_pearson"]
        b = np.concatenate([b0, br])
        p = np.concatenate([p0, pr])
    else:
        b, p = b0, p0
    order = np.argsort(b)
    return b[order], p[order]


def merge_cov(cov, covr, prefix):
    """Return (betas_sorted, array of eigvals per beta)."""
    keys = {k for k in cov.files if k.startswith(f"{prefix}_beta_")}
    d = {float(k.split("_")[-1]): cov[k] for k in keys}
    for k in covr.files:
        if k.startswith(f"{prefix}_beta_"):
            d[float(k.split("_")[-1])] = covr[k]
    betas = sorted(d)
    return betas, [d[b] for b in betas]


def beta_star_sbm(gamma, lam, c):
    n = min(len(lam), len(c))
    lk, ck = lam[:n], c[:n]
    mu = np.sum(lk * ck**2) / np.sum(ck**2)
    num = np.sum(lk**2 * ck**2)
    den = np.sum(lk * (lk - mu) * ck**4)
    return 1.0 + gamma * num / den if den > 0 else np.nan


def panel_pearson(ax, ts, tsr, gammas):
    cmap = mpl.colormaps["viridis_r"]
    n = len(gammas)
    opt_b, opt_p = [], []
    for i, g in enumerate(gammas):
        prefix = f"l2_{g:.4f}"
        b, p = merge_beta_curves(ts, tsr, prefix)
        mask = b <= 2.1
        ax.plot(b[mask], p[mask],
                marker="o", markersize=2.4, linewidth=1.0,
                color=cmap(i / max(n - 1, 1)))
        j = int(np.argmax(p))
        opt_b.append(float(b[j]))
        opt_p.append(float(p[j]))

    ax.scatter(opt_b, opt_p,
               s=16, facecolor="firebrick", edgecolor="black",
               linewidth=0.4, zorder=5,
               label=r"$\beta_{\mathrm{opt}}$")
    ax.set_xlabel(r"sampling inverse temperature $\beta$")
    ax.set_ylabel("Pearson correlation", labelpad=4)
    ax.set_xlim(0.45, 2.05)
    ax.set_ylim(0, 1)
    sm = mpl.cm.ScalarMappable(
        cmap=cmap,
        norm=mpl.colors.Normalize(vmin=gammas[0], vmax=gammas[-1]),
    )
    cbar = ax.figure.colorbar(sm, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label(r"regularization $\gamma$", fontsize=7, labelpad=2)
    cbar.ax.tick_params(labelsize=6.5, length=1.8)


def panel_a(ax, cov, covr, ts, tsr):
    prefix = f"l2_{GAMMA_PICK:.4f}"
    betas, evs = merge_cov(cov, covr, prefix)
    sig = np.array([s[:K_MODES] for s in evs]).T
    bulk_sum = np.array([float(s[K_MODES:].sum()) for s in evs])
    b_scan, p_scan = merge_beta_curves(ts, tsr, prefix)
    beta_opt = float(b_scan[np.argmax(p_scan)])

    cmap = mpl.colormaps["plasma"]
    for k in range(K_MODES):
        ax.plot(betas, sig[k],
                marker="o", markersize=3.0, linewidth=1.1,
                color=cmap(k / (K_MODES - 1)),
                label=rf"$k={k+1}$")

    ymax_d = float(sig.max())
    ymin_d = float(sig.min())
    ypad_top = 0.35 * (ymax_d - ymin_d)
    ax.set_ylim(ymin_d - 0.05 * (ymax_d - ymin_d), ymax_d + ypad_top)
    y_lo, y_hi = ax.get_ylim()
    y_pad = 0.04 * (y_hi - y_lo)
    if abs(beta_opt - round(beta_opt * 10) / 10) > 1e-6 or \
       round(beta_opt * 10) / 10 not in {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}:
        ax.plot([beta_opt, beta_opt], [y_lo + y_pad, y_hi - y_pad],
                color="firebrick", linestyle="--", linewidth=1.0)
    else:
        dx = 0.005 * (3.05 - 0.4)
        ax.plot([beta_opt + dx, beta_opt + dx],
                [y_lo + y_pad, y_hi - y_pad],
                color="firebrick", linestyle="--", linewidth=1.0)
    ax.text(beta_opt + 0.06, ymin_d + 0.04 * (ymax_d - ymin_d),
            rf"$\beta_{{\mathrm{{opt}}}}\!=\!{beta_opt:.2f}$",
            color="firebrick", fontsize=7, ha="left", va="bottom")

    ax.set_xlabel(r"sampling inverse temperature $\beta$")
    ax.set_ylabel(r"$\sigma_k^2(\beta)$", labelpad=4)
    ax.set_xlim(0.4, 3.05)

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.6)
    ax2.tick_params(axis="y", direction="in", length=2.5, width=0.6,
                    labelsize=8, pad=4.5, colors="#2a7f2a")
    (bulk_line,) = ax2.plot(betas, bulk_sum,
                            marker="s", markersize=2.6, linewidth=0.9,
                            color="#2a7f2a", linestyle="--",
                            label=rf"$\sum_{{k>{K_MODES}}} \sigma_k^2$")
    ax2.set_ylabel(rf"$\sum_{{k>{K_MODES}}} \sigma_k^2(\beta)$",
                   labelpad=4, color="#2a7f2a")

    h, l = ax.get_legend_handles_labels()
    h.append(bulk_line); l.append(bulk_line.get_label())
    ax.legend(h, l, fontsize=6.5, ncol=4, frameon=False,
              loc="lower center", bbox_to_anchor=(0.52, 1.02),
              handlelength=1.2, columnspacing=0.8, borderaxespad=0.0)


def panel_b(ax, ts, tsr, ts1, ev, c_data, gammas):
    # Pearson SE estimated from seed-0 vs seed-1 RMS on common β grid.
    # Each scan's SE of Pearson is about |diff|/sqrt(2).
    diffs = []
    for g in gammas:
        prefix = f"l2_{g:.4f}"
        if f"{prefix}_pearson" not in ts1.files: continue
        p0 = ts[f"{prefix}_pearson"]; p1 = ts1[f"{prefix}_pearson"]
        diffs.append(p0 - p1)
    pearson_se = float(np.sqrt(np.mean(np.concatenate(diffs) ** 2) / 2))

    beta_emp, err_lo, err_hi, beta_sbm, gs = [], [], [], [], []
    for g in gammas:
        prefix = f"l2_{g:.4f}"
        if prefix not in ev.files:
            continue
        # Central value: β_opt from merged (original + refined) grid.
        b, p = merge_beta_curves(ts, tsr, prefix)
        j = int(np.argmax(p))
        b_opt = float(b[j])
        p_max = float(p[j])

        # Asymmetric uncertainty: extent of β where P(β) ≥ P_max − pearson_se,
        # interpolated linearly between grid points around the peak.
        tol = p_max - pearson_se
        def interp_cross(b_arr, p_arr, j_center, direction):
            """Walk outward from j_center until p drops below tol, then
            linearly interpolate the crossing β. direction = +1 or −1."""
            if direction > 0:
                idxs = range(j_center, len(b_arr) - 1)
            else:
                idxs = range(j_center, 0, -1)
            for i in idxs:
                i_next = i + direction
                if p_arr[i_next] < tol:
                    # linear interpolation
                    frac = (p_arr[i] - tol) / (p_arr[i] - p_arr[i_next])
                    return float(b_arr[i] + frac * (b_arr[i_next] - b_arr[i]))
            return float(b_arr[-1 if direction > 0 else 0])
        b_hi = interp_cross(b, p, j, +1)
        b_lo = interp_cross(b, p, j, -1)

        beta_emp.append(b_opt)
        err_hi.append(b_hi - b_opt)
        err_lo.append(b_opt - b_lo)
        beta_sbm.append(beta_star_sbm(g, ev[prefix], c_data))
        gs.append(g)

    beta_emp = np.array(beta_emp)
    err_lo = np.array(err_lo); err_hi = np.array(err_hi)
    ax.errorbar(gs, beta_emp, yerr=[err_lo, err_hi],
                fmt="o-", markersize=4, linewidth=1.3,
                color="black", ecolor="gray", elinewidth=0.8, capsize=2.0,
                label=r"empirical $\beta_{\mathrm{opt}}$")
    ax.plot(gs, beta_sbm, marker="^", markersize=4, linewidth=1.3,
            linestyle="--", color="#1f77b4",
            label=r"SBM prediction $\beta^\star$")

    ax.set_xlabel(r"regularization $\gamma$")
    ax.set_ylabel(r"optimal sampling $\beta$", labelpad=6)
    ax.tick_params(axis="y", pad=7)
    ax.set_xlim(-0.04, max(gs) * 1.05)
    ymax = max(max(beta_emp), max(beta_sbm))
    ax.set_ylim(0.93, ymax + 0.12)
    ax.legend(fontsize=7, frameon=False, loc="upper left",
              handlelength=1.8, borderaxespad=0.3)


def main():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.pad": 5.5,
        "ytick.major.pad": 4.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 7,
    })

    ev, c_data, ts, cov, tsr, covr, ts1, gammas = load_data()

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.0))
    panel_pearson(axes[0], ts, tsr, gammas)
    panel_a(axes[1], cov, covr, ts, tsr)
    panel_b(axes[2], ts, tsr, ts1, ev, c_data, gammas)

    for lbl, ax in zip("ABC", axes):
        ax.text(-0.20, 1.05, lbl, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom", ha="left")

    fig.subplots_adjust(left=0.07, right=0.98, top=0.85,
                        bottom=0.19, wspace=0.50)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PDF.with_suffix(".png"), dpi=200)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
