#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "numpy>=2.4",
#     "matplotlib==3.10.9",
# ]
# ///
"""Publication 1x2 figure: SWAG reverse-KL/N vs paper-γ across K, for the
binary FVSBN trained on (A) the 2D Ising teacher and (B) the rank-1
Curie-Weiss teacher.

Output: writes `fig_fvsbn_K_dd.{pdf,png}` next to this script.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ISING = HERE / "data" / "ising"
CW    = HERE / "data" / "cw"
PAPER_FIG = HERE

K_VALUES = [4, 8, 16, 32]
N = 16
SEEDS = 10


def load_swag(results_dir: Path, K: int, name_template: str) -> dict:
    fname = results_dir / name_template.format(K=K)
    with open(fname) as f:
        return json.load(f)


def merged_curve(results_dir: Path, K: int,
                 base_template: str, ext_template: str | None):
    """Return (gammas, mean, sem) merging the base γ ≤ 25 sweep with the
    extension γ ∈ (25, 100] sweep when the latter exists. Sorted ascending."""
    d = load_swag(results_dir, K, base_template)
    g = np.array(d["agg"]["gammas"])
    m = np.array(d["agg"]["kl_swag_mean"])
    s = np.array(d["agg"]["kl_swag_sem"])
    if ext_template is not None:
        ext_path = results_dir / ext_template.format(K=K)
        if ext_path.exists():
            with open(ext_path) as f:
                d_ext = json.load(f)
            g_ext = np.array(d_ext["agg"]["gammas"])
            keep = g_ext > g.max()
            g = np.concatenate([g, g_ext[keep]])
            m = np.concatenate([m, np.array(d_ext["agg"]["kl_swag_mean"])[keep]])
            s = np.concatenate([s, np.array(d_ext["agg"]["kl_swag_sem"])[keep]])
    order = np.argsort(g)
    return g[order], m[order], s[order]


def panel(ax, results_dir: Path, base_template: str, ext_template: str | None,
          title: str):
    cmap = plt.colormaps["viridis"]
    colors = [cmap(i / max(1, len(K_VALUES) - 1))
              for i in range(len(K_VALUES))]
    for K, c in zip(K_VALUES, colors):
        g, m, s = merged_curve(results_dir, K, base_template, ext_template)
        # Replace γ=0 with the smallest positive grid point / 10 so the
        # left edge of the log axis carries the γ=0 baseline visibly.
        g_plot = np.where(g > 0, g, g[g > 0].min() / 10.0)
        ax.plot(g_plot, m, "-", lw=1.4, color=c,
                label=fr"$K={K}$  ($K/N={K / N:.2f}$)")
        ax.fill_between(g_plot, m - s, m + s, color=c, alpha=0.18,
                        linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"SWAG $D_{\mathrm{KL}}(q_{W}\Vert P^{*})\,/\,N$")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=7.5, loc="lower right", frameon=True,
              framealpha=0.85, handlelength=1.4, borderpad=0.3)


def main():
    PAPER_FIG.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    panel(
        axes[0],
        CW,
        "Ksweep_paperwd_K{K}_N16_beta2.0_T0.3_seeds10.json",
        "Ksweep_paperwd_ext_K{K}_N16_beta2.0_T0.3_seeds10.json",
        r"$\mathbf{A.}$  Rank-1 CW, $\beta=2.0$, $N=16$",
    )
    panel(
        axes[1],
        ISING,
        "Ksweep_paperwd_K{K}_L4_beta0.5_T0.3_seeds10.json",
        "Ksweep_paperwd_ext_K{K}_L4_beta0.5_T0.3_seeds10.json",
        r"$\mathbf{B.}$  2D Ising, $\beta=0.5$, $N=16$",
    )

    # Tighter shared y-range so the two panels are directly comparable.
    ymins = [ax.get_ylim()[0] for ax in axes]
    ymaxs = [ax.get_ylim()[1] for ax in axes]
    for ax in axes:
        ax.set_ylim(min(ymins), max(ymaxs))

    fig.tight_layout(pad=0.6)

    out_pdf = PAPER_FIG / "fig_fvsbn_K_dd.pdf"
    out_png = PAPER_FIG / "fig_fvsbn_K_dd.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"saved -> {out_pdf}")
    print(f"saved -> {out_png}")


if __name__ == "__main__":
    main()
