# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "numpy",
#     "matplotlib==3.10.9",
# ]
# ///
"""Publication figure: BayesGAN tempered-posterior sweep on a unimodal target,
both KL directions side by side. Compact two-panel layout.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def pool(field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    here = Path(__file__).resolve().parent
    z = np.load(here / "data" / "sigma_eta_unimodal.npz")
    sigmas, etas = z["sigmas"], z["etas"]
    base = z[field]
    extra = here / "data" / "sigma_eta_unimodal_extra.npz"
    if not extra.exists():
        return sigmas, etas, base
    ze = np.load(extra)
    e_sigmas = list(ze["sigmas"])
    e_etas = list(ze["etas"])
    n_total = base.shape[-1] + ze[field].shape[-1]
    out = np.full((base.shape[0], base.shape[1], n_total), np.nan)
    out[:, :, :base.shape[-1]] = base
    for si, sig in enumerate(sigmas):
        if sig not in e_sigmas:
            continue
        sj = e_sigmas.index(sig)
        for ei, e in enumerate(etas):
            if e not in e_etas:
                continue
            ej = e_etas.index(e)
            out[si, ei, base.shape[-1]:] = ze[field][sj, ej]
    return sigmas, etas, out


def main() -> None:
    sigmas, etas, kl_qp = pool("kl_qp")
    _, _, kl_pq = pool("kl_pq")

    fig = plt.figure(figsize=(5.6, 2.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, wspace=0.05)
    ax_qp = fig.add_subplot(gs[0, 0])
    ax_pq = fig.add_subplot(gs[0, 1])

    cmap = plt.get_cmap("viridis")
    colors = [cmap(0.15 + 0.7 * i / max(1, len(sigmas) - 1))
              for i in range(len(sigmas))]

    def draw(ax, kl, ylabel, ymax=None):
        for si, sig in enumerate(sigmas):
            m = np.nanmean(kl[si], axis=-1)
            valid = np.isfinite(kl[si]).sum(axis=-1).clip(min=1)
            s = np.nanstd(kl[si], axis=-1, ddof=1) / np.sqrt(valid)
            ax.plot(etas, m, "o-", color=colors[si], lw=1.3, ms=3.5,
                    label=rf"$\sigma_p={sig:g}$")
            ax.fill_between(etas, m - s, m + s, color=colors[si],
                            alpha=0.18, lw=0)
        ax.axvline(1.0, color="0.5", ls="--", lw=0.7, zorder=0)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(ylabel, labelpad=2)
        if ymax is not None:
            ax.set_ylim(0, ymax)

    draw(ax_qp, kl_qp, r"$D_{\mathrm{KL}}(P_{\rm pp}\,\Vert\,P^{*})$",
         ymax=1.7)
    draw(ax_pq, kl_pq, r"$D_{\mathrm{KL}}(P^{*}\,\Vert\,P_{\rm pp})$",
         ymax=1.0)

    panel_label_kw = dict(fontsize=9, fontweight="bold",
                          ha="left", va="top")
    ax_qp.text(-0.30, 1.06, "A", transform=ax_qp.transAxes, **panel_label_kw)
    ax_pq.text(-0.30, 1.06, "B", transform=ax_pq.transAxes, **panel_label_kw)

    ax_qp.legend(frameon=False, loc="lower left",
                 ncol=2, columnspacing=0.7, handlelength=1.0,
                 bbox_to_anchor=(0.0, 0.0), labelspacing=0.3)

    here = Path(__file__).resolve().parent
    out_local = here / "figures" / "fig_bayesgan_unimodal.pdf"
    out_local.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_local)
    print(f"wrote {out_local}")


if __name__ == "__main__":
    main()
