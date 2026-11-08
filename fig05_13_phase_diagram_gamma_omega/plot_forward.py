# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "numpy==2.4.4",
#     "matplotlib==3.11.2",
# ]
# ///
"""Forward-KL temperature-optimum phase diagram in the (omega*, gamma) plane.

Implements the camera-ready closed-form forward-KL classification for the
rank-one K=2 teacher.  At each (gamma, omega*), the minimizer eta_* of the
predictive-posterior forward KL D(eta) = D_KL(P* || P_pp,eta)/N is
classified by

    omega* > 1:
        gamma <  gamma_wc         warm flat optimum (eta_0 < 1, plus a flat warm interval)
        gamma_wc < gamma <= gamma_flat    mixed warm/cold tie (warm interval AND cold point both global)
        gamma_flat < gamma < gamma_inf    unique cold optimum (eta_0 > 1)
        gamma_inf <= gamma <= c1  MAP / no finite optimum
        gamma > c1                finite plateau eta >= eta_PM

At gamma = gamma_wc, the smooth representative is eta_0 = 1.

with closed-form thresholds

    c1 = 2 - 1/omega*,  c2 = 1/omega*,  b = c1 - (c1-c2)^2 / 2
    gamma_wc   = c2^2 = 1/omega*^2
    gamma_flat = (sqrt(c1) - sqrt(c1 - b))^2
    gamma_inf  = 2 (b/c1)^2 - c2 (b/c1)
               = (3 omega* - 2)(4 omega* - 3) / [omega*^2 (2 omega* - 1)^2].

The mixed/tie sliver gamma_wc < gamma <= gamma_flat is hatched onto the
warm/cold split — every (gamma, omega*) point in it has both a warm flat
sub-interval and a cold finite point achieving the same global minimum
D_fwd^pp(eta).

Outputs:
    reproduced/fig_fwd_pp_kl_phase_diagram.pdf
    reproduced/forward_boundaries.csv
    reproduced/forward_consistency_checks.txt

Panel B uses data/forward/pp_fwd_kl_4gamma.csv, produced by
generate_forward.jl through the released TeacherStudent.forward_kl_predictive.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Reproduce the forward KL figure from saved analytical curves.")
parser.add_argument("--outdir", type=Path, default=HERE / "reproduced")
args = parser.parse_args()
FIGDIR = args.outdir
RESDIR = HERE / "data" / "forward"
FIGDIR.mkdir(parents=True, exist_ok=True)


def teacher_params(omega: float) -> tuple[float, float, float]:
    """Empirical eigenvalues c1, c2 and the auxiliary b = c1 - (c1-c2)^2/2."""
    c1 = 2.0 - 1.0 / omega
    c2 = 1.0 / omega
    b = c1 - 0.5 * (c1 - c2) ** 2
    return c1, c2, b


def gamma_wc(omega: float) -> float:
    """Warm/cold (Bayesian-crossing) boundary on the smooth branch: eta_0 = 1."""
    return (1.0 / omega) ** 2


def gamma_flat(omega: float) -> float:
    """Upper boundary of the warm flat-minimum interval (mixed/tie sliver ends)."""
    c1, _, b = teacher_params(omega)
    return (np.sqrt(c1) - np.sqrt(c1 - b)) ** 2


def gamma_inf(omega: float) -> float:
    """Cold/MAP boundary: eta_0 -> +infinity on the smooth branch."""
    c1, c2, b = teacher_params(omega)
    return 2.0 * (b / c1) ** 2 - c2 * (b / c1)


def gamma_inf_closed(omega: float) -> float:
    """Sanity-check closed form: (3 w - 2)(4 w - 3) / (w^2 (2 w - 1)^2)."""
    return (3.0 * omega - 2.0) * (4.0 * omega - 3.0) / (omega**2 * (2.0 * omega - 1.0) ** 2)


def eta0_smooth(omega: float, gamma: float) -> float:
    """Smooth-branch representative eta_0(gamma, omega*).  Returns +inf past gamma_inf."""
    c1, c2, b = teacher_params(omega)
    g = (c2 + np.sqrt(c2 * c2 + 8.0 * gamma)) / 4.0
    den = b - c1 * g
    if den <= 0.0:
        return float("inf")
    return g * (1.0 - g) / den


# ------------------------------ build curves -------------------------------- #

OMEGA_MIN, OMEGA_MAX = 1.0, 3.0
GAMMA_MIN, GAMMA_MAX = 0.0, 1.5

omega_grid = np.linspace(OMEGA_MIN + 1e-6, OMEGA_MAX, 2001)
g_wc_curve = np.array([gamma_wc(w) for w in omega_grid])
g_flat_curve = np.array([gamma_flat(w) for w in omega_grid])
g_inf_curve = np.array([gamma_inf(w) for w in omega_grid])
g_deg_curve = 2.0 - 1.0 / omega_grid

np.savetxt(
    FIGDIR / "forward_boundaries.csv",
    np.column_stack([omega_grid, g_wc_curve, g_flat_curve, g_inf_curve]),
    delimiter=",",
    header="omega_star,gamma_wc,gamma_flat,gamma_inf",
    comments="",
)


# ------------------------------- plot --------------------------------------- #

C_WARM = "#F2D88E"     # warm yellow
C_COLD = "#7AABCC"     # medium blue (cold posterior)
C_MAP = "#CFE2EE"      # light blue (MAP)
C_TIE_HATCH = "#1A3F6B"   # hatching color over the warm/cold tie sliver
C_BAYES = "#0E8A8A"    # teal (eta_*=1 line)
C_DEG = "#C8C8C8"      # finite-temperature plateau, extending to MAP

fig, (ax, ax_b) = plt.subplots(
    1, 2,
    figsize=(4.6, 1.95),
    gridspec_kw={"width_ratios": [1.0, 1.0]},
    constrained_layout=True,
)
fig.set_constrained_layout_pads(h_pad=0.01, hspace=0.0,
                                w_pad=0.01, wspace=0.04)

# Three coarse regions split by the smooth-branch thresholds gamma_wc, gamma_inf.
ax.fill_between(
    omega_grid,
    np.full_like(omega_grid, GAMMA_MIN),
    g_wc_curve,
    color=C_WARM, alpha=0.55, linewidth=0,
)
ax.fill_between(
    omega_grid,
    g_wc_curve,
    np.minimum(g_inf_curve, GAMMA_MAX),
    color=C_COLD, alpha=0.55, linewidth=0,
)
ax.fill_between(
    omega_grid,
    np.minimum(g_inf_curve, GAMMA_MAX),
    np.minimum(g_deg_curve, GAMMA_MAX),
    color=C_MAP, alpha=0.55, linewidth=0,
)
ax.fill_between(
    omega_grid,
    np.minimum(g_deg_curve, GAMMA_MAX),
    np.full_like(omega_grid, GAMMA_MAX),
    color=C_DEG, alpha=0.7, linewidth=0,
)
# Mixed warm/cold tie sliver gamma_wc < gamma <= gamma_flat: hatched onto cold.
ax.fill_between(
    omega_grid,
    g_wc_curve,
    g_flat_curve,
    facecolor="none",
    edgecolor=C_TIE_HATCH,
    hatch="////",
    linewidth=0.0,
    alpha=0.7,
)

# Phase boundary lines.  warm/cold (eta_*=1) is the Bayes line, dashed teal;
# the gamma_flat upper edge of the tie sliver and the cold/MAP gamma_inf are
# both solid black.
LINE_KW = dict(color="black", lw=1.4, ls="-")
ax.plot(omega_grid, g_wc_curve, color=C_BAYES, lw=1.6, ls="--")
ax.plot(omega_grid, g_flat_curve, color="black", lw=1.0, ls=(0, (1, 1)))
ax.plot(omega_grid, g_inf_curve, **LINE_KW)
ax.plot(omega_grid, g_deg_curve, **LINE_KW)

# Region labels
ax.text(2.55, 0.07, "warm",
        ha="center", va="center", fontsize=7, color="#7A4D00")
ax.text(1.85, 0.45, "cold",
        ha="center", va="center", fontsize=7, color="#1A3F6B")
ax.text(2.50, 1.10, "MAP",
        ha="center", va="center", fontsize=8.5, color="#1A3F6B")
ax.text(1.27, 1.37, "deg.",
        ha="center", va="center", fontsize=8, color="#444444")
# Markers for the (omega*, gamma) points whose KL(eta) is shown in panel B.
C_MIX_DOT = "#1A3F6B"
C_MAP_DOT = "#9D7DB8"
ax.scatter([2.2], [0.10], color="#7A4D00", s=14, zorder=5, edgecolor="white", linewidth=0.4)
ax.scatter([2.2], [0.30], color="#1A3F6B", s=14, zorder=5, edgecolor="white", linewidth=0.4)
ax.scatter([2.2], [0.80], color=C_MAP_DOT, s=14, zorder=5, edgecolor="white", linewidth=0.4)

ax.set_xlim(OMEGA_MIN, OMEGA_MAX)
ax.set_ylim(GAMMA_MIN, GAMMA_MAX)
ax.set_xlabel(r"$\omega^*$", fontsize=8, labelpad=1)
ax.set_ylabel(r"$\gamma$", fontsize=8, labelpad=1, rotation=0)
ax.set_xticks([1.0, 1.5, 2.0, 2.5, 3.0])
ax.set_yticks([0.5, 1.0, 1.5])
ax.tick_params(labelsize=6, pad=1.5, length=2)

# ----- panel B: predictive-posterior forward KL vs eta at omega*=2.2 ----- #
ppkl_csv = RESDIR / "pp_fwd_kl_4gamma.csv"
if ppkl_csv.exists():
    data = np.genfromtxt(ppkl_csv, delimiter=",", names=True)
    eta = data["eta"]
    kl_warm  = data["kl_gamma_010"]
    kl_mix   = data["kl_gamma_215"]
    kl_cold  = data["kl_gamma_030"]
    kl_map   = data["kl_gamma_080"]

    ax_b.plot(eta, kl_warm, color="#7A4D00", lw=1.4, label=r"$\gamma=0.10$")
    ax_b.plot(eta, kl_cold, color="#1A3F6B", lw=1.4, label=r"$\gamma=0.30$")
    ax_b.plot(eta, kl_map,  color=C_MAP_DOT, lw=1.4, label=r"$\gamma=0.80$")

    # Mark the unique cold minimum at gamma=0.30 and the warm-interval level
    # at gamma=0.10 (the warm curve is flat across the whole I_warm; we mark
    # its smooth-branch representative eta_0 < 1).
    i_c = int(np.nanargmin(kl_cold))
    ax_b.scatter([eta[i_c]], [kl_cold[i_c]], color="#1A3F6B", s=14, zorder=5)
    eta0_warm = eta0_smooth(2.2, 0.10)
    kl_warm_level = float(np.nanmin(kl_warm))
    ax_b.scatter([eta0_warm], [kl_warm_level], color="#7A4D00", s=14, zorder=5)

    ax_b.axvline(1.0, color=C_BAYES, lw=1.4, ls="--")
    ax_b.text(0.98, 0.218, "Bayes", color=C_BAYES, fontsize=6,
              ha="right", va="top")

    ax_b.set_xlim(0.0, 3.0)
    ax_b.set_ylim(0.14, 0.22)
    ax_b.set_xlabel(r"$\eta$", fontsize=8, labelpad=1)
    ax_b.set_ylabel(r"$D_{\mathrm{KL}}(P^*\Vert P_{\mathrm{pp}})/N$",
                    fontsize=7, labelpad=1)
    ax_b.set_xticks([0, 1, 2, 3])
    ax_b.set_yticks([0.15, 0.20])
    ax_b.tick_params(labelsize=6, pad=1.5, length=2)
    # Legend in upper-right; nudge inwards so labels do not graze the curves.
    ax_b.legend(loc="upper right", fontsize=5.5, frameon=False,
                labelspacing=0.25, handlelength=1.3, handletextpad=0.3,
                borderpad=0.0, borderaxespad=0.0,
                bbox_to_anchor=(1.0, 1.02))
else:
    ax_b.set_visible(False)

LBL_KW = dict(fontsize=10, fontweight="bold", ha="right", va="top")
ax.text(-0.18, 1.06, "A)", transform=ax.transAxes, **LBL_KW)
if ppkl_csv.exists():
    ax_b.text(-0.18, 1.06, "B)", transform=ax_b.transAxes, **LBL_KW)

out = FIGDIR / "fig_fwd_pp_kl_phase_diagram.pdf"
fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print(f"wrote {out}")


# ---------------------------- consistency checks ---------------------------- #

checks: list[str] = []

# omega*=2.5 worked example from the note (Section 6).
w = 2.5
c1, c2, b = teacher_params(w)
checks.append(
    f"omega*={w}: c1={c1}, c2={c2}, b={b} "
    f"(expect 1.6, 0.4, 0.88)"
)
checks.append(
    f"omega*={w}: gamma_wc={gamma_wc(w):.6f} (expect 0.16), "
    f"gamma_flat={gamma_flat(w):.10f} (expect 0.1733747416), "
    f"gamma_inf={gamma_inf(w):.6f} (expect 0.385)"
)
# Cross-check the alternate closed form for gamma_inf.
checks.append(
    f"omega*={w}: gamma_inf_smooth={gamma_inf(w):.10f}, "
    f"gamma_inf_closed={gamma_inf_closed(w):.10f}"
)
# Note's explicit eta_* = 3.125 at omega*=2.5, gamma=0.3.
checks.append(
    f"omega*={w}, gamma=0.3: eta_0={eta0_smooth(w, 0.3):.6f} (expect 3.125)"
)

# Slice used in panel B.
w = 2.2
c1, c2, b = teacher_params(w)
checks.append(
    f"omega*={w}: c1={c1:.4f}, c2={c2:.4f}, b={b:.4f}, "
    f"gamma_wc={gamma_wc(w):.6f}, gamma_flat={gamma_flat(w):.6f}, "
    f"gamma_inf={gamma_inf(w):.6f}"
)
# At gamma=0.3 the smooth-branch formula gives the cold optimum; verify
# against Julia's numerical minimum (printed when running the JL script).
checks.append(
    f"omega*={w}, gamma=0.30: eta_0={eta0_smooth(w, 0.30):.6f} "
    f"(Julia argmin reported eta=1.6495)"
)

# Threshold ordering gamma_wc < gamma_flat < gamma_inf for omega*>1.
for w in (1.05, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0):
    g_wc = gamma_wc(w); g_fl = gamma_flat(w); g_inf = gamma_inf(w)
    ok = (g_wc < g_fl < g_inf)
    checks.append(
        f"omega*={w:5.2f}: gamma_wc={g_wc:.5f}, gamma_flat={g_fl:.5f}, "
        f"gamma_inf={g_inf:.5f}  (ordered: {ok})"
    )

report = "\n".join(checks)
(FIGDIR / "forward_consistency_checks.txt").write_text(report + "\n")
print(report)
