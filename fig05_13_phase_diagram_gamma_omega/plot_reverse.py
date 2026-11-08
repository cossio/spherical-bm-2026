# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "numpy==2.4.4",
#     "scipy==1.17.1",
#     "matplotlib==3.11.2",
# ]
# ///
"""Temperature-optimum phase diagram in the (omega*, gamma) plane.

Implements the camera-ready reverse-KL classification for the rank-one K=2 teacher.
At each (gamma, omega*), find the minimizer eta_* of the predictive-posterior
reverse KL D(eta) = D_KL(P_pp,eta || P*) / N and color the point by

    omega* > omega0 and gamma < c1(omega*):
        eta_* < 1            warm posterior,
        eta_* > 1            cold posterior,
        eta_* = +inf         MAP / no finite optimum.

    omega* <= omega0 and gamma < c1(omega*):
        SG plateau, eta_* in (0, gamma/c1^2].

    gamma >= c1(omega*) at any teacher strength:
        zero-overlap plateau, eta_* anywhere in (0, infty).

Outputs:
    reproduced/phase_diagram_gamma_omega.pdf
    reproduced/reverse_boundaries.csv
    reproduced/reverse_consistency_checks.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Reproduce the reverse KL figure from saved analytical curves.")
parser.add_argument("--outdir", type=Path, default=HERE / "reproduced")
args = parser.parse_args()
FIGDIR = args.outdir
RESDIR = HERE / "data" / "reverse"
FIGDIR.mkdir(parents=True, exist_ok=True)

OMEGA0 = 1.0 + 1.0 / np.sqrt(2.0)         # ~ 1.7071: a = 1 (positive-overlap turn-on)
OMEGA2 = (3.0 + np.sqrt(3.0)) / 2.0       # ~ 2.3660: d=2 vector criterion turn-on (internal)


def teacher_params(omega: float) -> tuple[float, float, float]:
    """Empirical eigenvalues c1, c2 and effective teacher strength a for K=2."""
    c1 = 2.0 - 1.0 / omega
    c2 = 1.0 / omega
    a = 2.0 * omega * (omega - 1.0) / (2.0 * omega - 1.0)
    return c1, c2, a


def gamma_inf(omega: float) -> float:
    """Cold/MAP boundary: gamma_inf(omega) = (2w-1) / (2w (w-1)^2)."""
    return (2.0 * omega - 1.0) / (2.0 * omega * (omega - 1.0) ** 2)


def gamma_wc_d1(omega: float) -> float:
    """Scalar d=1 warm/cold boundary, valid for omega0 < omega <= omega2."""
    c1, c2, a = teacher_params(omega)
    disc = (c2 + 3.0 * c1) ** 2 - 16.0 * c1 / a
    if disc < 0.0:
        return float("nan")
    g = ((c2 + 3.0 * c1) - np.sqrt(disc)) / 4.0
    return 2.0 * g * g - c2 * g


def gamma_wc_d2(omega: float) -> float:
    """Vector d=2 warm/cold boundary, valid for omega > omega2."""
    c1, c2, a = teacher_params(omega)

    def b_edge(t: float) -> float:
        return 2.0 * c1 * (c1 * c2 - t) / (t * (c1 - t) * (2.0 * c1 * c2 - t))

    eps = 1e-12
    f_lo = b_edge(eps) - a
    f_hi = b_edge(c2 - eps) - a
    if f_lo * f_hi > 0.0:
        return float("nan")
    t_star = brentq(lambda t: b_edge(t) - a, eps, c2 - eps, xtol=1e-12, rtol=1e-12)
    return t_star * t_star


def gamma_wc(omega: float) -> float:
    """Global warm/cold boundary, choosing the relevant d-branch by omega."""
    if omega <= OMEGA0:
        return float("nan")
    if omega <= OMEGA2:
        return gamma_wc_d1(omega)
    return gamma_wc_d2(omega)


# ------------------------------ build curves -------------------------------- #

OMEGA_MIN, OMEGA_MAX = 1.0, 3.0
GAMMA_MIN, GAMMA_MAX = 0.0, 1.5

# omega > omega0: warm/cold/MAP curves (positive-overlap optimum present)
omega_grid_signal = np.linspace(OMEGA0 + 1e-6, OMEGA_MAX, 2001)
g_inf_curve = np.array([gamma_inf(w) for w in omega_grid_signal])
g_wc_curve = np.array([gamma_wc(w) for w in omega_grid_signal])
c1_signal_curve = 2.0 - 1.0 / omega_grid_signal

# omega <= omega0: SG/PM split at gamma = c1(omega) = 2 - 1/omega
omega_grid_zero = np.linspace(OMEGA_MIN, OMEGA0, 401)
c1_curve = 2.0 - 1.0 / omega_grid_zero  # SG (below) vs PM (above)

np.savetxt(
    FIGDIR / "reverse_boundaries.csv",
    np.column_stack([omega_grid_signal, g_wc_curve, g_inf_curve]),
    delimiter=",",
    header="omega_star,gamma_warm_cold,gamma_inf",
    comments="",
)


# ------------------------------- plot --------------------------------------- #

C_WARM = "#F2D88E"     # warm yellow
C_COLD = "#7AABCC"     # medium blue (slightly darker than MAP)
C_MAP = "#CFE2EE"      # light blue
C_SG = "#EFB6B6"       # light red
C_PM = "#C8C8C8"       # gray
C_BOUND = "black"

fig, (ax, ax_b) = plt.subplots(
    2, 1,
    figsize=(2.1, 2.60),
    gridspec_kw={"height_ratios": [1.95, 1.20]},
    constrained_layout=True,
)
fig.set_constrained_layout_pads(h_pad=0.01, hspace=0.0,
                                w_pad=0.01, wspace=0.0)

# Zero-overlap strip omega* <= omega0: SG (gamma < c1) and PM (gamma >= c1).
# SG is rendered with the same fill as the warm phase so that the lower-omega
# strip merges visually with the warm region across omega = omega0.
ax.fill_between(
    omega_grid_zero,
    np.full_like(omega_grid_zero, GAMMA_MIN),
    np.minimum(c1_curve, GAMMA_MAX),
    color=C_WARM, alpha=0.55, linewidth=0,
)
ax.fill_between(
    omega_grid_zero,
    np.minimum(c1_curve, GAMMA_MAX),
    np.full_like(omega_grid_zero, GAMMA_MAX),
    color=C_PM, alpha=0.7, linewidth=0,
)

# omega* > omega0: warm / cold / MAP below c1; degenerate above c1.
ax.fill_between(
    omega_grid_signal,
    np.full_like(omega_grid_signal, GAMMA_MIN),
    g_wc_curve,
    color=C_WARM, alpha=0.55, linewidth=0,
)
ax.fill_between(
    omega_grid_signal,
    g_wc_curve,
    np.minimum(g_inf_curve, GAMMA_MAX),
    color=C_COLD, alpha=0.55, linewidth=0,
)
ax.fill_between(
    omega_grid_signal,
    np.minimum(g_inf_curve, GAMMA_MAX),
    np.minimum(c1_signal_curve, GAMMA_MAX),
    color=C_MAP, alpha=0.55, linewidth=0,
)
ax.fill_between(
    omega_grid_signal,
    np.minimum(c1_signal_curve, GAMMA_MAX),
    np.full_like(omega_grid_signal, GAMMA_MAX),
    color=C_PM, alpha=0.7, linewidth=0,
)

# Phase boundary lines.  Black solid for the non-Bayes boundaries; the warm/cold
# boundary (eta_*=1) is highlighted in green as the "Bayes" line.
LINE_KW = dict(color="black", lw=1.4, ls="-")
C_BAYES = "#0E8A8A"   # teal
ax.plot(omega_grid_signal, g_wc_curve, color=C_BAYES, lw=1.6, ls="--")
ax.plot(omega_grid_signal, g_inf_curve, **LINE_KW)
ax.plot(omega_grid_zero, c1_curve, **LINE_KW)
ax.plot(omega_grid_signal, c1_signal_curve, **LINE_KW)
# The omega = omega0 separator ends at gamma = c1(omega0) = sqrt(2).
# Above this point the zero-overlap degenerate region spans both sides.
GAMMA_JOIN = 2.0 - 1.0 / OMEGA0
ax.plot([OMEGA0, OMEGA0], [GAMMA_MIN, GAMMA_JOIN],
        color="black", lw=0.7, ls="--")

# Region labels
ax.text(1.95, 0.18, "warm",
        ha="center", va="center", fontsize=7, color="#7A4D00")
ax.text(2.20, 0.45, "cold",
        ha="center", va="center", fontsize=7, color="#1A3F6B",
        rotation=-30, rotation_mode="anchor")
ax.text(2.65, 1.00, "MAP",
        ha="center", va="center", fontsize=8.5, color="#1A3F6B")
ax.text(1.36, 0.55, "warm deg.", ha="center", va="center",
        fontsize=7, color="#7A4D00")
ax.text(1.27, 1.37, "deg.", ha="center", va="center",
        fontsize=8, color="#444444")

# Markers for the (omega*, gamma) points whose KL(eta) is shown in panel B.
C_MAP_DOT = "#9D7DB8"  # light purple, distinguishes the gamma=1 (MAP) curve
ax.scatter([2.2], [0.1], color="#7A4D00", s=14, zorder=5, edgecolor="white", linewidth=0.4)
ax.scatter([2.2], [0.3], color="#1A3F6B", s=14, zorder=5, edgecolor="white", linewidth=0.4)
ax.scatter([2.2], [1.0], color=C_MAP_DOT, s=14, zorder=5, edgecolor="white", linewidth=0.4)

ax.set_xlim(OMEGA_MIN, OMEGA_MAX)
ax.set_ylim(GAMMA_MIN, GAMMA_MAX)
ax.set_xlabel(r"$\omega^*$", fontsize=8, labelpad=1)
ax.set_ylabel(r"$\gamma$", fontsize=8, labelpad=1, rotation=0)
ax.set_xticks([1.0, 1.5, 2.0, 2.5, 3.0])
ax.set_yticks([0.5, 1.0, 1.5])
ax.tick_params(labelsize=6, pad=1.5, length=2)

# ----- panel B: predictive-posterior reverse KL vs eta at omega*=2.2 ----- #
# Data is generated by generate_reverse.jl into data/reverse/pp_rev_kl_2gamma.csv.
ppkl_csv = RESDIR / "pp_rev_kl_2gamma.csv"
data = np.genfromtxt(ppkl_csv, delimiter=",", names=True)
eta = data["eta"]
kl_warm = data["kl_gamma_01"]
kl_cold = data["kl_gamma_03"]
kl_map  = data["kl_gamma_10"]

ax_b.plot(eta, kl_warm, color="#7A4D00", lw=1.4)
ax_b.plot(eta, kl_cold, color="#1A3F6B", lw=1.4)
ax_b.plot(eta, kl_map,  color=C_MAP_DOT, lw=1.4)
i_w = int(np.nanargmin(kl_warm)); i_c = int(np.nanargmin(kl_cold))
ax_b.scatter([eta[i_w]], [kl_warm[i_w]], color="#7A4D00", s=14, zorder=5)
ax_b.scatter([eta[i_c]], [kl_cold[i_c]], color="#1A3F6B", s=14, zorder=5)
# Bayes vertical with inline label near the top of the plot.
ax_b.axvline(1.0, color=C_BAYES, lw=1.4, ls="--")
ax_b.text(0.98, 0.215, "Bayes", color=C_BAYES, fontsize=6,
          ha="right", va="top")

# Inline curve annotations (in each curve's color) replacing the legend.
ax_b.text(1.55, 0.213, r"$\gamma=0.1$ (warm)", color="#7A4D00",
          fontsize=6, ha="left", va="top")
ax_b.text(1.55, 0.155, r"$\gamma=0.3$ (cold)", color="#1A3F6B",
          fontsize=6, ha="left", va="bottom")
ax_b.text(2.05, 0.178, r"$\gamma=1$ (MAP)", color=C_MAP_DOT,
          fontsize=6, ha="left", va="bottom")

ax_b.set_xlim(0.0, 3.0)
ax_b.set_ylim(0.14, 0.22)
ax_b.set_xlabel(r"$\eta$", fontsize=8, labelpad=1)
ax_b.set_ylabel(r"$D_{\mathrm{KL}}(P_{\mathrm{pp}}\Vert P^*)/N$",
                fontsize=7, labelpad=1)
ax_b.set_xticks([0, 1, 2, 3])
ax_b.set_yticks([0.15, 0.20])
ax_b.tick_params(labelsize=6, pad=1.5, length=2)

# Panel labels A) and B) at top-left of each axes.  Using ax.transAxes so
# they are positioned identically (relative to each panel's plot box) and
# therefore align perfectly when both axes share the same x-extent.
LBL_KW = dict(fontsize=9, fontweight="bold", ha="right", va="top")
ax.text(-0.18, 1.05, "A)", transform=ax.transAxes, **LBL_KW)
ax_b.text(-0.18, 1.10, "B)", transform=ax_b.transAxes, **LBL_KW)

out = FIGDIR / "phase_diagram_gamma_omega.pdf"
fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print(f"wrote {out}")


# ---------------------------- consistency checks ---------------------------- #

checks: list[str] = []

w = 2.2
checks.append(f"omega*={w:.3f}: gamma_wc={gamma_wc(w):.5f} (expect 0.24611), "
              f"gamma_inf={gamma_inf(w):.5f} (expect 0.53662)")

w = OMEGA2
checks.append(f"omega*=omega2={w:.5f}: gamma_wc_d1={gamma_wc_d1(w):.6f}, "
              f"gamma_wc_d2={gamma_wc_d2(w + 1e-9):.6f}, c2^2={1.0/w**2:.6f}")

# Zero-overlap strip checks
for w in (1.0, 1.2, 1.5, OMEGA0):
    c1, _, a = teacher_params(w)
    checks.append(f"omega*={w:.4f}: a={a:.4f}, c1={c1:.4f} (SG below, PM at/above)")

for w in (3.0, 5.0, 10.0):
    checks.append(f"omega*={w:5.2f}: gamma_inf={gamma_inf(w):.5f}, gamma_wc={gamma_wc(w):.5f}")

report = "\n".join(checks)
(FIGDIR / "reverse_consistency_checks.txt").write_text(report + "\n")
print(report)
