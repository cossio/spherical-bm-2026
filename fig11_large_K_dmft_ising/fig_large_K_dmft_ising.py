#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.23",
#   "scipy>=1.9",
#   "matplotlib>=3.6",
#   "numba>=0.57",
# ]
# ///
"""Generate Fig. 10: large-K DMFT validation on 2D Ising data."""

from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.sparse.linalg import eigsh


SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR

L = 32
T_LOW = 2.2
T_HIGH = 3.2
K_DYN = 512
K_GRID = (16, 32, 64, 128, 256, 512, 1024, 2048)
N_REPS = 3
BURN_IN = 500
GAP = 5

GAMMA = 0.4
ETA = 10.0
NU = 0.85
T_MAX = 4.0
DT = 5e-4
S0 = 0.1
SEED = 123


# ── 2d Ising samples ─────────────────────────────────────────────────────

def critical_temp() -> float:
    return 2.0 / math.log(1.0 + math.sqrt(2.0))


def onsager_m2(T: float) -> float:
    if T >= critical_temp():
        return 0.0
    return (1.0 - math.sinh(2.0 / T) ** -4.0) ** 0.25


@njit
def seed_numba(seed: int) -> None:
    np.random.seed(seed)


@njit
def wolff_step(spins: np.ndarray, beta: float, L: int) -> None:
    N = L * L
    p_add = 1.0 - np.exp(-2.0 * beta)
    start = np.random.randint(N)
    spin = spins[start]

    seen = np.zeros(N, np.uint8)
    stack = np.empty(N, np.int64)
    cluster = np.empty(N, np.int64)

    top = 1
    count = 0
    stack[0] = start
    seen[start] = 1

    while top > 0:
        top -= 1
        i = stack[top]
        cluster[count] = i
        count += 1

        x = i // L
        y = i - L * x
        xp = 0 if x + 1 == L else x + 1
        xm = L - 1 if x == 0 else x - 1
        yp = 0 if y + 1 == L else y + 1
        ym = L - 1 if y == 0 else y - 1

        for j in (xp * L + y, xm * L + y, x * L + yp, x * L + ym):
            if seen[j] == 0 and spins[j] == spin and np.random.random() < p_add:
                seen[j] = 1
                stack[top] = j
                top += 1

    for q in range(count):
        spins[cluster[q]] = -spins[cluster[q]]


def ising_samples(L: int, K: int, T: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    spins = rng.choice(np.array([-1, 1], np.int8), L * L)
    samples = np.empty((K, L * L), np.int8)

    seed_numba(seed + 12345)
    wolff_step(spins, 1.0 / T, L)

    for _ in range(BURN_IN):
        wolff_step(spins, 1.0 / T, L)

    for k in range(K):
        for _ in range(GAP):
            wolff_step(spins, 1.0 / T, L)
        if rng.random() < 0.5:
            spins *= np.int8(-1)
        samples[k] = spins

    return samples


# ── Finite-N Langevin dynamics ─────────────────────────────────────────

def data_modes(samples: np.ndarray, n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    _, singular_values, Vt = np.linalg.svd(samples.astype(float), full_matrices=False)
    weights = singular_values[:n_modes] ** 2 / samples.shape[1]
    modes = np.sqrt(samples.shape[1]) * Vt[:n_modes].T

    for a in range(n_modes):
        if modes[:, a].sum() < 0:
            modes[:, a] *= -1

    return weights, modes


def initial_condition(modes: np.ndarray, s0: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    N = modes.shape[0]
    c = modes[:, 0] / np.sqrt(N)
    z = rng.normal(size=N)
    z -= c * (c @ z)
    z /= np.linalg.norm(z)
    return np.sqrt(N) * (s0 * c + np.sqrt(1.0 - s0 * s0) * z)


def goe_matrix(N: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(N, N))
    return (A + A.T) / np.sqrt(2.0)


def top_eigenpairs(A: np.ndarray, n_pairs: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        w, V = eigsh(
            A,
            k=n_pairs,
            which="LA",
            v0=np.ones(A.shape[0]) / np.sqrt(A.shape[0]),
            tol=1e-8,
            maxiter=max(1000, 20 * A.shape[0]),
        )
        order = np.argsort(w)[::-1]
        return w[order], V[:, order]
    except Exception:
        w, V = np.linalg.eigh(A)
        order = np.argsort(w)[::-1][:n_pairs]
        return w[order], V[:, order]


def simulate_dynamics(
    samples: np.ndarray,
    gamma: float,
    eta: float,
    nu: float,
    dt: float,
    t_max: float,
    x0: np.ndarray,
    seed: int,
    record_steps: np.ndarray,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    K, N = samples.shape
    covariance = samples.T @ samples / N
    _, modes = data_modes(samples, 2)
    W = goe_matrix(N, rng) / np.sqrt(gamma * eta * N)
    x = np.sqrt(N) * x0 / np.linalg.norm(x0)

    decay = np.exp(-0.5 * gamma * dt)
    noise = np.sqrt((1.0 - decay * decay) / (eta * gamma * N))
    keep = set(np.asarray(record_steps, int))
    rows = []
    n_steps = int(round(t_max / dt))

    for step in range(n_steps + 1):
        if step in keep:
            eigvals, eigvecs = top_eigenpairs(W, 2)
            rows.append(
                (
                    step * dt,
                    modes.T @ x / N,
                    eigvals,
                    (eigvecs.T @ modes) ** 2 / N,
                )
            )

        if step == n_steps:
            break

        xx = np.outer(x, x)
        Wx = W @ x
        z = rng.normal(size=N)
        tangent_noise = z - x * (x @ z / N)
        tangent_drift = Wx - x * (x @ Wx / N)
        y = x + dt * nu * tangent_drift + np.sqrt(2.0 * nu * dt) * tangent_noise
        x = np.sqrt(N) * y / np.linalg.norm(y)

        W_next = (
            decay * W
            + ((1.0 - decay) / gamma) * covariance
            - (K * (1.0 - decay) / (gamma * N)) * xx
            + noise * goe_matrix(N, rng)
        )
        W = 0.5 * (W_next + W_next.T)

    return {
        "t": np.array([row[0] for row in rows]),
        "s": np.stack([row[1] for row in rows]),
        "lambda_top": np.stack([row[2] for row in rows]),
        "u_sq": np.stack([row[3] for row in rows]),
    }


# ── DMFT prediction ──────────────────────────────────────

def trapz_weights(t: np.ndarray) -> np.ndarray:
    if len(t) < 2:
        return np.zeros(len(t))

    weights = np.zeros(len(t))
    weights[0] = (t[1] - t[0]) / 2.0
    weights[-1] = (t[-1] - t[-2]) / 2.0
    if len(t) > 2:
        weights[1:-1] = (t[2:] - t[:-2]) / 2.0
    return weights


def solve_dmft(
    c: np.ndarray,
    K: int,
    gamma: float,
    eta: float,
    nu: float,
    t_max: float,
    dt: float,
    s0: np.ndarray,
) -> dict[str, np.ndarray]:
    t = np.arange(0.0, t_max + 0.5 * dt, dt)
    n_times = len(t)
    n_modes = len(c)

    s = np.zeros((n_modes, n_times))
    Q = np.zeros((n_times, n_times))
    R = np.zeros((n_times, n_times))
    mu = np.zeros(n_times)

    s[:, 0] = s0
    Q[0, 0] = 1.0
    R[0, 0] = 1.0

    def compute_mu(n: int) -> float:
        kernel = np.exp(-0.5 * gamma * (t[n] - t[: n + 1]))
        memory = kernel * (
            (nu * nu / (eta * gamma)) * R[n, : n + 1]
            - 0.5 * K * nu * Q[n, : n + 1]
        )
        diffusion = (nu * nu / (eta * gamma)) * kernel * Q[n, : n + 1]
        signal = (nu / gamma) * (1.0 - np.exp(-0.5 * gamma * t[n]))
        weights = trapz_weights(t[: n + 1])
        return (
            signal * np.sum(c * s[:, n] ** 2)
            + np.sum(weights * (memory * Q[n, : n + 1] + diffusion * R[n, : n + 1]))
            + nu
        )

    mu[0] = compute_mu(0)

    for n in range(1, n_times):
        p = n - 1
        h = t[n] - t[p]
        tp = t[: p + 1]
        weights = trapz_weights(tp)
        kernel = np.exp(-0.5 * gamma * (t[p] - tp))
        memory = kernel * (
            (nu * nu / (eta * gamma)) * R[p, : p + 1]
            - 0.5 * K * nu * Q[p, : p + 1]
        )
        memory_weights = weights * memory
        signal = (nu / gamma) * (1.0 - np.exp(-0.5 * gamma * t[p]))

        s[:, n] = s[:, p] + h * (
            -mu[p] * s[:, p]
            + signal * c * s[:, p]
            + s[:, : p + 1] @ memory_weights
        )

        for j in range(n):
            weights_j = trapz_weights(t[j : p + 1])
            R[n, j] = R[p, j] + h * (
                -mu[p] * R[p, j]
                + np.sum(weights_j * memory[j : p + 1] * R[j : p + 1, j])
            )

        R[n, n] = 1.0
        MQ = memory_weights @ Q[: p + 1, : p + 1]
        diffusion = (nu * nu / (eta * gamma)) * kernel * Q[p, : p + 1]

        for j in range(n):
            diffusion_term = np.sum(
                trapz_weights(t[: j + 1]) * diffusion[: j + 1] * R[j, : j + 1]
            )
            noise_term = nu if j == p else 0.0
            q = Q[p, j] + h * (
                signal * np.sum(c * s[:, p] * s[:, j])
                - mu[p] * Q[p, j]
                + MQ[j]
                + diffusion_term
                + noise_term
            )
            Q[n, j] = q
            Q[j, n] = q

        Q[n, n] = 1.0
        mu[n] = compute_mu(n)

    return {"t": t, "s": s, "Q": Q, "R": R, "mu": mu}


def spectral_prediction(
    dmft: dict[str, np.ndarray],
    c: np.ndarray,
    gamma: float,
    eta: float,
    K: int,
    n_outliers: int = 2,
    output_stride: int = 2,
    history_stride: int = 4,
) -> dict[str, np.ndarray]:
    t = dmft["t"]
    s = dmft["s"]
    Q = dmft["Q"]
    beta = gamma * eta
    n_modes = len(c)
    edge = 2.0 / np.sqrt(beta)

    out_idx = np.unique(np.r_[np.arange(0, len(t), output_stride), len(t) - 1])
    lambdas = np.zeros((len(out_idx), n_outliers))
    overlaps = np.zeros((len(out_idx), n_outliers, n_modes))

    for out_pos, n in enumerate(out_idx):
        hist = np.unique(np.r_[np.arange(0, n + 1, history_stride), n])
        size = n_modes + len(hist)
        G = np.zeros((size, size))
        G[:n_modes, :n_modes] = np.eye(n_modes)
        G[:n_modes, n_modes:] = s[:, hist]
        G[n_modes:, :n_modes] = s[:, hist].T
        G[n_modes:, n_modes:] = Q[np.ix_(hist, hist)]

        B = np.r_[
            ((1.0 - np.exp(-0.5 * gamma * t[n])) / gamma) * c,
            -0.5
            * K
            * trapz_weights(t[hist])
            * np.exp(-0.5 * gamma * (t[n] - t[hist])),
        ]

        evals, evecs = np.linalg.eigh((G + G.T) / 2.0)
        keep = evals > 1e-10
        U = evecs[:, keep]
        sqrtG = (U * np.sqrt(evals[keep])) @ U.T
        invsqrtG = (U / np.sqrt(evals[keep])) @ U.T

        H = sqrtG @ (B[:, None] * sqrtG)
        theta, Z = np.linalg.eigh((H + H.T) / 2.0)
        order = np.argsort(theta)[::-1]

        for outlier, q in enumerate(order[:n_outliers]):
            if theta[q] <= 1.0 / np.sqrt(beta):
                lambdas[out_pos, outlier] = edge
                continue

            z = invsqrtG @ Z[:, q]
            z /= np.sqrt(max(z @ G @ z, 1e-300))
            mode_overlap = (G @ z)[:n_modes]
            strength = max(0.0, 1.0 - 1.0 / (beta * theta[q] * theta[q]))

            lambdas[out_pos, outlier] = theta[q] + 1.0 / (beta * theta[q])
            overlaps[out_pos, outlier] = strength * mode_overlap * mode_overlap

    return {
        "t": t[out_idx],
        "lambda": lambdas,
        "u_sq": overlaps,
        "edge": np.full(len(out_idx), edge),
    }


# ── Data generation ────────────────────────────────

def spectrum_scaling() -> dict[str, np.ndarray]:
    out = {key: [] for key in "bc1 bs1 bc2 bs2 ac1 as1 ac2 as2".split()}

    for tag, T, seed in (("b", T_LOW, SEED), ("a", T_HIGH, SEED + 333)):
        for K in K_GRID:
            print(f"spectrum scaling: T={T}, K={K}")
            vals = []
            for rep in range(N_REPS):
                X = ising_samples(L, K, T, seed + 1000 * rep + K).astype(np.float32)
                eigvals = np.linalg.eigvalsh(X @ X.T / (L * L))[::-1][:2] / K
                vals.append(eigvals)
            vals = np.array(vals)
            out[f"{tag}c1"].append(vals[:, 0].mean())
            out[f"{tag}s1"].append(vals[:, 0].std(ddof=1))
            out[f"{tag}c2"].append(vals[:, 1].mean())
            out[f"{tag}s2"].append(vals[:, 1].std(ddof=1))

    return {key: np.array(value) for key, value in out.items()} | {
        "K": np.array(K_GRID),
        "mlo": onsager_m2(T_LOW),
        "mhi": onsager_m2(T_HIGH),
    }


def dynamics_comparison() -> dict[str, object]:
    print("dynamics comparison")
    X = ising_samples(L, K_DYN, 1.8, SEED + 999).astype(float)
    c, modes = data_modes(X, 2)
    x0 = initial_condition(modes, S0, SEED + 1001)
    s0 = modes.T @ x0 / X.shape[1]

    dmft = solve_dmft(c, K_DYN, GAMMA, ETA, NU, T_MAX, DT, s0)
    spectrum = spectral_prediction(dmft, c, GAMMA, ETA, K_DYN)
    record_steps = np.unique(
        np.r_[
            np.arange(0, round(0.02 / DT) + 1),
            np.arange(round(0.02 / DT) + 1, round(T_MAX / DT) + 1, 20),
            round(T_MAX / DT),
        ]
    ).astype(int)
    simulation = simulate_dynamics(X, GAMMA, ETA, NU, DT, T_MAX, x0, SEED, record_steps)

    q = min(1.0, c[0] / K_DYN)
    theta = np.inf if q == 1.0 else 1.0 / (1.0 - q)
    edge = 2.0 / np.sqrt(GAMMA * ETA)
    lambda1 = theta + 1.0 / (GAMMA * ETA * theta) if theta > 1.0 / np.sqrt(GAMMA * ETA) else edge

    stationarity = {
        "edge": edge,
        "s1": np.sign(dmft["s"][0, -1]) * np.sqrt(q),
        "s2": 0.0,
        "u11": 1.0,
        "u22": 0.0,
        "lambda1": lambda1,
        "lambda2": edge,
    }
    return {"dmft": dmft, "spec": spectrum, "sim": simulation, "st": stationarity}


# ── Figure ────────────────────────────────────────────────────────────

def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.labelsize": 12,
            "legend.fontsize": 8,
        }
    )


def plot_spectrum_scaling(ax1, ax2, results: dict[str, np.ndarray]) -> None:
    K = results["K"]

    ax1.errorbar(K, results["bc1"], results["bs1"], marker="o", capsize=3, lw=1.6, ms=4,
                 label=r"$c_1(T_{\rm low})/K$")
    ax1.errorbar(K, results["ac1"], results["as1"], marker="s", capsize=3, lw=1.6, ms=4,
                 label=r"$c_1(T_{\rm high})/K$")
    ax1.axhline(results["mlo"], ls="--", lw=1.4, label=r"$m(T_{\rm low})^2$")
    ax1.axhline(results["mhi"], ls=":", lw=1.4, color="tab:orange",
                label=r"$m(T_{\rm high})^2$")
    ax1.set_xscale("log")
    ax1.set_xlim(0.9 * K.min(), 1.1 * K.max())
    ax1.set_ylabel(r"$c_1/K$")
    ax1.legend(frameon=False)

    ax2.errorbar(K, results["bc2"], results["bs2"], marker="o", capsize=3, lw=1.6, ms=4,
                 label=r"$c_2(T_{\rm low})/K$")
    ax2.errorbar(K, results["ac2"], results["as2"], marker="s", capsize=3, lw=1.6, ms=4,
                 label=r"$c_2(T_{\rm high})/K$")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$K$")
    ax2.set_ylabel(r"$c_2/K$")
    ax2.legend(frameon=False)


def plot_dynamics(axes, results: dict[str, object]) -> None:
    sim = results["sim"]
    dmft = results["dmft"]
    spec = results["spec"]
    st = results["st"]

    sim_mask = sim["t"] > 0
    dmft_mask = dmft["t"] > 0
    spec_mask = spec["t"] > 0
    t_min = min(a[a > 0].min() for a in (sim["t"], dmft["t"], spec["t"]))
    t_max = max(sim["t"].max(), dmft["t"].max(), spec["t"].max())

    ax = axes[0]
    ax.semilogx(sim["t"][sim_mask], sim["s"][sim_mask, 0], color="C3", lw=1.9, label=r"$s_1$")
    ax.semilogx(dmft["t"][dmft_mask], dmft["s"][0, dmft_mask], "--", color="C3", lw=1.9)
    ax.semilogx(sim["t"][sim_mask], sim["s"][sim_mask, 1], color="C4", lw=1.7, label=r"$s_2$")
    ax.semilogx(dmft["t"][dmft_mask], dmft["s"][1, dmft_mask], "--", color="C4", lw=1.7)
    ax.axhline(st["s1"], ls=":", color="C3")
    ax.axhline(st["s2"], ls=":", color="C4")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$s_k(t)$")

    ax = axes[1]
    ax.semilogx(sim["t"][sim_mask], sim["u_sq"][sim_mask, 0, 0], color="C5", lw=1.9,
                label=r"$u_{11}^2$")
    ax.semilogx(spec["t"][spec_mask], spec["u_sq"][spec_mask, 0, 0], "--", color="C5", lw=1.9)
    ax.semilogx(sim["t"][sim_mask], sim["u_sq"][sim_mask, 1, 1], color="C6", lw=1.6,
                label=r"$u_{22}^2$")
    ax.semilogx(spec["t"][spec_mask], spec["u_sq"][spec_mask, 1, 1], "--", color="C6", lw=1.6)
    ax.axhline(st["u11"], ls=":", color="C5")
    ax.axhline(st["u22"], ls=":", color="C6")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$u_{k\ell}^2(t)$")

    ax = axes[2]
    ax.semilogx(sim["t"][sim_mask], sim["lambda_top"][sim_mask, 0], color="C7", lw=1.9,
                label=r"$\lambda_1$")
    ax.semilogx(spec["t"][spec_mask], spec["lambda"][spec_mask, 0], "--", color="C7", lw=1.9)
    ax.semilogx(sim["t"][sim_mask], sim["lambda_top"][sim_mask, 1], color="C8", lw=1.6,
                label=r"$\lambda_2$")
    ax.semilogx(spec["t"][spec_mask], spec["lambda"][spec_mask, 1], "--", color="C8", lw=1.6)
    ax.axhline(st["edge"], ls="--", color="black", label="edge")
    ax.axhline(st["lambda1"], ls=":", color="C7")
    ax.axhline(st["lambda2"], ls=":", color="C8")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\lambda_i(t)$")

    for ax in axes:
        ax.set_xlim(t_min, t_max)
        ax.legend(frameon=False)


def make_figure(scaling: dict[str, np.ndarray], dynamics: dict[str, object]) -> None:
    set_style()
    fig = plt.figure(figsize=(13.0, 5.2))
    outer = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.15)
    left = outer[0].subgridspec(2, 1, hspace=0.30)
    right = outer[1].subgridspec(1, 3, wspace=0.38)

    ax1 = fig.add_subplot(left[0])
    ax2 = fig.add_subplot(left[1])
    dyn_axes = [fig.add_subplot(right[i]) for i in range(3)]

    plot_spectrum_scaling(ax1, ax2, scaling)
    plot_dynamics(dyn_axes, dynamics)

    for ax in [ax1, ax2, *dyn_axes]:
        ax.grid(True, which="both", alpha=0.22)
        ax.tick_params(direction="in", length=4.5, width=0.9)

    fig.suptitle(
        rf"$N={L * L}$, $T_c\simeq {critical_temp():.3f}$, "
        rf"$T_{{\rm low}}={T_LOW}$, $T_{{\rm high}}={T_HIGH}$, "
        rf"$K_{{\rm dyn}}={K_DYN}$, $\gamma={GAMMA}$, "
        rf"$\eta={ETA}$, $\nu={NU}$",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_pdf = FIG_DIR / "fig_large_K_dmft_ising.pdf"
    out_png = out_pdf.with_suffix(".png")
    fig.savefig(out_pdf, dpi=1200, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved → {out_pdf}")


def main() -> None:
    make_figure(spectrum_scaling(), dynamics_comparison())


if __name__ == "__main__":
    main()
