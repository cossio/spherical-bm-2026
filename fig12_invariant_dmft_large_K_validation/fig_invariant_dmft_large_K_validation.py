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
"""Generate Fig. 11: invariant large-K DMFT validation on Ising data."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from numba import njit
from scipy.sparse.linalg import eigsh


SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR

L = 45
K = 512
T = 1.8
BURN_IN = 500
GAP = 5

GAMMA = 0.4
ETA = 10.0
NU_GRID = (0.50, 0.85, 1.20)
SIM_SEEDS = (1001, 1002, 1003)
TAU_MAX = 30.0
DTAU = 1e-2
S0 = 0.1


# ── 2d Ising samples ─────────────────────────────────────────────────────

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


def record_steps(dt: float, t_max: float) -> np.ndarray:
    n_steps = round(t_max / dt)
    early = round(3.0 / K / dt)
    return np.unique(
        np.r_[
            np.arange(0, early + 1),
            np.arange(early + 1, n_steps + 1, 50),
            n_steps,
        ]
    ).astype(int)


def simulate_dynamics(
    samples: np.ndarray,
    gamma: float,
    eta: float,
    nu: float,
    dt: float,
    t_max: float,
    x0: np.ndarray,
    seed: int,
    keep_steps: np.ndarray,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    k_samples, N = samples.shape
    covariance = samples.T @ samples / N
    _, modes = data_modes(samples, 1)
    W = goe_matrix(N, rng) / np.sqrt(gamma * eta * N)
    x = np.sqrt(N) * x0 / np.linalg.norm(x0)

    decay = np.exp(-0.5 * gamma * dt)
    noise = np.sqrt((1.0 - decay * decay) / (eta * gamma * N))
    keep = set(keep_steps)
    rows = []
    n_steps = round(t_max / dt)

    for step in range(n_steps + 1):
        if step in keep:
            eigvals, eigvecs = top_eigenpairs(W, 1)
            rows.append(
                (
                    K * step * dt,
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
            - (k_samples * (1.0 - decay) / (gamma * N)) * xx
            + noise * goe_matrix(N, rng)
        )
        W = 0.5 * (W_next + W_next.T)

    return {
        "tau": np.array([row[0] for row in rows]),
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
    k_samples: int,
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
            - 0.5 * k_samples * nu * Q[n, : n + 1]
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
            - 0.5 * k_samples * nu * Q[p, : p + 1]
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

    return {"t": t, "s": s, "Q": Q}


def spectral_prediction(
    dmft: dict[str, np.ndarray],
    c: np.ndarray,
    gamma: float,
    eta: float,
    k_samples: int,
) -> dict[str, np.ndarray]:
    t = dmft["t"]
    s = dmft["s"]
    Q = dmft["Q"]
    beta = gamma * eta
    edge = 2.0 / np.sqrt(beta)

    out_idx = np.unique(np.r_[np.arange(0, len(t), 5), len(t) - 1])
    lambdas = np.zeros(len(out_idx))
    overlaps = np.zeros(len(out_idx))

    for out_pos, n in enumerate(out_idx):
        hist = np.unique(np.r_[np.arange(0, n + 1, 20), n])
        G = np.zeros((1 + len(hist), 1 + len(hist)))
        G[0, 0] = 1.0
        G[0, 1:] = s[0, hist]
        G[1:, 0] = s[0, hist]
        G[1:, 1:] = Q[np.ix_(hist, hist)]

        B = np.r_[
            ((1.0 - np.exp(-0.5 * gamma * t[n])) / gamma) * c[0],
            -0.5
            * k_samples
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
        q = np.argmax(theta)

        if theta[q] <= 1.0 / np.sqrt(beta):
            lambdas[out_pos] = edge
            continue

        z = invsqrtG @ Z[:, q]
        z /= np.sqrt(max(z @ G @ z, 1e-300))
        strength = max(0.0, 1.0 - 1.0 / (beta * theta[q] * theta[q]))
        lambdas[out_pos] = theta[q] + 1.0 / (beta * theta[q])
        overlaps[out_pos] = strength * (G @ z)[0] ** 2

    return {"t": t[out_idx], "lambda": lambdas, "u": overlaps}


# ── Data generation ──────────────────────────────────────────────────

def generate_results() -> dict[float, tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, np.ndarray]]]]:
    print(f"generating Ising data: L={L}, K={K}")
    samples = ising_samples(L, K, T, 12345 + K).astype(float)
    c, modes = data_modes(samples, 1)
    x0 = initial_condition(modes, S0, 54321 + K)
    s0 = modes[:, :1].T @ x0 / samples.shape[1]
    out = {}

    for i, nu in enumerate(NU_GRID):
        print(f"nu={nu}")
        c_bar = c[:1] / K
        dmft = solve_dmft(c_bar, 1, GAMMA, ETA, nu, TAU_MAX, DTAU, s0)
        pred = spectral_prediction(dmft, c_bar, GAMMA, ETA, 1)
        dt_finite = DTAU / K
        t_max_finite = TAU_MAX / K
        keep = record_steps(dt_finite, t_max_finite)
        sims = [
            simulate_dynamics(
                samples,
                K * GAMMA,
                ETA / K,
                K * nu,
                dt_finite,
                t_max_finite,
                x0,
                100000 + 10000 * i + seed,
                keep,
            )
            for seed in SIM_SEEDS
        ]
        out[nu] = (dmft, pred, sims)

    return out


# ── Figure ────────────────────────────────────────────────────────────

def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def color_map():
    nu_values = np.array(NU_GRID)
    cmap = plt.get_cmap("viridis")
    boundaries = np.r_[
        nu_values[0] - 0.5 * (nu_values[1] - nu_values[0]),
        0.5 * (nu_values[:-1] + nu_values[1:]),
        nu_values[-1] + 0.5 * (nu_values[-1] - nu_values[-2]),
    ]
    norm = mpl.colors.BoundaryNorm(boundaries, cmap.N)
    return lambda nu: cmap(norm(nu))


def plot_results(
    results: dict[
        float,
        tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, np.ndarray]]],
    ],
) -> None:
    set_style()
    color = color_map()
    fig, axes = plt.subplots(3, 1, figsize=(8.8, 9.0), sharex=True)

    panels = [
        (
            r"$\overline{s}_{1}^2(\overline{t})$",
            lambda dmft, pred: dmft["t"],
            lambda dmft, pred: dmft["s"][0] ** 2,
            lambda sims: np.stack([sim["s"][:, 0] ** 2 for sim in sims]),
            True,
        ),
        (
            r"$\overline{\lambda}_1(\overline{t})$",
            lambda dmft, pred: pred["t"],
            lambda dmft, pred: pred["lambda"],
            lambda sims: np.stack([sim["lambda_top"][:, 0] for sim in sims]),
            False,
        ),
        (
            r"$\overline{u}_{11}^2(\overline{t})$",
            lambda dmft, pred: pred["t"],
            lambda dmft, pred: pred["u"],
            lambda sims: np.stack([sim["u_sq"][:, 0, 0] for sim in sims]),
            True,
        ),
    ]

    for ax, (ylabel, times, theory, finite_n, nonnegative) in zip(axes, panels):
        for nu in NU_GRID:
            dmft, pred, sims = results[nu]
            samples = finite_n(sims)
            mean = samples.mean(axis=0)
            std = samples.std(axis=0, ddof=1)
            lower = np.maximum(mean - std, 0.0) if nonnegative else mean - std

            ax.fill_between(
                sims[0]["tau"],
                lower,
                mean + std,
                color=color(nu),
                alpha=0.28,
                lw=0,
            )
            ax.plot(times(dmft, pred), theory(dmft, pred), color=color(nu), lw=2.2)

        ax.set_ylabel(ylabel, fontsize=18)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", which="both", direction="out", labelsize=15)

    axes[1].axhline(2.0 / np.sqrt(GAMMA * ETA), color="gray", ls=":", lw=1.3, alpha=0.9)
    axes[2].set_xlabel(r"$\overline{t} = Kt/D$", fontsize=18)

    fig.subplots_adjust(left=0.10, right=0.84, bottom=0.08, top=0.93, hspace=0.10)
    fig.legend(
        handles=[
            Patch(facecolor="gray", alpha=0.28, edgecolor="none", label="Langevin"),
            Line2D([0], [0], color="black", lw=2.2, label="DMFT"),
        ],
        frameon=True,
        edgecolor="black",
        facecolor="white",
        fontsize=11,
        loc="upper left",
        bbox_to_anchor=(0.855, 0.93),
    )
    fig.legend(
        handles=[Line2D([0], [0], color=color(nu), lw=4, label=rf"${nu:.2f}$") for nu in NU_GRID],
        title=r"$\overline{\nu}$",
        title_fontsize=11,
        frameon=True,
        edgecolor="black",
        facecolor="white",
        fontsize=11,
        loc="upper left",
        bbox_to_anchor=(0.855, 0.73),
    )
    fig.suptitle(
        rf"Large-$K$ sweep over $\overline{{\nu}}$, $K={K}$, $N={L * L}$, $T={T}$",
        fontsize=16,
    )

    out_pdf = FIG_DIR / "fig_invariant_dmft_large_K_validation.pdf"
    out_png = out_pdf.with_suffix(".png")
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved → {out_pdf}")


def main() -> None:
    plot_results(generate_results())


if __name__ == "__main__":
    main()
