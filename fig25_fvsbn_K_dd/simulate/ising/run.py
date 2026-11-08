#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "torch>=2.6",
#     "numpy>=2.4",
#     "matplotlib>=3.10",
# ]
# ///
"""Typical reverse KL of a binary VAN on a 2D Ising teacher, vs. L2 reg.

Teacher: 2D Ising ferromagnet on an L x L lattice with PBC at inverse
temperature beta, exact log Z via the column transfer matrix.

Student: fully-visible sigmoid belief net (FVSBN) or one-hidden-layer
MADE over +/-1 spins, trained by maximum likelihood on K teacher samples
with L2 weight decay gamma.

Metric: D_KL(q_theta || p_teacher) estimated by Monte Carlo from the
student, using the exact teacher log Z.  Report average over R seeds
(each seed = independent teacher dataset + student init).

Usage:
    ./run.py --L 4 --beta 0.5 --K 4 --seeds 0 1 2 --model fvsbn
    ./run.py --L 4 --beta 0.44 --K 4 --seeds $(seq 0 19) --model made --H 32

The default grid mirrors the hopfield_van pilot for apples-to-apples
comparison with the rank-1 Curie-Weiss run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)


# =====================================================================
# Teacher: 2D Ising on L x L with PBC
# =====================================================================


def ising_energy(s: torch.Tensor, L: int) -> torch.Tensor:
    """E(s) = - sum_{<i,j>} s_i s_j on L x L with PBC (J = +1)."""
    s2 = s.view(*s.shape[:-1], L, L)
    horiz = (s2 * torch.roll(s2, shifts=-1, dims=-1)).sum(dim=(-1, -2))
    vert = (s2 * torch.roll(s2, shifts=-1, dims=-2)).sum(dim=(-1, -2))
    return -(horiz + vert)


def _ising_energy_numpy(s: np.ndarray, L: int) -> np.ndarray:
    s2 = s.reshape(*s.shape[:-1], L, L)
    horiz = (s2 * np.roll(s2, -1, axis=-1)).sum(axis=(-1, -2))
    vert = (s2 * np.roll(s2, -1, axis=-2)).sum(axis=(-1, -2))
    return -(horiz + vert)


def ising_log_Z(beta: float, L: int) -> float:
    """Exact log Z via the column transfer matrix.

    T[a,b] = exp( beta <a,b> + (beta/2)(R(a)+R(b)) ),  R(a) = sum_i a_i a_{i+1}.
    Z = Tr(T^L) = sum_k lambda_k^L.
    """
    idx = np.arange(2 ** L, dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(L, dtype=np.int64)) & 1).astype(np.int64)
    sigma = (2 * bits - 1).astype(np.float64)
    R = (sigma * np.roll(sigma, -1, axis=1)).sum(axis=1)
    dot = sigma @ sigma.T
    log_T = beta * dot + 0.5 * beta * (R[:, None] + R[None, :])
    m = float(log_T.max())
    evals = np.linalg.eigvalsh(np.exp(log_T - m))
    pos = evals[evals > 0]
    log_evals = np.log(pos) + m
    x = L * log_evals
    c = float(x.max())
    return float(c + math.log(float(np.exp(x - c).sum())))


def ising_log_prob(s: torch.Tensor, beta: float, L: int, log_Z: float) -> torch.Tensor:
    return -beta * ising_energy(s, L) - log_Z


def exact_sample(beta: float, L: int, K: int, rng: np.random.Generator) -> torch.Tensor:
    """Enumerate 2^{L^2} states and draw K i.i.d. samples.  OK up to L = 4."""
    N = L * L
    if N > 20:
        raise ValueError(f"exact_sample: 2^{N} states is too many; use metropolis_sample")
    idx = np.arange(2 ** N, dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(N, dtype=np.int64)) & 1).astype(np.int64)
    sigma = (2 * bits - 1).astype(np.int8)
    E = _ising_energy_numpy(sigma.astype(np.float64), L)
    log_w = -beta * E
    log_w -= log_w.max()
    p = np.exp(log_w)
    p /= p.sum()
    draws = rng.choice(2 ** N, size=K, p=p)
    return torch.tensor(sigma[draws], dtype=torch.float32)


def metropolis_sample(
    beta: float,
    L: int,
    K: int,
    n_sweeps: int = 3000,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Checkerboard single-spin Metropolis for larger L.  K parallel chains."""
    if rng is None:
        rng = np.random.default_rng()
    s = rng.choice([-1, 1], size=(K, L, L)).astype(np.int8)
    ij = np.indices((L, L)).sum(axis=0)
    even = (ij % 2 == 0)
    odd = ~even
    for _ in range(n_sweeps):
        for mask in (even, odd):
            nn_sum = (
                np.roll(s, 1, axis=1) + np.roll(s, -1, axis=1)
                + np.roll(s, 1, axis=2) + np.roll(s, -1, axis=2)
            )
            dE = 2 * s * nn_sum
            accept = rng.random(s.shape) < np.exp(-beta * dE)
            flip = accept & mask
            s = np.where(flip, -s, s)
    return torch.tensor(s.reshape(K, L * L).astype(np.float32))


# =====================================================================
# Student: FVSBN and MADE for +/-1 spins
# =====================================================================


class BinaryFVSBN(nn.Module):
    """Fully-visible sigmoid belief net: q(s_i=+1|s_{<i}) = sigmoid(b_i + W_i . s_{<i})."""

    def __init__(self, N: int):
        super().__init__()
        self.N = N
        self.W = nn.Parameter(torch.zeros(N, N))
        self.b = nn.Parameter(torch.zeros(N))
        self.register_buffer("mask", torch.tril(torch.ones(N, N), diagonal=-1))

    def _logits(self, s: torch.Tensor) -> torch.Tensor:
        return s @ (self.W * self.mask).T + self.b

    def log_prob(self, s: torch.Tensor) -> torch.Tensor:
        logits = self._logits(s)
        target = (s + 1.0) * 0.5
        return -F.binary_cross_entropy_with_logits(logits, target, reduction="none").sum(-1)

    @torch.no_grad()
    def sample_with_log_prob(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        W_eff = (self.W * self.mask).detach()
        b = self.b.detach()
        s = torch.zeros(n, self.N)
        log_q = torch.zeros(n)
        for i in range(self.N):
            logit_i = s @ W_eff[i] + b[i]
            p_up = torch.sigmoid(logit_i)
            s_i = torch.where(torch.rand(n) < p_up, 1.0, -1.0)
            s[:, i] = s_i
            log_q += torch.where(s_i > 0, F.logsigmoid(logit_i), F.logsigmoid(-logit_i))
        return s, log_q


class BinaryMADE(nn.Module):
    """One-hidden-layer MADE with tanh activation."""

    def __init__(self, N: int, H: int, mask_seed: int = 0):
        super().__init__()
        self.N, self.H = N, H
        self.W1 = nn.Parameter(torch.empty(H, N))
        self.b1 = nn.Parameter(torch.zeros(H))
        self.W2 = nn.Parameter(torch.empty(N, H))
        self.b2 = nn.Parameter(torch.zeros(N))
        nn.init.normal_(self.W1, std=1.0 / N ** 0.5)
        nn.init.normal_(self.W2, std=1.0 / H ** 0.5)
        g = torch.Generator().manual_seed(mask_seed)
        hidden_labels = torch.randint(1, N, (H,), generator=g)
        input_labels = torch.arange(1, N + 1)
        output_labels = torch.arange(1, N + 1)
        self.register_buffer(
            "m1", (hidden_labels[:, None] >= input_labels[None, :]).float()
        )
        self.register_buffer(
            "m2", (output_labels[:, None] > hidden_labels[None, :]).float()
        )

    def _logits(self, s: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(s @ (self.W1 * self.m1).T + self.b1)
        return h @ (self.W2 * self.m2).T + self.b2

    def log_prob(self, s: torch.Tensor) -> torch.Tensor:
        logits = self._logits(s)
        target = (s + 1.0) * 0.5
        return -F.binary_cross_entropy_with_logits(logits, target, reduction="none").sum(-1)

    @torch.no_grad()
    def sample_with_log_prob(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = torch.zeros(n, self.N)
        for i in range(self.N):
            logits = self._logits(s)
            p_up = torch.sigmoid(logits[:, i])
            s[:, i] = torch.where(torch.rand(n) < p_up, 1.0, -1.0)
        return s, self.log_prob(s)


def build_model(kind: str, N: int, H: int) -> nn.Module:
    if kind == "fvsbn":
        return BinaryFVSBN(N)
    if kind == "made":
        return BinaryMADE(N, H)
    raise ValueError(f"unknown model kind: {kind}")


# =====================================================================
# Training + evaluation
# =====================================================================


def train(
    model: nn.Module,
    s_data: torch.Tensor,
    gamma: float,
    n_steps: int,
    lr: float,
) -> float:
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=gamma)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_steps)
    for _ in range(n_steps):
        optim.zero_grad()
        nll = -model.log_prob(s_data).mean()
        nll.backward()
        optim.step()
        sched.step()
    return float(nll.item())


@torch.no_grad()
def reverse_kl(model: nn.Module, beta: float, L: int, log_Z: float, n_mc: int) -> dict:
    s, log_q = model.sample_with_log_prob(n_mc)
    log_p = ising_log_prob(s, beta, L, log_Z)
    kl = log_q - log_p
    N = L * L
    m = s.mean(dim=-1)
    ij = np.indices((L, L)).sum(axis=0) % 2
    stag = torch.tensor((1 - 2 * ij).astype(np.float32).ravel())
    m_stag = (s * stag).mean(dim=-1)
    return {
        "rev_kl_total": float(kl.mean().item()),
        "rev_kl_per_N": float(kl.mean().item()) / N,
        "rev_kl_sem_per_N": float(kl.std().item()) / (np.sqrt(n_mc) * N),
        "abs_m": float(m.abs().mean().item()),
        "m_sq": float((m ** 2).mean().item()),
        "abs_m_stag": float(m_stag.abs().mean().item()),
        "energy_per_N": float(ising_energy(s, L).mean().item()) / N,
    }


# =====================================================================
# Driver
# =====================================================================


GAMMA_GRID = np.concatenate([
    np.array([0.0]),
    np.geomspace(1e-4, 1.0, 30),
    np.geomspace(1.2, 50.0, 15),
])


def run_sweep(
    L: int, beta: float, K: int, kind: str, H: int,
    seed: int, n_steps: int, lr: float, n_mc: int,
    gammas: np.ndarray, n_mh_sweeps: int,
) -> dict:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    N = L * L
    log_Z = ising_log_Z(beta, L)

    if N <= 20:
        s_data = exact_sample(beta, L, K, rng=rng)
    else:
        s_data = metropolis_sample(beta, L, K, n_sweeps=n_mh_sweeps, rng=rng)

    E_data = float(ising_energy(s_data, L).mean().item()) / N
    m_data = float(s_data.mean(dim=-1).abs().mean().item())
    print(
        f"L={L} beta={beta} K={K} kind={kind} H={H if kind == 'made' else '-'} "
        f"seed={seed}"
    )
    print(f"  log Z = {log_Z:.4f}  per-N: {log_Z/N:.4f}")
    print(f"  data  <E>/N = {E_data:+.3f}  <|m|> = {m_data:.3f}")

    results = {
        "L": L, "N": N, "beta": beta, "K": K, "kind": kind, "H_hidden": H,
        "seed": seed, "log_Z": log_Z, "E_data": E_data, "m_data": m_data,
        "gammas": [], "rev_kl_per_N": [], "rev_kl_sem_per_N": [],
        "abs_m": [], "m_sq": [], "abs_m_stag": [], "energy_per_N": [],
        "final_nll": [],
    }

    for i, g in enumerate(gammas):
        t0 = time.time()
        model = build_model(kind, N, H)
        nll = train(model, s_data, float(g), n_steps=n_steps, lr=lr)
        stats = reverse_kl(model, beta, L, log_Z, n_mc=n_mc)
        dt = time.time() - t0
        print(
            f"  [{i+1:2d}/{len(gammas)}] gamma={g:.4f}  "
            f"rev_kl/N={stats['rev_kl_per_N']:+.4f}  "
            f"<|m|>={stats['abs_m']:.3f}  "
            f"nll={nll:.3f}  ({dt:.1f}s)"
        )
        results["gammas"].append(float(g))
        for k in ("rev_kl_per_N", "rev_kl_sem_per_N", "abs_m", "m_sq",
                  "abs_m_stag", "energy_per_N"):
            results[k].append(stats[k])
        results["final_nll"].append(nll)

    return results


def aggregate(runs: list[dict]) -> dict:
    gammas = np.array(runs[0]["gammas"])
    kl = np.array([r["rev_kl_per_N"] for r in runs])
    m = np.array([r["abs_m"] for r in runs])
    m_stag = np.array([r["abs_m_stag"] for r in runs])
    nll = np.array([r["final_nll"] for r in runs])
    return {
        **{k: runs[0][k] for k in ("L", "N", "beta", "K", "kind", "H_hidden", "log_Z")},
        "seeds": [r["seed"] for r in runs],
        "gammas": gammas.tolist(),
        "rev_kl_per_N_mean": kl.mean(0).tolist(),
        "rev_kl_per_N_sem": (kl.std(0, ddof=1) / np.sqrt(kl.shape[0])).tolist(),
        "rev_kl_per_N_all": kl.tolist(),
        "abs_m_mean": m.mean(0).tolist(),
        "abs_m_stag_mean": m_stag.mean(0).tolist(),
        "final_nll_mean": nll.mean(0).tolist(),
    }


def plot(agg: dict, outdir: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    g = np.array(agg["gammas"])
    g_plot = np.where(g > 0, g, g[g > 0].min() / 10)
    kl_mean = np.array(agg["rev_kl_per_N_mean"])
    kl_sem = np.array(agg["rev_kl_per_N_sem"])
    kl_all = np.array(agg["rev_kl_per_N_all"])

    for row in kl_all:
        axes[0].plot(g_plot, row, color="gray", alpha=0.25, lw=0.6)
    axes[0].errorbar(g_plot, kl_mean, yerr=kl_sem, fmt="o-", ms=3, capsize=2, color="C0")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$\gamma$ (L2 penalty)")
    axes[0].set_ylabel(r"$D_{KL}(q \| p_{\rm Ising})/N$")
    axes[0].set_title("Typical reverse KL per spin")

    axes[1].plot(g_plot, agg["abs_m_mean"], "o-", ms=3, label=r"$\langle |m| \rangle$")
    axes[1].plot(g_plot, agg["abs_m_stag_mean"], "s-", ms=3, label=r"$\langle |m_{\rm stag}| \rangle$")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$\gamma$")
    axes[1].set_title("Order parameters (seed-avg)")
    axes[1].legend(fontsize=9)

    axes[2].plot(g_plot, agg["final_nll_mean"], "o-", ms=3)
    axes[2].set_xscale("log")
    axes[2].set_xlabel(r"$\gamma$")
    axes[2].set_ylabel("training NLL")
    axes[2].set_title("Final training NLL (seed-avg)")

    tag = f"{agg['kind']}_L{agg['L']}_beta{agg['beta']}_K{agg['K']}_seeds{len(agg['seeds'])}"
    fig.suptitle(tag, fontsize=10)
    fig.tight_layout()
    out = outdir / f"ising2d_van_{tag}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  plot -> {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--model", choices=["fvsbn", "made"], default="fvsbn")
    p.add_argument("--H", type=int, default=32)
    p.add_argument("--n-steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--n-mc", type=int, default=20_000)
    p.add_argument("--n-mh-sweeps", type=int, default=3000)
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--figdir", type=str, default="figures")
    p.add_argument("--tag-suffix", type=str, default="")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    runs = [
        run_sweep(
            L=args.L, beta=args.beta, K=args.K, kind=args.model, H=args.H,
            seed=seed, n_steps=args.n_steps, lr=args.lr, n_mc=args.n_mc,
            gammas=GAMMA_GRID, n_mh_sweeps=args.n_mh_sweeps,
        )
        for seed in args.seeds
    ]

    suffix = f"_{args.tag_suffix}" if args.tag_suffix else ""
    if len(runs) == 1:
        tag = f"{args.model}_L{args.L}_beta{args.beta}_K{args.K}_seed{args.seeds[0]}{suffix}"
        with open(outdir / f"{tag}.json", "w") as f:
            json.dump(runs[0], f, indent=2)
        print(f"  saved -> {outdir / (tag + '.json')}")
        agg = aggregate(runs)
        try:
            plot(agg, figdir)
        except Exception as e:
            print(f"  plot failed: {e}", file=sys.stderr)
    else:
        agg = aggregate(runs)
        tag = f"{args.model}_L{args.L}_beta{args.beta}_K{args.K}_seeds{len(runs)}{suffix}"
        with open(outdir / f"{tag}.json", "w") as f:
            json.dump(agg, f, indent=2)
        print(f"  saved -> {outdir / (tag + '.json')}")
        try:
            plot(agg, figdir)
        except Exception as e:
            print(f"  plot failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
