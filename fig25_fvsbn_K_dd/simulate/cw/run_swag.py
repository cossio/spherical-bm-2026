#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "torch>=2.6",
#     "numpy>=2.4",
#     "matplotlib>=3.10",
# ]
# ///
"""MAP + SWAG posterior of an FVSBN student on a Curie-Weiss teacher.

Load-bearing bit: the SWAG protocol (burn-in, SGD collection, Langevin
noise, diagonal Gaussian fit, PP draws) is copied verbatim from
``double_descent/ising2d_van/run_swag.py``.  Only the teacher is swapped
from 2D Ising to rank-1 Hopfield / Curie-Weiss.

Pipeline per (seed, gamma):
  1. Adam burn-in with weight_decay = gamma on K teacher samples;
     record MAP reverse KL.
  2. Constant-LR SGD from that MAP, collecting snapshots every
     ``--collect-every`` steps.  Optional Langevin noise inflates width.
  3. Fit a diagonal Gaussian to the snapshots (SWAG-diag).
  4. Draw N_SWAG samples, evaluate reverse KL per sample, record stats.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from fvsbn import BinaryFVSBN  # noqa: E402
from teacher import (  # noqa: E402
    hopfield_energy,
    hopfield_log_Z,
    hopfield_log_prob,
    sample_teacher,
)


# Match ising2d_van defaults exactly.
GAMMA_GRID = np.concatenate([
    np.array([0.0]),
    np.geomspace(1e-4, 1.0, 24),
    np.geomspace(1.2, 50.0, 12),
])

BURN_IN = 1500
SGD_STEPS = 1500
COLLECT_EVERY = 15
N_SWAG_SAMPLES = 30
LR_ADAM = 0.02
LR_SGD = 0.005
MC_SAMPLES = 4000


def l2_of(model):
    return sum(p.pow(2).sum() for p in model.parameters())


def burn_in(model, s_data, gamma, steps, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=gamma)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        opt.zero_grad()
        (-model.log_prob(s_data).mean()).backward()
        opt.step()
        sched.step()


def collect_swag(model, s_data, gamma, steps, lr, every, langevin_temp=0.0,
                 implicit_l2=False):
    """SGD-with-Langevin SWAG collection.

    With ``implicit_l2=False`` (default): plain explicit Euler on the L2
    penalty as ``loss += 0.5 * gamma * ||theta||^2``. Numerically unstable
    when ``lr * gamma > 2`` (parameters oscillate and blow up).

    With ``implicit_l2=True``: a Strang-split update where each step is
    ``theta -> exp(-0.5*lr*gamma)*theta``, then a full explicit gradient
    + Langevin step on the data loss, then a second
    ``exp(-0.5*lr*gamma)*theta`` half-step. The L2 part is integrated
    exactly (it is just an OU contraction), so the scheme is uncondition-
    ally stable in lr*gamma and reduces to the explicit scheme to O(lr).
    """
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    mean = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    sq_mean = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    n_col = 0
    decay_half = float(np.exp(-0.5 * lr * gamma)) if implicit_l2 else None
    for step in range(steps):
        if implicit_l2:
            with torch.no_grad():
                for p in model.parameters():
                    p.mul_(decay_half)
        opt.zero_grad()
        if implicit_l2:
            loss = -model.log_prob(s_data).mean()
        else:
            loss = -model.log_prob(s_data).mean() + 0.5 * gamma * l2_of(model)
        loss.backward()
        opt.step()
        if langevin_temp > 0:
            scale = (2.0 * lr * langevin_temp) ** 0.5
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(scale * torch.randn_like(p))
        if implicit_l2:
            with torch.no_grad():
                for p in model.parameters():
                    p.mul_(decay_half)
        if step % every == 0:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    mean[n] += p.data
                    sq_mean[n] += p.data ** 2
            n_col += 1
    for n in mean:
        mean[n] /= n_col
        sq_mean[n] /= n_col
    return mean, sq_mean


def sample_from_swag(model, mean, sq_mean, scale=1.0):
    with torch.no_grad():
        for n, p in model.named_parameters():
            var = (sq_mean[n] - mean[n] ** 2).clamp(min=1e-12)
            p.data.copy_(mean[n] + scale * var.sqrt() * torch.randn_like(p))


@torch.no_grad()
def reverse_kl(model, beta, xi, log_Z, n_mc):
    s, log_q = model.sample_with_log_prob(n_mc)
    log_p = hopfield_log_prob(s, beta, xi, log_Z)
    kl = log_q - log_p
    N = xi.shape[0]
    m = (s @ xi) / N
    return {
        "kl_per_N": float(kl.mean().item()) / N,
        "abs_m": float(m.abs().mean().item()),
        "m2": float((m ** 2).mean().item()),
    }


def enumerate_configs(N: int) -> torch.Tensor:
    """All $2^N$ binary configurations in ±1, as a (2^N, N) float32 tensor."""
    idx = torch.arange(2 ** N, dtype=torch.int64)
    bits = ((idx[:, None] >> torch.arange(N, dtype=torch.int64)) & 1).to(torch.float32)
    return 2.0 * bits - 1.0


@torch.no_grad()
def reverse_kl_exact(model, beta, xi, log_Z, configs):
    """Noise-free reverse KL by enumeration over 2^N configs.

    Returns the same dict as ``reverse_kl`` but with zero estimator
    variance; only the student parameters' own variation remains.
    """
    N = xi.shape[0]
    log_q = model.log_prob(configs)
    log_p = hopfield_log_prob(configs, beta, xi, log_Z)
    q = log_q.exp()
    kl = (q * (log_q - log_p)).sum()
    m = (configs @ xi) / N
    abs_m = (q * m.abs()).sum()
    m2 = (q * (m ** 2)).sum()
    return {
        "kl_per_N": float(kl.item()) / N,
        "abs_m": float(abs_m.item()),
        "m2": float(m2.item()),
    }


def run_one_seed(
    N, beta, K, gammas, seed,
    burn_steps, sgd_steps, lr_adam, lr_sgd, collect_every, n_swag,
    langevin_temp, swag_scale, n_mc, configs=None, common_rng=False,
    implicit_l2=False, gamma_convention="bare",
):
    """``gamma_convention``: 'bare' interprets each γ as the literal
    PyTorch ``weight_decay`` (loss penalty 0.5*γ*Σ W²); 'paper' matches
    the paper's prior $P(W)\\propto e^{-N\\gamma\\,\\mathrm{Tr}\\,W^2/4}$
    (loss penalty (N/4)*γ*Σ W²). The mapping is γ_bare = (N/2)*γ_paper.
    JSON outputs always store the user-facing (input) γ values; the
    internal training uses the bare-equivalent.
    """
    # Paper prior P(W) ∝ exp(-Nγ Tr W²/4) ⇒ per-sample MAP loss has
    # (Nγ/4K)·||W||² ⇒ PyTorch weight_decay = N·γ_paper/(2K).
    gamma_scale = (N / (2.0 * K)) if gamma_convention == "paper" else 1.0
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    xi_np = rng.choice([-1, 1], size=N).astype(np.float32)
    xi = torch.tensor(xi_np)
    log_Z = hopfield_log_Z(beta, N)
    s_data = sample_teacher(beta, xi, K, rng=rng)
    E_data = float(hopfield_energy(s_data, xi).mean().item()) / N
    m_data = float(((s_data @ xi) / N).abs().mean().item())
    estimator = "exact" if configs is not None else "MC"
    print(
        f"  seed={seed}  logZ/N={log_Z/N:.4f}  "
        f"<E>/N_data={E_data:+.3f}  <|m|>={m_data:.3f}  estimator={estimator}"
        f"  common_rng={common_rng}"
    )

    def _kl(model):
        if configs is not None:
            return reverse_kl_exact(model, beta, xi, log_Z, configs)
        return reverse_kl(model, beta, xi, log_Z, n_mc)

    res = {
        "seed": seed, "gammas": [], "kl_map_per_N": [],
        "kl_swag_mean_per_N": [], "kl_swag_std_per_N": [],
        "abs_m_map": [], "abs_m_swag": [],
        "m2_map": [], "m2_swag": [],
    }

    for i, g in enumerate(gammas):
        t0 = time.time()
        torch.manual_seed(seed * 10_000 if common_rng else seed * 10_000 + i)
        model = BinaryFVSBN(N)
        g_internal = float(g) * gamma_scale
        burn_in(model, s_data, g_internal, burn_steps, lr_adam)
        map_stats = _kl(model)

        mean, sq_mean = collect_swag(
            model, s_data, g_internal, sgd_steps, lr_sgd,
            collect_every, langevin_temp=langevin_temp,
            implicit_l2=implicit_l2,
        )

        kls, ms, m2s = [], [], []
        for _ in range(n_swag):
            sample_from_swag(model, mean, sq_mean, scale=swag_scale)
            st = _kl(model)
            kls.append(st["kl_per_N"])
            ms.append(st["abs_m"])
            m2s.append(st["m2"])
        kl_arr = np.array(kls)
        dt = time.time() - t0
        print(
            f"    [{i+1:2d}/{len(gammas)}] g={g:.4f}  "
            f"MAP={map_stats['kl_per_N']:+.4f}  "
            f"SWAG={kl_arr.mean():+.4f}±{kl_arr.std():.4f}  "
            f"m_MAP={map_stats['abs_m']:.3f} m_SWAG={np.mean(ms):.3f}  "
            f"({dt:.1f}s)"
        )
        res["gammas"].append(float(g))
        res["kl_map_per_N"].append(map_stats["kl_per_N"])
        res["abs_m_map"].append(map_stats["abs_m"])
        res["m2_map"].append(map_stats["m2"])
        res["kl_swag_mean_per_N"].append(float(kl_arr.mean()))
        res["kl_swag_std_per_N"].append(float(kl_arr.std()))
        res["abs_m_swag"].append(float(np.mean(ms)))
        res["m2_swag"].append(float(np.mean(m2s)))
    return res


def _aggregate(runs):
    g = np.array(runs[0]["gammas"])
    map_kl = np.array([r["kl_map_per_N"] for r in runs])
    swag_kl = np.array([r["kl_swag_mean_per_N"] for r in runs])
    abs_m_map = np.array([r["abs_m_map"] for r in runs])
    abs_m_swag = np.array([r["abs_m_swag"] for r in runs])
    m2_map = np.array([r["m2_map"] for r in runs])
    m2_swag = np.array([r["m2_swag"] for r in runs])
    n = map_kl.shape[0]
    return {
        "gammas": g.tolist(),
        "kl_map_mean": map_kl.mean(0).tolist(),
        "kl_map_sem": (map_kl.std(0, ddof=1) / np.sqrt(n)).tolist() if n > 1 else [0.0] * len(g),
        "kl_swag_mean": swag_kl.mean(0).tolist(),
        "kl_swag_sem": (swag_kl.std(0, ddof=1) / np.sqrt(n)).tolist() if n > 1 else [0.0] * len(g),
        "abs_m_map_mean": abs_m_map.mean(0).tolist(),
        "abs_m_swag_mean": abs_m_swag.mean(0).tolist(),
        "m2_map_mean": m2_map.mean(0).tolist(),
        "m2_swag_mean": m2_swag.mean(0).tolist(),
        "kl_map_all": map_kl.tolist(),
        "kl_swag_all": swag_kl.tolist(),
    }


def _plot(agg, cfg, outdir):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    g = np.array(agg["gammas"])
    g_plot = np.where(g > 0, g, g[g > 0].min() / 10)

    for row in agg["kl_swag_all"]:
        axes[0].plot(g_plot, row, color="tab:orange", alpha=0.2, lw=0.5)
    axes[0].errorbar(g_plot, agg["kl_map_mean"], yerr=agg["kl_map_sem"],
                     fmt="o-", ms=3, capsize=2, color="tab:blue", label="MAP")
    axes[0].errorbar(g_plot, agg["kl_swag_mean"], yerr=agg["kl_swag_sem"],
                     fmt="s-", ms=3, capsize=2, color="tab:orange", label="SWAG posterior")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$\gamma$")
    axes[0].set_ylabel(r"$D_{KL}(q \| p_{\rm CW})/N$")
    axes[0].set_title("Reverse KL per spin")
    axes[0].legend(fontsize=9)

    axes[1].plot(g_plot, agg["abs_m_map_mean"], "o-", ms=3, color="tab:blue", label="MAP")
    axes[1].plot(g_plot, agg["abs_m_swag_mean"], "s-", ms=3, color="tab:orange", label="SWAG")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$\gamma$")
    axes[1].set_ylabel(r"$\langle |m| \rangle$")
    axes[1].set_title(r"Mean $|m|$")
    axes[1].legend(fontsize=9)

    axes[2].plot(g_plot, agg["m2_map_mean"], "o-", ms=3, color="tab:blue", label="MAP")
    axes[2].plot(g_plot, agg["m2_swag_mean"], "s-", ms=3, color="tab:orange", label="SWAG")
    axes[2].set_xscale("log")
    axes[2].set_xlabel(r"$\gamma$")
    axes[2].set_ylabel(r"$\langle m^2 \rangle$")
    axes[2].set_title(r"Mean $m^2$")
    axes[2].legend(fontsize=9)

    tag = (
        f"swag_fvsbn_cw_N{cfg['N']}_beta{cfg['beta']}_K{cfg['K']}"
        f"_T{cfg['langevin_temp']}_s{cfg['swag_scale']}"
        f"_seeds{len(cfg['seeds'])}"
    )
    fig.suptitle(tag, fontsize=10)
    fig.tight_layout()
    out = outdir / f"{tag}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  plot -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--burn-in", type=int, default=BURN_IN)
    p.add_argument("--sgd-steps", type=int, default=SGD_STEPS)
    p.add_argument("--lr-adam", type=float, default=LR_ADAM)
    p.add_argument("--lr-sgd", type=float, default=LR_SGD)
    p.add_argument("--collect-every", type=int, default=COLLECT_EVERY)
    p.add_argument("--n-swag", type=int, default=N_SWAG_SAMPLES)
    p.add_argument("--n-mc", type=int, default=MC_SAMPLES)
    p.add_argument("--langevin-temp", type=float, default=0.3)
    p.add_argument("--swag-scale", type=float, default=1.0)
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--figdir", type=str, default="figures")
    p.add_argument("--tag-suffix", type=str, default="")
    p.add_argument(
        "--exact-kl", choices=["auto", "true", "false"], default="auto",
        help="Exact enumeration of 2^N configs (zero MC variance). "
        "'auto' enables for N<=20.",
    )
    p.add_argument("--gamma-min", type=float, default=None,
                   help="If set, build a dense γ-grid geomspace(gamma-min, "
                        "gamma-max, n-gamma) + {0}.")
    p.add_argument("--gamma-max", type=float, default=100.0)
    p.add_argument("--n-gamma", type=int, default=60)
    p.add_argument("--high-max", type=float, default=50.0,
                   help="Upper bound of the high-γ block (default 50). "
                        "Setting e.g. 1000 extends the standard grid.")
    p.add_argument("--n-high", type=int, default=12,
                   help="Number of points in the high-γ block (default 12).")
    p.add_argument("--implicit-l2", action="store_true",
                   help="Use a Strang-split exp(-0.5*lr*gamma) update for "
                        "the L2 part of the SWAG SGD phase. Unconditionally "
                        "stable in lr*gamma (needed when extending γ-grid "
                        "beyond 50). Off by default to preserve the "
                        "explicit-Euler protocol used elsewhere.")
    p.add_argument("--gamma-convention", choices=["bare", "paper"], default="bare",
                   help="'bare' (default): γ is the literal weight_decay, "
                        "loss penalty is 0.5*γ*ΣW². 'paper': γ matches the "
                        "paper's prior P(W)∝exp(-Nγ TrW²/4), so the loss "
                        "penalty is (N/4)*γ*ΣW² and internally we multiply "
                        "by N/2 before passing to the optimizer.")
    p.add_argument("--common-rng", action="store_true",
                   help="Common Random Numbers across γ within a seed: re-seed "
                        "with seed*10_000 (γ-independent) at every γ so SGD "
                        "Langevin noise and SWAG draws are identical across γ. "
                        "Decorrelates the γ-axis at zero compute cost; the "
                        "averaged curve becomes much smoother.")
    args = p.parse_args()

    global GAMMA_GRID
    if args.gamma_min is not None:
        GAMMA_GRID = np.concatenate([
            np.array([0.0]),
            np.geomspace(args.gamma_min, args.gamma_max, args.n_gamma),
        ])
        print(f"  dense γ-grid: {len(GAMMA_GRID)} points, "
              f"[{args.gamma_min:g}, {args.gamma_max:g}] + {{0}}")
    elif args.high_max != 50.0 or args.n_high != 12:
        GAMMA_GRID = np.concatenate([
            np.array([0.0]),
            np.geomspace(1e-4, 1.0, 24),
            np.geomspace(1.2, args.high_max, args.n_high),
        ])
        print(f"  extended γ-grid: {len(GAMMA_GRID)} points, "
              f"high block [1.2, {args.high_max:g}] with {args.n_high} pts")

    use_exact = (args.exact_kl == "true") or (
        args.exact_kl == "auto" and args.N <= 20
    )
    configs = enumerate_configs(args.N) if use_exact else None
    if use_exact:
        print(f"  exact KL: enumerating 2^{args.N} = {2**args.N} configs")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    figdir = Path(args.figdir); figdir.mkdir(parents=True, exist_ok=True)

    print(
        f"N={args.N} beta={args.beta} K={args.K} "
        f"T_lang={args.langevin_temp} swag_scale={args.swag_scale} "
        f"seeds={args.seeds}"
    )

    runs = []
    for seed in args.seeds:
        r = run_one_seed(
            N=args.N, beta=args.beta, K=args.K, gammas=GAMMA_GRID, seed=seed,
            burn_steps=args.burn_in, sgd_steps=args.sgd_steps,
            lr_adam=args.lr_adam, lr_sgd=args.lr_sgd,
            collect_every=args.collect_every, n_swag=args.n_swag,
            langevin_temp=args.langevin_temp, swag_scale=args.swag_scale,
            n_mc=args.n_mc, configs=configs, common_rng=args.common_rng,
            implicit_l2=args.implicit_l2,
            gamma_convention=args.gamma_convention,
        )
        runs.append(r)

    agg = _aggregate(runs)
    cfg = {
        "N": args.N, "beta": args.beta, "K": args.K, "seeds": args.seeds,
        "langevin_temp": args.langevin_temp, "swag_scale": args.swag_scale,
        "burn_in": args.burn_in, "sgd_steps": args.sgd_steps,
        "lr_adam": args.lr_adam, "lr_sgd": args.lr_sgd,
        "collect_every": args.collect_every, "n_swag": args.n_swag,
        "common_rng": args.common_rng,
    }

    suffix = f"_{args.tag_suffix}" if args.tag_suffix else ""
    tag = (
        f"swag_fvsbn_cw_N{args.N}_beta{args.beta}_K{args.K}"
        f"_T{args.langevin_temp}_s{args.swag_scale}"
        f"_seeds{len(args.seeds)}{suffix}"
    )
    payload = {"config": cfg, "aggregated": agg, "per_seed": runs}
    with open(outdir / f"{tag}.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  saved -> {outdir / (tag + '.json')}")
    _plot(agg, cfg, figdir)


if __name__ == "__main__":
    main()
