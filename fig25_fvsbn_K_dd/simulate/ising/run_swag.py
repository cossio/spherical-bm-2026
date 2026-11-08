#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "torch>=2.6",
#     "numpy>=2.4",
#     "matplotlib>=3.10",
# ]
# ///
"""MAP + SWAG posterior of a binary VAN on a 2D Ising target, vs. L2 reg.

Mirrors ../hopfield_van/run_swag.py so the MAP vs SWAG comparison uses
the same knobs on both targets.  Reuses the teacher and student code in
run.py (same directory).

Pipeline per (seed, gamma):
  1. Adam burn-in with weight_decay = gamma on K teacher samples;
     record MAP reverse KL.
  2. Constant-LR SGD from that MAP, collecting snapshots every --collect-every
     steps.  Optional Langevin noise inflates the posterior width.
  3. Fit a diagonal Gaussian to the snapshots (SWAG-diag).
  4. Draw N_SWAG samples, evaluate reverse KL per sample, record mean/std.

Usage:
    ./run_swag.py --L 4 --beta 0.44 --K 4 --seeds 0 1 2 --langevin-temp 0.3
    ./run_swag.py --L 4 --beta 0.44 --K 4 --seeds $(seq 0 19) --langevin-temp 0.3
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

sys.path.insert(0, str(Path(__file__).parent))
from run import (  # noqa: E402
    BinaryFVSBN, BinaryMADE,
    exact_sample, metropolis_sample,
    ising_energy, ising_log_Z, ising_log_prob,
)


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


def build_model(kind: str, N: int, H: int) -> torch.nn.Module:
    if kind == "fvsbn":
        return BinaryFVSBN(N)
    if kind == "made":
        return BinaryMADE(N, H)
    raise ValueError(f"unknown model kind: {kind}")


def l2_of(model: torch.nn.Module) -> torch.Tensor:
    return sum(p.pow(2).sum() for p in model.parameters())


def burn_in(model, s_data, gamma, steps, lr, implicit_l2=False):
    """Adam burn-in with L2 regularization at strength ``gamma``.

    With ``implicit_l2=False`` (default): plain ``weight_decay=gamma`` on
    Adam, which adds ``gamma*theta`` to the gradient. Stable at moderate
    ``gamma`` because Adam normalizes by the second moment.

    With ``implicit_l2=True``: the L2 part is integrated exactly each step
    via a decoupled multiplicative half-step ``theta -> exp(-0.5*lr_eff*gamma)*theta``
    on either side of Adam's data-only update (Strang split). Here
    ``lr_eff`` follows the cosine schedule used by Adam, so the two
    contributions stay in lock-step. Unconditionally stable in
    ``lr*gamma``, which is required when ``gamma`` is large enough that
    the explicit decay would drive parameters to zero in one step.
    """
    if not implicit_l2:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=gamma)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for _ in range(steps):
            opt.zero_grad()
            (-model.log_prob(s_data).mean()).backward()
            opt.step()
            sched.step()
        return
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        lr_now = opt.param_groups[0]["lr"]
        decay_half = float(np.exp(-0.5 * lr_now * gamma))
        with torch.no_grad():
            for p in model.parameters():
                p.mul_(decay_half)
        opt.zero_grad()
        (-model.log_prob(s_data).mean()).backward()
        opt.step()
        with torch.no_grad():
            for p in model.parameters():
                p.mul_(decay_half)
        sched.step()


def collect_swag(model, s_data, gamma, steps, lr, every, langevin_temp=0.0,
                 implicit_l2=False):
    """SGD-with-Langevin SWAG collection.

    ``implicit_l2=True`` integrates the L2 piece via a Strang split,
    yielding an OU contraction factor ``exp(-0.5*lr*gamma)`` on either
    side of the data-likelihood gradient step. Required when
    ``lr*gamma >= 1``, which kills parameters under the explicit-Euler
    branch.
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
def reverse_kl(model, beta, L, log_Z, n_mc):
    s, log_q = model.sample_with_log_prob(n_mc)
    log_p = ising_log_prob(s, beta, L, log_Z)
    kl = log_q - log_p
    N = L * L
    return {
        "kl_per_N": float(kl.mean().item()) / N,
        "abs_m": float(s.mean(dim=-1).abs().mean().item()),
        "m2": float((s.mean(dim=-1) ** 2).mean().item()),
    }


def run_one_seed(
    L, beta, K, gammas, kind, H, seed,
    burn_steps, sgd_steps, lr_adam, lr_sgd, collect_every, n_swag,
    langevin_temp, swag_scale, n_mc, n_mh_sweeps, implicit_l2=False,
):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    N = L * L
    log_Z = ising_log_Z(beta, L)
    if N <= 20:
        s_data = exact_sample(beta, L, K, rng=rng)
    else:
        s_data = metropolis_sample(beta, L, K, n_sweeps=n_mh_sweeps, rng=rng)
    E_data = float(ising_energy(s_data, L).mean().item()) / N

    print(f"  seed={seed}  logZ/N={log_Z/N:.4f}  <E>/N_data={E_data:+.3f}")

    res = {
        "seed": seed, "gammas": [], "kl_map_per_N": [],
        "kl_swag_mean_per_N": [], "kl_swag_std_per_N": [],
        "abs_m_map": [], "abs_m_swag": [],
        "m2_map": [], "m2_swag": [],
    }

    for i, g in enumerate(gammas):
        t0 = time.time()
        torch.manual_seed(seed * 10_000 + i)
        model = build_model(kind, N, H)
        burn_in(model, s_data, float(g), burn_steps, lr_adam,
                implicit_l2=implicit_l2)
        map_stats = reverse_kl(model, beta, L, log_Z, n_mc)

        mean, sq_mean = collect_swag(
            model, s_data, float(g), sgd_steps, lr_sgd,
            collect_every, langevin_temp=langevin_temp,
            implicit_l2=implicit_l2,
        )

        kls, ms, m2s = [], [], []
        for _ in range(n_swag):
            sample_from_swag(model, mean, sq_mean, scale=swag_scale)
            st = reverse_kl(model, beta, L, log_Z, n_mc)
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
    axes[0].set_ylabel(r"$D_{KL}(q \| p_{\rm Ising})/N$")
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
        f"swag_{cfg['kind']}_L{cfg['L']}_beta{cfg['beta']}_K{cfg['K']}"
        f"_T{cfg['langevin_temp']}_s{cfg['swag_scale']}"
        f"_seeds{len(cfg['seeds'])}"
    )
    fig.suptitle(tag, fontsize=10)
    fig.tight_layout()
    out = outdir / f"ising2d_{tag}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  plot -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.44)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--model", choices=["fvsbn", "made"], default="fvsbn")
    p.add_argument("--H", type=int, default=32)
    p.add_argument("--burn-in", type=int, default=BURN_IN)
    p.add_argument("--sgd-steps", type=int, default=SGD_STEPS)
    p.add_argument("--lr-adam", type=float, default=LR_ADAM)
    p.add_argument("--lr-sgd", type=float, default=LR_SGD)
    p.add_argument("--collect-every", type=int, default=COLLECT_EVERY)
    p.add_argument("--n-swag", type=int, default=N_SWAG_SAMPLES)
    p.add_argument("--n-mc", type=int, default=MC_SAMPLES)
    p.add_argument("--n-mh-sweeps", type=int, default=3000)
    p.add_argument("--langevin-temp", type=float, default=0.3)
    p.add_argument("--swag-scale", type=float, default=1.0)
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--figdir", type=str, default="figures")
    p.add_argument("--tag-suffix", type=str, default="")
    args = p.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    figdir = Path(args.figdir); figdir.mkdir(parents=True, exist_ok=True)

    print(
        f"L={args.L} beta={args.beta} K={args.K} model={args.model} "
        f"T_lang={args.langevin_temp} swag_scale={args.swag_scale} "
        f"seeds={args.seeds}"
    )

    runs = []
    for seed in args.seeds:
        r = run_one_seed(
            L=args.L, beta=args.beta, K=args.K, gammas=GAMMA_GRID,
            kind=args.model, H=args.H, seed=seed,
            burn_steps=args.burn_in, sgd_steps=args.sgd_steps,
            lr_adam=args.lr_adam, lr_sgd=args.lr_sgd,
            collect_every=args.collect_every, n_swag=args.n_swag,
            langevin_temp=args.langevin_temp, swag_scale=args.swag_scale,
            n_mc=args.n_mc, n_mh_sweeps=args.n_mh_sweeps,
        )
        runs.append(r)

    agg = _aggregate(runs)
    cfg = {
        "L": args.L, "N": args.L * args.L, "beta": args.beta, "K": args.K,
        "kind": args.model, "H_hidden": args.H, "seeds": args.seeds,
        "langevin_temp": args.langevin_temp, "swag_scale": args.swag_scale,
        "burn_in": args.burn_in, "sgd_steps": args.sgd_steps,
        "lr_adam": args.lr_adam, "lr_sgd": args.lr_sgd,
        "collect_every": args.collect_every, "n_swag": args.n_swag,
    }

    suffix = f"_{args.tag_suffix}" if args.tag_suffix else ""
    tag = (
        f"swag_{args.model}_L{args.L}_beta{args.beta}_K{args.K}"
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
