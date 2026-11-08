# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy>=2.3", "scipy>=1.16", "threadpoolctl>=3.6"]
# ///
"""Metropolis-adjusted HMC cross-check and bridge for the eta=10 BM curve."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t
from threadpoolctl import threadpool_limits

from gamma import METRICS, MODEL_VERSION, state_space
from gamma_cold import direct_metrics, map_transform
from posterior import diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=768)
    parser.add_argument("--chains", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026110323)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gammas", type=float, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    n, omega, eta = 12, 3.0, 10.0
    space = state_space(n, omega)
    f = space["features"]
    dim = f.shape[1]
    gammas = np.array(args.gammas if args.gammas else np.concatenate([np.logspace(-2.2, -0.7, 16), [1.0, 10.0]]))
    ng, nc = gammas.size, args.chains
    rng = np.random.default_rng(args.seed)
    start = time.monotonic()
    maps, transforms, residuals = [], [], []
    with threadpool_limits(limits=args.threads):
        for gamma in gammas:
            jm, transform, _, residual = map_transform(space, n, gamma, eta)
            maps.append(jm)
            transforms.append(transform)
            residuals.append(residual)
        maps = np.array(maps)
        transforms = np.array(transforms)
        z = 2.0 * rng.normal(size=(ng, nc, dim))
        step = np.full((ng, 1, 1), 0.25)

        def evaluate(value):
            coupling = maps[:, None, None] + np.einsum("gci,gij->gcj", value, transforms, optimize=True)
            logits = coupling.reshape(-1, dim) @ f.T
            logz = logsumexp(logits, axis=-1)
            logp = logits - logz[:, None]
            p = np.exp(logp)
            mean = (p @ f).reshape(ng, nc, dim)
            energy = eta * (logz.reshape(ng, nc) - coupling.sum(axis=-1)
                            + n * gammas[:, None] / 2.0 * np.sum(coupling**2, axis=-1))
            gradj = eta * (mean - 1.0 + n * gammas[:, None, None] * coupling)
            gradz = np.einsum("gci,gij->gcj", gradj, transforms, optimize=True)
            return energy, gradz, logp, p

        energy, gradient, logp, probability = evaluate(z)
        traces = np.empty((ng, nc, args.draws, dim + 3))
        metric_traces = np.full((ng, nc, args.draws, len(METRICS)), np.nan)
        accepted = np.zeros((ng, nc))
        divergent = np.zeros((ng, nc), dtype=int)
        adaptation = np.zeros((ng, nc))
        for iteration in range(args.warmup + args.draws):
            epsilon = step * rng.uniform(0.85, 1.15)
            leaps = int(rng.integers(8, 17))
            momentum = rng.normal(size=z.shape)
            candidate = z.copy()
            pnew = momentum - epsilon / 2.0 * gradient
            for leap in range(leaps):
                candidate += epsilon * pnew
                e_new, g_new, lp_new, prob_new = evaluate(candidate)
                pnew -= epsilon * (0.5 if leap == leaps - 1 else 1.0) * g_new
            error = e_new - energy + np.sum(pnew**2 - momentum**2, axis=-1) / 2.0
            alpha = np.exp(np.minimum(-error, 0))
            alpha = np.where(np.isfinite(alpha), alpha, 0.0)
            accept = rng.random(size=alpha.shape) < alpha
            z = np.where(accept[..., None], candidate, z)
            gradient = np.where(accept[..., None], g_new, gradient)
            energy = np.where(accept, e_new, energy)
            flat_accept = accept.reshape(-1, 1)
            logp = np.where(flat_accept, lp_new, logp)
            probability = np.where(flat_accept, prob_new, probability)
            if iteration < args.warmup:
                adaptation += alpha
                if (iteration + 1) % 25 == 0:
                    rate = adaptation.mean(axis=1) / 25.0
                    step *= np.exp((rate - 0.8)[:, None, None] / np.sqrt(1 + iteration / 100.0))
                    step = np.clip(step, 0.005, 1.0)
                    adaptation.fill(0.0)
            else:
                draw = iteration - args.warmup
                metrics = direct_metrics(logp, probability, n, omega, space).reshape(ng, nc, -1)
                metric_traces[:, :, draw] = metrics
                traces[:, :, draw, :dim] = z
                traces[:, :, draw, dim] = energy
                traces[:, :, draw, dim + 1] = metrics[:, :, 0]
                traces[:, :, draw, dim + 2] = metrics[:, :, 4]
                accepted += accept
                divergent += (~np.isfinite(error)) | (np.abs(error) > 1000)
            if (iteration + 1) % 256 == 0:
                print(json.dumps({"iteration": iteration + 1, "seconds": time.monotonic() - start,
                                  "step_min": float(step.min()), "step_max": float(step.max())}), flush=True)
        cells = []
        rhats, esses = [], []
        for ig, gamma in enumerate(gammas):
            rhat, ess = diagnostics(traces[ig])
            rhats.append(rhat)
            esses.append(ess)
            flat = metric_traces[ig].reshape(-1, len(METRICS))
            summary = {}
            for im, metric in enumerate(METRICS):
                if np.isfinite(flat[:, im]).all():
                    chain_means = metric_traces[ig, :, :, im].mean(axis=1)
                    mean = float(chain_means.mean())
                    se = float(chain_means.std(ddof=1) / np.sqrt(nc))
                    critical = float(t.ppf(0.975, nc - 1))
                    summary[metric] = {"mean": mean, "mc_se": se,
                                       "mc_ci95": [mean - critical * se, mean + critical * se]}
            cell = {
                "eta": eta, "gamma": float(gamma), "statistics": summary,
                "max_rank_split_rhat": float(rhat.max()), "min_bulk_ess": float(ess.min()),
                "reverse_bulk_ess": float(ess[dim + 1]),
                "acceptance": accepted[ig].tolist(),
                "acceptance_rate": float(accepted[ig].sum() / (nc * args.draws)),
                "divergences": int(divergent[ig].sum()), "step_size": float(step[ig, 0, 0]),
                "map_residual": residuals[ig],
                "half_difference_reverse": float(metric_traces[ig, :, args.draws // 2:, 0].mean()
                                                   - metric_traces[ig, :, :args.draws // 2, 0].mean()),
            }
            cells.append(cell)
            print(json.dumps(cell), flush=True)
    raw_path = args.output / "gamma_cold_hmc_draws.npz"
    np.savez_compressed(raw_path, whitened_trace=traces, metric_trace=metric_traces,
                        map_coupling=maps, covariance_sqrt=transforms, gamma=gammas, eta=eta,
                        metric_names=np.array(METRICS), rank_split_rhat=rhats, bulk_ess=esses)
    summary = {
        "model_version": MODEL_VERSION, "method": "Metropolis-adjusted HMC in MAP-Hessian coordinates",
        "N": n, "K": 1, "omega": omega, "eta": eta, "seed": args.seed,
        "draws_per_chain": args.draws, "warmup": args.warmup, "chains": nc,
        "posterior_assumption": "exp(-eta*(logZ-data_energy+N*gamma/2*sum_unique_J_squared))",
        "data_average": "Exact population average over every K1 teacher dataset",
        "diagnostic_gate": "max rank/folded split Rhat<=1.05, min bulk ESS>=100, zero retained divergences",
        "uncertainty": "SE across independent chain means; mc_ci95 uses Student-t with chains-1 degrees of freedom",
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime_seconds": time.monotonic() - start, "cells": cells,
    }
    (args.output / "gamma_cold_hmc_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
