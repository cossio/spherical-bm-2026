# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy>=2.3", "scipy>=1.16", "threadpoolctl>=3.6"]
# ///
"""Direct-posterior Laplace importance check of the cold K=1 BM curve.

This independent proposal is useful precisely where prior-orbit importance
sampling loses ESS. Its validity is checked by ESS and agreement with adjusted
HMC, not assumed from the local Gaussian construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from gamma import METRICS, MODEL_VERSION, state_space, weighted_summary


def map_transform(space, n, gamma, eta):
    f = space["features"]
    fs = f.sum(axis=1)

    def stationarity(j):
        p = np.exp(j * fs - logsumexp(j * fs))
        return float(p @ f[:, 0] - 1 + n * gamma * j)

    jmap = brentq(stationarity, 0.0, 1.0 / (n * gamma), xtol=1e-14)
    p = np.exp(jmap * fs - logsumexp(jmap * fs))
    mean = p @ f
    hess = (f.T * p) @ f - np.outer(mean, mean) + n * gamma * np.eye(f.shape[1])
    eigen, vectors = np.linalg.eigh(eta * hess)
    transform = (vectors * eigen**-0.5) @ vectors.T
    return jmap, transform, float(np.log(eigen).sum()), abs(stationarity(jmap))


def direct_metrics(logp, p, n, omega, space):
    mu = p @ space["features"]
    mean_m2 = n + 2.0 * mu.sum(axis=-1)
    entropy_full = -np.sum(p * logp, axis=-1) + np.log(2.0)
    energy = omega / (2.0 * n) * (n + space["teacher_pair_correlation"] * (mean_m2 - n))
    reverse = -entropy_full + space["teacher_logz"] - energy
    values = np.full((p.shape[0], len(METRICS)), np.nan)
    values[:, 0] = reverse
    values[:, 3] = p[:, 0]
    values[:, 4] = mean_m2 / n**2
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=2026110313)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--gammas", type=float, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    n, omega, eta = 12, 3.0, 10.0
    space = state_space(n, omega)
    f = space["features"]
    d = f.shape[1]
    gammas = np.array(args.gammas if args.gammas else np.logspace(-3, 2, 51))
    rng = np.random.default_rng(args.seed)
    normals = rng.normal(size=(args.draws, d))
    cells, maps, transforms = [], [], []
    metrics = np.full((gammas.size, args.draws, len(METRICS)), np.nan)
    logweights = np.empty((gammas.size, args.draws))
    start = time.monotonic()
    with threadpool_limits(limits=args.threads):
        for ig, gamma in enumerate(gammas):
            jmap, transform, logdet, residual = map_transform(space, n, gamma, eta)
            coupling = jmap + normals @ transform
            logits = coupling @ f.T
            logz = logsumexp(logits, axis=-1)
            logp = logits - logz[:, None]
            prob = np.exp(logp)
            logprior = d / 2.0 * np.log(eta * n * gamma / (2.0 * np.pi)) - eta * n * gamma / 2.0 * np.sum(coupling**2, axis=-1)
            logproposal = logdet / 2.0 - d / 2.0 * np.log(2.0 * np.pi) - np.sum(normals**2, axis=-1) / 2.0
            lw = logprior + eta * logp[:, 0] - logproposal
            values = direct_metrics(logp, prob, n, omega, space)
            metrics[ig], logweights[ig] = values, lw
            summary = weighted_summary(values, lw, args.blocks)
            cells.append({"eta": eta, "gamma": float(gamma), "map_stationarity_residual": residual, **summary})
            maps.append(jmap)
            transforms.append(transform)
            print(json.dumps({"gamma": float(gamma), "reverse": summary["statistics"]["reverse_total"]["mean"],
                              "ess": summary["importance_ess"], "max_weight": summary["maximum_normalized_weight"],
                              "seconds": time.monotonic() - start}), flush=True)
    path = args.output / "gamma_cold_draws.npz"
    np.savez_compressed(path, standard_normals=normals, map_coupling=maps,
                        covariance_sqrt=transforms, gamma=gammas, eta=eta,
                        metrics=metrics, metric_names=np.array(METRICS), log_importance_weights=logweights)
    result = {
        "model_version": MODEL_VERSION, "method": "Direct Laplace Gaussian importance proposal for the K1 posterior",
        "assumptions": "Same posterior coefficient N*gamma/2 and data-population average as gamma.py",
        "N": n, "K": 1, "omega": omega, "eta": eta, "seed": args.seed, "draws": args.draws,
        "status": "Requires importance ESS and independent adjusted-HMC cross-check before interpretation",
        "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime_seconds": time.monotonic() - start, "cells": cells,
    }
    (args.output / "gamma_cold_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
