# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy>=2.3", "scipy>=1.16", "threadpoolctl>=3.6"]
# ///
"""Reconstruct the Appendix K=1 BM regularization experiment by gauge integration.

The assumed posterior is exp[-eta * (log Z(J) - E_J(xi)
                                    + N*gamma/2*sum_{i<j} J_ij**2)].
This is the summed-likelihood quadratic-prior convention used in the paper.
At eta=1 the gauge-orbit construction gives independent exact posterior draws;
at other eta it gives self-normalized importance estimates with reported ESS.
All states, all K=1 datasets, and all gauge orientations are summed exactly
inside each Gaussian-prior draw. Only the Gaussian matrix integral is sampled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits


MODEL_VERSION = "fully-visible-bm-v1"
METRICS = (
    "reverse_total",
    "reverse_parallel",
    "reverse_perpendicular",
    "retrieval_probability",
    "data_squared_overlap",
    "leading_mode_variance",
)


def fwht(values: np.ndarray) -> np.ndarray:
    """Unnormalized Walsh-Hadamard transform along the final, power-of-two axis."""
    result = np.array(values, dtype=np.float64, copy=True)
    nstates = result.shape[-1]
    if nstates < 1 or nstates & (nstates - 1):
        raise ValueError("FWHT dimension must be a positive power of two")
    block = 1
    while block < nstates:
        view = result.reshape(*result.shape[:-1], -1, 2, block)
        left = view[..., 0, :].copy()
        right = view[..., 1, :].copy()
        view[..., 0, :] = left + right
        view[..., 1, :] = left - right
        block *= 2
    return result


def state_space(n: int, omega: float) -> dict[str, np.ndarray | float]:
    """Use one representative per global-spin-flip pair, first spin +1."""
    codes = np.arange(1 << (n - 1), dtype=np.uint32)
    states = np.ones((codes.size, n), dtype=np.float64)
    states[:, 1:] = 1.0 - 2.0 * ((codes[:, None] >> np.arange(n - 1)) & 1)
    upper = np.triu_indices(n, 1)
    features = states[:, upper[0]] * states[:, upper[1]]
    m2 = np.square(states.sum(axis=1))
    teacher_logits = omega * m2 / (2.0 * n)
    teacher_lz_half = float(logsumexp(teacher_logits))
    teacher = np.exp(teacher_logits - teacher_lz_half)
    shell_values, shell_index = np.unique(m2, return_inverse=True)
    masks = np.array([shell_index == j for j in range(shell_values.size)], float)
    shell_probability = masks @ teacher
    entropy_teacher = float(-teacher @ np.log(teacher) + np.log(2.0))
    teacher_pair_correlation = float((teacher @ m2 - n) / (n * (n - 1)))
    uniform_reverse = float(
        teacher_lz_half + np.log(2.0) - n * np.log(2.0) - omega / 2.0
    )
    return {
        "states": states,
        "features": features,
        "m2": m2,
        "teacher": teacher,
        "teacher_hat": fwht(teacher),
        "teacher_logz": teacher_lz_half + np.log(2.0),
        "teacher_pair_correlation": teacher_pair_correlation,
        "teacher_entropy": entropy_teacher,
        "weak_gamma_reverse": entropy_teacher - np.log(2.0),
        "uniform_reverse": uniform_reverse,
        "shell_values": shell_values,
        "shell_masks": masks,
        "shell_hat": fwht(masks),
        "shell_probability": shell_probability,
        "upper_i": upper[0],
        "upper_j": upper[1],
    }


def longitudinal_average(
    probabilities: np.ndarray,
    orientation: np.ndarray,
    space: dict,
) -> np.ndarray:
    """Exact mean over teacher datasets and posterior gauge orientations."""
    # Keep the FWHT butterfly working set in cache. A single transform across
    # thousands of rows otherwise spends most of its time copying large arrays.
    if probabilities.shape[0] > 128:
        same_orientation = orientation is probabilities
        pieces = []
        for start in range(0, probabilities.shape[0], 128):
            pblock = probabilities[start:start + 128]
            rblock = pblock if same_orientation else orientation[start:start + 128]
            pieces.append(longitudinal_average(pblock, rblock, space))
        return np.concatenate(pieces)
    nstates = probabilities.shape[-1]
    phat = fwht(probabilities)
    rhat = phat if orientation is probabilities else fwht(orientation)
    gauge_weights = fwht(rhat * space["teacher_hat"]) / nstates
    gauge_weights = np.clip(gauge_weights, 0.0, None)
    gauge_weights /= gauge_weights.sum(axis=-1, keepdims=True)
    shell_kl = np.zeros_like(probabilities)
    for shell_hat, teacher_mass in zip(
        space["shell_hat"], space["shell_probability"], strict=True
    ):
        mass = fwht(phat * shell_hat) / nstates
        mass = np.clip(mass, 0.0, 1.0)
        positive = mass > 0.0
        term = np.zeros_like(mass)
        term[positive] = mass[positive] * np.log(mass[positive] / teacher_mass)
        shell_kl += term
    return np.sum(gauge_weights * shell_kl, axis=-1)


def cell_metrics(
    base_logits: np.ndarray,
    leading_square: np.ndarray,
    eta: float,
    gamma: float,
    n: int,
    omega: float,
    space: dict,
    decompose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Rao-Blackwellized conditional observables and log orbit weights."""
    logits = base_logits / math.sqrt(eta * n * gamma)
    logits -= logits.max(axis=-1, keepdims=True)
    logp = logits - logsumexp(logits, axis=-1, keepdims=True)
    p = np.exp(logp)
    if eta == 1.0:
        logweight = np.zeros(p.shape[0])
        orientation = p
    else:
        logweight = logsumexp(eta * logp, axis=-1)
        orientation = np.exp(eta * logp - logweight[:, None])
    mean_pairs = p @ space["features"]
    orientation_pairs = mean_pairs if eta == 1.0 else orientation @ space["features"]
    paired_sum = np.sum(mean_pairs * orientation_pairs, axis=-1)
    overlap_numerator = n + 2.0 * paired_sum
    entropy_full = -np.sum(p * logp, axis=-1) + np.log(2.0)
    teacher_energy = omega / (2.0 * n) * (
        n + 2.0 * space["teacher_pair_correlation"] * paired_sum
    )
    total = -entropy_full + space["teacher_logz"] - teacher_energy
    parallel = longitudinal_average(p, orientation, space) if decompose else np.full_like(total, np.nan)
    retrieval = np.sum(p * orientation, axis=-1)
    generated_leading_variance = np.sum(p * leading_square, axis=-1) / n
    values = np.stack(
        [total, parallel, total - parallel, retrieval, overlap_numerator / n**2,
         generated_leading_variance],
        axis=-1,
    )
    if decompose:
        if np.min(parallel) < -1e-9 or np.min(total - parallel) < -1e-9:
            raise RuntimeError("KL decomposition produced a negative component")
    return values, logweight


def weighted_summary(values: np.ndarray, logweights: np.ndarray, blocks: int) -> dict:
    w = np.exp(logweights - np.max(logweights))
    wsum = float(w.sum())
    mean = np.sum(w[:, None] * values, axis=0) / wsum
    residual = w[:, None] * (values - mean)
    n = values.shape[0]
    se = np.sqrt(np.sum(residual**2, axis=0) * n / (n - 1)) / wsum
    batch_estimates = []
    for indices in np.array_split(np.arange(n), blocks):
        wb = w[indices]
        batch_estimates.append(np.sum(wb[:, None] * values[indices], axis=0) / wb.sum())
    batch_se = np.std(batch_estimates, axis=0, ddof=1) / math.sqrt(blocks)
    result = {
        "importance_ess": float(wsum**2 / np.dot(w, w)),
        "maximum_normalized_weight": float(w.max() / wsum),
        "log_orbit_weight_min": float(logweights.min()),
        "log_orbit_weight_max": float(logweights.max()),
        "statistics": {},
    }
    for i, metric in enumerate(METRICS):
        if np.isfinite(mean[i]):
            result["statistics"][metric] = {
                "mean": float(mean[i]), "mc_se": float(se[i]),
                "batch_mc_se": float(batch_se[i]),
                "mc_ci95": [float(mean[i] - 1.96 * se[i]), float(mean[i] + 1.96 * se[i])],
            }
    return result


def validate() -> dict:
    """Independent N=4 finite sums check the orbit and Walsh formulas."""
    n, omega, eta = 4, 1.5, 0.6
    space = state_space(n, omega)
    rng = np.random.default_rng(90210)
    logits = space["features"] @ rng.normal(size=6)
    p = np.exp(logits - logsumexp(logits))
    r = p**eta
    r /= r.sum()
    teacher = space["teacher"]
    total_brute = 0.0
    parallel_brute = 0.0
    retrieval_brute = 0.0
    overlap_brute = 0.0
    for datum in range(p.size):
        for orientation in range(p.size):
            q = p[np.arange(p.size) ^ datum ^ orientation]
            weight = teacher[datum] * r[orientation]
            total_brute += weight * np.sum(q * np.log(q / teacher))
            qs = space["shell_masks"] @ q
            parallel_brute += weight * np.sum(qs * np.log(qs / space["shell_probability"]))
            retrieval_brute += weight * q[datum]
            overlap = (space["states"] @ space["states"][datum]) ** 2 / n**2
            overlap_brute += weight * np.dot(q, overlap)
    mu = p @ space["features"]
    mr = r @ space["features"]
    total_formula = (
        np.sum(p * np.log(p)) - np.log(2.0) + space["teacher_logz"]
        - omega / (2.0 * n) * (n + 2.0 * space["teacher_pair_correlation"] * np.dot(mu, mr))
    )
    parallel_fwht = float(longitudinal_average(p[None], r[None], space)[0])
    checks = {
        "fwht_involution": float(np.max(np.abs(fwht(fwht(p)) / p.size - p))),
        "total_bruteforce": abs(total_brute - total_formula),
        "parallel_bruteforce": abs(parallel_brute - parallel_fwht),
        "retrieval_bruteforce": abs(retrieval_brute - np.dot(p, r)),
        "overlap_bruteforce": abs(overlap_brute - (n + 2.0 * np.dot(mu, mr)) / n**2),
    }
    if max(checks.values()) > 2e-12:
        raise RuntimeError(f"Independent validation failed: {checks}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2026110303)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--omega", type=float, default=3.0)
    parser.add_argument("--etas", type=float, nargs="+", default=[0.2, 0.4, 0.6, 1.0, 10.0])
    parser.add_argument("--gammas", type=float, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--runtime-limit", type=float, default=180.0,
                        help="Save a resumable checkpoint before the tool process deadline")
    args = parser.parse_args()
    checks = validate()
    print(json.dumps({"validation": checks}), flush=True)
    if args.validate_only:
        return
    if args.draws < args.blocks * 2:
        raise ValueError("Need at least two draws per uncertainty block")
    args.output.mkdir(parents=True, exist_ok=True)
    source_at_start = Path(__file__).read_bytes()
    source_sha256 = hashlib.sha256(source_at_start).hexdigest()
    source_directory = args.output / "code_snapshots"
    source_directory.mkdir(exist_ok=True)
    (source_directory / f"{source_sha256}.py").write_bytes(source_at_start)
    gammas = np.array(args.gammas if args.gammas else sorted(set(
        np.concatenate([np.logspace(-11, -3, 9), np.logspace(-3, 2, 51),
                        np.logspace(2, 6, 9), [1.0]]).tolist()
    )))
    etas = np.array(args.etas)
    start = time.monotonic()
    space = state_space(args.n, args.omega)
    rng = np.random.default_rng(args.seed)
    normals = rng.normal(size=(args.draws, args.n * (args.n - 1) // 2))
    matrices = np.zeros((args.draws, args.n, args.n))
    matrices[:, space["upper_i"], space["upper_j"]] = normals
    matrices[:, space["upper_j"], space["upper_i"]] = normals
    with threadpool_limits(limits=args.threads):
        _, eigenvectors = np.linalg.eigh(matrices)
        leading_vectors = eigenvectors[:, :, -1]
        leading_square = np.square(leading_vectors @ space["states"].T)
        base_logits = normals @ space["features"].T
        metrics = np.full((etas.size, gammas.size, args.draws, len(METRICS)), np.nan)
        logweights = np.zeros((etas.size, gammas.size, args.draws))
        cells = []
        previous_runtime = 0.0
        if args.resume and (args.output / "gamma_summary.json").exists():
            previous = json.loads((args.output / "gamma_summary.json").read_text())
            saved = np.load(args.output / "gamma_draws.npz")
            if not (np.array_equal(saved["eta"], etas)
                    and np.array_equal(saved["gamma"], gammas)
                    and np.array_equal(saved["standard_normal_couplings"], normals)):
                raise ValueError("Resume settings or random normals differ from checkpoint")
            metrics = saved["metrics"].copy()
            logweights = saved["log_orbit_weights"].copy()
            cells = previous["cells"]
            if any("source_sha256" not in cell for cell in cells):
                raise ValueError("Resume checkpoint needs source attribution for every existing cell")
            previous_runtime = previous["runtime_seconds"]
        completed = {(cell["eta"], cell["gamma"]) for cell in cells}
        checkpoint_due = False
        for ie, eta in enumerate(etas):
            for ig, gamma in enumerate(gammas):
                if (float(eta), float(gamma)) in completed:
                    continue
                values, lw = cell_metrics(base_logits, leading_square, float(eta), float(gamma),
                                         args.n, args.omega, space, eta == 1.0)
                metrics[ie, ig] = values
                logweights[ie, ig] = lw
                summary = weighted_summary(values, lw, args.blocks)
                cells.append({"eta": float(eta), "gamma": float(gamma),
                              "source_sha256": source_sha256, **summary})
                if ig % 10 == 0 or ig == gammas.size - 1:
                    print(json.dumps({"eta": float(eta), "gamma": float(gamma),
                                      "reverse": summary["statistics"]["reverse_total"]["mean"],
                                      "ess": summary["importance_ess"],
                                      "elapsed_seconds": time.monotonic() - start}), flush=True)
                if args.runtime_limit > 0 and time.monotonic() - start >= args.runtime_limit:
                    checkpoint_due = True
                    break
            if checkpoint_due:
                break
    raw_path = args.output / "gamma_draws.npz"
    np.savez_compressed(raw_path, standard_normal_couplings=normals, eta=etas,
                        gamma=gammas, metric_names=np.array(METRICS), metrics=metrics,
                        log_orbit_weights=logweights, teacher_half=space["teacher"],
                        states_half=space["states"], teacher_shell_probability=space["shell_probability"])
    result = {
        "experiment": "Appendix small binary BM: K=1 double descent",
        "model_version": MODEL_VERSION,
        "parameters": {"N": args.n, "K": 1, "omega": args.omega,
                       "draws": args.draws, "seed": args.seed, "threads": args.threads,
                       "eta": etas.tolist(), "gamma": gammas.tolist()},
        "posterior_assumption": "exp(-eta*(K*logZ - sum_data_energy + N*gamma/2*sum_unique_J_squared))",
        "method": "Gaussian-prior gauge-orbit integration; exact iid orbit samples at eta=1; self-normalized importance otherwise",
        "data_average": "Exact sum over all K=1 teacher datasets; no finite dataset ensemble uncertainty",
        "gauge_average": "Exact sum over every orientation; moments for total and alignment, Walsh-Hadamard shell convolutions for eta=1 decomposition",
        "uncertainty": "Independent Gaussian-draw Monte Carlo SE; shared standard normal matrices across grid induce paired cell errors",
        "units": "Total nats; squared overlaps and generated variances normalized as figure labels",
        "uniform_reverse": space["uniform_reverse"],
        "weak_gamma_reverse": space["weak_gamma_reverse"],
        "teacher_pair_correlation": space["teacher_pair_correlation"],
        "validation": checks,
        "complete": len(cells) == etas.size * gammas.size,
        "completed_cells": len(cells),
        "expected_cells": int(etas.size * gammas.size),
        "runtime_seconds": previous_runtime + time.monotonic() - start,
        "software": {"numpy": np.__version__, "python": platform.python_version(), "platform": platform.platform()},
        "script_sha256": source_sha256,
        "source_provenance": "Each cell records its exact source snapshot hash captured at process startup; resumed cells can have different hashes",
        "source_snapshots_sha256": sorted({cell["source_sha256"] for cell in cells}),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "cells": cells,
    }
    (args.output / "gamma_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"saved": str(args.output), "seconds": result["runtime_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
