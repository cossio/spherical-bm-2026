# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy>=2.3", "scipy>=1.16", "numba>=0.63", "threadpoolctl>=3.6"]
# ///
"""Small binary-BM persistent-CD training with exact state-space evaluation.

Sampling and regularization conventions are CLI options saved in metadata.
The paper uses mean-loss ridge=0.8, equivalent to summed-loss gamma=3.2.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numba
import numpy as np
import scipy
from numba import njit, prange
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import t as student_t
from threadpoolctl import threadpool_limits


@njit(parallel=True, cache=True)
def train(cdata, seeds, rates, steps, checkpoints, particles, learning_rate, ridge):
    """One independently trained J per dataset, replicate, and sampling rate."""
    nd, nr = seeds.shape
    n = cdata.shape[1]
    nc = len(checkpoints)
    js = np.zeros((nd, nr, len(rates), nc, n, n))
    for job in prange(nd * nr * len(rates)):
        rate_idx = job % len(rates)
        replica = (job // len(rates)) % nr
        dataset = job // (len(rates) * nr)
        # Same initialization and random stream seed across sampling rates;
        # the differing scan counts consume different portions thereafter.
        np.random.seed(seeds[dataset, replica])
        spins = np.empty((particles, n), dtype=np.float64)
        for a in range(particles):
            for i in range(n):
                spins[a, i] = 2 * np.random.randint(2) - 1
        jmat = np.zeros((n, n))
        next_checkpoint = 1
        for step in range(1, steps + 1):
            for unused in range(rates[rate_idx]):
                for a in range(particles):
                    i = np.random.randint(n)
                    field = 0.0
                    for k in range(n):
                        field += jmat[i, k] * spins[a, k]
                    prob_plus = 0.5 * (1.0 + np.tanh(field))
                    spins[a, i] = 1.0 if np.random.random() < prob_plus else -1.0
            for i in range(n):
                for k in range(i + 1, n):
                    cmodel = 0.0
                    for a in range(particles):
                        cmodel += spins[a, i] * spins[a, k]
                    value = jmat[i, k] + learning_rate * (
                        cdata[dataset, i, k] - cmodel / particles - ridge * jmat[i, k]
                    )
                    jmat[i, k] = value
                    jmat[k, i] = value
            if next_checkpoint < nc and step == checkpoints[next_checkpoint]:
                js[dataset, replica, rate_idx, next_checkpoint] = jmat
                next_checkpoint += 1
    return js


def exact_setup(n, omega):
    states = 2 * ((np.arange(2**n)[:, None] >> np.arange(n)) & 1) - 1
    states = states.astype(np.float64)
    upper = np.triu_indices(n, 1)
    features = states[:, upper[0]] * states[:, upper[1]]
    logp = omega * np.square(states.sum(axis=1)) / (2 * n)
    logp -= logsumexp(logp)
    pstar = np.exp(logp)
    return states, upper, features, logp, pstar


def fit_map(features, empirical, ridge):
    def value_gradient(j):
        scores = features @ j
        logz = logsumexp(scores)
        mean = np.exp(scores - logz) @ features
        value = logz - empirical @ j + 0.5 * ridge * (j @ j)
        return value, mean - empirical + ridge * j

    opt = minimize(
        value_gradient,
        np.zeros(features.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"gtol": 1e-10, "ftol": 1e-14, "maxiter": 1000, "maxls": 50},
    )
    # L-BFGS can stop on machine-level objective changes before the requested
    # gradient tolerance. Convexity permits a direct exact-Hessian polish.
    polished = opt.x.copy()
    for unused in range(3):
        gradient = value_gradient(polished)[1]
        if np.linalg.norm(gradient, ord=np.inf) <= 1e-10:
            break
        scores = features @ polished
        probabilities = np.exp(scores - logsumexp(scores))
        mean = probabilities @ features
        hessian = features.T @ (probabilities[:, None] * features) - np.outer(mean, mean)
        hessian += ridge * np.eye(len(polished))
        polished -= np.linalg.solve(hessian, gradient)
    gradnorm = float(np.linalg.norm(value_gradient(polished)[1], ord=np.inf))
    if gradnorm > 2e-7:
        raise RuntimeError(f"MAP residual too large: {gradnorm}; {opt.message}")
    return polished, {"success": bool(opt.success), "message": str(opt.message), "gradient_inf": gradnorm, "iterations": int(opt.nit), "exact_hessian_polish": True}


def exact_metrics(matrices, features, logp, pstar, upper, vdata):
    shape = matrices.shape[:-2]
    n = matrices.shape[-1]
    flat = matrices.reshape((-1, n, n))
    pair_j = flat[:, upper[0], upper[1]]
    fwd = np.empty(len(flat))
    rev = np.empty(len(flat))
    entropy_term = pstar @ logp
    cstar = pstar @ features
    for start in range(0, len(flat), 512):
        end = min(start + 512, len(flat))
        scores = features @ pair_j[start:end].T
        logz = logsumexp(scores, axis=0)
        logq = scores - logz
        probs = np.exp(logq)
        fwd[start:end] = entropy_term - pair_j[start:end] @ cstar + logz
        rev[start:end] = np.sum(probs * (logq - logp[:, None]), axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(flat)
    leading = eigenvectors[:, :, -1].reshape(shape + (n,))
    broadcast_data = vdata.reshape((shape[0],) + (1,) * (len(shape) - 1) + (n,))
    align = np.square(np.sum(leading * broadcast_data, axis=-1))
    # At zero initialization the entire spectrum is degenerate, so the
    # eigensolver's arbitrary basis vector is not a learned direction.
    align.reshape(-1)[np.all(flat == 0.0, axis=(1, 2))] = np.nan
    return np.stack((eigenvalues[:, -1].reshape(shape), align, fwd.reshape(shape), rev.reshape(shape)), axis=-1)


def self_checks(n, omega, ridge):
    states, upper, features, logp, pstar = exact_setup(n, omega)
    rng = np.random.default_rng(48171)
    j = rng.normal(0, 0.03, features.shape[1])
    scores = features @ j
    jmat = np.zeros((n, n))
    jmat[upper] = j
    jmat += jmat.T
    matrix_error = float(np.max(np.abs(scores - np.einsum("bi,ij,bj->b", states, jmat, states) / 2)))
    empirical = features[rng.choice(len(states), 17)].mean(axis=0)
    def fun(x):
        return logsumexp(features @ x) - empirical @ x + ridge * (x @ x) / 2
    grad = np.exp(scores - logsumexp(scores)) @ features - empirical + ridge * j
    indices = np.arange(min(12, len(j)))
    differences = []
    for idx in indices:
        delta = np.zeros_like(j)
        delta[idx] = 1e-6
        differences.append(abs((fun(j + delta) - fun(j - delta)) / 2e-6 - grad[idx]))
    jt = np.full((n, n), omega / n)
    np.fill_diagonal(jt, 0)
    teacher_metrics = exact_metrics(jt[None], features, logp, pstar, upper, np.ones((1, n)) / np.sqrt(n))
    # Enumerating every source state verifies that one heat-bath site update
    # leaves the exact target distribution invariant, including exp(2*h).
    target = np.exp(scores - logsumexp(scores))
    site = min(3, n - 1)
    field = states @ jmat[site]
    pp = (1 + np.tanh(field)) / 2
    plus = np.arange(len(states)) | (1 << site)
    minus = np.arange(len(states)) & ~(1 << site)
    propagated = np.bincount(plus, weights=target * pp, minlength=len(states))
    propagated += np.bincount(minus, weights=target * (1 - pp), minlength=len(states))
    invariance_error = float(np.max(np.abs(propagated - target)))
    result = {
        "matrix_feature_energy_max_error": matrix_error,
        "objective_gradient_finite_difference_max_error": float(max(differences)),
        "teacher_kl_max_abs": float(np.max(np.abs(teacher_metrics[0, 2:]))),
        "teacher_eigenvalue_error": float(abs(teacher_metrics[0, 0] - omega * (1 - 1 / n))),
        "one_site_gibbs_stationarity_max_error": invariance_error,
        "state_count": int(len(states)),
    }
    assert matrix_error < 1e-12 and max(differences) < 1e-7
    assert result["teacher_kl_max_abs"] < 1e-12
    assert result["teacher_eigenvalue_error"] < 1e-12 and invariance_error < 1e-14
    return result


def exact_iid_control(map_matrices, empirical, features, states, upper, ridge, learning_rate, checkpoints, replicas, seed):
    """Single exactly independent negative sample per update, starting at MAP."""
    nd, n, _ = map_matrices.shape
    current = np.repeat(map_matrices[:, None], replicas, axis=1)
    pair_j = current[:, :, upper[0], upper[1]].reshape((-1, len(upper[0])))
    cdata = np.repeat(empirical[:, None, upper[0], upper[1]], replicas, axis=1).reshape(pair_j.shape)
    saved = np.zeros((nd, replicas, len(checkpoints), n, n))
    saved[:, :, 0] = current
    rng = np.random.default_rng(seed)
    next_checkpoint = 1
    for step in range(1, checkpoints[-1] + 1):
        scores = features @ pair_j.T
        scores -= scores.max(axis=0)
        weights = np.exp(scores)
        cdf = np.cumsum(weights, axis=0)
        draw = np.argmax(cdf >= rng.random(len(pair_j))[None, :] * cdf[-1], axis=0)
        pair_j += learning_rate * (cdata - features[draw] - ridge * pair_j)
        if next_checkpoint < len(checkpoints) and step == checkpoints[next_checkpoint]:
            current = np.zeros((nd * replicas, n, n))
            current[:, upper[0], upper[1]] = pair_j
            current += current.transpose(0, 2, 1).copy()
            saved[:, :, next_checkpoint] = current.reshape((nd, replicas, n, n))
            next_checkpoint += 1
    return saved


def interval(values):
    values = np.asarray(values)
    mean = values.mean(axis=0)
    halfwidth = student_t.ppf(0.975, len(values) - 1) * values.std(axis=0, ddof=1) / np.sqrt(len(values))
    return {"mean": mean.tolist(), "ci95_low": (mean - halfwidth).tolist(), "ci95_high": (mean + halfwidth).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=int, default=50)
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--particles", type=int, default=1)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--checkpoint", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026110304)
    parser.add_argument("--regularization-convention", choices=("mean-loss-gamma", "summed-loss-n-gamma"), default="mean-loss-gamma")
    parser.add_argument("--iid-control", action="store_true", help="Exact independent negative-phase sample, initialized at MAP")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    numba.set_num_threads(args.threads)
    n, k, omega, gamma, lr = 12, 48, 1.5, 0.8, 0.003
    ridge = gamma if args.regularization_convention == "mean-loss-gamma" else n * gamma / k
    rates = np.array([1, 3, 6, 12, 120])
    checkpoints = np.unique(np.r_[np.arange(0, args.steps + 1, args.checkpoint), args.steps]).astype(np.int64)
    started = time.monotonic()
    with threadpool_limits(limits=args.threads):
        checks = self_checks(n, omega, ridge)
        states, upper, features, logp, pstar = exact_setup(n, omega)
        rng = np.random.default_rng(args.seed)
        sample_indices = rng.choice(len(states), size=(args.datasets, k), p=pstar)
        data = states[sample_indices]
        cdata = np.einsum("dki,dkj->dij", data, data) / k
        _, eigvec = np.linalg.eigh(cdata)
        vdata = eigvec[:, :, -1]
        seeds = rng.integers(1, 2**31 - 1, size=(args.datasets, args.replicas))
        map_matrices = np.zeros((args.datasets, n, n))
        map_diagnostics = []
        for dataset in range(args.datasets):
            fit, diagnostic = fit_map(features, cdata[dataset][upper], ridge)
            map_matrices[dataset][upper] = fit
            map_matrices[dataset] += map_matrices[dataset].T.copy()
            map_diagnostics.append(diagnostic)
        map_metrics = exact_metrics(map_matrices, features, logp, pstar, upper, vdata)
        print(json.dumps({"stage": "map_complete", "seconds": time.monotonic() - started, "mean": map_metrics.mean(axis=0).tolist()}), flush=True)
        training_started = time.monotonic()
        js = train(cdata, seeds, rates, args.steps, checkpoints, args.particles, lr, ridge)
        print(json.dumps({"stage": "training_complete", "seconds": time.monotonic() - training_started}), flush=True)
        metrics = exact_metrics(js, features, logp, pstar, upper, vdata)
        control_arrays = {}
        if args.iid_control:
            control_started = time.monotonic()
            iid_js = exact_iid_control(map_matrices, cdata, features, states, upper, ridge, lr, checkpoints, args.replicas, args.seed + 99119)
            iid_metrics = exact_metrics(iid_js, features, logp, pstar, upper, vdata)
            control_arrays = {"iid_map_start_couplings": iid_js, "iid_map_start_metrics": iid_metrics}
            print(json.dumps({"stage": "iid_control_complete", "seconds": time.monotonic() - control_started}), flush=True)
    per_dataset = metrics.mean(axis=1)
    mean = per_dataset.mean(axis=0)
    if args.datasets > 1:
        halfwidth = student_t.ppf(0.975, args.datasets - 1) * per_dataset.std(axis=0, ddof=1) / np.sqrt(args.datasets)
    else:
        halfwidth = np.full_like(mean, np.nan)
    summary = []
    for ri, rate in enumerate(rates):
        fi, rvi = np.argmin(mean[ri, :, 2]), np.argmin(mean[ri, :, 3])
        summary.append({
            "single_site_updates": int(rate),
            "final_mean": mean[ri, -1].tolist(),
            "final_ci95_halfwidth": halfwidth[ri, -1].tolist(),
            "forward_minimum_step": int(checkpoints[fi]),
            "forward_minimum": float(mean[ri, fi, 2]),
            "reverse_minimum_step": int(checkpoints[rvi]),
            "reverse_minimum": float(mean[ri, rvi, 3]),
        })
    metadata = {
        "experiment": "independent reconstruction of camera-ready out-of-equilibrium small binary BM figure",
        "model_version": "fully-visible-bm-v1",
        "n": n, "k": k, "omega": omega, "gamma": gamma, "learning_rate": lr,
        "datasets": args.datasets, "replicas": args.replicas, "particles_per_replica": args.particles,
        "seed": args.seed, "steps": args.steps, "checkpoint_spacing": args.checkpoint,
        "ridge": ridge, "regularization_convention": args.regularization_convention,
        "mean_objective": "log Z - cdata dot j + (ridge/2) sum_{i<j} J_ij^2",
        "metric_order": ["lambda1", "alignment_squared_empirical", "kl_forward_total_nats", "kl_reverse_total_nats"],
        "array_axes": {"metrics": ["dataset", "replica", "sampling_rate", "checkpoint", "metric"], "couplings": ["dataset", "replica", "sampling_rate", "checkpoint", "i", "j"]},
        "assumed_conventions": [
            f"Regularizer {ridge}/2 times sum_{{i<j}} J_ij^2 in the MEAN negative log posterior; The paper convention has gamma_paper=(K/N)*ridge.",
            "Mean likelihood gradient cdata-cmodel-ridge*J, with the quoted learning rate; see regularization_convention.",
            "Each persistent-chain replica is an independent optimization run; particles_per_replica controls simultaneous negative-phase chains.",
            "Random-scan heat-bath Gibbs; each reported single-site update is applied once per particle.",
            "All coupling matrices start exactly zero; persistent particles start independently uniform.",
            "Gibbs negative phase precedes the coupling update; empirical direction uses uncentered second moment.",
            "Datasets and initialization/random-stream seeds are paired across sampling rates.",
            "Replicates averaged within datasets; pointwise 95% Student-t confidence intervals over independent datasets.",
            "No smoothing or time-axis rescaling; total KL in nats, not per spin.",
            "Leading-eigenvector alignment at exactly zero J is undefined and stored as NaN.",
        ],
        "checks": checks,
        "map_diagnostics": map_diagnostics,
        "map_mean": map_metrics.mean(axis=0).tolist(),
        "sampling_rate_summary": summary,
        "paired_final_slow_minus_fast": interval(per_dataset[:, 0, -1] - per_dataset[:, -1, -1]),
        "paired_final_slow_minus_map": interval(per_dataset[:, 0, -1] - map_metrics),
        "runtime_seconds": time.monotonic() - started,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "numba": numba.__version__},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "thread_limit": args.threads,
    }
    target = args.output / "dynamics.npz"
    if args.iid_control:
        metadata["iid_control"] = {
            "description": "One exact independent equilibrium draw per parameter update, starting at each dataset's unique MAP; same learning rate and regularizer.",
            "seed": args.seed + 99119,
            "final_interval": interval(iid_metrics[:, :, -1].mean(axis=1)),
            "paired_final_minus_map": interval(iid_metrics[:, :, -1].mean(axis=1) - map_metrics),
        }
    np.savez_compressed(target, checkpoints=checkpoints, single_site_updates=rates, metrics=metrics, dataset_metrics=per_dataset, mean=mean, ci95_halfwidth=halfwidth, map_metrics=map_metrics, map_couplings=map_matrices, couplings=js, sample_indices=sample_indices, datasets=data.astype(np.int8), empirical_covariances=cdata, seeds=seeds, teacher_logp=logp, **control_arrays)
    metadata["data_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"stage": "complete", "runtime_seconds": metadata["runtime_seconds"], "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
