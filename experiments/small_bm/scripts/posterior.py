# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy==2.4.6", "scipy==1.17.1", "threadpoolctl==3.6.0"]
# ///
"""Independent reconstruction of the current Appendix's small-BM posterior figure.

The posterior uses U = K log Z - sum_data f(s).j + N gamma ||j||²/2,
and pi_eta(j|D) proportional to exp(-eta U). Exact spin sums do NOT make the
coupling integral exact. Metropolis-adjusted HMC approximates that integral.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import logsumexp, ndtri
from scipy.stats import rankdata, t
from threadpoolctl import threadpool_limits

N = 8
K = 4
OMEGA = 1.5
PAIRS = np.array(np.triu_indices(N, 1)).T
STATES = 2 * ((np.arange(2**N)[:, None] >> np.arange(N)) & 1) - 1
FEATURES = np.asarray(STATES[:, PAIRS[:, 0]] * STATES[:, PAIRS[:, 1]], float)
# Global spin flip gives exactly duplicate pair features. No permutation
# symmetrization of the sampled empirical data is applied.
F = FEATURES[: 2 ** (N - 1)].copy()
TEACHER_LOGP = OMEGA / (2 * N) * STATES.sum(axis=1) ** 2
TEACHER_LOGP -= logsumexp(TEACHER_LOGP)
TEACHER = np.exp(TEACHER_LOGP)
SHELLS, SHELL_INDEX = np.unique(STATES.sum(axis=1) ** 2, return_inverse=True)
SHELL_MAP = np.eye(len(SHELLS))[SHELL_INDEX]
TEACHER_SHELL = TEACHER @ SHELL_MAP
METRIC_NAMES = ["forward", "reverse", "forward_parallel", "forward_perp", "reverse_parallel", "reverse_perp"]
ETAS = [0.05, 0.075, 0.1, 0.15, 0.25, 0.4, 0.6, 0.8, 1.0, 1.5, 3.0, 7.0, 20.0]


def cell_configuration(cfg, gamma, eta, ids):
    """Complete fingerprint for numerical and saved-output compatibility."""
    configuration = {name: cfg[name] for name in
                     ["draws", "warmup", "chains", "seed", "leapfrog_min", "leapfrog_max", "save_traces"]}
    configuration.update(gamma=gamma, eta=eta, N=N, K=K, omega=OMEGA,
                         dataset_indices_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                         sampler_script_sha256=cfg["sampler_script_sha256"])
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return encoded, hashlib.sha256(encoded.encode()).hexdigest()


def dataset_indices(count: int, offset: int = 1000) -> np.ndarray:
    return np.array([np.random.default_rng(offset + d).choice(2**N, K, p=TEACHER) for d in range(count)])


def objective(j: np.ndarray, data_sum: np.ndarray, gamma: float):
    logits = F @ j
    logz = logsumexp(logits)
    p = np.exp(logits - logz)
    mean = p @ F
    value = K * (logz + np.log(2)) - data_sum @ j + N * gamma / 2 * (j @ j)
    grad = K * mean - data_sum + N * gamma * j
    hess = K * ((F.T * p) @ F - np.outer(mean, mean)) + N * gamma * np.eye(len(j))
    return value, grad, hess


def map_fit(data_sum: np.ndarray, gamma: float):
    fit = minimize(lambda j: objective(j, data_sum, gamma)[:2], np.zeros(len(PAIRS)),
                   jac=True, hess=lambda j: objective(j, data_sum, gamma)[2],
                   method="trust-exact", options={"gtol": 1e-10})
    _, grad, hess = objective(fit.x, data_sum, gamma)
    if np.max(np.abs(grad)) > 1e-7:
        raise RuntimeError(f"MAP residual {np.max(np.abs(grad))}: {fit.message}")
    eig, vec = np.linalg.eigh(hess)
    invsqrt = (vec * eig**-0.5) @ vec.T
    return fit.x, invsqrt, float(np.max(np.abs(grad)))


def losses(p: np.ndarray) -> np.ndarray:
    p = p / p.sum(axis=-1, keepdims=True)
    logp = np.log(p)
    forward = np.sum(TEACHER * (TEACHER_LOGP - logp), axis=-1)
    reverse = np.sum(p * (logp - TEACHER_LOGP), axis=-1)
    pshell = p @ SHELL_MAP
    fp = np.sum(TEACHER_SHELL * (np.log(TEACHER_SHELL) - np.log(pshell)), axis=-1)
    rp = np.sum(pshell * (np.log(pshell) - np.log(TEACHER_SHELL)), axis=-1)
    return np.stack([forward, reverse, fp, forward - fp, rp, reverse - rp], axis=-1)


def expand_probability(p: np.ndarray) -> np.ndarray:
    return np.concatenate((p, p[..., ::-1]), axis=-1) / 2


def split_chains(x: np.ndarray) -> np.ndarray:
    # Input [chain, draw, observable]. Each half is treated as a chain.
    half = x.shape[1] // 2
    return np.concatenate([x[:, :half], x[:, -half:]], axis=0)


def ordinary_rhat(x: np.ndarray):
    size = x.shape[1]
    within = np.var(x, axis=1, ddof=1).mean(axis=0)
    between = size * np.var(x.mean(axis=1), axis=0, ddof=1)
    return np.sqrt(((size - 1) / size * within + between / size) / within)


def diagnostics(samples: np.ndarray):
    x = split_chains(samples)
    shape = x.shape
    ranks = rankdata(x.reshape(-1, shape[-1]), axis=0)
    ranked = ndtri((ranks - 0.375) / (ranks.shape[0] + 0.25)).reshape(shape)
    folded = np.abs(x - np.median(x, axis=(0, 1)))
    folded_rank = rankdata(folded.reshape(-1, shape[-1]), axis=0)
    folded_z = ndtri((folded_rank - 0.375) / (folded_rank.shape[0] + 0.25)).reshape(shape)
    rhat = np.maximum(ordinary_rhat(ranked), ordinary_rhat(folded_z))
    size = shape[1]
    centered = ranked - ranked.mean(axis=1, keepdims=True)
    fft = np.fft.rfft(centered, n=2 * size, axis=1)
    acov = np.fft.irfft(fft * fft.conj(), n=2 * size, axis=1)[:, :size] / size
    within = acov[:, 0].mean(axis=0) * size / (size - 1)
    between = size * np.var(ranked.mean(axis=1), axis=0, ddof=1)
    varplus = (size - 1) / size * within + between / size
    rho = 1 - (within - acov.mean(axis=0)) / varplus
    rho[0] = 1
    paired = rho[: 2 * (size // 2)].reshape(-1, 2, shape[-1]).sum(axis=1)
    positive = np.logical_and.accumulate(paired > 0, axis=0)
    monotone = np.minimum.accumulate(paired, axis=0)
    tau = -1 + 2 * np.sum(np.where(positive, monotone, 0), axis=0)
    ess = shape[0] * size / np.maximum(tau, 1)
    return rhat, ess


def run_cell(task):
    gamma, eta, ids, maps, transforms, cfg, outdir = task
    threadpool_limits(limits=1)
    start = time.monotonic()
    nd = len(ids)
    nc = cfg["chains"]
    draws = cfg["draws"]
    warmup = cfg["warmup"]
    seed = np.random.SeedSequence([cfg["seed"], int(round(gamma * 1000)), int(round(eta * 10000))])
    rng = np.random.default_rng(seed)
    sums = FEATURES[ids].sum(axis=1)
    scale = transforms / np.sqrt(eta)
    # Dispersed independent starts in local Hessian units, not reused chains.
    z = rng.normal(size=(nd, nc, len(PAIRS))) * 2.0
    eps = np.full((nd, 1, 1), 0.35)

    def evaluate(zvalue):
        j = maps[:, None, :] + np.einsum("dci,dij->dcj", zvalue, scale, optimize=True)
        logits = j.reshape(-1, len(PAIRS)) @ F.T
        logz = logsumexp(logits, axis=-1)
        prob = np.exp(logits - logz[:, None])
        mean = (prob @ F).reshape(nd, nc, -1)
        energy = eta * (K * (logz.reshape(nd, nc) + np.log(2))
                        - np.einsum("dci,di->dc", j, sums)
                        + N * gamma / 2 * np.sum(j * j, axis=-1))
        gradj = eta * (K * mean - sums[:, None, :] + N * gamma * j)
        gradz = np.einsum("dci,dij->dcj", gradj, scale, optimize=True)
        return energy, gradz, prob.reshape(nd, nc, -1)

    energy, grad, prob = evaluate(z)
    traces = np.empty((nd, nc, draws, len(PAIRS) + 2), dtype=np.float64)
    sums_prob = np.zeros((2, nd, nc, len(F)))
    accepted_count = np.zeros((nd, nc))
    divergent_count = np.zeros((nd, nc), dtype=int)
    adaptation_accept = np.zeros((nd, nc))
    for iteration in range(warmup + draws):
        # Each trajectory has a fixed jittered step; trajectory length is
        # randomized independently of states and Metropolis corrects its error.
        step = eps * rng.uniform(0.85, 1.15)
        nsteps = int(rng.integers(cfg["leapfrog_min"], cfg["leapfrog_max"] + 1))
        momentum = rng.normal(size=z.shape)
        prop_z = z.copy()
        prop_p = momentum - 0.5 * step * grad
        for leap in range(nsteps):
            prop_z += step * prop_p
            prop_energy, prop_grad, prop_prob = evaluate(prop_z)
            prop_p -= (0.5 if leap == nsteps - 1 else 1.0) * step * prop_grad
        delta = prop_energy - energy + 0.5 * np.sum(prop_p**2 - momentum**2, axis=-1)
        accept_prob = np.exp(np.minimum(-delta, 0))
        accept_prob = np.where(np.isfinite(accept_prob), accept_prob, 0)
        accept = rng.uniform(size=(nd, nc)) < accept_prob
        z = np.where(accept[..., None], prop_z, z)
        grad = np.where(accept[..., None], prop_grad, grad)
        prob = np.where(accept[..., None], prop_prob, prob)
        energy = np.where(accept, prop_energy, energy)
        if iteration < warmup:
            adaptation_accept += accept_prob
            if (iteration + 1) % 25 == 0:
                rate = adaptation_accept.mean(axis=1) / 25
                # Diminishing Robbins-Monro warmup. Frozen during sampling.
                eps *= np.exp((rate - 0.8)[:, None, None] / (1 + iteration / 100)**0.5)
                eps = np.clip(eps, 0.02, 1.0)
                adaptation_accept.fill(0)
        else:
            d = iteration - warmup
            traces[:, :, d, :len(PAIRS)] = z
            traces[:, :, d, -2] = energy
            traces[:, :, d, -1] = (prob @ (STATES[:len(F)].sum(axis=1)**2))
            sums_prob[int(d >= draws // 2)] += prob
            accepted_count += accept
            divergent_count += (~np.isfinite(delta)) | (np.abs(delta) > 1000)
    half_sizes = np.array([draws // 2, draws - draws // 2])
    half_predictive = expand_probability(sums_prob / half_sizes[:, None, None, None])
    chain_predictive = expand_probability(sums_prob.sum(axis=0) / draws)
    predictive = chain_predictive.mean(axis=1)
    metrics = losses(predictive)
    half_metrics = losses(half_predictive.mean(axis=2))
    chain_metrics = losses(chain_predictive)
    rhats, esses = zip(*(diagnostics(trace) for trace in traces))
    rhats, esses = np.array(rhats), np.array(esses)
    result = {
        "gamma": gamma, "eta": eta, "predictive": predictive, "metrics": metrics,
        "chain_predictive": chain_predictive, "half_predictive": half_predictive,
        "chain_metrics": chain_metrics, "half_metrics": half_metrics,
        "rank_split_rhat": rhats, "bulk_ess": esses,
        "acceptance": accepted_count / draws, "divergences": divergent_count,
        "step_size": eps[:, 0, 0], "dataset_indices": ids,
        "draws_per_chain": draws, "warmup": warmup, "chains": nc,
        "seed": cfg["seed"], "elapsed_seconds": time.monotonic() - start,
    }
    result["configuration_json"], result["configuration_sha256"] = cell_configuration(cfg, gamma, eta, ids)
    # Keep actual chain traces so diagnostics can be independently repeated.
    if cfg["save_traces"]:
        result["traces"] = traces.astype(np.float32)
    path = Path(outdir) / f"gamma-{gamma:g}_eta-{eta:g}.npz"
    np.savez_compressed(path, **result)
    print(json.dumps({"cell": path.name, "seconds": round(result["elapsed_seconds"], 1),
                      "accept": round(float(result["acceptance"].mean()), 3),
                      "max_rhat": float(rhats.max()), "min_ess": float(esses.min()),
                      "mean_kl": metrics[:, :2].mean(axis=0).tolist()}), flush=True)
    return str(path)


def aggregate(outdir: Path, gammas, etas, metadata):
    arrays = {}
    keys = ["predictive", "metrics", "chain_predictive", "half_predictive", "chain_metrics",
            "half_metrics", "rank_split_rhat", "bulk_ess", "acceptance", "divergences", "step_size"]
    for key in keys:
        arrays[key] = np.array([[np.load(outdir / f"gamma-{g:g}_eta-{e:g}.npz")[key]
                                for e in etas] for g in gammas])
    ids = np.load(outdir / f"gamma-{gammas[0]:g}_eta-{etas[0]:g}.npz")["dataset_indices"]
    arrays.update(gammas=gammas, etas=etas, dataset_indices=ids, metric_names=METRIC_NAMES,
                  teacher=TEACHER, states=STATES, shells=SHELLS)
    mean = arrays["metrics"].mean(axis=2)
    ci = t.ppf(0.995, len(ids) - 1) * arrays["metrics"].std(axis=2, ddof=1) / np.sqrt(len(ids))
    arrays.update(mean=mean, ci99_halfwidth=ci)
    np.savez_compressed(outdir / "posterior_results.npz", **arrays)
    rhats, esses = arrays["rank_split_rhat"], arrays["bulk_ess"]
    paired_half_differences = arrays["half_metrics"][:, :, 1] - arrays["half_metrics"][:, :, 0]
    # [gamma, eta, half, dataset, metric] above. All CIs use datasets only.
    summary = {
        "max_rank_split_rhat": float(rhats.max()), "min_bulk_ess": float(esses.min()),
        "rhat_p99": float(np.quantile(rhats, 0.99)),
        "dataset_cells_rhat_above_1_05": int(np.sum(rhats.max(axis=-1) > 1.05)),
        "dataset_cells_ess_below_100": int(np.sum(esses.min(axis=-1) < 100)),
        "divergences": int(arrays["divergences"].sum()),
        "max_half_draw_mean_KL_difference": float(np.max(np.abs(paired_half_differences[..., :2].mean(axis=2)))),
        "max_half_draw_dataset_KL_difference": float(np.max(np.abs(paired_half_differences[..., :2]))),
        "grid_minimum_eta": {str(g): {name: float(etas[np.argmin(mean[gi, :, mi])])
                                      for mi, name in enumerate(METRIC_NAMES[:2])}
                             for gi, g in enumerate(gammas)},
        "dataset_ci": "Two-sided 99% Student-t interval over independent datasets; posterior draws averaged within dataset",
        "diagnostic_gate": "max rank/folded split-Rhat <= 1.05; min bulk ESS >= 100; zero divergences; inspect predictive half-draw stability",
    }
    metadata["summary"] = summary
    metadata["files_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(outdir.glob("*.npz"))}
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def validate():
    rng = np.random.default_rng(2)
    j = rng.normal(size=len(PAIRS)) * 0.1
    data = dataset_indices(1)[0]
    sums = FEATURES[data].sum(axis=0)
    _, grad, hess = objective(j, sums, 0.1)
    direction = rng.normal(size=len(PAIRS))
    eps = 1e-5
    plus, gp, _ = objective(j + eps * direction, sums, 0.1)
    minus, gm, _ = objective(j - eps * direction, sums, 0.1)
    np.testing.assert_allclose((plus - minus) / (2 * eps), grad @ direction, rtol=1e-7)
    np.testing.assert_allclose((gp - gm) / (2 * eps), hess @ direction, atol=1e-7)
    p = np.exp(FEATURES @ j - logsumexp(FEATURES @ j))
    np.testing.assert_allclose(expand_probability(np.exp(F @ j - logsumexp(F @ j))), p, atol=1e-15)
    np.testing.assert_allclose(losses(TEACHER), 0, atol=1e-14)
    result = losses(p)
    direct_fperp = sum(TEACHER[mask] @ (np.log(1 / mask.sum()) - np.log(p[mask] / p[mask].sum()))
                       for mask in [SHELL_INDEX == si for si in range(len(SHELLS))])
    direct_rperp = sum(p[mask] @ (np.log(p[mask] / p[mask].sum()) - np.log(1 / mask.sum()))
                       for mask in [SHELL_INDEX == si for si in range(len(SHELLS))])
    np.testing.assert_allclose(result[[3, 5]], [direct_fperp, direct_rperp], atol=1e-14)
    iid = rng.normal(size=(4, 2000, 5))
    rhat, ess = diagnostics(iid)
    assert np.max(rhat) < 1.01 and np.min(ess) > 5000
    shifted = iid.copy()
    shifted[0] += 1
    assert diagnostics(shifted)[0].min() > 1.05
    print("PASS: directional gradient/Hessian, exact spin-flip reduction, KL shell chain rule, diagnostic sensitivity", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "posterior")
    parser.add_argument("--datasets", type=int, default=100)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.1, 1.5])
    parser.add_argument("--etas", type=float, nargs="+", default=ETAS)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--data-seed", type=int, default=1000)
    parser.add_argument("--leapfrog-min", type=int, default=5)
    parser.add_argument("--leapfrog-max", type=int, default=10)
    parser.add_argument("--save-traces", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    validate()
    if args.validate_only:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    ids = dataset_indices(args.datasets, args.data_seed)
    cfg = vars(args).copy()
    cfg["out"] = str(args.out)
    cfg["sampler_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata = {"configuration": cfg, "N": N, "K": K, "omega": OMEGA,
                "objective": "U=K log Z - sum_data f(s).j + N gamma/2 * ||j||²; pi_eta proportional exp(-eta U)",
                "regularization_normalization_is_assumption": True,
                "eta_grid_is_inferred_from_rendered_figure": True,
                "sampler": "Metropolis-adjusted HMC; fixed MAP-Hessian whitening; randomized trajectory length and step jitter; adaptation only in warmup",
                "diagnostics_observables": [f"whitened_coupling_{k}" for k in range(len(PAIRS))] + ["eta_U", "conditional_mean_M"],
                "numpy": np.__version__, "scipy": scipy.__version__,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "dataset_indices_sha256": hashlib.sha256(ids.tobytes()).hexdigest()}
    tasks = []
    for gamma in args.gammas:
        fits = [map_fit(FEATURES[d].sum(axis=0), gamma) for d in ids]
        maps = np.array([v[0] for v in fits])
        transforms = np.array([v[1] for v in fits])
        for eta in args.etas:
            path = args.out / f"gamma-{gamma:g}_eta-{eta:g}.npz"
            if args.resume and path.exists():
                with np.load(path) as old:
                    expected_json, expected_hash = cell_configuration(cfg, gamma, eta, ids)
                    if "configuration_sha256" not in old or "configuration_json" not in old:
                        raise RuntimeError(f"Cannot safely resume legacy cell without a complete fingerprint: {path}. Use a new output directory; preserve the original results and execution source.")
                    if str(old["configuration_sha256"]) != expected_hash or str(old["configuration_json"]) != expected_json or not np.array_equal(old["dataset_indices"], ids):
                        raise RuntimeError(f"Cannot resume {path}: dataset, seed, sampler source, trajectory settings, draws, warmup, chains, or saved-trace mode differs.")
                continue
            tasks.append((gamma, eta, ids, maps, transforms, cfg, str(args.out)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for _ in pool.map(run_cell, tasks):
            pass
    aggregate(args.out, args.gammas, args.etas, metadata)


if __name__ == "__main__":
    main()
