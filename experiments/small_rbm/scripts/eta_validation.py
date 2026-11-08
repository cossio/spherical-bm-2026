# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy>=2.4", "scipy>=1.17", "arviz>=0.23"]
# ///
"""Finite-state RBM numerical routines and reproducible training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from rbm_exact import log2cosh, log_partition, logpost_and_grad, spin_states


class ExactPosterior:
    def __init__(self, data: np.ndarray, M: int, gamma: float):
        self.data = np.asarray(data, dtype=float)
        self.N, self.M, self.gamma = self.data.shape[1], M, gamma
        self.a, self.b = np.zeros(self.N), np.zeros(M)
        self.H = spin_states(M)

    def __call__(self, W):
        return logpost_and_grad(
            W, self.data, self.a, self.b, 1.0, self.H, self.gamma, 1.0
        )


def array_hash(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def canonical_weights(W):
    """Remove only the exact hidden sign/permutation symmetries for start selection."""
    W = W.copy()
    for j in range(W.shape[1]):
        if W[np.abs(W[:, j]).argmax(), j] < 0:
            W[:, j] *= -1
    order = sorted(range(W.shape[1]), key=lambda j: tuple(np.round(W[:, j], 8)))
    return W[:, order]


def polish_candidates(target, starts, maxiter=2000):
    """Local optimization only: no global-optimum guarantee is implied."""
    weights, records = [], []
    for start in starts:
        shape = start.shape

        def objective(x):
            lp, grad = target(x.reshape(shape))
            return -float(lp), -grad.ravel()

        fit = minimize(
            objective, start.ravel(), jac=True, method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxls": 40},
        )
        candidate = fit.x.reshape(shape)
        # A failed line search must never replace a better supplied start.
        initial_lp = float(target(start)[0])
        lp, grad = target(candidate)
        if not np.isfinite(lp) or lp < initial_lp:
            candidate = start.copy()
            lp, grad = target(candidate)
        weights.append(candidate)
        records.append({
            "logpost": float(lp), "initial_logpost": initial_lp,
            "gradient_max_abs": float(np.abs(grad).max()),
            "success": bool(fit.success), "message": str(fit.message),
            "iterations": int(fit.nit),
        })
    order = np.argsort([-r["logpost"] for r in records])
    return np.stack(weights)[order], [records[i] for i in order]


def distinct_basins(weights, count):
    kept = []
    canon = []
    for W in weights:
        c = canonical_weights(W)
        if not any(np.linalg.norm(c - other) < 1e-3 for other in canon):
            kept.append(W)
            canon.append(c)
        if len(kept) == count:
            break
    return np.stack(kept)


def fixed_coordinate_scale(target, W_best, etas, metric="diag-map"):
    """Fixed positive diagonal whitening using local curvature as a proposal aid.

    Curvature is regularized below by gamma. This is NOT a posterior covariance
    estimate. Any fixed positive scale gives the same target after Metropolis
    correction. Scaling positions by s is equivalent to momentum mass 1/s**2.
    """
    if metric == "unit":
        curvature = np.ones_like(W_best)
    else:
        curvature = np.empty(W_best.size)
        for j in range(W_best.size):
            h = 1e-4 * max(1.0, abs(W_best.ravel()[j]))
            plus, minus = W_best.copy(), W_best.copy()
            plus.ravel()[j] += h
            minus.ravel()[j] -= h
            gp = target(plus)[1].ravel()[j]
            gm = target(minus)[1].ravel()[j]
            curvature[j] = -(gp - gm) / (2 * h)
        curvature = np.maximum(curvature.reshape(W_best.shape), target.gamma)
    return 1.0 / np.sqrt(np.asarray(etas)[:, None, None] * curvature), curvature


def hmc_transition(W, target, etas, scale, eps, rng, n_leapfrog):
    """Reversible leapfrog in x=W/scale, followed by exact Metropolis correction."""
    L, E, N, M = W.shape
    flat = W.reshape(L * E, N, M)
    lp0, grad0 = target(flat)
    lp0, grad0 = lp0.reshape(L, E), grad0.reshape(W.shape)
    eta = etas[None, :, None, None]
    steps = eps[:, :, None, None]
    p0 = rng.normal(size=W.shape)
    proposal = W.copy()
    p = p0 + 0.5 * steps * grad0 * eta * scale[None]
    lp = lp0.copy()
    for k in range(n_leapfrog):
        proposal += steps * p * scale[None]
        lp, grad = target(proposal.reshape(L * E, N, M))
        lp, grad = lp.reshape(L, E), grad.reshape(W.shape)
        p += (0.5 if k == n_leapfrog - 1 else 1.0) * steps * grad * eta * scale[None]
    delta = (lp - lp0) * etas[None] - 0.5 * ((p * p - p0 * p0).sum((-2, -1)))
    finite = np.isfinite(delta) & np.isfinite(lp)
    log_accept = np.where(finite, np.minimum(delta, 0.0), -np.inf)
    accept_prob = np.exp(log_accept)
    accepted = np.log(rng.uniform(size=delta.shape)) < log_accept
    result = np.where(accepted[:, :, None, None], proposal, W)
    base_lp = np.where(accepted, lp, lp0)
    divergent = (~finite) | (np.abs(delta) > 1000)
    return result, base_lp, accepted, accept_prob, divergent, delta


def swap_log_ratio(eta_left, eta_right, lp_left, lp_right):
    return (eta_left - eta_right) * (lp_right - lp_left)


def replica_exchange(W, lp, etas, ids, rng, parity):
    """Disjoint adjacent exchanges; lp is the UNtempered complete log posterior."""
    counts = np.zeros((W.shape[0], len(etas) - 1), dtype=np.int64)
    attempts = np.zeros_like(counts)
    for j in range(parity, len(etas) - 1, 2):
        ratio = swap_log_ratio(etas[j], etas[j + 1], lp[:, j], lp[:, j + 1])
        take = np.log(rng.uniform(size=W.shape[0])) < np.minimum(ratio, 0.0)
        attempts[:, j] = 1
        counts[:, j] = take
        for array in (W, lp, ids):
            left = array[take, j].copy()
            array[take, j] = array[take, j + 1]
            array[take, j + 1] = left
    return W, lp, ids, counts, attempts


def atomic_npz(path: Path, **arrays):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


def run_replica_hmc(
    target, initial, etas, scale, rng, *, warmup, sample, thin,
    n_leapfrog=12, initial_step=0.2, target_accept=0.8, swap_every=1,
    checkpoint_path=None, checkpoint_every=1000, progress_every=1000,
):
    """Sample independent ladders. Burn-in-only dual averaging of step sizes.

    Checkpoints contain the current state, adaptation state, RNG state and all
    retained samples so far, but automatic restart is deliberately unsupported.
    They allow recovery/audit after interruption without silently appending a
    new chain to an old one.
    """
    etas = np.asarray(etas, dtype=float)
    W = np.array(initial, dtype=float, copy=True)
    L, E, N, M = W.shape
    eps = np.full((L, E), initial_step)
    mu = np.log(10 * eps)
    logepsbar, hbar = np.log(eps), np.zeros_like(eps)
    accepted = np.zeros((L, E), dtype=np.int64)
    divergences = np.zeros_like(accepted)
    swap_counts = np.zeros((L, E - 1), dtype=np.int64)
    swap_attempts = np.zeros_like(swap_counts)
    ids = np.tile(np.arange(E), (L, 1))
    last_endpoint = np.full((L, E), -1, dtype=np.int64)
    traversals = np.zeros_like(last_endpoint)
    kept, lp_kept, delta_kept, id_kept = [], [], [], []
    lp = target(W.reshape(L * E, N, M))[0].reshape(L, E)
    parity = 0
    for iteration in range(warmup + sample):
        if iteration == warmup:
            eps = np.exp(logepsbar)
            # Count production traversals only.
            last_endpoint[:] = -1
        # Random integration length prevents fixed-trajectory resonances.
        steps = int(rng.integers(max(1, n_leapfrog // 2), n_leapfrog + 1))
        W, lp, take, aprob, div, delta = hmc_transition(
            W, target, etas, scale, eps, rng, steps
        )
        production = iteration >= warmup
        if production:
            accepted += take
            divergences += div
        else:
            t = iteration + 1
            weight = 1.0 / (t + 10.0)
            hbar = (1 - weight) * hbar + weight * (target_accept - aprob)
            logeps = np.clip(mu - np.sqrt(t) / 0.05 * hbar, np.log(1e-5), np.log(2.0))
            smooth = t ** -0.75
            logepsbar = smooth * logeps + (1 - smooth) * logepsbar
            eps = np.exp(logeps)
        if (iteration + 1) % swap_every == 0:
            W, lp, ids, hits, tries = replica_exchange(W, lp, etas, ids, rng, parity)
            parity = 1 - parity
            if production:
                swap_counts += hits
                swap_attempts += tries
        if production:
            endpoints = ((0, 0), (1, E - 1)) if E > 1 else ()
            for endpoint, position in endpoints:
                labels = ids[:, position]
                previous = last_endpoint[np.arange(L), labels]
                crossing = (previous != -1) & (previous != endpoint)
                traversals[np.arange(L), labels] += crossing
                last_endpoint[np.arange(L), labels] = endpoint
            if (iteration - warmup) % thin == 0:
                kept.append(W.copy())
                lp_kept.append(lp.copy())
                delta_kept.append(delta.copy())
                id_kept.append(ids.copy())
        if checkpoint_path is not None and (
            (iteration + 1) % checkpoint_every == 0 or iteration + 1 == warmup + sample
        ):
            atomic_npz(
                Path(checkpoint_path), W=W, base_logpost=lp, eps=eps,
                logepsbar=logepsbar, hbar=hbar, ids=ids,
                iteration=np.array(iteration + 1), etas=etas, coordinate_scale=scale,
                accepted=accepted, divergences=divergences,
                swap_counts=swap_counts, swap_attempts=swap_attempts,
                last_endpoint=last_endpoint, endpoint_traversals=traversals,
                samples=np.asarray(kept), logpost_trace=np.asarray(lp_kept),
                delta_h_trace=np.asarray(delta_kept), replica_ids=np.asarray(id_kept),
                rng_state=np.array(json.dumps(rng.bit_generator.state)),
                parity=np.array(parity),
            )
        if progress_every and (iteration + 1) % progress_every == 0:
            print(f"HMC {iteration + 1}/{warmup + sample} "
                  f"({'sample' if production else 'warmup'}), "
                  f"best log posterior {lp.max():.6f}", flush=True)
    return {
        "samples": np.stack(kept), "base_logpost": np.stack(lp_kept),
        # Proposal energy errors are indexed by pre-swap temperature slots;
        # retained samples and identities are the post-swap states.
        "delta_h": np.stack(delta_kept), "replica_ids": np.stack(id_kept),
        "acceptance": accepted / sample, "divergences": divergences,
        "step_size": eps, "swap_counts": swap_counts, "swap_attempts": swap_attempts,
        "endpoint_traversals": traversals, "coordinate_scale": scale,
    }


def invariant_traces(samples, target, xis, probe_states):
    """Cheap exact predictions via the hidden sum, retaining chain/time axes."""
    D, C, E, N, M = samples.shape
    flat = samples.reshape(-1, N, M)
    q = np.empty((len(flat), xis.shape[1]))
    probes = np.empty((len(flat), len(probe_states)))
    lpost = np.empty(len(flat))
    for start in range(0, len(flat), 256):
        W = flat[start:start + 256]
        fields = np.einsum("bnm,hm->bhn", W, target.H)
        logh = log2cosh(fields).sum(-1)
        logz = np.logaddexp.reduce(logh, axis=-1)
        ph = np.exp(logh - logz[:, None])
        conditional_mean = np.tanh(fields)
        projected = np.einsum("bhn,nr->bhr", conditional_mean, xis) / N
        variance = (1 - conditional_mean ** 2).sum(-1) / N ** 2
        q[start:start + len(W)] = np.einsum("bh,bhr->br", ph, projected ** 2 + variance[..., None])
        score = log2cosh(np.einsum("pn,bnm->bpm", probe_states, W)).sum(-1)
        probes[start:start + len(W)] = score - logz[:, None]
        lpost[start:start + len(W)] = target(W)[0]
    traces = {
        "base_logpost": lpost.reshape(D, C, E),
        "wnorm": np.sqrt((samples ** 2).sum((-2, -1))),
    }
    for j in range(q.shape[-1]):
        traces[f"q{j + 1}"] = q[:, j].reshape(D, C, E)
    for j in range(probes.shape[-1]):
        traces[f"logp_probe{j}"] = probes[:, j].reshape(D, C, E)
    return traces


def convergence_diagnostics(traces, etas, min_ess=400, max_rhat=1.01):
    import arviz as az

    output = {}
    for ei, eta in enumerate(etas):
        rows = {}
        for name, trace in traces.items():
            x = np.asarray(trace[:, :, ei]).T  # independent ladder, draw
            if x.shape[0] < 2 or x.shape[1] < 4:
                rows[name] = {"rhat_rank": None, "ess_bulk": None, "ess_tail": None, "pass": False}
                continue
            rhat = float(az.rhat(x, method="rank"))
            bulk = float(az.ess(x, method="bulk"))
            tail = float(az.ess(x, method="tail", prob=(0.05, 0.95)))
            rows[name] = {
                "rhat_rank": rhat, "ess_bulk": bulk, "ess_tail": tail,
                "pass": bool(np.isfinite(rhat) and rhat <= max_rhat and bulk >= min_ess and tail >= min_ess),
            }
        output[str(float(eta))] = rows
    return output
