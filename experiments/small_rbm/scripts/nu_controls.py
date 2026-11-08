# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy>=2.4"]
# ///
"""Finite-state RBM numerical routines and reproducible training."""

from __future__ import annotations

import hashlib
import time

import numpy as np

from rbm_exact import (
    gibbs_sweeps,
    kl,
    log2cosh,
    log_prob_visible,
    logpost_and_grad,
    logsumexp,
    neg_phase_chains,
    neg_phase_exact,
    spin_states,
)


def array_hash(array):
    array = np.ascontiguousarray(array, dtype="<f8")
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def hidden_probabilities(W, a, b, beta, H):
    fields = beta * (a[None, :] + H @ W.T)
    logits = log2cosh(fields).sum(axis=1) + beta * (H @ b)
    return np.exp(logits - logsumexp(logits))


def equilibrium_visible_sample(W, a, b, beta, H, count, rng):
    """Independent exact draws: sample P(h) over 2**M states, then P(v|h)."""
    h = H[rng.choice(len(H), size=count, p=hidden_probabilities(W, a, b, beta, H))]
    p_up = 0.5 * (1.0 + np.tanh(beta * (a + h @ W.T)))
    return np.where(rng.random(p_up.shape) < p_up, 1.0, -1.0)


def update_gradient(W, data, neg, beta, gamma):
    K = len(data)
    positive = data.T @ np.tanh(beta * (data @ W)) / K
    return beta * (positive - neg) - gamma / K * W


def time_grid(horizon, lr, interval):
    """A common continuous optimization time grid; reject silent rounding."""
    steps = int(round(horizon / lr))
    if not np.isclose(steps * lr, horizon, rtol=0, atol=1e-9):
        raise ValueError("horizon must be an integer multiple of every learning rate")
    times = np.arange(int(np.floor(horizon / interval)) + 1) * interval
    if not np.isclose(times[-1], horizon, rtol=0, atol=1e-9):
        times = np.append(times, horizon)
    sampled = np.rint(times / lr).astype(int)
    if not np.allclose(sampled * lr, times, rtol=0, atol=1e-9):
        raise ValueError("eval-interval must be an integer multiple of every learning rate")
    return steps, sampled, times


def observables(W, data, V, teacher_lp, xis, cvecs, beta, gamma, H):
    N, M = W.shape
    a, b = np.zeros(N), np.zeros(M)
    lp = log_prob_visible(W, a, b, beta, V)
    p = np.exp(lp)
    left, singular, _ = np.linalg.svd(W, full_matrices=False)
    teacher_dirs = xis / np.sqrt(N)
    rank = int(np.sum(singular > max(1e-12, singular[0] * 1e-10)))
    u_teacher = np.sum((left[:, :rank].T @ teacher_dirs) ** 2, axis=0)
    u_data = np.sum(left[:, :cvecs.shape[1]] * cvecs, axis=0) ** 2
    q_teacher = p @ (V @ xis / N) ** 2
    q_data = p @ (V @ cvecs / np.sqrt(N)) ** 2
    neg = neg_phase_exact(W, a, b, beta, H)
    return {
        "fwd": float(kl(teacher_lp, lp) / N),
        "rev": float(kl(lp, teacher_lp) / N),
        "lam": (singular**2).tolist(),
        "u_teacher": u_teacher.tolist(),
        "u_data": u_data.tolist(),
        "q_teacher": q_teacher.tolist(),
        "q_data": q_data.tolist(),
        "wnorm": float(np.linalg.norm(W)),
        "exact_gradient_norm": float(np.linalg.norm(update_gradient(W, data, neg, beta, gamma))),
    }


def train_arm(*, W0, data, V, teacher_lp, xis, cvecs, H, beta, gamma,
              lr, horizon, eval_interval, kind, nu, chains, rng,
              initial_visible, save_weights=False):
    started = time.perf_counter()
    N, M = W0.shape
    a, b = np.zeros(N), np.zeros(M)
    W = W0.copy()
    v = initial_visible.copy()
    updates, eval_steps, times = time_grid(horizon, lr, eval_interval)
    eval_set = set(eval_steps.tolist())
    history = []
    error_sum = np.zeros_like(W)
    error_sq_sum = 0.0
    error_last = np.zeros_like(W)
    count = 0
    for step in range(updates + 1):
        if step in eval_set:
            item = observables(W, data, V, teacher_lp, xis, cvecs, beta, gamma, H)
            item.update(step=step, time=float(step * lr))
            mean_error = error_sum / max(1, count)
            mse = error_sq_sum / max(1, count)
            item["negative_phase_residual"] = {
                "updates_in_window": count,
                "last": error_last.tolist() if count else None,
                "window_mean": mean_error.tolist() if count else None,
                "window_mean_norm": float(np.linalg.norm(mean_error)) if count else None,
                "window_rms_norm": float(np.sqrt(mse)) if count else None,
                "window_centered_rms_norm": float(np.sqrt(max(0.0, mse - np.sum(mean_error**2)))) if count else None,
            }
            if kind != "exact":
                item["q_chain_teacher"] = np.mean((v @ xis / N) ** 2, axis=0).tolist()
                item["q_chain_data"] = np.mean((v @ cvecs / np.sqrt(N)) ** 2, axis=0).tolist()
            if save_weights:
                item["W"] = W.tolist()
            history.append(item)
            error_sum.fill(0)
            error_sq_sum, count = 0.0, 0
        if step == updates:
            break
        exact = neg_phase_exact(W, a, b, beta, H)
        if kind == "exact":
            estimate = exact
        elif kind == "iid":
            v = equilibrium_visible_sample(W, a, b, beta, H, chains, rng)
            estimate = neg_phase_chains(v, W, a, b, beta)
        elif kind == "persistent":
            v = gibbs_sweeps(v, W, a, b, beta, nu, rng)
            estimate = neg_phase_chains(v, W, a, b, beta)
        else:
            raise ValueError(f"unknown arm: {kind}")
        error_last = estimate - exact
        error_sum += error_last
        error_sq_sum += float(np.sum(error_last**2))
        count += 1
        W += lr * update_gradient(W, data, estimate, beta, gamma)
        if not np.isfinite(W).all():
            raise FloatingPointError(f"nonfinite weights at update {step}")
    assert len(history) == len(times)
    return {
        "kind": kind, "nu": nu, "chains": chains, "lr": lr,
        "n_updates": updates, "initial_W_sha256": array_hash(W0),
        "final_W": W.tolist(), "final_W_sha256": array_hash(W),
        "trajectory": history, "elapsed_seconds": time.perf_counter() - started,
    }


def moment_noise_reference(W, a, b, beta, H, V):
    """Exact single-visible-state Rao--Blackwell moment variance, Frobenius trace."""
    lp = log_prob_visible(W, a, b, beta, V)
    p = np.exp(lp)
    hidden_mean = np.tanh(beta * (b + V @ W))
    exact = neg_phase_exact(W, a, b, beta, H)
    second = V.shape[1] * float(p @ np.sum(hidden_mean**2, axis=1))
    return exact, max(0.0, second - float(np.sum(exact**2)))


def fixed_weight_controls(W, *, beta, H, V, nus, chain_counts, steps, replicates, seed):
    """Frozen-W stationary controls: exact initialization removes startup bias.

    Each independent trajectory yields one time-mean residual. Its uncertainty
    is measured across these trajectories, not by pretending time steps are iid.
    Instantaneous variance has the same exact marginal value for iid and
    persistent samples; their time-mean variances can differ through correlation.
    """
    N, M = W.shape
    a, b = np.zeros(N), np.zeros(M)
    exact, variance = moment_noise_reference(W, a, b, beta, H, V)
    out = []
    for nc in chain_counts:
        for nu in [None] + list(nus):
            kind = "iid" if nu is None else "persistent"
            means, rms2, lag1, mean_norm2 = [], [], [], []
            for replicate in range(replicates):
                rng = np.random.default_rng(np.random.SeedSequence([seed, nc, nu or 0, replicate]))
                v = equilibrium_visible_sample(W, a, b, beta, H, nc, rng)
                residuals = np.empty((steps, N, M))
                for i in range(steps):
                    if nu is None:
                        v = equilibrium_visible_sample(W, a, b, beta, H, nc, rng)
                    else:
                        v = gibbs_sweeps(v, W, a, b, beta, nu, rng)
                    residuals[i] = neg_phase_chains(v, W, a, b, beta) - exact
                mean = residuals.mean(axis=0)
                means.append(mean)
                mean_norm2.append(float(np.sum(mean**2)))
                rms2.append(float(np.mean(np.sum(residuals**2, axis=(1, 2)))))
                centered = residuals - mean
                denom = float(np.sum(centered**2))
                lag1.append(float(np.sum(centered[1:] * centered[:-1]) / denom) if denom else 0.0)
            means = np.asarray(means)
            sem = means.std(axis=0, ddof=1) / np.sqrt(replicates) if replicates > 1 else None
            out.append({
                "kind": kind, "nu": nu, "chains": nc,
                "mean_residual": means.mean(axis=0).tolist(),
                "replicate_mean_residuals": means.tolist(),
                "mean_residual_sem": sem.tolist() if sem is not None else None,
                "instantaneous_rms": float(np.sqrt(np.mean(rms2))),
                "exact_stationary_instantaneous_rms": float(np.sqrt(variance / nc)),
                "time_mean_rms": float(np.sqrt(np.mean(mean_norm2))),
                "iid_time_mean_rms_reference": float(np.sqrt(variance / (nc * steps))),
                "lag1_frobenius_correlation": float(np.mean(lag1)),
            })
    return {"W_sha256": array_hash(W), "steps": steps, "replicates": replicates,
            "initialization": "independent exact equilibrium for each frozen-W trajectory",
            "controls": out}


def validate_sampling(sample_count=100_000):
    """Small independent enumeration and finite-difference checks; no fitting."""
    rng = np.random.default_rng(918273)
    N, M, beta, gamma = 5, 3, 0.8, 3.4
    V, H = spin_states(N), spin_states(M)
    W = rng.normal(scale=0.35, size=(N, M))
    a, b = rng.normal(scale=0.1, size=N), rng.normal(scale=0.1, size=M)
    pv = np.exp(log_prob_visible(W, a, b, beta, V))
    ph = hidden_probabilities(W, a, b, beta, H)
    field = beta * (a[None, :] + H @ W.T)
    conditional_lp = V @ field.T - log2cosh(field).sum(axis=1)[None, :]
    mixture = np.exp(conditional_lp) @ ph
    deterministic_error = float(np.max(np.abs(pv - mixture)))
    samples = equilibrium_visible_sample(W, a, b, beta, H, sample_count, rng)
    exact, _ = moment_noise_reference(W, a, b, beta, H, V)
    estimate = neg_phase_chains(samples, W, a, b, beta)
    values = V[:, :, None] * np.tanh(beta * (b + V @ W))[:, None, :]
    second = np.einsum("v,vnm->nm", pv, values**2)
    stderr = np.sqrt(np.maximum(0.0, second - exact**2) / sample_count)
    max_moment_z = float(np.max(np.abs(estimate - exact) / np.maximum(stderr, 1e-15)))
    indices = ((samples > 0).astype(np.int64) * (1 << np.arange(N))).sum(axis=1)
    empirical = np.bincount(indices, minlength=len(V)) / sample_count
    max_probability_z = float(np.max(np.abs(empirical - pv) / np.sqrt(pv * (1 - pv) / sample_count)))
    # Training uses zero fields: independently check its normalization and prior.
    a0, b0 = np.zeros(N), np.zeros(M)
    data = V[rng.integers(len(V), size=7)]
    negative = neg_phase_exact(W, a0, b0, beta, H)
    grad = update_gradient(W, data, negative, beta, gamma)
    _, existing = logpost_and_grad(W, data, a0, b0, beta, H, gamma)
    gradient_match = float(np.max(np.abs(grad - existing / len(data))))
    finite_difference = np.zeros_like(W)
    epsilon = 1e-6
    for idx in np.ndindex(W.shape):
        plus, minus = W.copy(), W.copy()
        plus[idx] += epsilon
        minus[idx] -= epsilon
        lp_plus = logpost_and_grad(plus, data, a0, b0, beta, H, gamma)[0]
        lp_minus = logpost_and_grad(minus, data, a0, b0, beta, H, gamma)[0]
        finite_difference[idx] = (lp_plus - lp_minus) / (2 * epsilon * len(data))
    finite_difference_error = float(np.max(np.abs(grad - finite_difference)))
    result = {
        "sample_count": sample_count, "hidden_mixture_max_error": deterministic_error,
        "sample_moment_max_z": max_moment_z, "sample_probability_max_z": max_probability_z,
        "existing_gradient_max_error": gradient_match,
        "finite_difference_gradient_max_error": finite_difference_error,
    }
    result["passed"] = bool(deterministic_error < 1e-12 and max_moment_z < 7
                            and max_probability_z < 7 and gradient_match < 1e-12
                            and finite_difference_error < 1e-7)
    if not result["passed"]:
        raise AssertionError(result)
    return result


def estimate_across_datasets(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = values.mean(axis=0)
    sem = values.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else None
    return {"n_datasets": n, "mean": mean.tolist(),
            "sem": sem.tolist() if sem is not None else None,
            "normal_95_interval": [np.asarray(mean - 1.96 * sem).tolist(), np.asarray(mean + 1.96 * sem).tolist()] if sem is not None else None}


def curve_metrics(rows, baseline, tail_fraction):
    """Compare replicate-averaged trajectories on the prespecified time grid."""
    end_time = baseline[-1]["time"]
    tail = np.asarray([r["time"] >= end_time * (1 - tail_fraction) for r in baseline])
    result = {}
    for field in ("fwd", "rev"):
        arm = np.mean([[r[field] for r in run] for run in rows], axis=0)
        ref = np.asarray([r[field] for r in baseline])
        result[f"{field}_tail_excess"] = float((arm - ref)[tail].mean())
        result[f"{field}_final_excess"] = float(arm[-1] - ref[-1])
        result[f"{field}_tail_minus_grid_min_excess"] = float(arm[tail].mean() - arm.min() - ref[tail].mean() + ref.min())
        result[f"{field}_tail_drift"] = float(arm[tail][-1] - arm[tail][0])
        result[f"{field}_tail_drift_excess"] = float(arm[tail][-1] - arm[tail][0] - ref[tail][-1] + ref[tail][0])
    lam = np.mean([[r["lam"][0] for r in run] for run in rows], axis=0)
    lam_ref = np.asarray([r["lam"][0] for r in baseline])
    result["lambda1_max_pointwise_excess"] = float(np.max(lam - lam_ref))
    result["lambda1_peak_excess"] = float(lam.max() - lam_ref.max())
    result["lambda1_tail_excess"] = float((lam - lam_ref)[tail].mean())
    result["lambda1_peak_minus_reference_tail"] = float(lam.max() - lam_ref[tail].mean())
    result["lambda1_tail_drift"] = float(lam[tail][-1] - lam[tail][0])
    return result


def aggregate_datasets(records, tail_fraction=0.2):
    """One independent unit is one dataset, regardless of chain replication."""
    if not records:
        raise ValueError("no dataset records")
    seeds = [r["dataset_seed"] for r in records]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate dataset seeds; refuse pseudoreplication")
    if len({r["protocol_sha256"] for r in records}) != 1:
        raise ValueError("incompatible experiment protocols")
    grouped = {}
    deterministic = {}
    reference_groups = None
    for record in records:
        if "complete" in record and not record["complete"]:
            raise ValueError(f"incomplete dataset {record['dataset_seed']}")
        exact = {run["lr"]: run for run in record["runs"] if run["kind"] == "exact"}
        groups = {}
        for run in record["runs"]:
            if run["kind"] != "exact":
                key = f"lr={run['lr']:g}|{run['kind']}|nu={run['nu']}|chains={run['chains']}"
                groups.setdefault(key, []).append(run)
        arm_counts = {key: len(runs) for key, runs in groups.items()}
        if reference_groups is None:
            reference_groups = arm_counts
        elif arm_counts != reference_groups:
            raise ValueError("datasets contain different stochastic arms or replicate counts")
        for key, runs in groups.items():
            exemplar = runs[0]
            baseline = exact[exemplar["lr"]]["trajectory"]
            rows = [r["trajectory"] for r in runs]
            times = [row["time"] for row in baseline]
            if any([row["time"] for row in run] != times for run in rows):
                raise ValueError("unaligned diagnostic time grids")
            metrics = curve_metrics(rows, baseline, tail_fraction)
            target = grouped.setdefault(key, {"per_dataset": [], "pointwise": []})
            entry = {"dataset_seed": record["dataset_seed"], "n_replicates": len(runs), "vs_exact": metrics}
            if exemplar["kind"] == "persistent":
                iid_key = f"lr={exemplar['lr']:g}|iid|nu=None|chains={exemplar['chains']}"
                iid_rows = [r["trajectory"] for r in groups[iid_key]]
                iid_mean = []
                for i, row in enumerate(iid_rows[0]):
                    iid_mean.append({"time": row["time"],
                                     "fwd": float(np.mean([s[i]["fwd"] for s in iid_rows])),
                                     "rev": float(np.mean([s[i]["rev"] for s in iid_rows])),
                                     "lam": np.mean([s[i]["lam"] for s in iid_rows], axis=0).tolist()})
                entry["vs_iid"] = curve_metrics(rows, iid_mean, tail_fraction)
            entry["replicate_final_rev_sd"] = float(np.std([r[-1]["rev"] for r in rows], ddof=1)) if len(rows) > 1 else None
            target["per_dataset"].append(entry)
            pointwise = {f: (np.mean([[row[f] for row in run] for run in rows], axis=0) - np.asarray([row[f] for row in baseline])).tolist() for f in ("rev", "fwd")}
            pointwise["lambda1"] = (np.mean([[row["lam"][0] for row in run] for run in rows], axis=0) - np.asarray([row["lam"][0] for row in baseline])).tolist()
            target["pointwise"].append(pointwise)
            target["time"] = [row["time"] for row in baseline]
        for lr, run in exact.items():
            rows = run["trajectory"]
            tail = [r for r in rows if r["time"] >= rows[-1]["time"] * (1 - tail_fraction)]
            deterministic.setdefault(str(lr), []).append({
                "dataset_seed": record["dataset_seed"],
                "final_exact_gradient_norm": rows[-1]["exact_gradient_norm"],
                "tail_rev_range": max(r["rev"] for r in tail) - min(r["rev"] for r in tail),
                "tail_rev_endpoint_change": tail[-1]["rev"] - tail[0]["rev"],
                "tail_fwd_endpoint_change": tail[-1]["fwd"] - tail[0]["fwd"],
            })
    for group in grouped.values():
        entries = group["per_dataset"]
        for contrast in ("vs_exact", "vs_iid"):
            if contrast in entries[0]:
                group[contrast] = {metric: estimate_across_datasets([entry[contrast][metric] for entry in entries]) for metric in entries[0][contrast]}
        group["pointwise_vs_exact"] = {field: estimate_across_datasets([entry[field] for entry in group["pointwise"]]) for field in ("rev", "fwd", "lambda1")}
        del group["pointwise"]
    return {"dataset_seeds": seeds, "n_datasets": len(seeds), "groups": grouped,
            "deterministic_convergence": deterministic, "tail_fraction": tail_fraction,
            "inference_unit": "dataset; stochastic replicates averaged within dataset",
            "interval_note": "Normal 1.96*SEM pointwise intervals; exploratory maxima/minima are selected on the saved time grid and are not corrected significance tests.",
            "residual_note": "Training residual windows describe noise plus path-dependent mismatch; they do not isolate conditional bias. Frozen-W controls separately measure stationary sample noise and time correlations."}
