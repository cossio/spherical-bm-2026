# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy>=2.4"]
# ///
"""Finite-state RBM numerical routines and reproducible training."""

from __future__ import annotations

import hashlib
import time

import numpy as np

from nu_controls import array_hash, equilibrium_visible_sample, observables, update_gradient
from rbm_exact import gibbs_sweeps, log2cosh, neg_phase_chains, neg_phase_exact


def mixed_time_grid(horizon, lr, dense_until=20.0, dense_interval=0.1, coarse_interval=10.0):
    """Dense early diagnostics and a global coarse grid, including the endpoint."""
    if min(horizon, lr, dense_interval, coarse_interval) <= 0 or dense_until < 0:
        raise ValueError("Positive horizon, learning rate and intervals are required")
    updates = int(round(horizon / lr))
    if not np.isclose(updates * lr, horizon, rtol=0, atol=1e-9):
        raise ValueError("horizon must be an integer multiple of lr")
    early = np.arange(int(np.floor(min(horizon, dense_until) / dense_interval)) + 1) * dense_interval
    coarse = np.arange(int(np.floor(horizon / coarse_interval)) + 1) * coarse_interval
    proposed = np.r_[early, coarse[coarse > dense_until], horizon]
    steps = np.unique(np.rint(proposed / lr).astype(int))
    if not np.allclose(np.rint(proposed / lr) * lr, proposed, rtol=0, atol=1e-9):
        raise ValueError("Every diagnostic time must be an integer multiple of lr")
    return updates, steps, steps * lr


def advance_visible(v, W, *, beta, H, kind, refresh_mask, rng):
    """One full blocked Gibbs sweep or an exact redraw on selected chains."""
    if kind not in {"gibbs", "refresh"}:
        raise ValueError(f"Unknown transition kernel {kind}")
    mask = np.asarray(refresh_mask, dtype=bool)
    if mask.shape != (len(v),):
        raise ValueError("Refresh mask must contain one decision per visible chain")
    result = v.copy()
    count = int(mask.sum())
    if count:
        a, b = np.zeros(W.shape[0]), np.zeros(W.shape[1])
        if kind == "gibbs":
            result[mask] = gibbs_sweeps(v[mask], W, a, b, beta, 1, rng)
        else:
            result[mask] = equilibrium_visible_sample(W, a, b, beta, H, count, rng)
    return result


def expected_negative_phase(v, W, *, beta, H, V, kind, p):
    """E[next v tanh(beta vW)^T | current W, retained visible states].

    This is a conditional transition expectation, not an assertion of bias in
    an ensemble of training trajectories. Gibbs uses exact visible enumeration
    only at saved diagnostic times; training still uses one sampled state.
    """
    if not 0 <= p <= 1:
        raise ValueError("Refresh probability must be in [0,1]")
    a, b = np.zeros(W.shape[0]), np.zeros(W.shape[1])
    exact = neg_phase_exact(W, a, b, beta, H)
    if kind == "iid":
        if p != 1:
            raise ValueError("Fresh iid sampling requires p=1")
        return exact
    held = neg_phase_chains(v, W, a, b, beta)
    if kind == "refresh":
        return (1 - p) * held + p * exact
    if kind != "gibbs":
        raise ValueError(f"Unknown sampled arm {kind}")
    if p == 0:
        return held
    # P(h|v_current), followed by P(v_next|h), then the same RB statistic.
    fields_current = beta * (v @ W)
    ph = np.exp(fields_current @ H.T - log2cosh(fields_current).sum(1)[:, None])
    fields_hidden = beta * (H @ W.T)
    pv_given_h = np.exp(V @ fields_hidden.T - log2cosh(fields_hidden).sum(1)[None, :])
    next_hidden_mean = np.tanh(beta * (V @ W))
    moments = (V[:, :, None] * next_hidden_mean[:, None, :]).reshape(len(V), -1)
    refreshed = (ph.mean(0) @ (pv_given_h.T @ moments)).reshape(W.shape)
    return (1 - p) * held + p * refreshed


def _growth_projection(W, moment_error, beta):
    left, singular, right = np.linalg.svd(W, full_matrices=False)
    return float(-2 * beta * singular[0] * (left[:, 0] @ moment_error @ right[0]))


def train_slow_arm(*, W0, data, V, teacher_lp, xis, cvecs, H, beta, gamma,
                   lr, horizon, kind, p, chains, rng, schedule_rng,
                   initial_visible, dense_until=20.0, dense_interval=0.1,
                   coarse_interval=10.0):
    """Run a single arm with diagnostics tied to the actual pre-update W.

    Observables and W refer to update n before its parameter change. The
    retained state, transition expectation and realized negative phase are
    evaluated at that same W. The terminal point consumes no extra transition.
    """
    if kind not in {"exact", "iid", "gibbs", "refresh"}:
        raise ValueError(f"Unknown arm {kind}")
    if kind != "exact" and (chains < 1 or not 0 < p <= 1):
        raise ValueError("Sampled arms require chains>=1 and 0<p<=1")
    if kind == "iid" and p != 1:
        raise ValueError("Fresh iid arm requires p=1")
    if kind != "exact" and initial_visible.shape != (chains, W0.shape[0]):
        raise ValueError("Initial visible states have the wrong shape")
    started = time.perf_counter()
    W, v = W0.copy(), initial_visible.copy()
    N, M = W.shape
    a, b = np.zeros(N), np.zeros(M)
    updates, eval_steps, _ = mixed_time_grid(horizon, lr, dense_until, dense_interval, coarse_interval)
    eval_set = set(eval_steps.tolist())
    history, schedule_digest = [], hashlib.sha256()
    refresh_counts = np.zeros(chains, dtype=int)
    for step in range(updates + 1):
        observe = step in eval_set
        schedule_before = schedule_digest.hexdigest() if observe and kind != "exact" else None
        exact = neg_phase_exact(W, a, b, beta, H)
        retained = v.copy() if observe and kind != "exact" else None
        mask, estimate = None, None
        if step < updates:
            if kind == "exact":
                estimate = exact
            else:
                mask = np.ones(chains, dtype=bool) if p == 1 else schedule_rng.random(chains) < p
                refresh_counts += mask
                schedule_digest.update(mask.tobytes())
                v = advance_visible(v, W, beta=beta, H=H,
                                    kind="refresh" if kind == "iid" else kind,
                                    refresh_mask=mask, rng=rng)
                estimate = neg_phase_chains(v, W, a, b, beta)
        if observe:
            item = observables(W, data, V, teacher_lp, xis, cvecs, beta, gamma, H)
            item.update(step=step, time=float(step * lr), W=W.tolist(),
                        schedule_prefix_sha256=schedule_before)
            if kind != "exact":
                expected = expected_negative_phase(retained, W, beta=beta, H=H, V=V, kind=kind, p=p)
                residual = None if estimate is None else estimate - exact
                item["sampler_at_current_W"] = {
                    "has_next_update": step < updates,
                    "retained_visible_before_transition": retained.tolist(),
                    "visible_after_transition": v.tolist() if step < updates else None,
                    "refresh_mask": mask.tolist() if mask is not None else None,
                    "exact_negative_phase": exact.tolist(),
                    "conditional_expected_residual": (expected - exact).tolist(),
                    "realized_residual": residual.tolist() if residual is not None else None,
                    "conditional_expected_lambda1_rate_error": _growth_projection(W, expected - exact, beta),
                    "realized_lambda1_rate_error": _growth_projection(W, residual, beta) if residual is not None else None,
                    "retained_q_data": np.mean((retained @ cvecs / np.sqrt(N)) ** 2, axis=0).tolist(),
                    "retained_q_teacher": np.mean((retained @ xis / N) ** 2, axis=0).tolist(),
                    "sampled_q_data": np.mean((v @ cvecs / np.sqrt(N)) ** 2, axis=0).tolist() if step < updates else None,
                    "sampled_q_teacher": np.mean((v @ xis / N) ** 2, axis=0).tolist() if step < updates else None,
                    "refresh_counts_before_update": (refresh_counts - mask if mask is not None else refresh_counts).tolist(),
                }
            history.append(item)
        if step < updates:
            W += lr * update_gradient(W, data, estimate, beta, gamma)
            if not np.isfinite(W).all():
                raise FloatingPointError(f"Nonfinite weights at update {step}")
    return {
        "kind": kind, "p": p, "chains": chains, "lr": lr, "n_updates": updates,
        "initial_W_sha256": array_hash(W0), "initial_visible_sha256": array_hash(initial_visible),
        "final_W": W.tolist(), "final_W_sha256": array_hash(W),
        "refresh_counts": refresh_counts.tolist(),
        "schedule_sha256": schedule_digest.hexdigest() if kind != "exact" else None,
        "trajectory": history, "elapsed_seconds": time.perf_counter() - started,
    }
