#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy>=2.4"]
# ///
"""Bounded correctness tests for the matched negative-phase control protocol."""

import copy
import json
import os

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

from rbm_exact import Teacher, log_prob_visible, spin_states
from nu_controls import (
    aggregate_datasets,
    array_hash,
    equilibrium_visible_sample,
    fixed_weight_controls,
    time_grid,
    train_arm,
    validate_sampling,
)


def main():
    validation = validate_sampling(100_000)
    print("independent sampling and gradient checks:", json.dumps(validation))
    for lr in (0.1, 0.02):
        updates, steps, times = time_grid(10, lr, 1)
        assert updates == round(10 / lr)
        np.testing.assert_allclose(steps * lr, times)
    try:
        time_grid(1, 0.03, 0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("inexact optimization horizon was silently rounded")

    N, M, K = 4, 2, 4
    V, H = spin_states(N), spin_states(M)
    teacher = Teacher.hadamard(N, [1.0, 0.5])
    teacher_lp = log_prob_visible(teacher.W, np.zeros(N), np.zeros(2), 1.0, V)
    data = teacher.sample(K, teacher_lp, V, np.random.default_rng(1000))
    W0 = np.random.default_rng(20000).normal(scale=0.01, size=(N, M))
    _, vectors = np.linalg.eigh(data.T @ data / N)
    initial_v = equilibrium_visible_sample(W0, np.zeros(N), np.zeros(M), 1.0, H, 2, np.random.default_rng(17))
    common = dict(W0=W0, data=data, V=V, teacher_lp=teacher_lp,
                  xis=teacher.xis, cvecs=vectors[:, ::-1][:, :M], H=H,
                  beta=1.0, gamma=3.4, horizon=0.4, eval_interval=0.1, lr=0.1,
                  initial_visible=initial_v)
    exact = train_arm(**common, kind="exact", nu=None, chains=0, rng=None)
    runs = [exact]
    for replicate in range(2):
        for kind, nu in (("iid", None), ("persistent", 1)):
            runs.append(train_arm(**common, kind=kind, nu=nu, chains=2,
                                  rng=np.random.default_rng(500 + replicate + 20 * (nu or 0))))
    assert all(run["initial_W_sha256"] == array_hash(W0) for run in runs)
    assert all(len(run["trajectory"]) == 5 for run in runs)
    assert all(run["trajectory"][-1]["step"] == 4 for run in runs)
    assert exact["trajectory"][-1]["negative_phase_residual"]["window_rms_norm"] == 0
    # The saved final W must correspond to the final diagnostic, with no extra update.
    final_lp = log_prob_visible(np.asarray(exact["final_W"]), np.zeros(N), np.zeros(M), 1, V)
    direct_rev = np.sum(np.exp(final_lp) * (final_lp - teacher_lp)) / N
    np.testing.assert_allclose(exact["trajectory"][-1]["rev"], direct_rev, atol=1e-14)
    record = {"dataset_seed": 0, "protocol_sha256": "test", "runs": runs}
    record2 = copy.deepcopy(record)
    record2["dataset_seed"] = 1
    # Identical copies only test inference accounting, not scientific replication.
    summary = aggregate_datasets([record, record2])
    assert summary["n_datasets"] == 2
    for group in summary["groups"].values():
        assert all(r["n_replicates"] == 2 for r in group["per_dataset"])
        assert group["vs_exact"]["rev_tail_excess"]["n_datasets"] == 2
    try:
        aggregate_datasets([record, record])
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate datasets were accepted")
    singleton = aggregate_datasets([record])
    for group in singleton["groups"].values():
        assert group["vs_exact"]["rev_tail_excess"]["sem"] is None
    frozen = fixed_weight_controls(teacher.W, beta=1.0, H=H, V=V,
                                   nus=[1, 4], chain_counts=[1, 4], steps=128,
                                   replicates=8, seed=101)
    iid = [row for row in frozen["controls"] if row["kind"] == "iid"]
    for row in iid:
        assert abs(row["instantaneous_rms"] / row["exact_stationary_instantaneous_rms"] - 1) < 0.15
        # No pseudoreplicated SE from correlated time steps: one mean per trajectory.
        assert len(row["replicate_mean_residuals"]) == 8
    json.dumps(summary, allow_nan=False)
    json.dumps(frozen, allow_nan=False)
    print("PASS: matched horizon/init, final-state consistency, frozen-W variance, dataset-level aggregation")


if __name__ == "__main__":
    main()
