#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy>=2.4"]
# ///
"""Finite-state RBM numerical routines and reproducible training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

from nu_controls import array_hash, equilibrium_visible_sample, validate_sampling
from nu_slow import mixed_time_grid, train_slow_arm
from rbm_exact import Teacher, data_spectrum, log_prob_visible, spin_states


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--M", type=int, default=3)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--A1", type=float, default=1.0)
    parser.add_argument("--A2", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=3.4)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--refresh-probabilities", nargs="+", type=float, default=[1.0, 0.25, 0.0625])
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--horizon", type=float, default=3000.0)
    parser.add_argument("--dense-until", type=float, default=20.0)
    parser.add_argument("--dense-interval", type=float, default=0.1)
    parser.add_argument("--coarse-interval", type=float, default=10.0)
    parser.add_argument("--init-scale", type=float, default=0.01)
    parser.add_argument("--validation-samples", type=int, default=100_000)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    if args.N < 4 or args.N & (args.N - 1) or not 1 <= args.M <= args.N or args.K < 1:
        parser.error("N must be a power of two >=4, 1<=M<=N, and K>=1")
    if min(args.lr, args.beta, args.init_scale) <= 0 or args.gamma < 0 or min(args.replicates, args.chains) < 1:
        parser.error("Require positive learning rate, beta, initialization scale and counts; gamma>=0")
    if any(not 0 < p <= 1 for p in args.refresh_probabilities) or 1.0 not in args.refresh_probabilities:
        parser.error("Refresh probabilities must be in (0,1] and include1")
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0 or len(set(args.refresh_probabilities)) != len(args.refresh_probabilities):
        parser.error("Seeds and probabilities must be unique; seeds nonnegative")
    mixed_time_grid(args.horizon, args.lr, args.dense_until, args.dense_interval, args.coarse_interval)
    args.outdir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        if (args.outdir / f"dataset_{seed:04d}.json").exists():
            raise FileExistsError("Use a fresh output directory; completed datasets are never overwritten")
    validation = validate_sampling(args.validation_samples)
    here = Path(__file__).resolve().parent
    sources = {name: hashlib.sha256((here / name).read_bytes()).hexdigest()
               for name in ("30_nu_slow.py", "nu_slow.py", "nu_controls.py", "rbm_exact.py")}
    protocol = {k: v for k, v in vars(args).items() if k not in {"seeds", "outdir"}}
    protocol.update(
        schema="matched_slow_nu_v1", source_sha256=sources,
        dataset_rng="default_rng(1000+seed)", initialization_rng="default_rng(20000+seed)",
        initial_visible_rng="SeedSequence([20261103,seed,replicate,chains,99]); original nu convention",
        clock_rng="SeedSequence([20261105,seed,replicate,chains,p_as_float64_uint32_words,71])",
        sampler_rng="SeedSequence([20261105,seed,replicate,chains,p_as_float64_uint32_words,kind_code,73])",
        gradient="beta*(mean_data_v_tanh(beta*vW)-sampled_negative)-gamma/K*W",
        kernel="with probability p per chain and update perform one blocked Gibbs sweep or exact equilibrium refresh; otherwise retain v but recompute its RB moment using current W",
        matching="same dataset/W0/initialv; Gibbs and exact-refresh share Bernoulli clocks at matching p, transition random streams independent",
        observation_time="W_n before update n; transition and exact/conditional/realized moments all evaluated at W_n; terminal point draws no extra transition",
        limitations=[
            "Conditional expected residual given W and retained state is not unconditional training bias; average across replicate states before interpreting signed projections.",
            "Instantaneous residual norms measure noise as well as lag and are not bias estimates.",
            "Lazy exact refresh measures cadence and aging; matching Gibbs effects can still support generic out-of-equilibrium lag.",
            "Dataset, not stochastic replica, is the independent unit for reported population contrasts.",
            "WW^T is a spectral diagnostic, not the exact visible coupling of a binary-hidden RBM.",
        ])
    protocol_sha = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    environment = {"numpy": np.__version__, "python": platform.python_version(), "platform": platform.platform(),
                   "argv": sys.argv, "blas_threads": {k: os.environ[k] for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
    V, H = spin_states(args.N), spin_states(args.M)
    teacher = Teacher.hadamard(args.N, [args.A1, args.A2], beta=args.beta)
    teacher_lp = log_prob_visible(teacher.W, np.zeros(args.N), np.zeros(2), args.beta, V)
    for seed in args.seeds:
        start = time.perf_counter()
        data = teacher.sample(args.K, teacher_lp, V, np.random.default_rng(1000 + seed))
        W0 = np.random.default_rng(20_000 + seed).normal(scale=args.init_scale, size=(args.N, args.M))
        _, directions = np.linalg.eigh(data.T @ data / args.N)
        cvecs = directions[:, ::-1][:, :min(args.M, 3)]
        common = dict(W0=W0, data=data, V=V, teacher_lp=teacher_lp, xis=teacher.xis, cvecs=cvecs,
                      H=H, beta=args.beta, gamma=args.gamma, lr=args.lr, horizon=args.horizon,
                      dense_until=args.dense_until, dense_interval=args.dense_interval, coarse_interval=args.coarse_interval)
        record = {"dataset_seed": seed, "protocol": protocol, "protocol_sha256": protocol_sha,
                  "environment": environment, "validation": validation, "data": data.tolist(),
                  "data_sha256": array_hash(data), "initial_W": W0.tolist(), "initial_W_sha256": array_hash(W0),
                  "teacher_W_sha256": array_hash(teacher.W), "data_spectrum": data_spectrum(data, args.N).tolist(), "runs": []}
        exact = train_slow_arm(**common, kind="exact", p=None, chains=0, rng=None, schedule_rng=None,
                               initial_visible=np.empty((0, args.N)))
        exact["replicate"] = None
        record["runs"].append(exact)
        print(f"dataset={seed} exact elapsed={exact['elapsed_seconds']:.2f}s", flush=True)
        arms = [("iid", 1.0)] + [("gibbs", p) for p in args.refresh_probabilities] + [("refresh", p) for p in args.refresh_probabilities if p < 1]
        for replicate in range(args.replicates):
            init_seed = [20261103, seed, replicate, args.chains, 99]
            initial_v = equilibrium_visible_sample(W0, np.zeros(args.N), np.zeros(args.M), args.beta, H, args.chains,
                                                  np.random.default_rng(np.random.SeedSequence(init_seed)))
            clocks = {}
            for kind, p in arms:
                probability_code = np.array([p], dtype="<f8").view("<u4").astype(int).tolist()
                clock_seed = [20261105, seed, replicate, args.chains, *probability_code, 71]
                sampler_seed = [20261105, seed, replicate, args.chains, *probability_code,
                                {"iid": 1, "gibbs": 2, "refresh": 3}[kind], 73]
                run = train_slow_arm(**common, kind=kind, p=p, chains=args.chains,
                                     rng=np.random.default_rng(np.random.SeedSequence(sampler_seed)),
                                     schedule_rng=np.random.default_rng(np.random.SeedSequence(clock_seed)), initial_visible=initial_v)
                run.update(replicate=replicate, initial_visible_seed_sequence=init_seed,
                           sampler_seed_sequence=sampler_seed, clock_seed_sequence=clock_seed)
                if p in clocks:
                    if clocks[p] != run["schedule_sha256"]:
                        raise AssertionError("Matched Bernoulli schedules disagree")
                clocks[p] = run["schedule_sha256"]
                record["runs"].append(run)
                print(f"dataset={seed} rep={replicate} {kind} p={p:g} elapsed={run['elapsed_seconds']:.2f}s rev={run['trajectory'][-1]['rev']:.6g}", flush=True)
        if any(run["initial_W_sha256"] != record["initial_W_sha256"] for run in record["runs"]):
            raise AssertionError("Initial weights differ across arms")
        record.update(complete=True, elapsed_seconds=time.perf_counter() - start)
        output = args.outdir / f"dataset_{seed:04d}.json"
        write_json(output, record)
        print(f"Complete: {output} ({record['elapsed_seconds']:.2f}s)", flush=True)


if __name__ == "__main__":
    main()
