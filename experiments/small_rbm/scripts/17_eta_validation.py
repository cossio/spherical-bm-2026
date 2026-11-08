#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy>=2.4", "scipy>=1.17", "arviz>=0.23"]
# ///
"""Finite-state RBM numerical routines and reproducible training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np

from eta_validation import (
    ExactPosterior, array_hash, atomic_npz, convergence_diagnostics,
    distinct_basins, fixed_coordinate_scale, invariant_traces,
    polish_candidates, run_replica_hmc,
)
from rbm_exact import Teacher, data_spectrum, kl, log_prob_visible, spin_states


def finite_json(value):
    """JSON remains standards-compliant when diagnostics produce NaN."""
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(finite_json(value), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def predictive_evaluation(samples, etas, wanted, V, lp_teacher, group_size, n_eval, chunk, rng):
    """Visible-state sum is exact; averaging over W remains Monte Carlo.

    Balanced subsets take the same draw count from every independent ladder.
    Draws span the whole retained chain; nested random order assesses finite
    mixture bias separately from the first-half/second-half comparison.
    """
    D, C, _, N, M = samples.shape
    n = min(n_eval, D)
    selected = np.linspace(0, D - 1, n, dtype=int)
    selected = selected[rng.permutation(n)]
    counts = sorted(set([x for x in (16, 32, 64, 128, 256, 512, 1024) if x <= n] + [n]))
    a, b = np.zeros(N), np.zeros(M)
    output, kl_trace = {}, {}
    target_probability = np.exp(lp_teacher)

    def metrics(prob):
        prob = np.maximum(prob, np.finfo(float).tiny)
        prob /= prob.sum()
        lp = np.log(prob)
        return {"reverse": float(kl(lp, lp_teacher) / N),
                "forward": float(np.sum(target_probability * (lp_teacher - lp)) / N)}

    for eta in wanted:
        ei = int(np.flatnonzero(np.isclose(etas, eta, rtol=0, atol=1e-12))[0])
        sums = np.zeros((len(counts), C, len(V)))
        half_sums = np.zeros((2, C, len(V)))
        half_counts = np.zeros((2, C), dtype=int)
        reverse = np.empty((n, C))
        forward = np.empty((n, C))
        for chain in range(C):
            cumulative = np.zeros(len(V))
            pos = 0
            for start in range(0, n, chunk):
                indices = selected[start:start + chunk]
                lp = log_prob_visible(samples[indices, chain, ei], a, b, 1.0, V, chunk=chunk)
                probability = np.exp(lp)
                reverse[start:start + len(lp), chain] = kl(lp, lp_teacher) / N
                forward[start:start + len(lp), chain] = kl(lp_teacher, lp) / N
                for j in range(len(lp)):
                    cumulative += probability[j]
                    pos += 1
                    if pos in counts:
                        sums[counts.index(pos), chain] = cumulative
                    half = int(indices[j] >= D // 2)
                    half_sums[half, chain] += probability[j]
                    half_counts[half, chain] += 1
        curves = []
        for ci, count in enumerate(counts):
            row = {"draws_per_ladder": count, "total_draws": count * C,
                   "all": metrics(sums[ci].sum(0) / (count * C)),
                   "dispersed": metrics(sums[ci, :group_size].sum(0) / (count * group_size)),
                   "optimized": metrics(sums[ci, group_size:].sum(0) / (count * group_size))}
            curves.append(row)
        halves = [metrics(half_sums[h].sum(0) / half_counts[h].sum())
                  if half_counts[h].sum() else None for h in range(2)]
        # Restore chronological order before autocorrelation diagnostics.
        chronological = np.argsort(selected)
        kl_trace[str(float(eta))] = {
            "reverse": reverse[chronological], "forward": forward[chronological],
            "draw_indices": selected[chronological],
        }
        output[str(float(eta))] = {
            "counts": curves, "halves": halves,
            "typical_reverse": float(reverse.mean()),
            "typical_forward": float(forward.mean()),
            "individual_ladder_predictive": [metrics(sums[-1, c] / n) for c in range(C)],
        }
        print(f"Predictive eta={eta:g}: reverse={curves[-1]['all']['reverse']:.6f}, "
              f"{n} draws per ladder", flush=True)
    return output, kl_trace


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True, help="Original dataset seed; uses RNG 1000+seed")
    ap.add_argument("--sampler-seed", type=int, default=20261103, help="Independent sampler replication seed")
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--A1", type=float, default=1.0)
    ap.add_argument("--A2", type=float, default=0.5)
    ap.add_argument("--etas", type=float, nargs="+", default=[0.2, 0.5, 1, 2, 5, 20])
    ap.add_argument("--buffers", type=float, nargs="*", default=[0.05, 0.075, 0.1, 0.15, 0.3, 0.75, 1.5, 3, 7.5, 10, 15])
    ap.add_argument("--ladders-per-group", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=3000)
    ap.add_argument("--sample", type=int, default=6000)
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--leapfrog", type=int, default=12)
    ap.add_argument("--step-size", type=float, default=0.2)
    ap.add_argument("--target-accept", type=float, default=0.8)
    ap.add_argument("--swap-every", type=int, default=1)
    ap.add_argument("--metric", choices=["unit", "diag-map"], default="diag-map")
    ap.add_argument("--map-restarts", type=int, default=24)
    ap.add_argument("--map-maxiter", type=int, default=2000)
    ap.add_argument("--n-eval", type=int, default=256, help="Visible-enumeration draws PER ladder and requested eta")
    ap.add_argument("--eval-chunk", type=int, default=16)
    ap.add_argument("--max-rhat", type=float, default=1.01)
    ap.add_argument("--min-ess", type=float, default=400)
    ap.add_argument("--predictive-tolerance", type=float, default=0.002, help="nats/spin; a diagnostic tolerance, not significance")
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    ap.add_argument("--outdir", type=Path, default=Path("results/eta_validation"))
    args = ap.parse_args()
    if args.gamma <= 0 or args.N < 4 or args.M < 1 or args.K < 1:
        ap.error("gamma>0, N>=4 (power of 2), M>=1 and K>=1 are required")
    if min(args.sample, args.thin, args.leapfrog, args.swap_every, args.n_eval,
           args.map_restarts, args.eval_chunk, args.checkpoint_every) <= 0:
        ap.error("iteration counts, thinning, restarts, and chunks must be positive")
    if args.warmup < 1 or args.ladders_per_group < 2:
        ap.error("warmup>=1 and at least two ladders per initialization group are required")
    if not 0 < args.target_accept < 1 or args.step_size <= 0:
        ap.error("target acceptance must lie in (0,1), and step size must be positive")
    wanted = np.array(sorted(set(args.etas + [1.0])))
    etas = np.array(sorted(set(wanted.tolist() + args.buffers)))
    if np.any(etas <= 0) or not np.all(np.isfinite(etas)):
        ap.error("all etas must be finite and positive")
    stamp = f"K{args.K}_N{args.N}_M{args.M}_A{args.A1:g}-{args.A2:g}_g{args.gamma:g}_s{args.seed}_r{args.sampler_seed}"
    out = args.outdir / stamp
    # Deliberately refuse to overwrite the evidence for an earlier run.
    out.mkdir(parents=True, exist_ok=False)
    start_time = time.monotonic()
    N, M, C = args.N, args.M, 2 * args.ladders_per_group
    teacher = Teacher.hadamard(N, [args.A1, args.A2])
    V = spin_states(N)
    lp_teacher = log_prob_visible(teacher.W, np.zeros(N), np.zeros(2), 1.0, V)
    data = teacher.sample(args.K, lp_teacher, V, np.random.default_rng(1000 + args.seed))
    target = ExactPosterior(data, M, args.gamma)
    seed_sequence = np.random.SeedSequence([args.sampler_seed, args.seed])
    rng_map, rng_init, rng_hmc, rng_eval = [np.random.default_rng(s) for s in seed_sequence.spawn(4)]
    sources = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
               for name in (Path(__file__).name, "eta_validation.py", "rbm_exact.py")}
    cfg = vars(args).copy()
    cfg["outdir"] = str(args.outdir)
    cfg.update({
        "schema": "rbm_eta_validation_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "etas_requested": wanted.tolist(), "eta_ladder": etas.tolist(),
        "data_rng_seed": 1000 + args.seed, "data_sha256": array_hash(data),
        "source_sha256": sources, "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "arviz")},
        "argv": sys.argv, "gamma_convention": "eta * (sum log likelihood - gamma ||W||^2/2)",
        "initial_groups": ["dispersed"] * args.ladders_per_group + ["optimized"] * args.ladders_per_group,
        "checkpoint_restart": "state is saved; automatic resume is unsupported",
        "delta_h_convention": "pre-swap HMC proposal log acceptance ratio, per temperature slot; weights/replica IDs are retained after swaps",
        "claimed_global_map": False,
    })
    write_json(out / "config.json", cfg)
    atomic_npz(out / "data.npz", data=data, teacher_weights=teacher.W,
               teacher_patterns=teacher.xis, covariance_spectrum=data_spectrum(data, N))
    print(f"Output {out}\n{C} independent ladders x {len(etas)} replicas; "
          f"data hash {cfg['data_sha256']}", flush=True)
    start_scales = np.geomspace(0.05, max(3.0, 2 / np.sqrt(args.gamma)), args.map_restarts)
    starts = rng_map.normal(size=(args.map_restarts, N, M)) * start_scales[:, None, None]
    candidates, map_records = polish_candidates(target, starts, args.map_maxiter)
    basins = distinct_basins(candidates, args.ladders_per_group)
    scale, curvature = fixed_coordinate_scale(target, candidates[0], etas, args.metric)
    initial = np.empty((C, len(etas), N, M))
    for chain in range(args.ladders_per_group):
        # Deliberately overdispersed across independent ladders and temperatures.
        multiplier = np.geomspace(0.1, 1.0, args.ladders_per_group)[chain]
        initial[chain] = rng_init.normal(size=(len(etas), N, M)) * (
            multiplier / np.sqrt(args.gamma * etas[:, None, None])
        )
        initial[chain + args.ladders_per_group] = basins[chain % len(basins)] + (
            0.1 * rng_init.normal(size=(len(etas), N, M)) * scale
        )
    atomic_npz(out / "initialization.npz", initial=initial, candidate_weights=candidates,
               coordinate_scale=scale, curvature=curvature, basins=basins)
    write_json(out / "initial_map.json", map_records)
    print(f"Optimization candidates {len(candidates)}, distinct symmetry-reduced starts {len(basins)}, "
          f"best objective {map_records[0]['logpost']:.6f}", flush=True)
    result = run_replica_hmc(
        target, initial, etas, scale, rng_hmc, warmup=args.warmup, sample=args.sample,
        thin=args.thin, n_leapfrog=args.leapfrog, initial_step=args.step_size,
        target_accept=args.target_accept, swap_every=args.swap_every,
        checkpoint_path=out / "checkpoint.npz", checkpoint_every=args.checkpoint_every,
    )
    atomic_npz(out / "samples.npz", **result, etas=etas)
    samples = result["samples"]
    probes = np.stack([teacher.xis[:, 0], teacher.xis[:, 1], data[0]])
    traces = invariant_traces(samples, target, teacher.xis, probes)
    atomic_npz(out / "observable_traces.npz", **traces, probe_states=probes, etas=etas)
    diagnostics = convergence_diagnostics(traces, etas, args.min_ess, args.max_rhat)
    group_diagnostics = {
        label: convergence_diagnostics(
            {k: v[:, section] for k, v in traces.items()}, etas, args.min_ess, args.max_rhat
        ) for label, section in (("dispersed", slice(0, args.ladders_per_group)),
                                 ("optimized", slice(args.ladders_per_group, C)))
    }
    # Refine from the best retained state of EACH ladder, in addition to initial
    # candidates. No unproved claim that the best objective is globally maximal.
    post_starts = []
    for chain in range(C):
        objective = result["base_logpost"][:, chain]
        di, ei = np.unravel_index(objective.argmax(), objective.shape)
        post_starts.append(samples[di, chain, ei])
    polished, post_records = polish_candidates(
        target, np.concatenate([candidates[:max(2, args.ladders_per_group)], post_starts]), args.map_maxiter
    )
    lp_map = log_prob_visible(polished[0], np.zeros(N), np.zeros(M), 1.0, V)
    map_summary = {
        "status": "best found local optimum; global optimality not established",
        "logpost": float(target(polished[0])[0]),
        "best_retained_logpost": float(result["base_logpost"].max()),
        "reverse": float(kl(lp_map, lp_teacher) / N),
        "forward": float(kl(lp_teacher, lp_map) / N),
        "candidate_records": post_records,
    }
    atomic_npz(out / "map_weights.npz", weights=polished, log_prob_visible=lp_map)
    predictive, kl_traces = predictive_evaluation(
        samples, etas, wanted, V, lp_teacher, args.ladders_per_group,
        args.n_eval, args.eval_chunk, rng_eval,
    )
    kl_diagnostics = {}
    kl_arrays = {}
    for eta, rows in kl_traces.items():
        e = float(eta)
        diag = convergence_diagnostics(
            {k: rows[k][:, :, None] for k in ("reverse", "forward")}, [e], args.min_ess, args.max_rhat
        )
        kl_diagnostics[eta] = diag[eta]
        for name, array in rows.items():
            kl_arrays[f"eta_{eta}_{name}"] = array
    atomic_npz(out / "kl_traces.npz", **kl_arrays)
    gates = {}
    for eta, prediction in predictive.items():
        last = prediction["counts"][-1]
        group_difference = {d: abs(last["dispersed"][d] - last["optimized"][d])
                            for d in ("reverse", "forward")}
        convergence_delta = None
        if len(prediction["counts"]) >= 2:
            previous = prediction["counts"][-2]["all"]
            convergence_delta = {d: abs(last["all"][d] - previous[d]) for d in ("reverse", "forward")}
        half_difference = None
        if all(x is not None for x in prediction["halves"]):
            half_difference = {d: abs(prediction["halves"][0][d] - prediction["halves"][1][d])
                               for d in ("reverse", "forward")}
        checks = {
            "observable_diagnostics": all(x["pass"] for x in diagnostics[eta].values()),
            "within_group_diagnostics": all(x["pass"] for group in group_diagnostics.values() for x in group[eta].values()),
            "kl_diagnostics": all(x["pass"] for x in kl_diagnostics[eta].values()),
            "no_production_divergences": bool(np.all(result["divergences"][:, int(np.flatnonzero(np.isclose(etas, float(eta)))[0])] == 0)),
            "initial_group_agreement": max(group_difference.values()) <= args.predictive_tolerance,
            "draw_count_stability": convergence_delta is not None and max(convergence_delta.values()) <= args.predictive_tolerance,
            "time_half_stability": half_difference is not None and max(half_difference.values()) <= args.predictive_tolerance,
        }
        gates[eta] = {
            "status": "diagnostics_pass" if all(checks.values()) else "needs_followup",
            "checks": checks, "initial_group_difference": group_difference,
            "last_draw_count_difference": convergence_delta, "time_half_difference": half_difference,
            "map_minus_predictive": {d: map_summary[d] - last["all"][d] for d in ("reverse", "forward")},
        }
    write_json(out / "summary.json", {
        "schema": cfg["schema"], "config_sha256": hashlib.sha256((out / "config.json").read_bytes()).hexdigest(),
        "elapsed_seconds": time.monotonic() - start_time, "map": map_summary,
        "diagnostics": diagnostics, "group_diagnostics": group_diagnostics,
        "kl_diagnostics": kl_diagnostics, "predictive": predictive, "gates": gates,
        "sampler": {k: result[k] for k in ("acceptance", "divergences", "step_size", "swap_counts", "swap_attempts", "endpoint_traversals")},
        "limitations": [
            "One dataset per job: no across-dataset significance or optimum selection is performed here.",
            "Predictive KL uses an estimated posterior mixture; visible-state enumeration alone does not eliminate its Monte Carlo error.",
            "Diagnostic thresholds are screening rules, not a proof of equilibrium or global optimization.",
            "Temperature replicas within one ladder are correlated; independent ladders provide the diagnostic chains.",
        ],
    })
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}
    write_json(out / "manifest.json", {"sha256": files, "complete": True})
    print(f"Complete in {time.monotonic() - start_time:.1f}s: {out / 'summary.json'}", flush=True)
    for eta, row in gates.items():
        print(f"eta={eta}: {row['status']} {row['checks']}", flush=True)
    return 0  # A completed experiment with failed diagnostics is still a valid artifact.


if __name__ == "__main__":
    raise SystemExit(main())
