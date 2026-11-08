# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy==2.4.4"]
# ///
"""Recompute the saved numerical comparison for every retained setting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load_run(path):
    with np.load(path) as source:
        data = {key: source[key] for key in source.files if key != "metadata"}
        data["metadata"] = json.loads(str(source["metadata"]))
    data["path"] = str(path)
    return data


def theory_data(directory, case, step="0p0125"):
    return np.genfromtxt(directory / f"{case}_dt{step}_optimized.csv", delimiter=",", names=True)


def sem(values):
    return np.std(values, axis=0, ddof=1) / np.sqrt(len(values))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--theory-dt", default="0p0125")
    p.add_argument("--require-complete", action="store_true")
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "generated" / "summary.json")
    args = p.parse_args()
    root = args.root
    runs = [load_run(path) for path in sorted((root / "results").glob("*.npz")) if not path.name.startswith("pilot")]
    if args.require_complete:
        expected = {(case, n, dt) for case in "ABCD" for n in (1000,4000) for dt in (.02,.01,.005)}
        actual = [(run["metadata"]["case"], run["metadata"]["N"], run["metadata"]["dt"]) for run in runs]
        if set(actual) != expected or len(actual) != len(expected):
            raise ValueError(f"Incomplete or duplicated experiment: missing={expected-set(actual)}, extra={set(actual)-expected}")
    summaries = []
    for run in runs:
        meta = run["metadata"]
        theory = theory_data(root / "theory", meta["case"], args.theory_dt)
        if run["t"][0] < theory["t"][0] or run["t"][-1] > theory["t"][-1] + 1e-10:
            raise ValueError(f"Reference does not cover simulation times: {run['path']}")
        names = [f"s{j+1}" for j in range(meta["K"])]
        reference_s = np.stack([np.interp(run["t"], theory["t"], theory[name]) for name in names], axis=1)
        reference_k = np.interp(run["t"], theory["t"], theory["kappa"])
        avg_s = run["s"].mean(axis=1)
        avg_k = run["kappa"].mean(axis=1)
        pair_s = run["s"][0] - run["s"][1]
        pair_k = run["kappa"][0] - run["kappa"][1]
        summaries.append({
            "path": str(Path(run["path"]).relative_to(root)), "sha256": hashlib.sha256(Path(run["path"]).read_bytes()).hexdigest(),
            "case": meta["case"], "N": meta["N"], "dt": meta["dt"], "replicas": meta["replicas"],
            "time_horizon": float(run["t"][-1]), "source_sha256": meta["source_sha256"],
            "theory_path": str(Path("theory") / f"{meta['case']}_dt{args.theory_dt}_optimized.csv"),
            "s_rmse_vs_dmft": np.sqrt(np.mean((avg_s-reference_s)**2, axis=(1,2))).tolist(),
            "kappa_rmse_vs_dmft": np.sqrt(np.mean((avg_k-reference_k)**2, axis=1)).tolist(),
            "max_s_mean_error_vs_dmft": np.max(np.abs(avg_s-reference_s), axis=(1,2)).tolist(),
            "max_kappa_mean_error_vs_dmft": np.max(np.abs(avg_k-reference_k), axis=1).tolist(),
            "terminal_s_mean": avg_s[:, -1].tolist(),
            "terminal_s_sem": np.stack([sem(arm[:, -1]) for arm in run["s"]]).tolist(),
            "terminal_s_dmft": reference_s[-1].tolist(),
            "terminal_kappa_mean": avg_k[:, -1].tolist(),
            "terminal_kappa_sem": [float(sem(arm[:, -1])) for arm in run["kappa"]],
            "terminal_kappa_dmft": float(reference_k[-1]),
            "paired_terminal_s_difference": pair_s.mean(0)[-1].tolist(),
            "paired_terminal_s_sem": sem(pair_s)[-1].tolist(),
            "paired_s_mean_difference_rms": float(np.sqrt(np.mean(pair_s.mean(0)**2))),
            "paired_kappa_mean_difference_rms": float(np.sqrt(np.mean(pair_k.mean(0)**2))),
            "max_trace_error": meta["max_trace_error"], "max_norm_error": meta["max_norm_error"],
            "data_sampling_frequencies": (run["data_counts"] / run["data_counts"].sum(axis=1,keepdims=True)).tolist(),
        })
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"runs": summaries, "error_bar": "SEM across trajectories; no simultaneous coverage claim"}, indent=2)+"\n")
    print(f"Wrote {len(summaries)} run summaries to {out}")
    for item in summaries:
        print(f"{item['case']} N={item['N']} dt={item['dt']}: "
              f"SGD overlap RMSE={item['s_rmse_vs_dmft'][0]:.6f}, "
              f"kappa RMSE={item['kappa_rmse_vs_dmft'][0]:.6f}")


if __name__ == "__main__":
    main()
