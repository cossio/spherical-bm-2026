# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy==2.4.4", "scipy==1.17.1", "arviz==0.22.0"]
# ///
"""Audit saved exact-SBM chains and export autocorrelation-aware diagnostics.

uv run --locked scripts/analyze.py --input results/hpc_pilot --output validation/pilot
Paths are relative to the current directory. Completed metadata files identify
chains; unfinished CSVs are reported and never included in posterior estimates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import sys
import tarfile
import tomllib
from collections import defaultdict
from pathlib import Path

import arviz as az
import numpy as np
from scipy.stats import norm

REQUIRED = (
    "lambda1", "lambda2", "gap12", "h_top2", "projector_c1", "projector_c2",
    "second_moment", "w11", "w22", "sample_c1", "sample_c2",
)
OPTIONAL = (
    "lambda3", "top2_mean", "u11", "u22", "h1", "h2", "centered_lambda1", "trace_shift",
)
MATRIX_GATES = {
    "lambda1", "lambda2", "gap12", "projector_c1", "projector_c2", "second_moment", "w11", "w22",
}
SOURCE_FILES = ("run_chain.jl", "bingham.jl", "config.toml")


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def source_catalog(experiment: Path, extra_archives: list[Path]) -> dict:
    """Verify source bytes, never treating a checksum manifest as its own proof."""
    catalog = defaultdict(lambda: defaultdict(list))
    for name in SOURCE_FILES:
        path = experiment / (name if name == "config.toml" else f"scripts/{name}")
        if path.is_file():
            catalog[name][digest(path.read_bytes())].append(str(path.resolve()))
    archives = sorted(set((experiment / "provenance").glob("*.tar.gz")) | set(extra_archives))
    for path in archives:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                name = Path(member.name).name
                if member.isfile() and name in SOURCE_FILES:
                    handle = archive.extractfile(member)
                    if handle is None:
                        raise ValueError(f"Cannot read archived source {path}:{member.name}")
                    catalog[name][digest(handle.read())].append(f"{path.resolve()}:{member.name}")
    return catalog


def load_chain(metadata_path: Path, catalog: dict) -> dict:
    meta = tomllib.loads(metadata_path.read_text())
    expected_hashes = meta.get("source_sha256", {})
    if set(expected_hashes) != set(SOURCE_FILES):
        raise ValueError(f"Missing or unexpected source hash names in {metadata_path}")
    verified = {}
    for name, expected in expected_hashes.items():
        matches = catalog.get(name, {}).get(expected, [])
        if not matches:
            raise ValueError(f"Source hash cannot be verified: {metadata_path}: {name}={expected}")
        verified[name] = matches
    for key in ("acceptance", "auxiliary_acceptance"):
        if not 0 < meta[key] <= 1:
            raise ValueError(f"Invalid {key}={meta[key]} in {metadata_path}")
    if not 0 < meta["rho"] < 1:
        raise ValueError(f"Invalid fixed production rho in {metadata_path}")
    path = metadata_path.with_name("draws.csv")
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float, ndmin=1)
    names = data.dtype.names or ()
    missing = set(REQUIRED) - set(names)
    if missing:
        raise ValueError(f"Missing mandatory observables {missing} in {path}")
    expected_iterations = np.arange(meta["thin"], meta["iterations"] + 1, meta["thin"])
    if not np.array_equal(data["iteration"], expected_iterations):
        raise ValueError(f"Saved iteration count/order does not match metadata in {path}")
    if len(data) < 20:
        raise ValueError(f"Too few saved draws for diagnostics in {path}")
    for name in names:
        if name == "lambda3" and meta["N"] == 2 and np.isnan(data[name]).all():
            continue
        if not np.isfinite(data[name]).all():
            raise ValueError(f"Nonfinite {name} values in {path}")
    if not np.allclose(data["rho"], meta["rho"], rtol=0, atol=1e-14):
        raise ValueError(f"Adaptation continued in production or rho metadata disagrees: {path}")
    if np.any(data["lambda1"] < data["lambda2"]) or np.any(data["gap12"] < 0):
        raise ValueError(f"Eigenvalue ordering is violated in {path}")
    if not np.allclose(data["gap12"], data["lambda1"] - data["lambda2"], atol=1e-12):
        raise ValueError(f"Saved eigenvalue gap is inconsistent in {path}")
    return {
        "metadata": meta, "data": data, "path": str(path.resolve()),
        "metadata_path": str(metadata_path.resolve()), "draws_sha256": digest(path.read_bytes()),
        "metadata_sha256": digest(metadata_path.read_bytes()), "source_verification": verified,
    }


def finite_float(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def sampling_diagnostics(values: np.ndarray, constant_allowed: bool) -> dict:
    chains, draws = values.shape
    mean = float(np.mean(values))
    constant = float(np.ptp(values)) <= 64 * np.finfo(float).eps * max(1.0, abs(mean))
    if constant:
        return {
            "mean": mean, "sd": 0.0, "mcse_mean": 0.0,
            "rhat_rank": 1.0 if constant_allowed else None,
            "rhat_z_scale": 1.0 if constant_allowed else None,
            "rhat_folded": 1.0 if constant_allowed else None,
            "rhat_max": 1.0 if constant_allowed else None,
            "ess_bulk": float(values.size) if constant_allowed else 0.0,
            "ess_tail": float(values.size) if constant_allowed else 0.0,
            "constant": True, "constant_expected": constant_allowed,
        }
    rank = finite_float(az.rhat(values, method="rank")) if chains >= 2 else None
    z_scale = finite_float(az.rhat(values, method="z_scale")) if chains >= 2 else None
    folded = finite_float(az.rhat(values, method="folded")) if chains >= 2 else None
    return {
        "mean": mean, "sd": float(np.std(values, ddof=1)),
        "mcse_mean": finite_float(az.mcse(values, method="mean")),
        "rhat_rank": rank, "rhat_z_scale": z_scale, "rhat_folded": folded,
        "rhat_max": max(rank, z_scale, folded) if all(v is not None for v in (rank, z_scale, folded)) else None,
        "ess_bulk": finite_float(az.ess(values, method="bulk")),
        "ess_tail": finite_float(az.ess(values, method="tail")),
        "constant": False, "constant_expected": False,
    }


def half_comparison(values: np.ndarray) -> dict:
    """Descriptive stationarity check; correlation between halves is not modeled."""
    midpoint = values.shape[1] // 2
    first, last = values[:, :midpoint], values[:, -midpoint:]
    difference = float(last.mean() - first.mean())
    constant = float(np.ptp(values)) <= 64 * np.finfo(float).eps * max(1.0, abs(float(values.mean())))
    if constant:
        first_mcse = last_mcse = 0.0
    else:
        first_mcse = float(az.mcse(first, method="mean"))
        last_mcse = float(az.mcse(last, method="mean"))
    scale = math.hypot(first_mcse, last_mcse)
    return {
        "first_half_mean": float(first.mean()), "last_half_mean": float(last.mean()),
        "half_difference": difference, "first_half_mcse": finite_float(first_mcse),
        "last_half_mcse": finite_float(last_mcse),
        "half_difference_z_descriptive": finite_float(difference / scale) if scale > 0 else None,
        "chain_first_half_means": first.mean(axis=1).tolist(),
        "chain_last_half_means": last.mean(axis=1).tolist(),
    }


def analyze_group(chains: list[dict], args) -> tuple[list[dict], dict]:
    chains.sort(key=lambda chain: chain["metadata"]["chain"])
    first = chains[0]["metadata"]
    ids = [chain["metadata"]["chain"] for chain in chains]
    seeds = [chain["metadata"]["seed"] for chain in chains]
    if len(set(ids)) != len(ids) or len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate chain identifiers or seeds in {first['case']} N={first['N']}")
    for key in ("N", "case", "gamma", "eta", "c", "thin", "iterations", "warmup", "measurement_spins", "source_sha256", "target"):
        if any(chain["metadata"][key] != first[key] for chain in chains):
            raise ValueError(f"Inconsistent {key} across {first['case']} N={first['N']} chains")
    common_names = set.intersection(*(set(chain["data"].dtype.names) for chain in chains))
    names = list(REQUIRED) + [name for name in OPTIONAL if name in common_names and not (name == "lambda3" and first["N"] == 2)]
    names += ["score_residual_c1", "score_residual_c2"]
    rows = []
    for name in names:
        if name.startswith("score_residual_c"):
            index = int(name[-1])
            data_eigenvalue = first["c"][index - 1] if index <= len(first["c"]) else 0.0
            values = np.stack([
                data_eigenvalue - len(first["c"]) * chain["data"][f"sample_c{index}"]
                - first["gamma"] * chain["data"][f"w{index}{index}"]
                for chain in chains
            ])
        else:
            values = np.stack([chain["data"][name] for chain in chains])
        allowed = first["N"] == 2 and name in {"projector_c1", "projector_c2", "h_top2"}
        row = {"case": first["case"], "N": first["N"], "gamma": first["gamma"], "eta": first["eta"],
               "observable": name, "chains": len(chains), "draws_per_chain": values.shape[1],
               "gate_required": name in MATRIX_GATES, "noisy_spin_measurement": name in {"h1", "h2", "h_top2", "sample_c1", "sample_c2"}}
        row.update(sampling_diagnostics(values, allowed))
        row.update(half_comparison(values))
        half_width = norm.ppf(0.975) * row["mcse_mean"] if row["mcse_mean"] is not None else None
        row["mc_ci95_low"] = row["mean"] - half_width if half_width is not None else None
        row["mc_ci95_high"] = row["mean"] + half_width if half_width is not None else None
        # Stable concise aliases consumed by the comparison-figure script.
        row["mcse"] = row["mcse_mean"]
        row["ci_low"] = row["mc_ci95_low"]
        row["ci_high"] = row["mc_ci95_high"]
        row["rhat"] = row["rhat_max"]
        score = name.startswith("score_residual_c")
        row["score_mean_z"] = row["mean"] / row["mcse"] if score and row["mcse"] is not None and row["mcse"] > 0 else None
        row["score_identity_flag_4sigma"] = abs(row["score_mean_z"]) > 4 if row["score_mean_z"] is not None else False
        reasons = []
        if len(chains) < args.min_chains:
            reasons.append(f"fewer_than_{args.min_chains}_chains")
        if row["rhat_max"] is None or row["rhat_max"] > args.rhat_limit:
            reasons.append("rhat")
        for statistic in ("ess_bulk", "ess_tail"):
            if row[statistic] is None or row[statistic] < args.ess_min:
                reasons.append(statistic)
        if row["constant"] and not row["constant_expected"]:
            reasons.append("unexpected_constant")
        row["diagnostic_pass"] = not reasons
        row["failure_reasons"] = reasons
        rows.append(row)
    failing = [row["observable"] for row in rows if row["gate_required"] and not row["diagnostic_pass"]]
    group = {
        "case": first["case"], "N": first["N"], "chains": len(chains), "chain_ids": ids,
        "matrix_diagnostics_pass": not failing, "failing_matrix_observables": failing,
        "acceptance": [chain["metadata"]["acceptance"] for chain in chains],
        "auxiliary_acceptance": [chain["metadata"]["auxiliary_acceptance"] for chain in chains],
        "rho": [chain["metadata"]["rho"] for chain in chains],
        "source_sha256": first["source_sha256"],
        "score_identity_flags": [row["observable"] for row in rows if row["score_identity_flag_4sigma"]],
    }
    return rows, group


def write_outputs(output: Path, report: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    rows = report["summary"]
    if rows:
        with (output / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()} for row in rows)
    lines = ["# Exact finite-N SBM diagnostics", "", f"Input: `{report['input']}`.", "",
             "The intervals below quantify Monte Carlo uncertainty in posterior means, not posterior fluctuation widths or finite-size bias.", "",
             f"Matrix-observable gates: at least {report['gates']['min_chains']} independent chains, rank-normalized split/folded R-hat ≤ {report['gates']['rhat_limit']}, bulk and tail ESS ≥ {report['gates']['ess_min']}.", "",
             "Generated-spin observables have an additional independent measurement layer. Their diagnostics are reported but do not replace matrix-observable equilibration checks. Initial/final-half differences are descriptive; their standardized values neglect covariance across the boundary and are not additional pass/fail tests.", "",
             "| Case | N | Chains | Matrix gates | Failing observables | Acceptance range |", "|---|---:|---:|---|---|---|"]
    for group in report["groups"]:
        rates = group["acceptance"]
        lines.append(f"| {group['case']} | {group['N']} | {group['chains']} | {'PASS' if group['matrix_diagnostics_pass'] else 'FAIL'} | {', '.join(group['failing_matrix_observables']) or '—'} | {min(rates):.3f}–{max(rates):.3f} |")
    lines += ["", f"Source-byte verification passed for all {len(report['chains'])} included chains. Full source, metadata and draw checksums are in `summary.json`.", "",
              "Incomplete traces omitted: " + str(len(report["incomplete_traces"])) + ".", "",
              "| Case | N | Observable | Mean ± MCSE | SD | R-hat | Bulk ESS | Tail ESS | Half difference / approximate SE |", "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    def fmt(value, spec=".4g"):
        return format(value, spec) if value is not None else "unavailable"
    for row in rows:
        lines.append(f"| {row['case']} | {row['N']} | {row['observable']} | {fmt(row['mean'])} ± {fmt(row['mcse_mean'])} | {fmt(row['sd'])} | {fmt(row['rhat_max'])} | {fmt(row['ess_bulk'], '.0f')} | {fmt(row['ess_tail'], '.0f')} | {fmt(row['half_difference_z_descriptive'], '.2f')} |")
    lines += ["", "Posterior-score identities use each joint saved draw to form `c_j - K*sample_cj - gamma*wjj`, with missing data eigenvalues set to zero. Their exact posterior expectations vanish. The reported MCSE retains covariance between the two measured terms; a four-MCSE discrepancy is flagged for investigation, without treating two-MCSE deviations across many comparisons as failures.", "",
              "| Case | N | Score direction | Residual / MCSE | Flag above 4 |", "|---|---:|---|---:|---|"]
    for row in rows:
        if row["observable"].startswith("score_residual_c"):
            lines.append(f"| {row['case']} | {row['N']} | {row['observable']} | {fmt(row['score_mean_z'], '.3f')} | {'FLAG' if row['score_identity_flag_4sigma'] else '—'} |")
    lines += ["", "ArviZ 0.22.0 supplies rank/split/folded R-hat, bulk/tail ESS and spectral-variance Monte Carlo standard errors. Expected N=2 constant projector and total-spin-mass observables are identified explicitly; other constant traces fail diagnostics."]
    (output / "diagnostics.md").write_text("\n".join(lines) + "\n")


def self_test() -> None:
    """Controls detect independent sampling, separated chains and correlation."""
    rng = np.random.default_rng(6110603)
    iid = rng.normal(size=(4, 4000))
    good = sampling_diagnostics(iid, False)
    assert good["rhat_max"] < 1.01 and good["ess_bulk"] > 4000
    separated = sampling_diagnostics(iid + np.arange(4)[:, None], False)
    assert separated["rhat_max"] > 1.1
    ar = np.zeros((4, 6000))
    noise = rng.normal(size=ar.shape)
    for draw in range(1, ar.shape[1]):
        ar[:, draw] = 0.95 * ar[:, draw - 1] + math.sqrt(1 - 0.95**2) * noise[:, draw]
    correlated = sampling_diagnostics(ar[:, 1000:], False)
    naive = float(np.std(ar[:, 1000:], ddof=1) / math.sqrt(ar[:, 1000:].size))
    assert correlated["mcse_mean"] > 4 * naive
    assert correlated["ess_bulk"] < 1500
    assert sampling_diagnostics(np.ones((4, 100)), True)["rhat_max"] == 1
    assert sampling_diagnostics(np.ones((4, 100)), False)["rhat_max"] is None
    print(json.dumps({"self_test": "PASS", "iid_rhat": good["rhat_max"],
                      "separated_rhat": separated["rhat_max"],
                      "ar1_mcse_over_naive": correlated["mcse_mean"] / naive,
                      "ar1_bulk_ess": correlated["ess_bulk"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-archive", type=Path, action="append", default=[])
    parser.add_argument("--min-chains", type=int, default=4)
    parser.add_argument("--rhat-limit", type=float, default=1.01)
    parser.add_argument("--ess-min", type=float, default=400)
    parser.add_argument("--require-pass", action="store_true", help="Exit 2 when any matrix group fails; outputs are always written.")
    parser.add_argument("--complete-only", action="store_true", help="Include only groups with at least --min-chains completed chains.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.input is None or args.output is None:
        parser.error("--input and --output are required unless --self-test is used")
    experiment = Path(__file__).resolve().parents[1]
    catalog = source_catalog(experiment, args.source_archive)
    metadata_paths = sorted(args.input.rglob("metadata.toml"))
    if not metadata_paths:
        parser.error("No completed metadata.toml files found in input")
    chains = [load_chain(path, catalog) for path in metadata_paths]
    groups = defaultdict(list)
    for chain in chains:
        meta = chain["metadata"]
        groups[(meta["case"], meta["N"])].append(chain)
    rows, summaries = [], []
    pending_groups = []
    for key in sorted(groups):
        if args.complete_only and len(groups[key]) < args.min_chains:
            pending_groups.append({"case": key[0], "N": key[1], "completed_chains": len(groups[key])})
            continue
        group_rows, summary = analyze_group(groups[key], args)
        rows.extend(group_rows)
        summaries.append(summary)
    report = {
        "schema_version": 1, "input": str(args.input.resolve()),
        "analysis_source_sha256": digest(Path(__file__).read_bytes()),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "arviz")},
        "python_version": sys.version,
        "gates": {"min_chains": args.min_chains, "rhat_limit": args.rhat_limit, "ess_min": args.ess_min},
        "summary": rows, "groups": summaries,
        "pending_groups": pending_groups,
        "chains": [{key: value for key, value in chain.items() if key != "data"} for chain in chains],
        "incomplete_traces": [str(path.resolve()) for path in sorted(args.input.rglob("draws.csv")) if not path.with_name("metadata.toml").is_file()],
    }
    write_outputs(args.output, report)
    passed = sum(group["matrix_diagnostics_pass"] for group in summaries)
    print(f"Verified {len(chains)} complete chains; matrix diagnostic gates passed for {passed}/{len(summaries)} groups. Wrote {args.output}.")
    if args.require_pass and passed != len(summaries):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
