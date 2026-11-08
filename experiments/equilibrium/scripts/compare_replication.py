# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy==2.4.4", "scipy==1.17.1", "arviz==0.22.0"]
# ///
"""Compare independent full-length runs without discarding or replacing draws."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import norm

import analyze

COMPARISON_NAMES = analyze.MATRIX_GATES | {"h_top2", "score_residual_c1", "score_residual_c2"}
COMPARISON_Z_LIMIT = float(norm.ppf(1 - 0.05 / (2 * len(COMPARISON_NAMES))))


def values_for(chains, name):
    first = chains[0]["metadata"]
    if name.startswith("score_residual_c"):
        direction = int(name[-1])
        c = first["c"][direction - 1]
        return np.stack([
            c - len(first["c"]) * chain["data"][f"sample_c{direction}"]
            - first["gamma"] * chain["data"][f"w{direction}{direction}"]
            for chain in chains
        ])
    return np.stack([chain["data"][name] for chain in chains])


def load_run(path, catalog, gates):
    files = sorted(path.glob("chain*/metadata.toml"))
    if len(files) != 4:
        raise ValueError(f"Exactly four complete chains are required in {path}; found {len(files)}")
    chains = [analyze.load_chain(file, catalog) for file in files]
    rows, group = analyze.analyze_group(chains, gates)
    return chains, {row["observable"]: row for row in rows}, group


def main():
    experiment = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=experiment / "results/production/deep_coalesced/N128")
    parser.add_argument("--replication", type=Path, default=experiment / "results/independent_replication/deep_coalesced/N128")
    parser.add_argument("--output", type=Path, default=experiment / "validation/replication_comparison.json")
    parser.add_argument("--write-protocol", action="store_true", help="Save the fixed comparison rule before examining replication results")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    gates = SimpleNamespace(min_chains=4, rhat_limit=1.01, ess_min=400)
    protocol_path = experiment / "validation/replication_protocol.json"
    protocol_rules = {
        "case": "deep_coalesced", "N": 128, "chains": 4, "warmup": 50000,
        "retained_period_transitions": 600000, "thin": 20,
        "seeds": [91718801, 91718802, 91718803, 91718804],
        "comparison_names": sorted(COMPARISON_NAMES), "comparison_count": len(COMPARISON_NAMES),
        "difference_mcse": "sqrt(original_mcse^2 + replication_mcse^2)",
        "difference_abs_z_limit": COMPARISON_Z_LIMIT, "familywise_alpha": 0.05,
        "rhat_limit": gates.rhat_limit, "ess_bulk_min": gates.ess_min, "ess_tail_min": gates.ess_min,
        "cohort_score_abs_z_limit": 4, "pooled_diagnostics_are_descriptive": True,
        "maximum_replication_cohorts": 1,
        "primary_estimates": "Original complete 24-group production grid, unchanged",
        "quarter_and_half_diagnostics": "Descriptive; inspect coherent drift without using them to discard draws",
    }
    if args.write_protocol:
        if protocol_path.exists():
            raise ValueError(f"Refusing to overwrite predeclared protocol {protocol_path}")
        manifest = experiment / "manifests/replication.tsv"
        protocol = {
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "reason": "Descriptive late-quarter diagnostics in deep-coalesced N128 W22 exposed a slow mode despite passing full-run gates",
            "rules": protocol_rules,
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "original_summary_sha256": hashlib.sha256((experiment / "validation/production/summary.json").read_bytes()).hexdigest(),
            "comparison_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "analysis_source_sha256": hashlib.sha256(Path(analyze.__file__).read_bytes()).hexdigest(),
        }
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
        print(json.dumps(protocol, indent=2))
        return
    protocol = json.loads(protocol_path.read_text())
    if protocol["rules"] != protocol_rules:
        raise ValueError("The comparison rules differ from the saved predeclared protocol")
    if protocol["analysis_source_sha256"] != hashlib.sha256(Path(analyze.__file__).read_bytes()).hexdigest():
        raise ValueError("The diagnostic implementation changed after replication protocol declaration")
    manifest = experiment / "manifests/replication.tsv"
    if protocol["manifest_sha256"] != hashlib.sha256(manifest.read_bytes()).hexdigest():
        raise ValueError("The replication manifest changed after protocol declaration")
    comparison_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if comparison_hash != protocol["comparison_source_sha256"]:
        amendment_path = experiment / "validation/replication_protocol_amendment.json"
        amendment = json.loads(amendment_path.read_text())
        if (amendment["original_protocol_sha256"] != hashlib.sha256(protocol_path.read_bytes()).hexdigest()
                or amendment["original_comparison_source_sha256"] != protocol["comparison_source_sha256"]
                or amendment["updated_comparison_source_sha256"] != comparison_hash
                or amendment["decision_rules_changed"]):
            raise ValueError("Comparator source changed without a matching implementation-only amendment")
    catalog = analyze.source_catalog(experiment, [])
    original, a, group_a = load_run(args.original, catalog, gates)
    replication, b, group_b = load_run(args.replication, catalog, gates)
    first = original[0]["metadata"]
    second = replication[0]["metadata"]
    # Warmup length and final proposal scales can differ; the posterior cannot.
    for key in ("N", "case", "gamma", "eta", "c", "thin", "iterations", "measurement_spins", "source_sha256", "target"):
        if first[key] != second[key]:
            raise ValueError(f"The original and replication disagree on {key}")
    all_chains = original + replication
    seeds = [chain["metadata"]["seed"] for chain in all_chains]
    if len(set(seeds)) != 8:
        raise ValueError("All eight chains must have distinct simulation seeds")
    if (sorted(seeds[4:]) != protocol_rules["seeds"] or second["warmup"] != protocol_rules["warmup"]
            or second["iterations"] != protocol_rules["retained_period_transitions"]
            or first["case"] != protocol_rules["case"] or first["N"] != protocol_rules["N"]
            or second["thin"] != protocol_rules["thin"]):
        raise ValueError("Replication case, size, seeds, thinning, or run lengths differ from the predeclared protocol")
    rows = []
    for name in a:
        values = np.concatenate((values_for(original, name), values_for(replication, name)))
        pooled = analyze.sampling_diagnostics(values, False)
        pooled_pass = (
            pooled["rhat_max"] is not None and pooled["rhat_max"] <= gates.rhat_limit
            and pooled["ess_bulk"] >= gates.ess_min and pooled["ess_tail"] >= gates.ess_min
        )
        difference = b[name]["mean"] - a[name]["mean"]
        difference_mcse = math.hypot(a[name]["mcse"], b[name]["mcse"])
        z = difference / difference_mcse
        score = name.startswith("score_residual_c")
        score_z = pooled["mean"] / pooled["mcse_mean"] if score else None
        monitored = name in COMPARISON_NAMES
        rows.append({
            "observable": name, "gate_required": name in analyze.MATRIX_GATES,
            "comparison_monitored": monitored,
            "original_mean": a[name]["mean"], "original_mcse": a[name]["mcse"],
            "original_rhat": a[name]["rhat"], "original_ess_bulk": a[name]["ess_bulk"],
            "original_ess_tail": a[name]["ess_tail"], "original_pass": a[name]["diagnostic_pass"],
            "replication_mean": b[name]["mean"], "replication_mcse": b[name]["mcse"],
            "replication_rhat": b[name]["rhat"], "replication_ess_bulk": b[name]["ess_bulk"],
            "replication_ess_tail": b[name]["ess_tail"], "replication_pass": b[name]["diagnostic_pass"],
            "difference": difference, "difference_mcse": difference_mcse,
            "difference_ci95_low": difference - 1.959963984540054 * difference_mcse,
            "difference_ci95_high": difference + 1.959963984540054 * difference_mcse,
            "difference_z": z, "difference_flag_bonferroni": monitored and abs(z) > COMPARISON_Z_LIMIT,
            "pooled_mean": pooled["mean"], "pooled_mcse": pooled["mcse_mean"],
            "pooled_rhat": pooled["rhat_max"], "pooled_ess_bulk": pooled["ess_bulk"],
            "pooled_ess_tail": pooled["ess_tail"], "pooled_pass": pooled_pass,
            "pooled_score_z": score_z,
            "pooled_score_flag_4sigma": abs(score_z) > 4 if score else False,
        })
    matrix_pooled_pass = all(row["pooled_pass"] for row in rows if row["gate_required"])
    difference_flags = [row["observable"] for row in rows if row["difference_flag_bonferroni"]]
    score_flags = [row["observable"] for row in rows if row["pooled_score_flag_4sigma"]]
    passed = (group_a["matrix_diagnostics_pass"] and group_b["matrix_diagnostics_pass"]
              and not difference_flags
              and not group_a["score_identity_flags"] and not group_b["score_identity_flags"])
    inputs = {chain["path"]: chain["draws_sha256"] for chain in all_chains}
    inputs.update({chain["metadata_path"]: chain["metadata_sha256"] for chain in all_chains})
    result = {
        "passed": passed, "case": first["case"], "N": first["N"],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "protocol": protocol,
        "analysis_source_sha256": hashlib.sha256(Path(analyze.__file__).read_bytes()).hexdigest(),
        "input_sha256": inputs, "source_verification": [chain["source_verification"] for chain in all_chains],
        "original_input": str(args.original.resolve()), "replication_input": str(args.replication.resolve()),
        "original_group": group_a, "replication_group": group_b,
        "original_warmup": first["warmup"], "replication_warmup": second["warmup"],
        "iterations_per_chain": first["iterations"], "thin": first["thin"],
        "original_seeds": seeds[:4], "replication_seeds": seeds[4:],
        "pooled_matrix_pass": matrix_pooled_pass, "difference_flags": difference_flags,
        "pooled_score_flags": score_flags,
        "gates": {"rhat_limit": gates.rhat_limit, "ess_min": gates.ess_min,
                  "comparison_names": sorted(COMPARISON_NAMES), "comparison_count": len(COMPARISON_NAMES),
                  "familywise_alpha": 0.05, "difference_abs_z_limit": COMPARISON_Z_LIMIT,
                  "cohort_score_abs_z_limit": 4, "pooled_diagnostics_are_descriptive": True},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Independent replication of the slowest setting", "",
             f"Status: {'PASS' if passed else 'requires investigation'}. Four new independent chains repeat {first['case']}, N={first['N']}, with {second['warmup']:,} warmup and {first['iterations']:,} retained-period transitions each. The original four chains used {first['warmup']:,} warmup transitions. All eight seeds are distinct.", "",
             "The additional run was requested after a descriptive quarter-run check exposed a slow mode in the original W22 trace. Both complete original and replication datasets are retained. Main production estimates are not replaced or silently pooled with the replication. The pooled eight-chain results below are an additional diagnostic.", "",
             f"For independent runs, difference MCSE is sqrt(MCSE_original² + MCSE_replication²), with each MCSE accounting for within-chain autocorrelation. The predeclared comparison covers 11 observables: eight matrix monitors, top-two sample mass, and both joint score residuals. The two-sided Bonferroni threshold is norm.ppf(1−0.05/(2×11))={COMPARISON_Z_LIMIT:.6f}; all absolute difference z values must be below it. Both individual runs must pass the unchanged matrix R-hat/ESS gates, and each cohort's score means must be within four MCSE of zero. The pooled eight-chain diagnostics are descriptive and cannot rescue a failed individual run. Only one fixed replication is performed; it is not repeated until a pass is obtained.", "",
             "| Observable | Original mean ± MCSE | Replication mean ± MCSE | Difference / MCSE | Original R-hat | Replication R-hat | Pooled R-hat | Pooled bulk/tail ESS |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if not row["comparison_monitored"]:
            continue
        lines.append(f"| {row['observable']} | {row['original_mean']:.6f} ± {row['original_mcse']:.6f} | {row['replication_mean']:.6f} ± {row['replication_mcse']:.6f} | {row['difference_z']:+.2f} | {row['original_rhat']:.5f} | {row['replication_rhat']:.5f} | {row['pooled_rhat']:.5f} | {row['pooled_ess_bulk']:.0f}/{row['pooled_ess_tail']:.0f} |")
    lines += ["", "All measured observables, separate and pooled diagnostics, exact score residual checks, raw-data hashes, and distinct seeds are recorded in the companion JSON/CSV.", "",
              "Agreement between independently seeded runs supports reproducibility of the reported posterior means. It does not prove the absence of arbitrarily slow or exponentially rare modes."]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    monitored = [row for row in rows if row["comparison_monitored"]]
    print(json.dumps({"passed": passed, "max_monitored_difference_z": max(abs(row["difference_z"]) for row in monitored),
                      "original_matrix_pass": group_a["matrix_diagnostics_pass"], "replication_matrix_pass": group_b["matrix_diagnostics_pass"],
                      "pooled_matrix_pass": matrix_pooled_pass, "difference_flags": difference_flags,
                      "output": str(args.output)}, indent=2))
    if args.require_pass and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
