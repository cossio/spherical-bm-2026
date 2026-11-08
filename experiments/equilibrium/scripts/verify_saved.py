# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Check release hashes, recorded diagnostics, seeds and Figure 8 estimates.

This checks the compact release; raw-chain diagnostics require the documented
simulation and analyze.py workflow. No raw draws are inferred from summaries.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(name):
    return json.loads((ROOT / name).read_text())


def main():
    checksums = (ROOT / "provenance/release.sha256").read_text().splitlines()
    for line in checksums:
        expected, name = line.split("  ", 1)
        path = ROOT / name
        assert path.is_file(), name
        assert sha(path) == expected, f"Changed release file: {name}"

    config = tomllib.loads((ROOT / "config.toml").read_text())
    reports = {}
    for cohort, expected_chains, expected_groups in [
        ("production", 96, 24), ("n256", 24, 6),
        ("independent_replication", 4, 1), ("controls", 8, 2)
    ]:
        report = read_json(f"validation/{cohort}/summary.json")
        reports[cohort] = report
        assert len(report["chains"]) == expected_chains
        assert len(report["groups"]) == expected_groups
        assert not report["pending_groups"] and not report["incomplete_traces"]
        assert all(g["matrix_diagnostics_pass"] and not g["score_identity_flags"] for g in report["groups"])
        for chain in report["chains"]:
            meta = chain["metadata"]
            for name, expected in meta["source_sha256"].items():
                source = ROOT / (name if name == "config.toml" else f"scripts/{name}")
                assert sha(source) == expected, (cohort, name)
            assert meta["measurement_spins"] == 16
            assert meta["blas_threads"] == 1
            assert len(chain["draws_sha256"]) == len(chain["metadata_sha256"]) == 64
        assert len({c["metadata"]["seed"] for c in report["chains"]}) == expected_chains
        for row in report["summary"]:
            if row["gate_required"]:
                assert row["rhat"] <= 1.01 and min(row["ess_bulk"], row["ess_tail"]) >= 400
            assert math.isclose(row["ci_low"], row["mean"] - NormalDist().inv_cdf(.975)*row["mcse"], abs_tol=1e-12)
            assert math.isclose(row["ci_high"], row["mean"] + NormalDist().inv_cdf(.975)*row["mcse"], abs_tol=1e-12)

    production = reports["production"]
    for chain in production["chains"]:
        m = chain["metadata"]
        assert m["N"] in [32, 64, 128] and m["warmup"] == 20000 and m["thin"] == 20
        assert m["iterations"] == {32:150000,64:300000,128:600000}[m["N"]]
    assert set(config["sizes"]) == {32,64,128}
    n256 = reports["n256"]
    protocol256 = read_json("validation/n256_protocol.json")
    assert sha(ROOT / "manifests/n256.tsv") == protocol256["manifest_sha256"]
    with (ROOT / "manifests/n256.tsv").open() as handle:
        manifest256 = list(csv.DictReader(handle, delimiter="\t"))
    seeds256 = {chain["metadata"]["seed"] for chain in n256["chains"]}
    assert seeds256 == {int(row["seed"]) for row in manifest256}
    assert not seeds256.intersection(chain["metadata"]["seed"] for chain in production["chains"])
    for name, expected in protocol256["source_sha256"].items():
        assert sha(ROOT / (name if name == "config.toml" else f"scripts/{name}")) == expected
    for chain in n256["chains"]:
        m = chain["metadata"]
        assert m["N"] == 256 and m["warmup"] == 100000 and m["thin"] == 20
        assert m["iterations"] == protocol256["design"]["production_by_case"][m["case"]]
        assert m["eta"] == 4 and m["c"] == [1.25,0.75]
    table = {(r["case"],r["N"],r["observable"]):r
             for cohort in [production,n256] for r in cohort["summary"]}
    for cohort in ["production","n256"]:
        with (ROOT / f"validation/{cohort}/summary.csv").open() as handle:
            csv_rows = list(csv.DictReader(handle))
        for row in csv_rows:
            recorded = table[row["case"],int(row["N"]),row["observable"]]
            for key in ["mean","ci_low","ci_high","mcse"]:
                assert float(row[key]) == recorded[key]
    points = read_json("data/plot_points.json")
    assert len(points["plotted_points"]) == 72
    assert [p for p in points["plotted_points"] if p["N"] != 256] == read_json("data/plot_points_previous.json")["plotted_points"]
    assert sha(ROOT / "validation/production/summary.csv") == points["summary_sha256"]
    assert sha(ROOT / "validation/n256/summary.csv") == points["n256_summary_sha256"]
    assert sha(ROOT / "figures/equilibrium_comparison.pdf") == points["pdf_sha256"]
    for row in points["plotted_points"]:
        recorded = table[row["case"],row["N"],row["observable"]]
        assert recorded["eta"] == 4
        for key in ["mean","ci_low","ci_high"]:
            assert row[key] == recorded[key]

    replication = read_json("validation/replication_comparison.json")
    protocol = read_json("validation/replication_protocol.json")
    amendment = read_json("validation/replication_protocol_amendment.json")
    assert replication["passed"] and replication["protocol"] == protocol
    assert sha(ROOT / "scripts/analyze.py") == protocol["analysis_source_sha256"]
    assert sha(ROOT / "scripts/compare_replication.py") == amendment["updated_comparison_source_sha256"]
    assert sha(ROOT / "manifests/replication.tsv") == protocol["manifest_sha256"]
    assert sha(ROOT / "validation/replication_protocol.json") == amendment["original_protocol_sha256"]
    assert not amendment["decision_rules_changed"]
    assert len(set(replication["original_seeds"]+replication["replication_seeds"])) == 8
    assert replication["replication_seeds"] == protocol["rules"]["seeds"]
    monitored = [r for r in replication["rows"] if r["comparison_monitored"]]
    assert len(monitored) == 11
    cutoff = NormalDist().inv_cdf(1-.05/(2*11))
    for row in monitored:
        combined = math.hypot(row["original_mcse"], row["replication_mcse"])
        z = (row["replication_mean"]-row["original_mean"])/combined
        assert abs(z-row["difference_z"]) < 1e-12
        assert abs(z) < cutoff
    oracle = read_json("results/n2_reference_eta4_gamma0p25.json")
    assert max(abs(v) for v in oracle["checks"].values()) < 1e-9
    figure = read_json("validation/comparison_pdf.json")
    assert figure["status"] == "PASS" and not figure["coordinate_normalized_issues"]
    assert figure["embedded_images"] == 0
    print(json.dumps({"status":"PASS", "release_files":len(checksums),
        "production_chains":120,"production_groups":30,"figure8_points":72,
        "n256_chains":24,"n256_groups":6,
        "replication_chains":4,"control_chains":8,
        "maximum_replication_difference_z":max(abs(r["difference_z"]) for r in monitored),
        "scope":"Release integrity and saved statistics; raw-chain diagnostics are reproduced by analyze.py after simulation."}, indent=2))


if __name__ == "__main__":
    main()
