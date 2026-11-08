# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy==2.4.4"]
# ///
"""Read-only independent protocol, sign, and finite-N/step result checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE = "395143aa453763a0bbe71e7fb4d5a7786e04db55377ec3f5b251545842137ed2"
CASES = {"A": ([1.5, .5], [.3, .25], .5), "B": ([1.8, .2], [.3, .15], .5),
         "C": ([1.], [.3], .5), "D": ([1.], [.1], 1.5)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "generated" / "independent_results_report.json")
    args = parser.parse_args()
    assert digest(ROOT / "scripts" / "run_sgd.py") == EXPECTED_SOURCE, "Production source hash changed"
    files = sorted(p for p in (ROOT / "results").glob("*.npz") if not p.name.startswith("pilot"))
    expected = {(c, n, dt) for c in CASES for n in [1000, 4000] for dt in [.02, .01, .005]}
    seen = set()
    reviews = []
    for path in files:
        with np.load(path) as f:
            m = json.loads(str(f["metadata"]))
            identity = m["case"], m["N"], m["dt"]
            assert identity in expected and identity not in seen, identity
            seen.add(identity)
            c, s0, gamma = CASES[m["case"]]
            k, n, r, dt = len(c), m["N"], m["replicas"], m["dt"]
            assert r >= 8 and m["K"] == k and m["gamma"] == gamma
            assert m["eta"] == "infinity" and m["parameter_noise"] == "none" and m["W_initial"] == "zero"
            assert m["nu"] == .3 and m["beta"] == 1 and m["Tmax"] == 30
            assert m["source_sha256"] == EXPECTED_SOURCE
            assert m["modes"] == ["data_minibatch_SGD", "full_data_PCD_control"]
            assert m["c"] == c and m["s0"] == s0
            t, s, kap = f["t"], f["s"], f["kappa"]
            assert np.all(np.diff(t) > 0) and t[0] == 0 and t[-1] == 30
            assert s.shape == (2, r, len(t), k) and kap.shape == (2, r, len(t))
            assert np.max(np.abs(s[:, :, 0] - s0)) < 1e-14
            assert np.max(np.abs(kap[:, :, 0] - .3*(n-1)/n)) < 1e-14
            assert np.max(np.abs(f["weight_signal_block"][:, :, 0])) == 0
            assert np.max(np.abs(f["trace"])) < 1e-9
            assert np.max(np.abs(f["norm_squared_over_N"] - 1)) < 1e-12
            assert np.all(f["data_counts"].sum(axis=1) == round(30/dt))
            data = f["normalized_data"]
            assert np.max(np.abs(np.einsum('ki,kj->ij', data, data)-np.diag(c))) < 1e-14
            assert np.max(np.abs(np.sum(data*data, axis=1)-1)) < 1e-14
            for name in f.files:
                if name != "metadata":
                    assert np.isfinite(f[name]).all(), (path.name, name)
            if k == 1:
                for name in ["s", "kappa", "trace", "norm_squared_over_N", "weight_signal_block"]:
                    assert np.max(np.abs(f[name][0]-f[name][1])) < 1e-12, (path.name, name)
            theory_path = ROOT / "theory" / f"{m['case']}_dt0p0125_optimized.csv"
            theory = np.genfromtxt(theory_path, names=True, delimiter=",")
            assert np.all(np.diff(theory["t"]) > 0)
            assert theory["t"][0] <= t[0] and theory["t"][-1] >= t[-1]
            ref_s = np.column_stack([np.interp(t, theory["t"], theory[f"s{i+1}"]) for i in range(k)])
            ref_k = np.interp(t, theory["t"], theory["kappa"])
            coarse = np.genfromtxt(ROOT / "theory" / f"{m['case']}_dt0p025_optimized.csv", names=True, delimiter=",")
            theory_differences = {name: float(np.max(np.abs(np.interp(t, coarse["t"], coarse[name]) - np.interp(t, theory["t"], theory[name])))) for name in [*(f"s{i+1}" for i in range(k)), "kappa"]}
            review = {"file": path.name, "sha256": digest(path), "case":m["case"], "N":n, "dt":dt,
                      "replicas":r, "seed":m["seed"], "source_sha256":m["source_sha256"],
                      "theory_sha256":digest(theory_path), "theory_refinement_max_abs":theory_differences,
                      "arms": []}
            for arm in range(2):
                # Stack replica means explicitly, independently of analyzer's broadcast expressions.
                mean_s = sum(s[arm, j] for j in range(r))/r
                mean_k = sum(kap[arm, j] for j in range(r))/r
                se_s = np.sqrt(sum((s[arm,j]-mean_s)**2 for j in range(r))/(r*(r-1)))
                se_k = np.sqrt(sum((kap[arm,j]-mean_k)**2 for j in range(r))/(r*(r-1)))
                review["arms"].append({
                    "name":m["modes"][arm],
                    "signal_rmse_by_mode":np.sqrt(np.mean((mean_s-ref_s)**2,axis=0)).tolist(),
                    "signal_rmse_pooled":float(np.sqrt(np.mean((mean_s-ref_s)**2))),
                    "kappa_rmse":float(np.sqrt(np.mean((mean_k-ref_k)**2))),
                    "signal_max_abs_error_by_mode":np.max(np.abs(mean_s-ref_s),axis=0).tolist(),
                    "kappa_max_abs_error":float(np.max(np.abs(mean_k-ref_k))),
                    "terminal_signal_mean":mean_s[-1].tolist(),"terminal_signal_sem":se_s[-1].tolist(),
                    "terminal_signal_theory":ref_s[-1].tolist(),
                    "terminal_kappa_mean":float(mean_k[-1]),"terminal_kappa_sem":float(se_k[-1]),
                    "terminal_kappa_theory":float(ref_k[-1]),
                    "negative_terminal_replica_counts":np.sum(s[arm,:,-1,:]<0,axis=0).tolist(),
                    "ever_negative_replica_counts":np.sum(np.any(s[arm]<0,axis=1),axis=0).tolist(),
                    "negative_while_theory_above_three_noise_floors":np.sum(np.any((s[arm]<0)&(ref_s>3/np.sqrt(n)),axis=1),axis=0).tolist(),
                })
            pair_s = s[0]-s[1]
            pair_k = kap[0]-kap[1]
            review["paired_mean_signal_rms"] = float(np.sqrt(np.mean(np.mean(pair_s,axis=0)**2)))
            review["paired_mean_kappa_rms"] = float(np.sqrt(np.mean(np.mean(pair_k,axis=0)**2)))
            review["paired_terminal_signal"] = np.mean(pair_s[:,-1],axis=0).tolist()
            review["paired_terminal_signal_sem"] = (np.std(pair_s[:,-1],axis=0,ddof=1)/np.sqrt(r)).tolist()
            reviews.append(review)
    analyzer_checks, analyzer_max_error = 0, 0.0
    summary_path = ROOT / "validation" / "summary.json"
    if summary_path.exists():
        other_runs = json.loads(summary_path.read_text())["runs"]
        other_map = {(x["case"],x["N"],x["dt"]):x for x in other_runs}
        for review in reviews:
            identity = review["case"],review["N"],review["dt"]
            if identity not in other_map:
                continue  # A concurrent analyzer may still describe an earlier retrieval.
            other = other_map[identity]
            assert other["sha256"] == review["sha256"]
            comparisons = [
                (other["s_rmse_vs_dmft"], [a["signal_rmse_pooled"] for a in review["arms"]]),
                (other["kappa_rmse_vs_dmft"], [a["kappa_rmse"] for a in review["arms"]]),
                (other["max_s_mean_error_vs_dmft"], [max(a["signal_max_abs_error_by_mode"]) for a in review["arms"]]),
                (other["max_kappa_mean_error_vs_dmft"], [a["kappa_max_abs_error"] for a in review["arms"]]),
                (other["terminal_s_mean"], [a["terminal_signal_mean"] for a in review["arms"]]),
                (other["terminal_s_sem"], [a["terminal_signal_sem"] for a in review["arms"]]),
                (other["terminal_kappa_mean"], [a["terminal_kappa_mean"] for a in review["arms"]]),
                (other["terminal_kappa_sem"], [a["terminal_kappa_sem"] for a in review["arms"]]),
                (other["paired_s_mean_difference_rms"], review["paired_mean_signal_rms"]),
                (other["paired_kappa_mean_difference_rms"], review["paired_mean_kappa_rms"]),
                (other["paired_terminal_s_difference"], review["paired_terminal_signal"]),
                (other["paired_terminal_s_sem"], review["paired_terminal_signal_sem"]),
            ]
            for first, second in comparisons:
                error = float(np.max(np.abs(np.asarray(first)-np.asarray(second))))
                assert error < 1e-13, (identity,error)
                analyzer_max_error = max(analyzer_max_error,error)
                analyzer_checks += 1
    complete = seen == expected
    report = {"complete":complete, "run_count":len(seen),
              "missing":[list(x) for x in sorted(expected-seen)],
              "protocol_and_array_checks_passed":True,
              "analyzer_summary_fields_compared":analyzer_checks,
              "analyzer_summary_max_absolute_difference":analyzer_max_error,
              "interpretation":"Errors are descriptive; time points are dependent, SEM is pointwise, and no sign alignment is applied.",
              "runs":reviews}
    target = args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({"complete":complete,"run_count":len(seen),"missing":report["missing"],"report":str(target)},indent=2))
    if args.require_complete:
        assert complete, report["missing"]


if __name__ == '__main__':
    main()
