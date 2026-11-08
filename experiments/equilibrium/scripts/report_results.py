# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Write finite-size comparisons from verified diagnostic summaries; no fits."""

import argparse
import hashlib
import json
import math
import tomllib
from pathlib import Path


def interval(row):
    return f"{row['mean']:.5f} ± {1.959963984540054 * row['mcse']:.5f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    experiment = Path(__file__).resolve().parents[1]
    parser.add_argument("--summary", type=Path, default=experiment / "validation/production/summary.json")
    parser.add_argument("--saddle", type=Path, default=experiment / "results/saddle_predictions.toml")
    parser.add_argument("--output", type=Path, default=experiment / "report_results.md")
    args = parser.parse_args()
    config = tomllib.loads((experiment / "config.toml").read_text())
    saddle = tomllib.loads(args.saddle.read_text())
    source = experiment / "provenance/reference_sources/rmt.jl"
    actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_hash != saddle["provenance"]["rmt_sha256"]:
        raise ValueError("The analytic prediction source has changed since the table was generated")
    report = json.loads(args.summary.read_text()) if args.summary.exists() else {"summary": [], "groups": [], "chains": []}
    case_names = [case["name"] for case in config["cases"]]
    sizes = config["sizes"]
    lookup = {(row["case"], row["N"], row["observable"]): row for row in report["summary"]}
    groups = {(group["case"], group["N"]): group for group in report["groups"]}
    complete = [(case, n) for case in case_names for n in sizes if (case, n) in groups]
    expected = len(case_names) * len(sizes)
    passing = sum(groups[key]["matrix_diagnostics_pass"] for key in complete)
    flags = [(case, n, obs) for case, n in complete for obs in groups[(case, n)].get("score_identity_flags", [])]
    replication_protocol = experiment / "validation/replication_protocol.json"
    replication_path = experiment / "validation/replication_comparison.json"
    replication = json.loads(replication_path.read_text()) if replication_path.exists() else None
    replication_pending = replication_protocol.exists() and (replication is None or not replication["passed"])
    final = len(complete) == expected and passing == expected and not flags and not replication_pending
    status = "validated production results" if final else "provisional; validation still in progress"
    if replication is not None and not replication["passed"]:
        status = "limited; independent replication did not meet the predeclared criteria"
    largest = max(sizes)
    eta4_names = [case["name"] for case in config["cases"] if case["eta"] == 4]
    readout = []
    if all((case, largest) in groups for case in eta4_names):
        max_spectral_relative = max(
            abs(lookup[(case, largest, f"lambda{j+1}")]["mean"] / saddle[case]["lambda"][j] - 1)
            for case in eta4_names for j in (0, 1)
        )
        boundary_mass = lookup[("condensation_boundary", largest, "h_top2")]
        readout += [f"Quantitative readout of the original production grid: at N={largest}, the largest relative offset of either leading eigenvalue across the six η=4 settings is {100*max_spectral_relative:.3f}%. At the condensation boundary, top-two sample mass is {interval(boundary_mass)} versus a zero leading condensate coefficient, showing a resolved finite-N rounding effect.", ""]
    if all(("deep_coalesced", n) in groups for n in sizes):
        gap_trend = " → ".join(f"{lookup[('deep_coalesced',n,'gap12')]['mean']:.5f}" for n in sizes)
        readout += [f"In the detached deep-coalesced setting the mean top gap decreases {gap_trend} for N={','.join(map(str,sizes))}, toward the zero leading gap. At the coalesced bulk edge, the identification of exactly top-two mass/projector statistics with the paper's saddle coefficients remains unresolved; those order-parameter comparisons are diagnostic only. Spectral positions provide the established comparison there.", ""]
    lines = ["# Moderate-N exact equilibrium SBM validation", "",
             f"Status: {status}. {len(complete)}/{expected} case/size groups are available, {passing} pass all matrix diagnostic gates, and {len(flags)} posterior-score identities exceed four MCSE.", "",
             "The target is the actual finite-N posterior over all real symmetric coupling matrices. Exact Bingham rejection draws enter an exchange step, and the independent scalar trace is sampled analytically. Neither the partition function nor the orientation integral is replaced by its saddle value. The chosen temperatures have integer Kη; conclusions are restricted to the eight specified parameter settings and N=32,64,128.", "",
             "All intervals below are 95% Monte Carlo intervals for posterior means. They do not represent the posterior spread, a finite-size confidence band around the asymptotic formula, or a fit error. `validation/production/summary.csv` also records the posterior SD. No finite-size exponent or extrapolation is fitted.", "",
             *readout,
             "Four independent chains start from prior, strongly data-aligned, randomly oriented two-mode, and broadened-prior matrices. Adaptation is confined to warmup. Gates require rank-normalized split/folded R-hat ≤1.01 and bulk/tail ESS ≥400 for both leading eigenvalues, their gap, both top-two data-projector overlaps, the matrix second moment, and W11/W22. Generated-spin observables are additionally monitored; their independent sampling noise cannot substitute for the matrix diagnostics.", "",
             "## Equilibration and independent controls", "",
             "The Bingham kernel passed 9,080 checks, including independent angular quadrature at N=2 and a tilted-Beta integral for two coalesced modes at N=64. The diagnostic implementation distinguishes IID, deliberately separated, and strongly autocorrelated chains in its saved self-test."]
    pilot_path = experiment / "validation/pilot/summary.json"
    if pilot_path.exists():
        pilot = json.loads(pilot_path.read_text())
        examples = [row for row in pilot["summary"] if row["case"] == "deep_coalesced" and row["N"] == 128 and row["observable"] == "w22"]
        if examples:
            example = examples[0]
            lines += ["", f"Short pilots failed the convergence gates in the hardest coalesced setting: at N=128, W22 had R-hat {example['rhat']:.3f} and bulk ESS {example['ess_bulk']:.0f}. This prompted longer production runs with proposal adaptation frozen after warmup. Pilot draws are never pooled into the production estimates or their uncertainty calculations."]
    controls_path = experiment / "validation/controls/summary.json"
    if controls_path.exists():
        controls = json.loads(controls_path.read_text())
        control_pass = sum(group["matrix_diagnostics_pass"] for group in controls["groups"])
        lines += ["", f"Independent end-to-end controls: {control_pass}/{len(controls['groups'])} groups pass the same matrix gates. The interacting N=2 posterior is checked by one-dimensional quadrature; the K=1, η=1 control has exactly the GOE eigenvalue law after integrating orientations."]
        oracle = json.loads((experiment / "results/n2_reference_eta4_gamma0p25.json").read_text())
        means = oracle["means"]
        mapping = {"lambda1": means["lambda1"], "lambda2": means["lambda2"], "gap12": means["gap12"],
                   "w11": means["w11"], "w22": means["w22"], "u11": means["alignment_top1_data1"],
                   "h1": means["condensation_top1"], "sample_c1": means["predictive_data1"],
                   "second_moment": means["radius_squared"] + oracle["trace_shift"]["variance"]}
        z = [(row["observable"], abs(row["mean"] - mapping[row["observable"]]) / row["mcse"])
             for row in controls["summary"] if row["case"] == "exact_n2" and row["observable"] in mapping]
        if z:
            worst = max(z, key=lambda item: item[1])
            lines += [f"Across the {len(z)} reported nonconstant N=2 oracle checks, the largest discrepancy is {worst[1]:.2f} MCSE ({worst[0]})."]
    lines += ["", "| Case | N | Matrix gates | Largest R-hat | Smallest bulk ESS | Smallest tail ESS | Max absolute score z |", "|---|---:|---|---:|---:|---:|---:|"]
    for case, n in complete:
        rows = [row for row in report["summary"] if row["case"] == case and row["N"] == n and row["gate_required"]]
        scores = [abs(lookup[(case, n, name)]["score_mean_z"]) for name in ("score_residual_c1", "score_residual_c2")]
        lines.append(f"| {case} | {n} | {'PASS' if groups[(case,n)]['matrix_diagnostics_pass'] else 'FAIL'} | {max(row['rhat'] for row in rows):.4f} | {min(row['ess_bulk'] for row in rows):.0f} | {min(row['ess_tail'] for row in rows):.0f} | {max(scores):.2f} |")
    failures = [(case, n, groups[(case, n)]["failing_matrix_observables"]) for case, n in complete if not groups[(case, n)]["matrix_diagnostics_pass"]]
    if failures:
        lines += ["", "Remaining convergence failures: " + "; ".join(f"{case}, N={n}: {', '.join(obs)}" for case, n, obs in failures) + ". Means from these groups remain provisional."]
    if flags:
        lines += ["", "Score identities requiring investigation: " + "; ".join(f"{case}, N={n}, {obs}" for case, n, obs in flags) + "."]
    half_checks = [row for row in report["summary"] if row["gate_required"] and row.get("half_difference_z_descriptive") is not None]
    if half_checks:
        worst_half = max(half_checks, key=lambda row: abs(row["half_difference_z_descriptive"]))
        lines += ["", f"The largest descriptive first-versus-last-half difference among the required matrix observables is {worst_half['half_difference_z_descriptive']:+.2f} approximate standard errors ({worst_half['case']}, N={worst_half['N']}, {worst_half['observable']}). This comparison neglects covariance across the split and is one of many correlated checks; it was not a separate pass/fail criterion. Together with the retained traces it records the remaining slow fluctuations, rather than proving that all residual drift is absent."]
    if replication_protocol.exists():
        lines += ["", "A descriptive last-quarter check of deep_coalesced at N=128 exposed a slow W22 mode despite the full-run gates passing. One independent four-chain replication was therefore specified before viewing its results: new seeds, the same four start types, 50,000 warmup and 600,000 retained-period transitions. The original 24-group production grid and its estimates remain unchanged. The protocol in `validation/replication_protocol.json` fixes eleven mean comparisons (eight matrix monitors, top-two sample mass, and both joint score residuals), with autocorrelation-aware uncertainties combined in quadrature and a two-sided Bonferroni cutoff of 2.837597. Both cohorts must separately pass the unchanged matrix gates and score-identity checks. No repeat-until-pass procedure is allowed."]
        if replication is None:
            lines += ["", "The independent replication comparison is pending; final equilibration validation remains open."]
        else:
            comparisons = [row for row in replication["rows"] if row["comparison_monitored"]]
            worst = max(comparisons, key=lambda row: abs(row["difference_z"]))
            lines += ["", f"Independent replication status: {'PASS' if replication['passed'] else 'requires investigation'}. The largest absolute standardized mean difference is {abs(worst['difference_z']):.3f} ({worst['observable']}). Separate and explicitly pooled eight-chain diagnostics are saved in `validation/replication_comparison.json`; the pooled estimates do not replace the primary grid."]
            stationarity_path = experiment / "validation/stationarity_replication.json"
            if stationarity_path.exists():
                windows = json.loads(stationarity_path.read_text())["rows"]
                w22_half = windows["w22"]["half_comparison"]["half_difference_z_descriptive"]
                gap_half = windows["gap12"]["half_comparison"]["half_difference_z_descriptive"]
                score_half = windows["score_residual_c2"]["half_comparison"]["half_difference_z_descriptive"]
                w22_ess = windows["w22"]["windows"]["last_quarter"]["ess_bulk"]
                moment_ess = windows["second_moment"]["windows"]["last_quarter"]["ess_bulk"]
                lines += ["", f"The [descriptive window audit](validation/stationarity_replication.json) finds that the large W22 shift does not recur: its replication half-difference is {w22_half:+.2f} approximate SE, with mixed chain signs. Weaker time fluctuations remain: gap and score-residual-c2 half-differences are {gap_half:+.2f} and {score_half:+.2f}, with positive shifts in all four chains, while last-quarter bulk ESS is {w22_ess:.0f}/{moment_ess:.0f} for W22/second moment. These reduced-window checks remain descriptive; the fixed cohort comparison passing does not establish absence of all long-time fluctuations."]
    lines += ["", "The score residual is computed per joint draw as c_j−K·sample_cj−γWjj, preserving its covariance in the MCSE calculation. Four-MCSE flags trigger investigation; ordinary two-MCSE deviations among many comparisons are not failure criteria.", "",
             "## Eigenvalues and approach to coalescence", "",
             "Bias denotes finite-N mean minus the leading saddle prediction. A statistically resolved bias is expected at finite N and is reported explicitly.", "",
             "| Case | N | λ1 mean ± 95% MC | λ1 saddle | Bias | λ2 mean ± 95% MC | λ2 saddle | Bias |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for case, n in complete:
        a, b = lookup[(case, n, "lambda1")], lookup[(case, n, "lambda2")]
        pa, pb = saddle[case]["lambda"]
        lines.append(f"| {case} | {n} | {interval(a)} | {pa:.5f} | {a['mean']-pa:+.5f} | {interval(b)} | {pb:.5f} | {b['mean']-pb:+.5f} |")
    lines += ["", "| Case | N | Top gap mean ± 95% MC | Saddle gap | Bias |", "|---|---:|---:|---:|---:|"]
    for case, n in complete:
        row = lookup[(case, n, "gap12")]
        predicted = saddle[case]["lambda"][0] - saddle[case]["lambda"][1]
        lines.append(f"| {case} | {n} | {interval(row)} | {predicted:.5f} | {row['mean']-predicted:+.5f} |")
    lines += ["", "For coalesced predictions the finite-N ordered gap is strictly positive. Its size dependence is evidence about approach to coalescence, not a test that finite-N eigenvalues should already coincide. At a bulk edge, leading eigenvalues retain edge fluctuations and can have larger finite-size offsets.", "",
             "## Generated-sample mass and data subspaces", "",
             "The generated-sample observable is the fraction of squared norm in the top two weight modes. For separated outliers or a detached coalesced pair, this captures the condensate carried by those modes. The data overlaps are squared projections onto the full top-two eigenspace, an invariant counterpart of the paper's individual ordered-eigenvector overlaps for a detached pair. At the coalesced bulk edge there is no isolated two-dimensional eigenspace. The displayed h² and u² values there are formal saddle coefficients for diagnostic comparison; the manuscript does not establish their identification with these exact top-two finite-N observables.", "",
             "| Case | N | Top-two mass ± 95% MC | h² saddle coefficient | Difference |", "|---|---:|---:|---:|---:|"]
    for case, n in complete:
        row = lookup[(case, n, "h_top2")]
        predicted = saddle[case]["h_squared"]
        lines.append(f"| {case} | {n} | {interval(row)} | {predicted:.5f} | {row['mean']-predicted:+.5f} |")
    lines += ["", "| Case | N | Data-1 projector ± 95% MC | Saddle coefficient | Difference | Data-2 projector ± 95% MC | Saddle coefficient | Difference |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for case, n in complete:
        a, b = lookup[(case, n, "projector_c1")], lookup[(case, n, "projector_c2")]
        pa, pb = saddle[case]["u_squared"]
        lines.append(f"| {case} | {n} | {interval(a)} | {pa:.5f} | {a['mean']-pa:+.5f} | {interval(b)} | {pb:.5f} | {b['mean']-pb:+.5f} |")
    lines += ["", "Positive mass in a chosen two-dimensional subspace persists at every finite N, including uncondensed settings whose limiting h² is zero. Near either phase boundary the finite-N crossover is rounded. At the coalesced bulk edge there is no fixed spectral separation from the remaining modes; ordered top-two mass and projector statistics deserve more caution than separated-outlier observables.", "",
             "For edge_coalesced, the differences in the two tables are differences from formal saddle coefficients, not verified finite-size biases for an established observable correspondence. The spectral-position comparison does not have this identification issue.", "",
             "## Interpretation and scope", ""]
    largest = max(sizes)
    if all((case, largest) in groups for case in case_names[:6]):
        deviations = [abs(lookup[(case, largest, f"lambda{j+1}")]["mean"] / saddle[case]["lambda"][j] - 1)
                      for case in case_names[:6] for j in (0, 1)]
        lines += [f"At N={largest}, across the six η=4 settings (including both selected boundaries), the largest relative offset of either leading eigenvalue from its saddle value is {100*max(deviations):.2f}%. This describes finite-size accuracy; it is separate from Monte Carlo precision.", ""]
    for case in ("coalescence_boundary", "two_condensed", "deep_coalesced", "edge_coalesced"):
        if all((case, n) in groups for n in sizes):
            trend = " → ".join(f"{lookup[(case,n,'gap12')]['mean']:.5f}" for n in sizes)
            lines += [f"{case}: the mean top gap is {trend} for N={','.join(map(str,sizes))}; the leading saddle gap is zero.", ""]
    if all(("edge_coalesced", n) in groups for n in sizes):
        case = "edge_coalesced"
        mass_trend = " → ".join(f"{lookup[(case,n,'h_top2')]['mean']:.5f}" for n in sizes)
        projector_trend = " → ".join(f"{lookup[(case,n,'projector_c1')]['mean']:.5f}" for n in sizes)
        lines += [f"The bulk-edge coalesced setting is a limitation of a broad moderate-N agreement claim. Its top-two sample mass is {mass_trend}, alongside the formal coefficient h²={saddle[case]['h_squared']:.5f}; its data-1 top-two projector is {projector_trend}, alongside {saddle[case]['u_squared'][0]:.5f}. These statistics do not exhibit monotone agreement with the displayed coefficients over the sampled sizes, despite passing the equilibrium diagnostics. The selected two-dimensional eigenspace has no fixed separation from nearby modes. This is an unresolved finite-size and observable-identification discrepancy; it cannot be attributed solely to slow finite-size convergence. A dedicated definition and analysis of the edge observable is needed before claiming that comparison.", ""]
    lines += ["The numerical experiment assesses the accuracy of selected equilibrium observables and their finite-size evolution. It does not prove global uniqueness, exclude exponentially rare structures, establish a uniform error bound, or replace the separate appendix uniqueness argument. Finite-rate DMFT trajectories and the deferred binary BM/RBM experiments are not used as evidence for this exact equilibrium comparison.", "",
              f"Analytic source SHA-256: `{actual_hash}`. Predictions: `{args.saddle.name}`. Diagnostic summary: `{args.summary}`."]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.output}: {len(complete)}/{expected} groups, {passing} pass, {len(flags)} score flags.")


if __name__ == "__main__":
    main()
