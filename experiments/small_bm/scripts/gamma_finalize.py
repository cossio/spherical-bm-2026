# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy>=2.3"]
# ///
"""Select validated BM gamma estimates and create the common plotting product."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def nearest(cells, gamma, eta=10.0):
    eligible = [cell for cell in cells if cell["eta"] == eta]
    cell = min(eligible, key=lambda entry: abs(np.log(entry["gamma"] / gamma)))
    if abs(np.log(cell["gamma"] / gamma)) > 1e-10:
        raise ValueError(f"Missing cell gamma={gamma}, eta={eta}")
    return cell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "orbit": args.results / "gamma/gamma_summary.json",
        "cold_orbit": args.results / "gamma_cold_orbit/gamma_summary.json",
        "cold_laplace": args.results / "gamma_cold_laplace/gamma_cold_summary.json",
        "cold_hmc": args.results / "gamma_cold_hmc/gamma_cold_hmc_summary.json",
    }
    runs = {name: json.loads(path.read_text()) for name, path in paths.items()}
    grid_path = args.results / "gamma/grid.npz"
    if not grid_path.exists():
        grid_path = args.results / "gamma/gamma_draws.npz"
    raw = np.load(grid_path)
    etas = np.append(raw["eta"], 10.0)
    gammas = raw["gamma"]
    names = raw["metric_names"]
    mean = np.full((len(etas), len(gammas), len(names)), np.nan)
    mc_se = np.full_like(mean, np.nan)
    method = np.empty((len(etas), len(gammas)), dtype="U24")
    ess = np.full((len(etas), len(gammas)), np.nan)
    rhat = np.full_like(ess, np.nan)
    selected = []
    for ie, eta in enumerate(etas):
        for ig, gamma in enumerate(gammas):
            if eta <= 1:
                cell = nearest(runs["orbit"]["cells"], gamma, eta)
                source = "orbit_iid" if eta == 1 else "orbit_importance"
                if cell["importance_ess"] < 500:
                    raise ValueError("Unexpected low ESS on warm posterior")
            elif gamma < 0.0063095734448:
                cell = nearest(runs["cold_orbit"]["cells"], gamma)
                source = "orbit_importance"
                if cell["importance_ess"] < 500:
                    raise ValueError("Cold orbit tail fails ESS gate")
            elif gamma < 0.2:
                cell = nearest(runs["cold_hmc"]["cells"], gamma)
                source = "adjusted_hmc"
                if cell["max_rank_split_rhat"] > 1.05 or cell["min_bulk_ess"] < 100 or cell["divergences"]:
                    raise ValueError("Cold bridge HMC fails convergence gate")
            elif gamma <= 100:
                cell = nearest(runs["cold_laplace"]["cells"], gamma)
                source = "laplace_importance"
                if cell["importance_ess"] < 1000 or cell["maximum_normalized_weight"] > 0.01:
                    raise ValueError("Cold Laplace tail fails importance gate")
            else:
                cell = nearest(runs["cold_orbit"]["cells"], gamma)
                source = "orbit_importance"
                if cell["importance_ess"] < 1000:
                    raise ValueError("Cold high-gamma tail fails ESS gate")
            method[ie, ig] = source
            ess[ie, ig] = cell.get("importance_ess", cell.get("min_bulk_ess", np.nan))
            rhat[ie, ig] = cell.get("max_rank_split_rhat", np.nan)
            for im, name in enumerate(names):
                if name in cell["statistics"]:
                    stat = cell["statistics"][name]
                    mean[ie, ig, im], mc_se[ie, ig, im] = stat["mean"], stat["mc_se"]
            selected.append({"eta": float(eta), "gamma": float(gamma), "method": source,
                             "diagnostic_ess": float(ess[ie, ig]),
                             "statistics": cell["statistics"]})
    crosschecks = []
    for hc in runs["cold_hmc"]["cells"]:
        gamma = hc["gamma"]
        compare = []
        if gamma <= 0.0100000001:
            compare.append(("orbit", nearest(runs["cold_orbit"]["cells"], gamma)))
        if gamma >= 0.15:
            compare.append(("laplace", nearest(runs["cold_laplace"]["cells"], gamma)))
        h = hc["statistics"]["reverse_total"]
        for kind, other in compare:
            o = other["statistics"]["reverse_total"]
            delta = h["mean"] - o["mean"]
            se = float(np.hypot(h["mc_se"], o["mc_se"]))
            crosschecks.append({"gamma": gamma, "method_compared_to_hmc": kind,
                                "difference": delta, "combined_mc_se": se, "z_score": delta / se})
    if max(abs(check["z_score"]) for check in crosschecks) > 4:
        raise ValueError("Independent cold proposal crosscheck failed")
    uniform = runs["orbit"]["uniform_reverse"]
    peaks = []
    for ie, eta in enumerate(etas):
        index = int(np.argmax(mean[ie, :, 0]))
        peaks.append({"eta": float(eta), "gamma_peak": float(gammas[index]),
                      "reverse_peak": float(mean[ie, index, 0]),
                      "overshoot_above_uniform": float(mean[ie, index, 0] - uniform),
                      "peak_mc_se": float(mc_se[ie, index, 0])})
    # Eight independent HMC chain means estimate those cells' SE. The
    # Student-t critical value avoids calling 1.96*SE an exact finite-chain CI.
    critical = np.where(method == "adjusted_hmc", 2.3646242515927844, 1.96)
    mc_ci95_halfwidth = critical[..., None] * mc_se
    final_path = args.results / "gamma_final.npz"
    np.savez_compressed(final_path, eta=etas, gamma=gammas, metric_names=names,
                        mean=mean, mc_se=mc_se, mc_ci95_halfwidth=mc_ci95_halfwidth,
                        method=method, diagnostic_ess=ess,
                        diagnostic_rhat=rhat, uniform_reverse=uniform,
                        weak_gamma_reverse=runs["orbit"]["weak_gamma_reverse"])
    summary = {
        "model_version": "fully-visible-bm-v1",
        "posterior_assumption": runs["orbit"]["posterior_assumption"],
        "N": 12, "K": 1, "omega": 3.0,
        "eta": etas.tolist(), "gamma": gammas.tolist(), "metric_names": names.tolist(),
        "data_average": "Exact teacher-population average over all K1 datasets",
        "uncertainty": "Monte Carlo SE of independent importance samples or independent HMC chain means; not a dataset confidence interval",
        "mc_ci95_halfwidth": "1.96*SE for independent iid/importance estimates; Student-t7 critical value2.364624*SE for the eight-chain HMC means",
        "raw_hmc_interval_note": (runs["cold_hmc"].get("uncertainty") or
                                  "Raw HMC mc_ci95 fields use a normal approximation; final NPZ mc_ci95_halfwidth uses Student-t7 for these cells"),
        "cold_method_selection": "Prior-orbit below gamma .00631; adjusted HMC .00631 through .19953; Laplace importance .25119 through100; prior-orbit above100",
        "uniform_reverse": uniform, "weak_gamma_reverse": runs["orbit"]["weak_gamma_reverse"],
        "min_selected_importance_ess": float(min(entry["diagnostic_ess"] for entry in selected if entry["method"] != "adjusted_hmc")),
        "cold_hmc_max_rhat": max(cell["max_rank_split_rhat"] for cell in runs["cold_hmc"]["cells"]),
        "cold_hmc_min_ess": min(cell["min_bulk_ess"] for cell in runs["cold_hmc"]["cells"]),
        "cold_hmc_divergences": sum(cell["divergences"] for cell in runs["cold_hmc"]["cells"]),
        "cold_crosschecks": crosschecks,
        "peaks": peaks,
        "validation": runs["orbit"]["validation"],
        "source_summary_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
        "plot_data_sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
        "cells": selected,
    }
    (args.results / "gamma_final.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: summary[key] for key in ["peaks", "min_selected_importance_ess", "cold_hmc_max_rhat", "cold_hmc_min_ess", "cold_crosschecks"]}, indent=2))


if __name__ == "__main__":
    main()
