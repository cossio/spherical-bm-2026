# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy==2.4.6", "scipy==1.17.1", "threadpoolctl==3.6.0"]
# ///
"""Archive initial posterior estimates and incorporate longer warm chains."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import posterior as bm
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parents[1] / "results"
    parser.add_argument("--initial", type=Path, default=base / "posterior")
    parser.add_argument("--refinement", type=Path, default=base / "posterior-warm-refinement")
    args = parser.parse_args()
    metadata = json.loads((args.initial / "metadata.json").read_text())
    initial_draws = metadata.get("refinement", {}).get("initial_draws_per_chain", metadata["configuration"]["draws"])
    if not isinstance(initial_draws, int):
        raise RuntimeError("Cannot determine initial campaign draw count")
    archive = args.initial / f"initial-{initial_draws}"
    archive.mkdir(exist_ok=True)
    refined_meta = json.loads((args.refinement / "metadata.json").read_text())
    execution_hash = metadata["script_sha256"]
    if execution_hash != refined_meta["script_sha256"]:
        raise RuntimeError("Initial and refinement campaigns used different sampler source files")
    provenance_dir = args.initial.parent / "provenance"
    provenance_dir.mkdir(exist_ok=True)
    candidates = list(provenance_dir.glob("posterior_execution_source*.py")) + [Path(bm.__file__)]
    matching = [p for p in candidates if hashlib.sha256(p.read_bytes()).hexdigest() == execution_hash]
    if not matching:
        raise RuntimeError("No preserved or maintained sampler source matches the campaign hash")
    execution_source = matching[0]
    if execution_source == Path(bm.__file__):
        execution_source = provenance_dir / f"posterior_execution_source-{execution_hash[:12]}.py"
        if execution_source.exists() and hashlib.sha256(execution_source.read_bytes()).hexdigest() != execution_hash:
            raise RuntimeError("Refusing to overwrite a different preserved execution source")
        if not execution_source.exists():
            shutil.copy2(bm.__file__, execution_source)
    gammas = metadata["configuration"]["gammas"]
    etas = metadata["configuration"]["etas"]
    original_result = archive / "posterior_results.npz"
    if not original_result.exists():
        original_result = args.initial / "posterior_results.npz"
    with np.load(original_result) as old:
        old_metrics = old["metrics"].copy()
        old_ids = old["dataset_indices"].copy()
    # Refuse to erase the saved initial campaign on a repeated invocation.
    for name in ["posterior_results.npz", "metadata.json", "predictive_validation.json", "run.log", "validation.log"]:
        source = args.initial / name
        target = archive / name
        if source.exists() and not target.exists():
            shutil.copy2(source, target)
    refinements = []
    for source in sorted(args.refinement.glob("gamma-*_eta-*.npz")):
        with np.load(source) as result:
            if not np.array_equal(result["dataset_indices"], old_ids):
                raise RuntimeError(f"Dataset mismatch: {source}")
            gamma, eta = float(result["gamma"]), float(result["eta"])
            gi, ei = gammas.index(gamma), etas.index(eta)
            refinements.append({"gamma": gamma, "eta": eta,
                                "draws_per_chain": int(result["draws_per_chain"]),
                                "mean_KL_change_from_initial": (result["metrics"][:, :2] - old_metrics[gi, ei, :, :2]).mean(axis=0).tolist()})
        old_path = args.initial / source.name
        if not (archive / source.name).exists():
            shutil.copy2(old_path, archive / source.name)
        shutil.copy2(source, old_path)
    metadata["configuration"]["draws"] = "cell-dependent; see cell_configuration"
    metadata["execution_source_snapshot"] = {"path": str(execution_source), "sha256": execution_hash}
    metadata["maintained_sampler_script_sha256"] = hashlib.sha256(Path(bm.__file__).read_bytes()).hexdigest()
    metadata["refinement"] = {
        "reason": "Initial estimated predictive KL plug-in bias was comparable to adjacent weak-regularization grid minima",
        "initial_draws_per_chain": initial_draws,
        "refined_cells": refinements,
        "nested_draws": "Same seed and warmup; longer trajectories extend the original trajectories" if all(metadata["configuration"][key] == refined_meta["configuration"][key] for key in ["seed", "warmup", "chains", "leapfrog_min", "leapfrog_max"]) else "Seed or trajectory settings changed; extensions are not claimed to be nested",
        "sampler_script_sha256": refined_meta["script_sha256"],
        "merge_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "initial_results_archive": str(archive),
    }
    metadata["cell_configuration"] = []
    draws = np.empty((len(gammas), len(etas)), int)
    for gi, gamma in enumerate(gammas):
        for ei, eta in enumerate(etas):
            with np.load(args.initial / f"gamma-{gamma:g}_eta-{eta:g}.npz") as result:
                draws[gi, ei] = int(result["draws_per_chain"])
                has_fingerprint = "configuration_json" in result
                if has_fingerprint:
                    encoded = str(result["configuration_json"])
                    if hashlib.sha256(encoded.encode()).hexdigest() != str(result["configuration_sha256"]) or json.loads(encoded)["sampler_script_sha256"] != execution_hash:
                        raise RuntimeError("Cell configuration fingerprint or sampler source mismatch")
                metadata["cell_configuration"].append({"gamma": gamma, "eta": eta,
                    "draws_per_chain": int(result["draws_per_chain"]), "warmup": int(result["warmup"]),
                    "chains": int(result["chains"]), "seed": int(result["seed"]),
                    "leapfrog_min": metadata["configuration"]["leapfrog_min"],
                    "leapfrog_max": metadata["configuration"]["leapfrog_max"],
                    "save_traces": metadata["configuration"]["save_traces"],
                    "sampler_script_sha256": execution_hash,
                    "provenance": "Verified cell configuration fingerprint and sampler source" if has_fingerprint else "Recorded original campaign arguments and verified execution source snapshot; legacy cells intentionally remain non-resumable by the maintained code"})
    bm.aggregate(args.initial, gammas, etas, metadata)
    final = args.initial / "posterior_results.npz"
    with np.load(final) as result:
        arrays = {name: result[name] for name in result.files}
    arrays["draws_per_chain"] = draws
    np.savez_compressed(final, **arrays)
    metadata["files_sha256"][final.name] = hashlib.sha256(final.read_bytes()).hexdigest()
    (args.initial / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata["refinement"], indent=2))


if __name__ == "__main__":
    main()
