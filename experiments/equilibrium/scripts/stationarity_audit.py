# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy==2.4.4", "scipy==1.17.1", "arviz==0.22.0"]
# ///
"""Save descriptive quarter/block checks without selecting or discarding draws."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    experiment = Path(__file__).resolve().parents[1]
    catalog = analyze.source_catalog(experiment, [])
    paths = sorted(args.input.glob("chain*/metadata.toml"))
    if len(paths) != 4:
        raise ValueError("Exactly four completed chains are required")
    chains = [analyze.load_chain(path, catalog) for path in paths]
    _, group = analyze.analyze_group(
        chains, SimpleNamespace(min_chains=4, rhat_limit=1.01, ess_min=400)
    )
    meta = chains[0]["metadata"]
    names = sorted(analyze.MATRIX_GATES | {"h_top2", "score_residual_c1", "score_residual_c2"})
    rows = {}
    for name in names:
        if name.startswith("score_residual_c"):
            direction = int(name[-1])
            values = np.stack([
                meta["c"][direction - 1]
                - len(meta["c"]) * chain["data"][f"sample_c{direction}"]
                - meta["gamma"] * chain["data"][f"w{direction}{direction}"]
                for chain in chains
            ])
        else:
            values = np.stack([chain["data"][name] for chain in chains])
        length = values.shape[1]
        if length % 12:
            raise ValueError("Saved draw count must be divisible by 12")
        blocks = values.reshape(4, 12, length // 12).mean(axis=2)
        quarters = values.reshape(4, 4, length // 4).mean(axis=2)
        windows = {
            "full": values,
            "first_half": values[:, : length // 2],
            "last_half": values[:, length // 2 :],
            "last_quarter": values[:, 3 * length // 4 :],
            "drop_first_quarter": values[:, length // 4 :],
        }
        rows[name] = {
            "chain_means": values.mean(axis=1).tolist(),
            "chain_half_differences": (
                windows["last_half"].mean(axis=1) - windows["first_half"].mean(axis=1)
            ).tolist(),
            "chain_quarter_means": quarters.tolist(),
            "pooled_quarter_means": quarters.mean(axis=0).tolist(),
            "chain_12_block_means": blocks.tolist(),
            "pooled_12_block_means": blocks.mean(axis=0).tolist(),
            "half_comparison": analyze.half_comparison(values),
            "windows": {
                label: analyze.sampling_diagnostics(window, False)
                for label, window in windows.items()
            },
        }
    fingerprints = {chain["path"]: chain["draws_sha256"] for chain in chains}
    fingerprints.update({chain["metadata_path"]: chain["metadata_sha256"] for chain in chains})
    result = {
        "scope": "Descriptive fixed windows only; no additional pass gate, draw selection, or replacement of primary estimates",
        "input": str(args.input.resolve()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "analysis_source_sha256": hashlib.sha256(Path(analyze.__file__).read_bytes()).hexdigest(),
        "input_sha256": fingerprints,
        "group": group,
        "N": meta["N"], "case": meta["case"], "warmup": meta["warmup"],
        "iterations": meta["iterations"], "thin": meta["thin"],
        "seeds": [chain["metadata"]["seed"] for chain in chains],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name in ("w22", "gap12", "score_residual_c2"):
        row = rows[name]
        print(name, json.dumps({
            "quarters": row["pooled_quarter_means"],
            "chain_half_differences": row["chain_half_differences"],
            "half_comparison": row["half_comparison"],
        }))
    print(f"Saved descriptive audit: {args.output}")


if __name__ == "__main__":
    main()
