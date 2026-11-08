# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy>=2,<3", "scipy>=1.16,<2"]
# ///
"""Audit exact-posterior controls using independent N=2 quadrature and GOE draws."""

import argparse
import hashlib
import json
import math
import tomllib
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata

from exact_reference import reference


def split_chains(x: np.ndarray) -> np.ndarray:
    half = x.shape[1] // 2
    return np.concatenate([x[:, :half], x[:, -half:]], axis=0)


def basic_rhat(x: np.ndarray) -> float:
    m, n = x.shape
    within = np.var(x, axis=1, ddof=1).mean()
    if within < 1e-25:
        return 1.0
    between = n * np.var(x.mean(axis=1), ddof=1)
    return float(math.sqrt(((n - 1) / n * within + between / n) / within))


def rank_transform(x: np.ndarray) -> np.ndarray:
    ranks = rankdata(x.ravel(), method="average").reshape(x.shape)
    return norm.ppf((ranks - 3 / 8) / (x.size + 1 / 4))


def rank_rhat(x: np.ndarray) -> float:
    split = split_chains(x)
    return max(
        basic_rhat(rank_transform(split)),
        basic_rhat(rank_transform(np.abs(split - np.median(split)))),
    )


def effective_size(x: np.ndarray) -> float:
    """Split-chain Geyer initial-positive, monotone paired autocorrelation sum."""
    x = split_chains(x)
    m, n = x.shape
    centered = x - x.mean(axis=1, keepdims=True)
    nfft = 1 << (2 * n - 1).bit_length()
    ft = np.fft.rfft(centered, n=nfft, axis=1)
    acov = np.fft.irfft(ft * ft.conj(), n=nfft, axis=1)[:, :n].real / n
    within = np.var(x, axis=1, ddof=1).mean()
    if within < 1e-25:
        return float(x.size)
    between = n * np.var(x.mean(axis=1), ddof=1)
    variance = (n - 1) / n * within + between / n
    rho = 1 - (within - acov.mean(axis=0)) / variance
    rho[0] = 1.0
    pairs = rho[: 2 * (n // 2)].reshape(-1, 2).sum(axis=1)
    nonpositive = np.flatnonzero(pairs <= 0)
    if len(nonpositive):
        pairs = pairs[: nonpositive[0]]
    if len(pairs) == 0:
        return float(x.size)
    pairs = np.minimum.accumulate(pairs)
    tau = max(1.0, -1 + 2 * pairs.sum())
    return float(min(x.size, x.size / tau))


def summarize(x: np.ndarray) -> dict:
    """Conservative MCSE: autocorrelation, batches, and chain-mean dispersion."""
    m, n = x.shape
    ess = effective_size(x)
    ess_se = float(np.std(x, ddof=1) / math.sqrt(ess))
    chain_se = float(np.std(x.mean(axis=1), ddof=1) / math.sqrt(m))
    batch = max(20, int(math.sqrt(n)))
    nbatch = n // batch
    batch_means = x[:, : nbatch * batch].reshape(m, nbatch, batch).mean(axis=2)
    batch_se = float(np.std(batch_means, ddof=1) / math.sqrt(batch_means.size))
    return {
        "mean": float(x.mean()),
        "mcse": max(ess_se, chain_se, batch_se),
        "mcse_autocorrelation": ess_se,
        "mcse_batches": batch_se,
        "mcse_between_chains": chain_se,
        "ess": ess,
        "rank_normalized_split_rhat": rank_rhat(x),
        "chain_means": x.mean(axis=1).tolist(),
        "retained_per_chain": n,
    }


def load_chains(root: Path, case: str, n: int) -> tuple[dict, list[dict], dict]:
    arrays, metadata, hashes = [], [], {}
    for chain in range(1, 5):
        folder = root / case / f"N{n}" / f"chain{chain}"
        meta = tomllib.loads((folder / "metadata.toml").read_text())
        path = folder / "draws.csv"
        data = np.genfromtxt(path, names=True, delimiter=",")
        assert len(data) == meta["iterations"] // meta["thin"]
        arrays.append(data)
        metadata.append(meta)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata_path = folder / "metadata.toml"
        hashes[str(metadata_path)] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    lengths = {len(x) for x in arrays}
    if len(lengths) != 1:
        raise ValueError(f"Unequal retained lengths in {case}: {lengths}")
    columns = {key: np.stack([a[key] for a in arrays]) for key in arrays[0].dtype.names}
    return columns, metadata, hashes


def compare(samples: np.ndarray, target: float, reference_se: float = 0.0) -> dict:
    stats = summarize(samples)
    stats["reference"] = target
    stats["reference_mcse"] = reference_se
    total_se = math.hypot(stats["mcse"], reference_se)
    delta = stats["mean"] - target
    stats["difference"] = delta
    stats["z_score"] = delta / total_se if total_se > 1e-14 else 0.0
    stats["passed"] = bool(
        abs(stats["z_score"]) <= 4
        and stats["rank_normalized_split_rhat"] <= 1.01
        and stats["ess"] >= 400
    )
    return stats


def independent_goe(n: int, gamma: float, count: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    spectra = np.empty((count, n))
    for offset in range(0, count, 1000):
        size = min(1000, count - offset)
        z = rng.normal(size=(size, n, n))
        matrices = (z + z.transpose(0, 2, 1)) / math.sqrt(2 * n * gamma)
        spectra[offset : offset + size] = np.linalg.eigvalsh(matrices)[:, ::-1]
    return {
        "spectra": spectra,
        "lambda1": spectra[:, 0],
        "lambda2": spectra[:, 1],
        "lambda3": spectra[:, 2],
        "gap12": spectra[:, 0] - spectra[:, 1],
        "top2_mean": spectra[:, :2].mean(axis=1),
        "second_moment": np.square(spectra).mean(axis=1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parents[1]
    parser.add_argument("--controls", type=Path, default=base / "results/controls")
    parser.add_argument("--goe-draws", type=int, default=50000)
    parser.add_argument("--goe-seed", type=int, default=611069991)
    args = parser.parse_args()

    n2, n2_metadata, n2_hashes = load_chains(args.controls, "exact_n2", 2)
    exact = reference(4.0, 0.25, 1.25, 2)
    means = exact["means"]
    targets = {
        "lambda1": means["lambda1"],
        "lambda2": means["lambda2"],
        "gap12": means["gap12"],
        "u11": means["alignment_top1_data1"],
        "u22": means["alignment_top2_data2"],
        "h1": means["condensation_top1"],
        "h2": 1 - means["condensation_top1"],
        "h_top2": 1.0,
        "sample_c1": means["predictive_data1"],
        "sample_c2": 1 - means["predictive_data1"],
        "second_moment": means["radius_squared"] + exact["trace_shift"]["variance"],
        "w11": means["w11"],
        "w22": means["w22"],
        "trace_shift": 0.0,
    }
    n2_results = {key: compare(n2[key], value) for key, value in targets.items()}
    n2_results["trace_shift_variance"] = compare(
        n2["trace_shift"] ** 2, exact["trace_shift"]["variance"]
    )

    goe, goe_metadata, goe_hashes = load_chains(args.controls, "goe_k1", 16)
    independent = independent_goe(16, 0.7, args.goe_draws, args.goe_seed)
    spectra_path = args.controls / "independent_goe_N16.npz"
    np.savez_compressed(
        spectra_path,
        eigenvalues=independent["spectra"],
        seed=np.array(args.goe_seed),
        gamma=np.array(0.7),
    )
    goe_results = {}
    for key in ["lambda1", "lambda2", "lambda3", "gap12", "top2_mean"]:
        values = independent[key]
        goe_results[key] = compare(
            goe[key], float(values.mean()), float(values.std(ddof=1) / math.sqrt(len(values)))
        )
    goe_results["second_moment"] = compare(goe["second_moment"], 17 / (16 * 0.7))
    goe_results["trace_shift"] = compare(goe["trace_shift"], 0.0)
    goe_results["trace_shift_variance"] = compare(goe["trace_shift"] ** 2, 2 / (16**2 * 0.7))
    result = {
        "description": "Independent exact finite-N equilibrium sampler controls",
        "passed": all(v["passed"] for v in [*n2_results.values(), *goe_results.values()]),
        "criteria": {
            "absolute_z_score_max": 4,
            "rank_normalized_split_rhat_max": 1.01,
            "ess_min": 400,
            "mcse": "maximum of FFT initial-monotone autocorrelation, batch-means, and chain-mean SE",
            "constant_observables": "rank Rhat=1, ESS=retained_count, z=0 for numerical zero discrepancy",
        },
        "N2": {"comparisons": n2_results, "metadata": n2_metadata},
        "GOE_N16": {
            "comparisons": goe_results,
            "metadata": goe_metadata,
            "independent_reference": {"seed": args.goe_seed, "draws": args.goe_draws, "path": str(spectra_path)},
        },
        "input_sha256": n2_hashes | goe_hashes | {
            str(spectra_path): hashlib.sha256(spectra_path.read_bytes()).hexdigest(),
        },
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("validate_controls.py", "exact_reference.py")
        },
    }
    output = args.controls / "validation.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Exact equilibrium control validation",
        "",
        f"Overall status: {'PASS' if result['passed'] else 'REVIEW REQUIRED'}.",
        "",
        "Four independent chains per control. N=2 uses independent radial quadrature;",
        f"N=16 uses {args.goe_draws:,} independent GOE draws and an analytic second moment.",
        "The z score combines MCMC and reference uncertainty. The MCMC standard error",
        "accounts for serial correlation and is enlarged if batch or chain variability is greater.",
        "",
    ]
    for title, comparisons in [("N=2 exact posterior", n2_results), ("K=1, eta=1 exact GOE spectrum at N=16", goe_results)]:
        lines += [f"## {title}", "", "| Observable | Estimate | MCSE | Reference | z | Rhat | ESS |", "|---|---:|---:|---:|---:|---:|---:|"]
        for key, stats in comparisons.items():
            lines.append(
                f"| {key} | {stats['mean']:.8f} | {stats['mcse']:.6f} | {stats['reference']:.8f} | "
                f"{stats['z_score']:.2f} | {stats['rank_normalized_split_rhat']:.4f} | {stats['ess']:.0f} |"
            )
        lines.append("")
    lines += ["The saved validation binds every control chain's draws and metadata, the independent GOE reference archive, and both validation/quadrature source files by SHA-256.", "",
              "Passing these controls supports implementation correctness; it does not establish production-chain equilibration.", ""]
    (args.controls / "validation.md").write_text("\n".join(lines))
    print(json.dumps({"passed": result["passed"], "report": str(output)}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
