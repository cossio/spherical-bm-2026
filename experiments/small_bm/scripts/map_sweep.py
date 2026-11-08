# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy>=2.3", "scipy>=1.16", "threadpoolctl>=3.6"]
# ///
"""Exact K=1 MAP comparison for current Appendix G.1, including full-space audits.

For each observed xi, minimize over ALL 66 independent couplings:
    L(J) = log Z(J) - sum_{i<j} J_ij xi_i xi_j
           + N gamma/2 sum_{i<j} J_ij**2.
Strict convexity for gamma>0 and signed permutation symmetry imply the unique
global minimizer J_ij=a xi_i xi_j. This is a consequence, not a rank constraint.
Independent state enumeration audits gradients and KL averages, and unconstrained
66-dimensional optimization checks selected cases from random initial conditions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.optimize import brentq, minimize
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

N = 12
OMEGA = 3.0
DIM = N * (N - 1) // 2


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shell_space() -> dict:
    m = N - 2 * np.arange(N + 1)
    multiplicity = np.array([math.comb(N, k) for k in range(N + 1)])
    pair_sum = (m * m - N) / 2.0
    return {"m": m, "multiplicity": multiplicity,
            "log_multiplicity": np.log(multiplicity), "pair_sum": pair_sum,
            "deficit": (N * N - m * m) / (N * (N - 1))}


def scalar_model(a: float, space: dict) -> dict:
    # Center energy at the maximum to avoid cancellation at small gamma.
    shifted_logits = a * (space["pair_sum"] - DIM)
    shifted_logz = float(logsumexp(space["log_multiplicity"] + shifted_logits))
    shell_p = np.exp(space["log_multiplicity"] + shifted_logits - shifted_logz)
    deficit = float(shell_p @ space["deficit"])
    correlation = 1.0 - deficit
    mean_pair_sum = DIM * correlation
    variance_pair_sum = float(shell_p @ (space["pair_sum"] - mean_pair_sum) ** 2)
    logz = a * DIM + shifted_logz
    entropy = shifted_logz + a * DIM * deficit
    return {"correlation": correlation, "deficit": deficit, "logz": logz,
            "entropy": entropy, "variance_pair_sum": variance_pair_sum,
            "shell_p": shell_p, "retrieval": 2.0 * math.exp(-shifted_logz),
            "shifted_logz": shifted_logz}


def solve_map(gamma: float, space: dict) -> tuple[float, dict]:
    def residual(a):
        return N * gamma * a - scalar_model(a, space)["deficit"]

    upper = min(1.0, 1.0 / (N * gamma))
    while residual(upper) < 0:
        upper *= 2.0
    a = brentq(residual, 0.0, upper, xtol=1e-18, rtol=1e-14)
    result = scalar_model(a, space)
    result["stationarity_residual"] = abs(residual(a))
    return a, result


def enumerated_space() -> dict:
    codes = np.arange(1 << N, dtype=np.int64)
    states = 1.0 - 2.0 * ((codes[:, None] >> np.arange(N)) & 1)
    upper_i, upper_j = np.triu_indices(N, 1)
    features = states[:, upper_i] * states[:, upper_j]
    logits = OMEGA / N * features.sum(axis=1)
    logp = logits - logsumexp(logits)
    return {"codes": codes, "states": states, "features": features,
            "upper_i": upper_i, "upper_j": upper_j,
            "teacher_logp": logp, "teacher_p": np.exp(logp)}


def full_objective(j: np.ndarray, gamma: float, space: dict, datum: int = 0,
                   hessian: bool = False):
    f = space["features"]
    logits = f @ j
    lz = logsumexp(logits)
    p = np.exp(logits - lz)
    mu = p @ f
    value = float(lz - f[datum] @ j + N * gamma / 2 * (j @ j))
    gradient = mu - f[datum] + N * gamma * j
    if hessian:
        centered = f - mu
        h = (centered.T * p) @ centered + N * gamma * np.eye(DIM)
        return value, gradient, h
    return value, gradient


def independently_check(space: dict, shell: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    results = {"full_dimensional_gradient_checks": [], "unconstrained_fits": [],
               "all_dataset_direct_kl_checks": []}
    for gamma in [1e-11, 1e-5, 0.001, 0.01, 0.1, 1.0, 10.0, 1e3, 1e6]:
        a, model = solve_map(gamma, shell)
        datum = int(rng.integers(0, 1 << N))
        j = a * space["features"][datum]
        value, gradient, hessian = full_objective(j, gamma, space, datum, True)
        eigen = np.linalg.eigvalsh(hessian)
        check = {"gamma": gamma, "datum_code": datum,
                 "gradient_inf": float(np.max(np.abs(gradient))),
                 "hessian_min_eigenvalue": float(eigen[0]),
                 "strict_convexity_lower_bound": N * gamma,
                 "objective": value,
                 "objective_gap_bound_from_gradient": float(gradient @ gradient / (2 * N * gamma)),
                 "shell_objective_difference": abs(value - (model["shifted_logz"] + N * gamma / 2 * DIM * a * a))}
        results["full_dimensional_gradient_checks"].append(check)
        if check["gradient_inf"] > 2e-10 or eigen[0] < N * gamma - 2e-10:
            raise RuntimeError(f"Full-space audit failed: {check}")

    # Small and moderate gamma exercise distinct highly magnetized/uniform limits.
    for gamma in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        a, _ = solve_map(gamma, shell)
        datum = int(rng.integers(0, 1 << N))
        target = a * space["features"][datum]
        target_value = full_objective(target, gamma, space, datum)[0]
        for initialization in range(2):
            initial = rng.normal(0.0, 0.2, DIM)
            fit = minimize(lambda j: full_objective(j, gamma, space, datum),
                           initial, method="trust-exact", jac=True,
                           hess=lambda j: full_objective(j, gamma, space, datum, True)[2],
                           options={"gtol": 2e-10, "maxiter": 200})
            # Newton polish independently in the full parameter space. scipy's
            # stopping rule can terminate when roundoff masks objective decrease.
            polished = fit.x.copy()
            for _ in range(3):
                _, grad, hess = full_objective(polished, gamma, space, datum, True)
                polished -= np.linalg.solve(hess, grad)
            value, grad = full_objective(polished, gamma, space, datum)
            check = {"gamma": gamma, "datum_code": datum, "initialization": initialization,
                     "scipy_success": bool(fit.success), "scipy_message": str(fit.message),
                     "iterations": int(fit.nit), "newton_polish_steps": 3,
                     "gradient_inf": float(np.max(np.abs(grad))),
                     "coupling_inf_error": float(np.max(np.abs(polished - target))),
                     "objective_difference": value - target_value}
            results["unconstrained_fits"].append(check)
            if check["gradient_inf"] > 2e-9 or check["coupling_inf_error"] > 2e-8:
                raise RuntimeError(f"Unconstrained optimization check failed: {check}")

    teacher = scalar_model(OMEGA / N, shell)
    cstar = teacher["correlation"]
    for gamma in [0.001, 0.01, 0.1, 1.0, 10.0]:
        a, model = solve_map(gamma, shell)
        logq = a * space["features"].sum(axis=1) - model["logz"]
        q = np.exp(logq)
        direct_fwd = 0.0
        direct_rev = 0.0
        for start in range(0, 1 << N, 64):
            indices = space["codes"][start:start + 64, None] ^ space["codes"][None]
            lp = logq[indices]
            p = q[indices]
            fwd = np.sum(space["teacher_p"] * (space["teacher_logp"] - lp), axis=1)
            rev = np.sum(p * (lp - space["teacher_logp"]), axis=1)
            direct_fwd += float(space["teacher_p"][start:start + 64] @ fwd)
            direct_rev += float(space["teacher_p"][start:start + 64] @ rev)
        formula_fwd = -teacher["entropy"] + model["logz"] - a * DIM * cstar * cstar
        formula_rev = -model["entropy"] + teacher["logz"] - OMEGA / N * DIM * cstar * model["correlation"]
        check = {"gamma": gamma, "forward_error": abs(direct_fwd - formula_fwd),
                 "reverse_error": abs(direct_rev - formula_rev),
                 "datasets_enumerated": 1 << N, "states_per_dataset": 1 << N}
        results["all_dataset_direct_kl_checks"].append(check)
        if max(check["forward_error"], check["reverse_error"]) > 1e-10:
            raise RuntimeError(f"Exact dataset average check failed: {check}")

    # Analytic gradients and Hessians of the full objective have a separate
    # central-difference check at a generic nonsymmetric point.
    gamma = 0.17
    j = rng.normal(0, 0.08, DIM)
    direction = rng.normal(size=DIM)
    direction /= np.linalg.norm(direction)
    eps = 2e-5
    _, gradient, hessian = full_objective(j, gamma, space, 87, True)
    plus, gplus = full_objective(j + eps * direction, gamma, space, 87)
    minus, gminus = full_objective(j - eps * direction, gamma, space, 87)
    results["finite_difference_check"] = {
        "directional_gradient_error": abs((plus - minus) / (2 * eps) - gradient @ direction),
        "hessian_vector_inf_error": float(np.max(np.abs((gplus - gminus) / (2 * eps) - hessian @ direction))),
    }
    if max(results["finite_difference_check"].values()) > 1e-7:
        raise RuntimeError("Full-space finite difference check failed")
    derivative_checks = []
    for gamma in [0.001, 0.01, 0.1, 1.0, 10.0]:
        a, model = solve_map(gamma, shell)
        cprime = model["variance_pair_sum"] / DIM
        aprime = -N * a / (N * gamma + cprime)
        derivative = DIM * (a - OMEGA * cstar / N) * cprime * aprime
        def risk(g):
            _, trial = solve_map(g, shell)
            return -trial["entropy"] + teacher["logz"] - OMEGA / N * DIM * cstar * trial["correlation"]
        step = gamma * 1e-4
        finite_difference = (risk(gamma + step) - risk(gamma - step)) / (2 * step)
        error = abs(finite_difference - derivative) / max(1.0, abs(derivative))
        derivative_checks.append({"gamma": gamma, "analytic_derivative": derivative,
                                  "central_difference": finite_difference,
                                  "relative_or_absolute_error": error})
        if error > 2e-7:
            raise RuntimeError("MAP reverse KL derivative check failed")
    results["reverse_kl_derivative_checks"] = derivative_checks
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-grid", type=Path)
    parser.add_argument("--dense-points", type=int, default=681)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026110701)
    parser.add_argument("--skip-independent-checks", action="store_true")
    args = parser.parse_args()
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    shell = shell_space()
    states = enumerated_space()
    teacher = scalar_model(OMEGA / N, shell)
    cstar = teacher["correlation"]
    optimal_a = OMEGA * cstar / N
    optimal_gamma = scalar_model(optimal_a, shell)["deficit"] / (N * optimal_a)
    # The original 67-point grid is exactly the old production grid; the optional
    # file supplies its bitwise float representation for exact array indexing.
    old_grid = (np.load(args.old_grid)["gamma"] if args.old_grid else
                np.concatenate([np.logspace(-11, -3, 9), np.logspace(-3, 2, 51)[1:],
                                np.logspace(2.5, 6, 8)]))
    gamma = np.unique(np.concatenate([np.logspace(-11, 6, args.dense_points), old_grid,
                                      [optimal_gamma]]))
    columns = {name: [] for name in ["a", "forward_kl", "reverse_kl", "objective",
                                    "stationarity_residual", "correlation", "entropy",
                                    "retrieval_probability", "data_squared_overlap",
                                    "leading_mode_variance", "coupling_leading_eigenvalue",
                                    "coupling_subleading_eigenvalue", "coupling_gap",
                                    "forward_kl_derivative_gamma", "reverse_kl_derivative_gamma"]}
    dataset_forward = []
    dataset_reverse = []
    datum_pair_sum = states["features"].sum(axis=1)
    with threadpool_limits(limits=args.threads):
        for g in gamma:
            a, model = solve_map(float(g), shell)
            c = model["correlation"]
            cprime = model["variance_pair_sum"] / DIM
            aprime = -N * a / (N * g + cprime)
            forward = -teacher["entropy"] + model["logz"] - a * DIM * cstar * cstar
            reverse = -model["entropy"] + teacher["logz"] - OMEGA / N * DIM * cstar * c
            values = [a, forward, reverse, model["shifted_logz"] + N * g / 2 * DIM * a * a,
                      model["stationarity_residual"], c, model["entropy"], model["retrieval"],
                      (1 + (N - 1) * c) / N, (1 + (N - 1) * c) / N,
                      (N - 1) * a, -a, N * a, DIM * (c - cstar * cstar) * aprime,
                      DIM * (a - optimal_a) * cprime * aprime]
            for name, value in zip(columns, values, strict=True):
                columns[name].append(value)
            dataset_forward.append(-teacher["entropy"] + model["logz"] - a * cstar * datum_pair_sum)
            dataset_reverse.append(-model["entropy"] + teacher["logz"] - OMEGA / N * c * datum_pair_sum)
        columns = {name: np.array(values) for name, values in columns.items()}
        checks = ({} if args.skip_independent_checks else independently_check(states, shell, args.seed))

    if max(columns["stationarity_residual"]) > 2e-12:
        raise RuntimeError("Scalar stationarity tolerance failed")
    dataset_forward = np.asarray(dataset_forward)
    dataset_reverse = np.asarray(dataset_reverse)
    checks["dataset_array_forward_weighted_mean_error"] = float(np.max(np.abs(
        dataset_forward @ states["teacher_p"] - columns["forward_kl"])))
    checks["dataset_array_reverse_weighted_mean_error"] = float(np.max(np.abs(
        dataset_reverse @ states["teacher_p"] - columns["reverse_kl"])))
    checks["scalar_stationarity_max"] = float(max(columns["stationarity_residual"]))
    checks["unique_coupling_count"] = DIM
    if max(checks["dataset_array_forward_weighted_mean_error"],
           checks["dataset_array_reverse_weighted_mean_error"]) > 1e-10:
        raise RuntimeError("Saved dataset arrays do not reproduce exact average")

    raw_path = args.output / "bm_map_sweep.npz"
    np.savez_compressed(raw_path, gamma=gamma, old_gamma=old_grid, **columns,
                        canonical_unique_couplings=np.repeat(columns["a"][:, None], DIM, axis=1),
                        dataset_states=states["states"].astype(np.int8),
                        dataset_gauge_features=states["features"].astype(np.int8),
                        dataset_probability=states["teacher_p"],
                        dataset_forward_kl=dataset_forward, dataset_reverse_kl=dataset_reverse,
                        unique_i=states["upper_i"], unique_j=states["upper_j"],
                        teacher_correlation=cstar, teacher_entropy=teacher["entropy"],
                        teacher_logz=teacher["logz"], reverse_optimal_gamma=optimal_gamma,
                        uniform_reverse=teacher["logz"] - N * np.log(2),
                        uniform_forward=N * np.log(2) - teacher["entropy"])
    curve_digest = hashlib.sha256()
    for name, value in [("gamma", gamma), *columns.items()]:
        curve_digest.update(name.encode())
        curve_digest.update(np.asarray(value, dtype="<f8").tobytes())
    summary = {
        "experiment": "Current Appendix G.1 fully-visible binary BM MAP gamma sweep",
        "N": N, "K": 1, "omega": OMEGA,
        "parameter_dimension": DIM, "posterior_inverse_temperature": "infinity (MAP)",
        "posterior_loss": "logZ(J)-sum_unique(J_ij*xi_i*xi_j)+N*gamma/2*sum_unique(J_ij^2)",
        "matrix_prior_equivalent": "N*gamma/4*Tr(J^2); zero diagonal symmetric J",
        "optimization_scope": "All 66 independent offdiagonal couplings; symmetry reduction follows from strict convexity, without a rank constraint",
        "global_optimality_certificate": "Hessian=Cov_J(pair_features)+N*gamma*I >= N*gamma*I; zero full gradient identifies the unique global minimum",
        "method": "Brent scalar root of N*gamma*a=1-c(a), exact binomial shell sums; independent full-state/full-parameter audits",
        "data_average": "Exact average over all 4096 one-sample datasets with their teacher probabilities",
        "units": "Total nats; no nats-per-spin normalization",
        "uncertainty": "Deterministic exact finite sums to numerical precision; no Monte Carlo or dataset sampling uncertainty",
        "coupling_reconstruction": "J_unique[gamma,datum]=canonical_unique_couplings[gamma]*dataset_gauge_features[datum]; symmetric matrix via unique_i/unique_j; diagonal zero",
        "gamma_count": int(gamma.size), "gamma_min": float(gamma[0]), "gamma_max": float(gamma[-1]),
        "old_gamma_count": int(old_grid.size), "old_grid_included_exactly": bool(np.all(np.isin(old_grid, gamma))),
        "reverse_kl_minimum": {"gamma": optimal_gamma, "a": optimal_a,
                               "value": float(columns["reverse_kl"][np.argmin(columns["reverse_kl"])])},
        "uniform_reverse": float(teacher["logz"] - N * np.log(2)),
        "weak_gamma_reverse_limit": float(teacher["entropy"] - np.log(2)),
        "gamma_zero_note": "At gamma=0 there is no finite maximum-likelihood coupling; only a->infinity and a limiting distribution on +/-xi. No finite gamma=0 MAP is plotted.",
        "no_second_descent_certificate": {
            "derivative_with_respect_to_a": "d R/da = DIM*(a-omega*c_teacher/N)*c_prime(a)",
            "facts": "c_prime(a)>0 and da/dgamma<0 for finite gamma>0; hence exactly one minimum and thereafter strict increase to the uniform limit from below",
            "second_descent_detected": False,
            "finite_size_scope": "Exact for this K=1 zero-field full BM and teacher, without thermodynamic or cold-posterior approximation"},
        "independent_checks_performed": not args.skip_independent_checks,
        "checks": checks, "source_sha256": sha256(Path(__file__)),
        "input_grid_sha256": sha256(args.old_grid) if args.old_grid else None,
        "raw_sha256": sha256(raw_path), "curve_arrays_sha256": curve_digest.hexdigest(),
        "runtime_seconds": time.monotonic() - start, "seed_for_checks": args.seed,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "platform": platform.platform()},
        "execution": {"threads": args.threads},
    }
    (args.output / "bm_map_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    curve = {"gamma": gamma.tolist(), "reverse_kl_mean": columns["reverse_kl"].tolist(),
             "forward_kl_mean": columns["forward_kl"].tolist(), "reverse_kl_interval95": None,
             "description": "Exact teacher dataset average at the unique global BM MAP; total nats",
             "source_sha256": summary["source_sha256"], "raw_sha256": summary["raw_sha256"]}
    (args.output / "curve.json").write_text(json.dumps(curve, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: summary[key] for key in ["gamma_count", "reverse_kl_minimum", "uniform_reverse",
                                                 "runtime_seconds", "raw_sha256", "curve_arrays_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
