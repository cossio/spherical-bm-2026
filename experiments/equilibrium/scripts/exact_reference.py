# /// script
# requires-python = ">=3.14"
# dependencies = ["scipy>=1.16,<2"]
# ///
"""Independent exact N=2 posterior benchmark by one-dimensional quadrature.

Run with uv; this script does not import the exchange/Bingham implementation.
The default is the K=2, eta=4, gamma=0.25 coalesced-regime control.
"""

import argparse
import json
import math
from pathlib import Path

from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import i0e, ive


def reference(eta: float, gamma: float, c1: float, k: int) -> dict:
    """Integrate the exact traceless 2x2 posterior in polar coordinates."""
    c2 = k - c1
    if eta <= 0 or gamma <= 0 or not 0 <= c2 <= c1:
        raise ValueError("Require eta,gamma > 0 and 0 <= c2 <= c1 with c1+c2=K")
    tilt = eta * (c1 - c2)
    replicas = k * eta

    def log_i0(x: float) -> float:
        return abs(x) + math.log(float(i0e(x)))

    def ratio(order: int, x: float) -> float:
        return float(ive(order, x) / i0e(x))

    def log_weight(r: float) -> float:
        if r <= 0:
            return -math.inf
        return (
            math.log(r)
            - gamma * eta * r * r
            + log_i0(tilt * r)
            - replicas * log_i0(r)
        )

    upper = max(20 / math.sqrt(gamma * eta), (tilt + replicas) / (gamma * eta))
    mode = minimize_scalar(
        lambda r: -log_weight(r), bounds=(1e-12, upper), method="bounded"
    )
    if not mode.success:
        raise RuntimeError("Could not locate a scaling point for quadrature")
    shift = log_weight(float(mode.x))

    def weight(r: float) -> float:
        return math.exp(log_weight(r) - shift)

    normalizer, normalizer_error = quad(
        weight, 0, math.inf, epsabs=2e-12, epsrel=2e-12, limit=400
    )

    functions = {
        "lambda1": lambda r: r,
        "lambda2": lambda r: -r,
        "gap12": lambda r: 2 * r,
        "radius_squared": lambda r: r * r,
        "w11": lambda r: r * ratio(1, tilt * r),
        "w22": lambda r: -r * ratio(1, tilt * r),
        "w11_squared_traceless": lambda r: r * r * (1 + ratio(2, tilt * r)) / 2,
        "w12_squared": lambda r: r * r * (1 - ratio(2, tilt * r)) / 2,
        "alignment_top1_data1": lambda r: (1 + ratio(1, tilt * r)) / 2,
        "alignment_top2_data2": lambda r: (1 + ratio(1, tilt * r)) / 2,
        "alignment_subspace2_data1": lambda r: 1.0,
        "alignment_subspace2_data2": lambda r: 1.0,
        "condensation_top1": lambda r: (1 + ratio(1, r)) / 2,
        "condensation_top2_sum": lambda r: 1.0,
        "predictive_data1": lambda r: (1 + ratio(1, r) * ratio(1, tilt * r)) / 2,
        "log_z_normalized_traceless": log_i0,
    }
    means = {}
    errors = {}
    for name, function in functions.items():
        value, error = quad(
            lambda r: weight(r) * function(r),
            0,
            math.inf,
            epsabs=2e-12,
            epsrel=2e-12,
            limit=400,
        )
        means[name] = value / normalizer
        errors[name] = (error + abs(means[name]) * normalizer_error) / normalizer

    # This is an exact finite-N posterior integration-by-parts identity.
    score_residual = c1 - k * means["predictive_data1"] - gamma * means["w11"]
    if abs(score_residual) > 1e-9:
        raise AssertionError(f"Exact posterior score identity failed: {score_residual}")
    trace_shift_variance = 1 / (2 * gamma * eta)
    return {
        "model": "finite_N_spherical_Boltzmann_machine",
        "method": "independent_N2_radial_quadrature",
        "parameters": {"N": 2, "K": k, "eta": eta, "gamma": gamma, "c": [c1, c2]},
        "radial_density": "r*exp(-gamma*eta*r^2)*I0(eta*(c1-c2)*r)/I0(r)^(K*eta)",
        "trace_shift": {"mean": 0.0, "variance": trace_shift_variance},
        "means": means,
        "quadrature_error_estimates": errors,
        "checks": {"posterior_score_residual_data1": score_residual},
        "notes": [
            "lambda1/2 include the independent trace shift in expectation; its mean is zero",
            "w11_squared_traceless omits trace_shift.variance; add it for full-matrix W11^2",
            "I0 is the modified Bessel function, with normalized spherical base measure",
            "Quadrature estimates are numerical integration errors, not MCMC uncertainty",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eta", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--c1", type=float, default=1.25)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = reference(args.eta, args.gamma, args.c1, args.k)

    # K=1, eta=1 has an exactly GOE spectral posterior at every N.
    goe = reference(1.0, 0.7, 1.0, 1)
    known_mean = math.sqrt(math.pi) / (2 * math.sqrt(0.7))
    known_second = 1 / 0.7
    residuals = {
        "K1_eta1_goe_radius_mean": goe["means"]["lambda1"] - known_mean,
        "K1_eta1_goe_radius_second_moment": goe["means"]["radius_squared"] - known_second,
    }
    if max(map(abs, residuals.values())) > 1e-10:
        raise AssertionError(residuals)
    result["checks"].update(residuals)
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
