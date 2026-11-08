# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "torch>=2.3"]
# ///
"""Actual data-minibatch PCD SGD and a paired full-data control; no weight noise.

The loss is -log p(D|W)/N + gamma*tr(W^2)/4. A random datum is
selected independently at each update, and one spherical persistent particle
estimates the negative phase. All updates use pre-step W and x. The sampler
uses a tangential Euler step followed by radial retraction to the sphere.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

CASES = {
    "A": {"c": [1.5, 0.5], "s0": [0.30, 0.25], "gamma": 0.5},
    "B": {"c": [1.8, 0.2], "s0": [0.30, 0.15], "gamma": 0.5},
    "C": {"c": [1.0], "s0": [0.30], "gamma": 0.5},
    "D": {"c": [1.0], "s0": [0.10], "gamma": 1.5},
}


def normalized_dataset(c: list[float], *, device: str = "cpu") -> torch.Tensor:
    """Rows are y/sqrt(N); each row has unit norm and sum yy'/N=diag(c)."""
    if len(c) == 1:
        if c != [1.0]:
            raise ValueError("A single spherical datum requires c=1")
        return torch.ones((1, 1), dtype=torch.float64, device=device)
    if len(c) != 2 or not math.isclose(sum(c), 2.0):
        raise ValueError("This experiment implements physical K=1 or K=2 data")
    a, b = math.sqrt(c[0] / 2), math.sqrt(c[1] / 2)
    return torch.tensor([[a, b], [a, -b]], dtype=torch.float64, device=device)


def selftest() -> dict:
    """Independent finite-N log-likelihood differentiation and update invariants."""
    torch.set_num_threads(1)
    dtype = torch.float64
    tests = {}
    for name, case in CASES.items():
        y = normalized_dataset(case["c"])
        tests[f"{name}_norm_error"] = float((y.square().sum(1) - 1).abs().max())
        tests[f"{name}_covariance_error"] = float(
            (y.T @ y - torch.diag(torch.tensor(case["c"], dtype=dtype))).abs().max()
        )
    # N=2 spherical partition function by a 16384-point periodic quadrature.
    n, k, gamma = 2, 2, 0.5
    y = normalized_dataset(CASES["A"]["c"]) * math.sqrt(n)
    angle = torch.arange(16384, dtype=dtype) * (2 * math.pi / 16384)
    states = math.sqrt(n) * torch.stack((angle.cos(), angle.sin()), dim=1)
    w = torch.tensor([[0.3, -0.2], [-0.2, -0.1]], dtype=dtype)
    h = torch.tensor([[0.7, 0.4], [0.4, -0.3]], dtype=dtype)

    def objective(matrix):
        data_energy = 0.5 * torch.einsum("ki,ij,kj->", y, matrix, y)
        logits = 0.5 * torch.einsum("ki,ij,kj->k", states, matrix, states)
        logz = torch.logsumexp(logits, 0) - math.log(len(states))
        return (data_energy - k * logz) / n - gamma * matrix.square().sum() / 4

    logits = 0.5 * torch.einsum("ki,ij,kj->k", states, w, states)
    probs = logits.softmax(0)
    negative = torch.einsum("k,ki,kj->ij", probs, states, states)
    gradients = k / (2 * n) * (y[:, :, None] * y[:, None, :] - negative) - gamma * w / 2
    derivative = (gradients.mean(0) * h).sum()
    eps = 1e-5
    finite_difference = (objective(w + eps * h) - objective(w - eps * h)) / (2 * eps)
    tests["likelihood_gradient_absolute_error"] = float((derivative - finite_difference).abs())
    # Independent noise support: for K=2, only off-diagonal data gradient fluctuates.
    fluctuation = gradients - gradients.mean(0)
    tests["data_noise_diagonal_error"] = float(torch.diagonal(fluctuation, dim1=1, dim2=2).abs().max())
    tests["data_noise_variance_error"] = abs(float(fluctuation[:, 0, 1].square().mean()) - 1.5 * 0.5 / 4)
    # An arbitrary pre-step spherical particle preserves tr W=0 exactly in SGD.
    x = torch.tensor([math.sqrt(0.6 * n), math.sqrt(0.4 * n)], dtype=dtype)
    sample_gradient = k / (2 * n) * (torch.outer(y[0], y[0]) - torch.outer(x, x))
    tests["trace_gradient_error"] = float(sample_gradient.trace().abs())
    # K=1 estimator equals the full data gradient for every draw.
    one = normalized_dataset([1.0])
    tests["K1_minibatch_fullbatch_error"] = float((one.T @ one - torch.eye(1, dtype=dtype)).abs().max())
    assert max(tests.values()) < 1e-8, tests
    return {"passed": True, "tests": tests, "torch": torch.__version__}


@torch.no_grad()
def simulate(args) -> None:
    case = CASES[args.case]
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU requested but unavailable")
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float64)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    n, replicas, dt = args.N, args.replicas, args.dt
    k, gamma, nu = len(case["c"]), case["gamma"], args.nu
    steps = round(args.Tmax / dt)
    save_every = round(args.save_dt / dt)
    if not math.isclose(steps * dt, args.Tmax) or save_every < 1 or not math.isclose(save_every * dt, args.save_dt):
        raise ValueError("Tmax and save_dt must be integer multiples of dt")
    if not 0 < dt * gamma / 2 < 1:
        raise ValueError("Unstable weight decay step")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    data_generator = torch.Generator(device=device).manual_seed(args.seed + 1000003)
    data = normalized_dataset(case["c"], device=device)
    c = torch.tensor(case["c"], device=device)
    s0 = torch.tensor(case["s0"], device=device)
    x0 = torch.randn((replicas, n), generator=generator, device=device)
    x0[:, :k] = math.sqrt(n) * s0
    x0[:, k:] *= (n * (1 - s0.square().sum()) / x0[:, k:].square().sum(1, keepdim=True)).sqrt()
    # Modes 0 and 1 share initial particles and each subsequent Brownian draw.
    # Mode 0 randomly samples data; mode 1 evaluates the full data covariance.
    x = torch.cat((x0, x0.clone()), dim=0)
    w = torch.zeros((2 * replicas, n, n), device=device)
    diagonal = torch.arange(k, device=device)
    counts = torch.zeros((replicas, k), dtype=torch.int64, device=device)
    times, overlaps, kappas, traces, norms, weight_blocks = [], [], [], [], [], []
    start = time.monotonic()

    def record(step, wx):
        times.append(step * dt)
        overlaps.append((x[:, :k] / math.sqrt(n)).cpu().numpy().copy())
        energy = (x * wx).sum(1) / n
        kappas.append((nu * (energy + (n - 1) / n)).cpu().numpy().copy())
        traces.append(w.diagonal(dim1=1, dim2=2).sum(1).cpu().numpy().copy())
        norms.append((x.square().sum(1) / n).cpu().numpy().copy())
        weight_blocks.append(w[:, :k, :k].cpu().numpy().copy())

    for step in range(steps + 1):
        wx = torch.bmm(w, x[:, :, None]).squeeze(2)
        if step % save_every == 0:
            record(step, wx)
        if step == steps:
            break
        # Explicit unbiased DATA minibatch gradient, one actual datum per update.
        indices = torch.randint(k, (replicas,), generator=data_generator, device=device)
        counts.scatter_add_(1, indices[:, None], torch.ones((replicas, 1), dtype=torch.int64, device=device))
        selected = data[indices]
        # Save old-W sampler drift before changing W.
        radial_energy = (x * wx).sum(1, keepdim=True) / n
        tangent_drift = nu * (wx - x * radial_energy)
        raw_noise = torch.randn((replicas, n), generator=generator, device=device)
        raw_noise = torch.cat((raw_noise, raw_noise), dim=0)
        tangent_noise = raw_noise - x * (x * raw_noise).sum(1, keepdim=True) / n
        # SGD on the regularized log likelihood divided by N. No GOE increment.
        w.mul_(1 - gamma * dt / 2)
        w.baddbmm_(x[:, :, None], x[:, None, :], beta=1, alpha=-dt * k / (2 * n))
        w[:replicas, :k, :k] += dt * k / 2 * selected[:, :, None] * selected[:, None, :]
        w[replicas:, diagonal, diagonal] += dt * c / 2
        x.add_(tangent_drift, alpha=dt)
        x.add_(tangent_noise, alpha=math.sqrt(2 * nu * dt))
        x.mul_((n / x.square().sum(1, keepdim=True)).sqrt())
        if (step + 1) % max(1, round(5 / dt)) == 0:
            if device == "cuda":
                torch.cuda.synchronize()
            print(json.dumps({"case": args.case, "N": n, "dt": dt, "t": (step + 1) * dt, "elapsed_s": time.monotonic() - start}), flush=True)

    if device == "cuda":
        torch.cuda.synchronize()
    arrays = {
        "t": np.asarray(times),
        "s": np.stack(overlaps).reshape(-1, 2, replicas, k).transpose(1, 2, 0, 3),
        "kappa": np.stack(kappas).reshape(-1, 2, replicas).transpose(1, 2, 0),
        "trace": np.stack(traces).reshape(-1, 2, replicas).transpose(1, 2, 0),
        "norm_squared_over_N": np.stack(norms).reshape(-1, 2, replicas).transpose(1, 2, 0),
        "weight_signal_block": np.stack(weight_blocks).reshape(-1, 2, replicas, k, k).transpose(1, 2, 0, 3, 4),
        "data_counts": counts.cpu().numpy(),
        "normalized_data": data.cpu().numpy(),
    }
    assert all(np.isfinite(array).all() for array in arrays.values())
    assert np.max(np.abs(arrays["norm_squared_over_N"] - 1)) < 1e-12
    assert np.max(np.abs(arrays["trace"])) < 1e-9
    if k == 1:
        assert np.max(np.abs(arrays["s"][0] - arrays["s"][1])) < 1e-12
    metadata = {
        **vars(args), **case, "K": k, "eta": "infinity", "beta": 1.0,
        "modes": ["data_minibatch_SGD", "full_data_PCD_control"],
        "parameter_noise": "none", "W_initial": "zero",
        "sampler": "projected Euler with sphere retraction; old W and x",
        "weight_update": "explicit Euler ascent on logposterior/N",
        "replicate_rng": "independent draws within batched pseudorandom stream; shared across paired controls",
        "elapsed_s": time.monotonic() - start, "torch": torch.__version__,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "max_norm_error": float(np.max(np.abs(arrays["norm_squared_over_N"] - 1))),
        "max_trace_error": float(np.max(np.abs(arrays["trace"]))),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    np.savez_compressed(out, **arrays, metadata=json.dumps(metadata))
    out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"saved": str(out), "elapsed_s": metadata["elapsed_s"], "max_trace_error": metadata["max_trace_error"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, default="A")
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--Tmax", type=float, default=30)
    parser.add_argument("--save-dt", type=float, default=0.1)
    parser.add_argument("--nu", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", default="results/pilot.npz")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        print(json.dumps(selftest(), indent=2))
    else:
        simulate(args)


if __name__ == "__main__":
    main()
