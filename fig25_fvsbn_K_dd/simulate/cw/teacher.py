# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy", "torch"]
# ///
"""Rank-1 Hopfield / Curie-Weiss Ising teacher.

Energy (self-coupling subtracted):

    E(s) = -(1/2N) (s . xi)^2 + 1/2,    s_i in {-1,+1}.

The gauge s_i -> xi_i s_i maps this to the ferromagnetic Curie-Weiss
energy  -(N/2) m^2  in the gauged variables, so the partition function
depends only on the magnetization histogram.

Local copy of the rank-1 subset of
``11_hopfield_bm/scripts/teacher.py`` so this experiment folder is
self-contained; if the parent layout changes again we do not need to
re-point any imports.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def hopfield_energy(s: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """Energy E(s) = -(1/2N) (s.xi)^2 + 1/2 for rank-1 pattern xi."""
    N = xi.shape[0]
    proj = s @ xi
    return -0.5 * proj ** 2 / N + 0.5


def hopfield_log_Z(beta: float, N: int) -> float:
    """Exact log Z for the rank-1 Hopfield teacher in N spins."""
    ks = np.arange(N + 1)
    log_binom = (
        math.lgamma(N + 1)
        - np.array([math.lgamma(k + 1) + math.lgamma(N - k + 1) for k in ks])
    )
    log_terms = log_binom + beta * (N - 2 * ks) ** 2 / (2.0 * N)
    m = log_terms.max()
    logZ = m + math.log(np.exp(log_terms - m).sum())
    return float(logZ - 0.5 * beta)


def hopfield_log_prob(
    s: torch.Tensor, beta: float, xi: torch.Tensor, log_Z: float
) -> torch.Tensor:
    """log p_teacher(s) = -beta E(s) - log Z."""
    return -beta * hopfield_energy(s, xi) - log_Z


def sample_teacher(
    beta: float,
    xi: torch.Tensor,
    K: int,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Exact K samples from the rank-1 Hopfield teacher via the gauge."""
    if rng is None:
        rng = np.random.default_rng()
    N = xi.shape[0]

    ks = np.arange(N + 1)
    log_binom = (
        math.lgamma(N + 1)
        - np.array([math.lgamma(k + 1) + math.lgamma(N - k + 1) for k in ks])
    )
    log_p = log_binom + beta * (N - 2 * ks) ** 2 / (2.0 * N)
    log_p -= log_p.max()
    p = np.exp(log_p)
    p /= p.sum()

    xi_np = xi.detach().cpu().numpy().astype(np.int8)
    samples = np.empty((K, N), dtype=np.int8)
    k_draws = rng.choice(N + 1, size=K, p=p)
    for idx, k in enumerate(k_draws):
        tilde = np.ones(N, dtype=np.int8)
        flip = rng.choice(N, size=int(k), replace=False)
        tilde[flip] = -1
        samples[idx] = tilde * xi_np
    return torch.tensor(samples, dtype=torch.float32)
