# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "torch",
#     "numpy",
# ]
# ///
"""Bayesian GAN (Saatchi & Wilson 2017) with a tempered SGHMC posterior.

The conditional posteriors over generator and discriminator weights are
sampled with stochastic gradient Hamiltonian Monte Carlo, modulated by a
single inverse-temperature scalar `eta` that scales the joint log-density
(likelihood + Gaussian prior). `eta = 1` is the Bayesian posterior;
`eta > 1` is cold (peaked, MAP-like); `eta < 1` is warm.

The discriminator is parameterised by raw logits `s = D_logit(x)`. With
`log D = logsigmoid(s)` and `log(1 - D) = logsigmoid(-s)`, the unsupervised
posteriors of Saatchi & Wilson (eq. 1, 2) read

    log p(theta_g | z, theta_d) = w_g · sum_i logsigmoid( s(G(z_i)) )
                                  + log p(theta_g | sigma_p),
    log p(theta_d | z, X, theta_g) = w_d · sum_i logsigmoid( s(x_i) )
                                  + w_g · sum_i logsigmoid(-s(G(z_i)))
                                  + log p(theta_d | sigma_p),

where `w_g, w_d` are mini-batch likelihood weights (Appendix A.1 of the
paper). For a synthetic experiment we set `w_g = w_d = 1`, recovering the
classical-GAN gradient scale and putting the Bayesian flavour in the
explicit Gaussian prior.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace

# Cap intra-op parallelism: tiny MLPs (32 units) thrash on many-core boxes.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

torch.set_num_threads(2)
torch.set_num_interop_threads(2)


# ---------- target: 2-D ring of Gaussians ---------------------------------

def ring_mixture_centers(n_modes: int = 8, radius: float = 2.0) -> torch.Tensor:
    angles = torch.linspace(0, 2 * math.pi, n_modes + 1)[:-1]
    return torch.stack([radius * angles.cos(), radius * angles.sin()], dim=-1)


def sample_ring_mixture(
    n: int,
    n_modes: int = 8,
    radius: float = 2.0,
    sigma: float = 0.05,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    centers = ring_mixture_centers(n_modes, radius)
    idx = torch.randint(0, n_modes, (n,), generator=rng)
    eps = torch.randn(n, 2, generator=rng) * sigma
    return centers[idx] + eps


def ring_mixture_logpdf(x: torch.Tensor, n_modes: int = 8, radius: float = 2.0,
                        sigma: float = 0.05) -> torch.Tensor:
    centers = ring_mixture_centers(n_modes, radius).to(x)
    diff = x.unsqueeze(-2) - centers
    log_comp = -0.5 * (diff.pow(2).sum(-1) / sigma ** 2) \
               - math.log(2 * math.pi * sigma ** 2)
    return torch.logsumexp(log_comp, dim=-1) - math.log(n_modes)


# ---------- networks ------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, dims: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-2], dims[1:-1]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------- log-posterior evaluation -------------------------------------

@dataclass
class BGANConfig:
    z_dim: int = 4
    x_dim: int = 2
    g_hidden: tuple[int, ...] = (32, 32)
    d_hidden: tuple[int, ...] = (32, 32)
    n_real: int = 64             # n_d
    n_fake: int = 64             # n_g
    w_real: float = 1.0          # discriminator real-term likelihood weight
    w_fake: float = 1.0          # generator/discriminator fake-term weight
    prior_sigma: float = 1.0     # Gaussian prior std on weights
    sghmc_lr: float = 5e-5       # eta in Algorithm 1 (SGHMC step size)
    sghmc_alpha: float = 0.1     # SGHMC friction
    inner_M: int = 2             # SGHMC inner iterations
    n_chains_g: int = 4          # J_g (paper: 10 for images, 100 for D=100 synth)
    n_chains_d: int = 1          # J_d
    # target spec: ring-of-Gaussians; n_modes=1, radius=0 gives unimodal
    target_n_modes: int = 8
    target_radius: float = 2.0
    target_sigma: float = 0.05


def make_chains(cfg: BGANConfig, device: str = "cpu",
                seed: int = 0) -> tuple[list[MLP], list[MLP]]:
    g_chains, d_chains = [], []
    for j in range(cfg.n_chains_g):
        torch.manual_seed(seed + j + 1)
        g_chains.append(MLP([cfg.z_dim, *cfg.g_hidden, cfg.x_dim]).to(device))
    for j in range(cfg.n_chains_d):
        torch.manual_seed(seed + 1000 + j)
        d_chains.append(MLP([cfg.x_dim, *cfg.d_hidden, 1]).to(device))
    return g_chains, d_chains


def gaussian_log_prior(net: nn.Module, sigma: float) -> torch.Tensor:
    s = torch.zeros((), device=next(net.parameters()).device)
    for p in net.parameters():
        s = s + p.pow(2).sum() / (2 * sigma ** 2)
    return -s   # missing constant doesn't affect gradients


def log_post_generator(g_net: MLP, d_nets: list[MLP], cfg: BGANConfig,
                       z: torch.Tensor) -> torch.Tensor:
    """log p(theta_g | z, theta_d), averaged over discriminator chains."""
    fake = g_net(z)
    log_d = torch.zeros((), device=z.device)
    for d in d_nets:
        log_d = log_d + F.logsigmoid(d(fake)).sum()
    log_d = log_d * cfg.w_fake / len(d_nets)
    return log_d + gaussian_log_prior(g_net, cfg.prior_sigma)


def log_post_discriminator(d_net: MLP, g_nets: list[MLP], cfg: BGANConfig,
                           x_real: torch.Tensor,
                           z: torch.Tensor) -> torch.Tensor:
    """log p(theta_d | z, X, theta_g), averaged over generator chains."""
    real_logit = d_net(x_real)
    real_term = F.logsigmoid(real_logit).sum() * cfg.w_real
    fake_term = torch.zeros((), device=z.device)
    for g in g_nets:
        fake_logit = d_net(g(z))
        fake_term = fake_term + F.logsigmoid(-fake_logit).sum()
    fake_term = fake_term * cfg.w_fake / len(g_nets)
    return real_term + fake_term + gaussian_log_prior(d_net, cfg.prior_sigma)


# ---------- SGHMC update --------------------------------------------------

class SGHMC:
    """One-chain SGHMC with momentum buffers, targeting p^eta.

    Update (Chen, Fox, Guestrin 2014; eq. 15 with M = I, hat-beta = 0):

        v <- (1 - alpha) v + lr * eta * grad log p + N(0, 2 alpha lr).

    The eta tempering scales the gradient term only, so the stationary
    distribution becomes p(theta)^eta.
    """

    def __init__(self, net: nn.Module, lr: float, alpha: float):
        self.net = net
        self.lr = lr
        self.alpha = alpha
        self.v = [torch.zeros_like(p) for p in net.parameters()]

    def step(self, log_post: torch.Tensor, eta: float) -> None:
        params = list(self.net.parameters())
        grads = torch.autograd.grad(log_post, params, create_graph=False)
        noise_std = math.sqrt(2 * self.alpha * self.lr)
        for p, v, g in zip(params, self.v, grads):
            n = torch.randn_like(p) * noise_std
            v.mul_(1 - self.alpha).add_(self.lr * eta * g).add_(n)
            p.data.add_(v)


# ---------- diagnostics ---------------------------------------------------

def kde_logpdf(samples: torch.Tensor, query: torch.Tensor,
               bandwidth: float = 0.1) -> torch.Tensor:
    diff = query.unsqueeze(1) - samples.unsqueeze(0)         # [Q, S, d]
    sq = diff.pow(2).sum(-1) / (2 * bandwidth ** 2)
    log_kernel = -sq - query.shape[-1] * math.log(
        bandwidth * math.sqrt(2 * math.pi)
    )
    return torch.logsumexp(log_kernel, dim=1) - math.log(samples.shape[0])


def kl_to_target(samples: torch.Tensor, n_grid: int = 200,
                 lim: float = 3.0,
                 bandwidth: float = 0.1,
                 n_modes: int = 8, radius: float = 2.0,
                 sigma: float = 0.05) -> tuple[float, float]:
    """KL(KDE(samples) || target) and reverse, on a 2-D grid."""
    g = torch.linspace(-lim, lim, n_grid)
    grid = torch.stack(torch.meshgrid(g, g, indexing="xy"), dim=-1).reshape(-1, 2)
    finite = torch.isfinite(samples).all(dim=-1)
    samples = samples[finite]
    if samples.shape[0] == 0:
        return float("nan"), float("nan")
    log_q = kde_logpdf(samples, grid, bandwidth)
    log_p = ring_mixture_logpdf(grid, n_modes=n_modes, radius=radius,
                                sigma=sigma)
    cell = (2 * lim / (n_grid - 1)) ** 2
    q = log_q.exp()
    p = log_p.exp()
    q = q / (q.sum() * cell)
    p = p / (p.sum() * cell)
    eps = 1e-12
    kl_qp = (q * (q.add(eps).log() - p.add(eps).log())).sum() * cell
    kl_pq = (p * (p.add(eps).log() - q.add(eps).log())).sum() * cell
    return float(kl_qp), float(kl_pq)


def mode_coverage(samples: torch.Tensor, threshold: float = 0.3,
                  n_modes: int = 8, radius: float = 2.0,
                  min_count: int = 5) -> int:
    centers = ring_mixture_centers(n_modes, radius)
    finite = torch.isfinite(samples).all(dim=-1)
    samples = samples[finite]
    if samples.shape[0] == 0:
        return 0
    d2 = (samples.unsqueeze(1) - centers.unsqueeze(0)).pow(2).sum(-1)
    nearest = d2.argmin(dim=1)
    nearest_d2 = d2.min(dim=1).values
    on_mode = nearest_d2 < threshold ** 2
    counts = torch.bincount(nearest[on_mode], minlength=n_modes)
    return int((counts >= min_count).sum())


# ---------- top-level training loop ---------------------------------------

def train_bayes_gan(cfg: BGANConfig, eta: float, n_outer: int,
                    seed: int = 0, device: str = "cpu",
                    burn_in: int = 1000, sample_every: int = 50,
                    adam_warmup: int = 0,
                    prior_sigma: float | None = None,
                    verbose: bool = False) -> dict[str, np.ndarray]:
    """Bayesian-GAN posterior sampling at temperature eta.

    Optionally pre-train with Adam for `adam_warmup` outer iterations to
    avoid SGHMC blow-up from large initial gradients (paper: "we found it
    helps to speed up the burn-in process by replacing the SGD part of
    this algorithm with Adam for the first few thousand iterations").
    """
    if prior_sigma is not None:
        cfg = replace(cfg, prior_sigma=prior_sigma)
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    g_chains, d_chains = make_chains(cfg, device=device, seed=seed)

    g_adam = [torch.optim.Adam(g.parameters(), lr=2e-4, betas=(0.5, 0.9))
              for g in g_chains]
    d_adam = [torch.optim.Adam(d.parameters(), lr=2e-4, betas=(0.5, 0.9))
              for d in d_chains]
    g_sghmc = [SGHMC(g, cfg.sghmc_lr, cfg.sghmc_alpha) for g in g_chains]
    d_sghmc = [SGHMC(d, cfg.sghmc_lr, cfg.sghmc_alpha) for d in d_chains]

    real_pool = sample_ring_mixture(
        20000, n_modes=cfg.target_n_modes, radius=cfg.target_radius,
        sigma=cfg.target_sigma, rng=rng,
    ).to(device)
    snapshots: list[torch.Tensor] = []

    for it in range(n_outer):
        in_warmup = it < adam_warmup
        for _ in range(cfg.inner_M):
            for j, (g_net, opt_a, opt_h) in enumerate(zip(g_chains, g_adam,
                                                          g_sghmc)):
                z = torch.randn(cfg.n_fake, cfg.z_dim, device=device)
                lp = log_post_generator(g_net, d_chains, cfg, z)
                if in_warmup:
                    opt_a.zero_grad()
                    (-eta * lp).backward()
                    opt_a.step()
                else:
                    opt_h.step(lp, eta)
        for _ in range(cfg.inner_M):
            for d_net, opt_a, opt_h in zip(d_chains, d_adam, d_sghmc):
                idx = torch.randint(0, real_pool.shape[0], (cfg.n_real,))
                x_real = real_pool[idx]
                z = torch.randn(cfg.n_fake, cfg.z_dim, device=device)
                lp = log_post_discriminator(d_net, g_chains, cfg, x_real, z)
                if in_warmup:
                    opt_a.zero_grad()
                    (-eta * lp).backward()
                    opt_a.step()
                else:
                    opt_h.step(lp, eta)

        if it >= burn_in and (it - burn_in) % sample_every == 0:
            with torch.no_grad():
                z = torch.randn(200, cfg.z_dim, device=device)
                fake = torch.cat([g(z) for g in g_chains], dim=0).cpu()
            snapshots.append(fake)
            if verbose:
                kl_qp, _ = kl_to_target(
                    fake, n_modes=cfg.target_n_modes,
                    radius=cfg.target_radius, sigma=cfg.target_sigma,
                )
                cov = mode_coverage(fake, n_modes=cfg.target_n_modes,
                                    radius=cfg.target_radius)
                print(f"  iter {it:5d}  KL={kl_qp:.3f}  "
                      f"modes={cov}/{cfg.target_n_modes}")

    if not snapshots:
        with torch.no_grad():
            z = torch.randn(200, cfg.z_dim, device=device)
            snapshots = [
                torch.cat([g(z) for g in g_chains], dim=0).cpu()
            ]
    samples = torch.cat(snapshots, dim=0)
    kl_qp, kl_pq = kl_to_target(
        samples, n_modes=cfg.target_n_modes, radius=cfg.target_radius,
        sigma=cfg.target_sigma,
    )
    cov = mode_coverage(samples, n_modes=cfg.target_n_modes,
                        radius=cfg.target_radius)
    return {
        "samples": samples.numpy(),
        "kl_qp": np.array(kl_qp),
        "kl_pq": np.array(kl_pq),
        "modes": np.array(cov),
        "eta": np.array(eta),
        "seed": np.array(seed),
    }


if __name__ == "__main__":
    cfg = BGANConfig()
    out = train_bayes_gan(cfg, eta=1.0, n_outer=2000, seed=0,
                          burn_in=1500, sample_every=50,
                          adam_warmup=1000, verbose=True)
    print("final:", out["kl_qp"], out["kl_pq"], out["modes"])
