# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy==2.5.4"]
# ///
"""Finite-state RBM numerical routines and reproducible training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "spin_states",
    "log2cosh",
    "logsumexp",
    "Teacher",
    "log_partition",
    "grad_log_partition",
    "neg_phase_exact",
    "log_prob_visible",
    "kl",
    "overlaps",
    "spectrum",
    "data_spectrum",
    "sbm_thresholds",
    "beta_star_cov",
    "subspace_overlap",
    "loglik_and_grad",
    "logpost_and_grad",
    "map_estimate",
    "hmc_posterior",
    "gibbs_sweeps",
    "neg_phase_chains",
]


# --------------------------------------------------------------------------
# elementary pieces
# --------------------------------------------------------------------------


def spin_states(n: int) -> np.ndarray:
    """All ``2^n`` configurations in ``{-1,+1}^n``, shape ``(2^n, n)``."""
    idx = np.arange(1 << n, dtype=np.int64)
    bits = (idx[:, None] >> np.arange(n, dtype=np.int64)[None, :]) & 1
    return 2.0 * bits - 1.0


def log2cosh(x: np.ndarray) -> np.ndarray:
    """``log(2 cosh x)``, overflow-free (single ufunc pass)."""
    return np.logaddexp(x, -x)


def logsumexp(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def _mm_right(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """``X (..., n, k) @ Y (k, m)`` as a *single* GEMM.

    ``np.matmul`` on a stacked operand issues one BLAS call per matrix in the
    batch; with matrices this small the dispatch overhead dominates.  Folding
    the batch into the row index turns thousands of tiny GEMMs into one large
    one, which is the difference between an hour and a few minutes over the
    whole HMC grid.
    """
    sh = X.shape
    return (X.reshape(-1, sh[-1]) @ Y).reshape(*sh[:-1], Y.shape[-1])


def _mm_left(D: np.ndarray, X: np.ndarray) -> np.ndarray:
    """``D (k, n) @ X (..., n, m)`` as a single GEMM, returning ``(..., k, m)``."""
    if X.ndim == 2:
        return D @ X
    n, m = X.shape[-2], X.shape[-1]
    lead = X.shape[:-2]
    # (..., n, m) -> (n, ... * m) so the contraction is one big GEMM
    Xt = np.moveaxis(X, -2, 0).reshape(n, -1)
    return np.moveaxis((D @ Xt).reshape(D.shape[0], *lead, m), 0, -2)


# --------------------------------------------------------------------------
# partition function and exact negative phase  (cost O(2^M N))
# --------------------------------------------------------------------------


def _hidden_terms(W, a, b, beta, H):
    """Shared inner block: ``arg = beta (a + W h)`` and the per-``h`` log-weights."""
    WH = _mm_right(W, H.T)  # (..., N, S)
    arg = beta * (np.asarray(a)[:, None] + WH)  # (..., N, S)
    log_s = log2cosh(arg).sum(-2) + beta * (H @ np.asarray(b))  # (..., S)
    return arg, log_s


def log_partition(W, a, b, beta, H) -> np.ndarray:
    """``log Z_beta`` by exact summation over the ``2^M`` hidden states."""
    _, log_s = _hidden_terms(W, a, b, beta, H)
    return logsumexp(log_s, axis=-1)


def logZ_and_grad(W, a, b, beta, H):
    """``log Z`` and ``dlogZ/dW`` in a single pass over the hidden states.

    Computing them together halves the cost of the HMC inner loop, which is the
    hot path of the whole study.
    """
    arg, log_s = _hidden_terms(W, a, b, beta, H)
    lz = logsumexp(log_s, axis=-1)
    w = _softmax(log_s, axis=-1)  # (..., S)
    tw = np.tanh(arg) * w[..., None, :]  # (..., N, S)
    return lz, beta * _mm_right(tw, H)  # (..., N, M)


def grad_log_partition(W, a, b, beta, H) -> np.ndarray:
    """``dlogZ/dW``, shape ``(..., N, M)``.  Equals ``beta <v h^T>_model``."""
    return logZ_and_grad(W, a, b, beta, H)[1]


def neg_phase_exact(W, a, b, beta, H) -> np.ndarray:
    """Exact model moment ``<v h^T>``, shape ``(..., N, M)``."""
    return grad_log_partition(W, a, b, beta, H) / beta


# --------------------------------------------------------------------------
# full visible distribution  (cost O(2^N M))
# --------------------------------------------------------------------------


def log_prob_visible(W, a, b, beta, V, chunk: int | None = None) -> np.ndarray:
    """Normalised ``log P_beta(v)`` on every enumerated visible state.

    Returns shape ``(..., 2^N)``.  ``chunk`` splits the leading batch axis to
    bound peak memory (the intermediate is ``batch x 2^N x M``).
    """
    W = np.asarray(W, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if W.ndim == 2 or chunk is None:
        A = _mm_left(V, W)
        logits = beta * (V @ a) + log2cosh(beta * (b + A)).sum(-1)
        return logits - logsumexp(logits, axis=-1, keepdims=True)

    flat = W.reshape(-1, W.shape[-2], W.shape[-1])
    out = np.empty((flat.shape[0], V.shape[0]), dtype=np.float64)
    for i in range(0, flat.shape[0], chunk):
        out[i : i + chunk] = log_prob_visible(flat[i : i + chunk], a, b, beta, V)
    return out.reshape(W.shape[:-2] + (V.shape[0],))


def kl(logp: np.ndarray, logq: np.ndarray) -> np.ndarray:
    """``KL(P||Q)`` in nats; both are normalised log-pmfs over the same support."""
    p = np.exp(logp)
    return np.sum(p * (logp - logq), axis=-1)


def overlaps(logp: np.ndarray, mags: np.ndarray) -> np.ndarray:
    """``q_r = E_P[m_r^2]`` with ``m_r = xi_r.v / N``; ``mags`` is ``(2^N, R)``.

    This is the *generation* diagnostic: whether generated samples express mode
    ``r``, the RBM counterpart of the manuscript's sample condensation ``h_k^2``.
    """
    p = np.exp(logp)
    return np.einsum("...v,vr->...r", p, mags**2)


# --------------------------------------------------------------------------
# gauge-invariant weight diagnostics
# --------------------------------------------------------------------------


def spectrum(W) -> np.ndarray:
    """Non-zero eigenvalues of ``J_eff = W W^T / N``, descending.

    Equals ``s_mu^2 / N`` with ``s_mu`` the singular values of ``W``.
    """
    W = np.asarray(W, dtype=np.float64)
    s = np.linalg.svd(W, compute_uv=False)
    return s**2 / W.shape[-2]


def data_spectrum(data, N: int) -> np.ndarray:
    """Eigenvalues ``c_k`` of the SBM data covariance ``C = (1/N) sum_k x^k x^{k T}``.

    For ``+-1`` spins ``||x||^2 = N`` exactly, so the hypercube is inscribed in
    the sphere the manuscript's theory lives on and ``tr C = K`` identically --
    the SBM normalisation transfers with no rescaling.  Returns the non-zero
    eigenvalues in descending order.
    """
    C = np.asarray(data).T @ np.asarray(data) / N
    ev = np.linalg.eigvalsh(C)[::-1]
    return ev[ev > 1e-10]


def sbm_thresholds(ck, gamma_paper, eta, N: int) -> dict:
    """Manuscript thresholds evaluated on a measured data spectrum ``c_k``.

    All gammas are in the manuscript convention ``Pi(W) prop exp(-N gamma tr W^2/4)``;
    ``gamma_code = N * gamma_paper / 2`` is also returned for the local convention
    ``exp(-gamma_code ||W||_F^2 / 2)``.

    * alignment / BBP (mode ``k`` detaches):  ``gamma < eta c_k^2``
    * condensation (``h^2 > 0``, ``d`` coalesced modes): ``gamma < (1/d) sum_{k<=d} c_k``,
      so the global boundary -- and the predicted reverse-KL spike -- is ``gamma = c_1``
    * unaligned edge branch condenses iff ``gamma eta < 1``
    """
    ck = np.asarray(ck, dtype=np.float64)
    g_align = eta * ck**2
    g_cond = np.array([ck[:d].sum() / d for d in range(1, len(ck) + 1)])
    return {
        "c_k": ck.tolist(),
        "trC_should_equal_K": float(ck.sum()),
        "gamma_align_paper": g_align.tolist(),
        "gamma_align_code": (N * g_align / 2).tolist(),
        "gamma_cond_paper": g_cond.tolist(),
        "gamma_cond_code": (N * g_cond / 2).tolist(),
        "gamma_edge_cond_paper": 1.0 / eta,
        "aligned_modes_at_gamma": int(np.sum(g_align > gamma_paper)),
        "condensed_at_gamma": bool(gamma_paper < g_cond[0]),
    }


def beta_star_cov(W, C, gamma_paper: float, mu_prime: float = 0.0) -> float:
    """Manuscript Eq. (192) prediction for the tuned sampling temperature.

    ``beta* = 1 + gamma * sum_k lam_k^2 c_k^2 / sum_k lam_k (lam_k - mu') c_k^4``,
    with ``lam_k`` the trained-coupling eigenvalues and ``c_k`` the data-covariance
    eigenvalues *in a common eigenbasis*.  Here the common basis is taken to be the
    eigenbasis of ``J_eff = W W^T`` and ``c_k`` is the Rayleigh quotient of ``C``
    along each of its eigenvectors.  ``mu' = 0`` is the per-site-normalised
    (non-spherical) choice, which is the right one for a ``+-1`` RBM.
    """
    W = np.asarray(W, dtype=np.float64)
    lam, vec = np.linalg.eigh(W @ W.T)
    order = np.argsort(lam)[::-1]
    lam, vec = lam[order], vec[:, order]
    keep = lam > 1e-10
    lam, vec = lam[keep], vec[:, keep]
    ck = np.einsum("nk,nm,mk->k", vec, np.asarray(C), vec)
    den = float(np.sum(lam * (lam - mu_prime) * ck**4))
    if abs(den) < 1e-300:
        return float("nan")
    return 1.0 + gamma_paper * float(np.sum(lam**2 * ck**2)) / den


def subspace_overlap(W, xis) -> np.ndarray:
    """``u_r^2 = ||Proj_{col(W)} xi_r/sqrt(N)||^2`` for each planted pattern.

    ``xis`` is ``(N, R)`` with ``+-1`` entries.  ``u_r^2 = 1`` means the pattern
    lies entirely in the weight column space, ``0`` that it is absent.  This is
    the *representation* diagnostic, the RBM counterpart of the manuscript's
    weight--data alignment ``u_k^2``; ``overlaps`` is the generation one.
    """
    W = np.asarray(W, dtype=np.float64)
    N = W.shape[-2]
    xn = np.asarray(xis, dtype=np.float64) / np.sqrt(N)  # (N, R)
    Q, _ = np.linalg.qr(W)  # (..., N, M)
    proj = np.einsum("...nm,nr->...mr", Q, xn)  # (..., M, R)
    return np.sum(proj**2, axis=-2)


# --------------------------------------------------------------------------
# teacher
# --------------------------------------------------------------------------


@dataclass
class Teacher:
    """Planted two-mode teacher ``W* = [A_1 xi_1, A_2 xi_2]`` with zero fields."""

    W: np.ndarray
    xis: np.ndarray
    amps: np.ndarray
    N: int
    beta: float = 1.0

    @classmethod
    def hadamard(cls, N: int, amps, cols=(1, 2), beta: float = 1.0) -> "Teacher":
        """Orthogonal ``+-1`` patterns taken as columns of a Sylvester Hadamard matrix."""
        k = int(round(np.log2(N)))
        if 1 << k != N:
            raise ValueError("Hadamard construction needs N a power of two")
        Hd = np.ones((1, 1))
        for _ in range(k):
            Hd = np.block([[Hd, Hd], [Hd, -Hd]])
        xis = Hd[:, list(cols)].astype(np.float64)  # (N, R)
        amps = np.asarray(amps, dtype=np.float64)
        return cls(W=xis * amps[None, :], xis=xis, amps=amps, N=N, beta=beta)

    @classmethod
    def from_lambdas(cls, N: int, lambdas, cols=(1, 2), beta: float = 1.0) -> "Teacher":
        """Parameterise by the eigenvalues ``lambda_r = A_r^2 N`` of ``W W^T``.

        ``lambda_r`` is the diagnostic that maps onto the manuscript's spectrum
        (it is ``N`` times the ``r``-th eigenvalue of ``J_eff``), but for
        *Bernoulli* hidden units it is **not** the control parameter for mode
        activation.  With ``h_mu = +-1`` a single hidden unit injects a field of
        magnitude ``A_r`` per visible spin, so the large-``N`` saddle point of
        ``(1/N) log 2 cosh(A_r N m) + s(m)`` gives ``m_r^* = tanh(A_r)``: the
        activation threshold sits at ``A_r = O(1)``, i.e. ``lambda_r = O(N)``,
        and ``lambda_r = 1`` is deep in the paramagnetic phase.

        The naive ``lambda_r = 1`` BBP threshold applies to the *Gaussian*-hidden
        RBM, which integrates out exactly to a rank-``M`` Ising model with
        ``J = W W^T`` (see ``05_gauss_hidden.py``).  For the Bernoulli model the
        amplitudes are calibrated empirically against the exactly enumerated
        ``q_r`` in ``01_teacher.py``.
        """
        lambdas = np.asarray(lambdas, dtype=np.float64)
        return cls.hadamard(N, np.sqrt(lambdas / N), cols=cols, beta=beta)

    def sample(self, K: int, logp: np.ndarray, V: np.ndarray, rng) -> np.ndarray:
        """``K`` i.i.d. draws from the exact teacher visible distribution."""
        idx = rng.choice(logp.shape[-1], size=K, p=np.exp(logp))
        return V[idx].copy()


# --------------------------------------------------------------------------
# likelihood, posterior, MAP
# --------------------------------------------------------------------------


def loglik_and_grad(W, data, a, b, beta, H):
    """Exact ``sum_k log P_W(v^k)`` and its ``W``-gradient.

    The positive phase is closed form; the negative phase uses
    ``grad_log_partition``, i.e. the *exact* model moment.
    """
    W = np.asarray(W, dtype=np.float64)
    A = beta * (b + _mm_left(data, W))  # (..., K, M)
    ll = beta * (data @ a).sum() + log2cosh(A).sum((-1, -2))
    g = beta * _mm_left(data.T, np.tanh(A))  # (..., N, M)
    K = data.shape[0]
    lz, glz = logZ_and_grad(W, a, b, beta, H)
    return ll - K * lz, g - K * glz


def logpost_and_grad(W, data, a, b, beta, H, gamma, eta=1.0):
    """Tempered posterior ``Pi_{eta,gamma}(W|D) prop [prod_k P_W(v^k) e^{-gamma||W||^2/2}]^eta``.

    ``gamma`` and ``eta`` may be scalars or arrays broadcastable to the leading
    batch shape of ``W``.  Per-chain values let a whole ``(gamma, eta)`` grid run
    inside a single vectorised HMC sweep, which is what makes the multi-seed
    grid affordable: the chains are independent, so only the scalars differ.
    """
    ll, g = loglik_and_grad(W, data, a, b, beta, H)
    W = np.asarray(W, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    lp = eta * (ll - 0.5 * gamma * np.sum(W**2, axis=(-1, -2)))
    gg = gamma[..., None, None] if gamma.ndim else gamma
    ee = eta[..., None, None] if eta.ndim else eta
    return lp, ee * (g - gg * W)


def map_estimate(
    data,
    a,
    b,
    beta,
    H,
    gammas,
    N: int,
    M: int,
    rng,
    n_restarts: int = 64,
    n_steps: int = 6000,
    lr: float = 0.05,
    init_scale: float = 0.1,
    W_extra=None,
):
    """MAP (the ``eta -> inf`` limit) by Adam with random restarts.

    ``W_extra`` of shape ``(G, n_extra, N, M)`` supplies additional starting
    points.  Spreading random restarts over many scales is not by itself enough
    at small ``gamma``: it narrows the shortfall against cold HMC from 16 nats to
    2.6, but does not close it.  Seeding from cold-HMC draws, which already sit
    in the dominant basin, and polishing them does -- the returned optimum is
    then at least as good as the best sample handed in.

    ``gammas`` is an array of shape ``(G,)``; the whole ``gamma`` grid and all
    restarts are optimised in one batched sweep.  Returns ``(W_map, logpost)``
    with shapes ``(G, N, M)`` and ``(G,)``, keeping the best restart per
    ``gamma``.

    The objective is badly non-convex at small ``gamma``, where the likelihood
    saturates and the optimum sits at large ``||W||``.  Restarts drawn from a
    single small scale get trapped: against very cold HMC (which concentrates
    near the true optimum) an 8-restart, scale-0.1 search fell short by 16 nats
    of log-posterior at ``gamma = 0.01``.  Restarts are therefore spread over
    several orders of magnitude in initial scale, which closes that gap.
    """
    gammas = np.atleast_1d(np.asarray(gammas, dtype=np.float64))
    G = gammas.shape[0]
    gg = gammas[:, None]                                   # (G, 1) over restarts
    scales = init_scale * np.logspace(0, 2.5, n_restarts)   # 0.1 .. ~30
    W = rng.normal(size=(G, n_restarts, N, M)) * scales[None, :, None, None]
    if W_extra is not None:
        W = np.concatenate([W, np.asarray(W_extra, dtype=np.float64)], axis=1)
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, n_steps + 1):
        _, g = logpost_and_grad(W, data, a, b, beta, H, gg, eta=1.0)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        W = W + lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
    lp, _ = logpost_and_grad(W, data, a, b, beta, H, gg, eta=1.0)
    best = np.argmax(lp, axis=1)                           # (G,)
    return W[np.arange(G), best], lp[np.arange(G), best]


def map_estimate_seeded(
    data, a, b, beta, H, gammas, N, M, rng, *, n_restarts=48, n_steps=6000,
    n_seed=8, eta_cold=1000.0, n_chains=32, n_warmup=2000, n_sample=3000,
):
    """MAP with cold-HMC seeding: the reliable version.

    Runs a very cold posterior (``eta_cold``), which concentrates in the
    dominant basin, keeps the ``n_seed`` highest-log-posterior draws per
    ``gamma``, and polishes them alongside multi-scale random restarts.  This is
    what the MAP curve should be computed with; plain multi-scale restarts alone
    still fall short at small ``gamma``.
    """
    gammas = np.atleast_1d(np.asarray(gammas, dtype=np.float64))
    G = gammas.shape[0]
    gam_vec = np.repeat(gammas, n_chains)
    smp, _ = hmc_posterior(data, a, b, beta, H, gam_vec,
                           np.full(G * n_chains, eta_cold), N, M, rng,
                           n_chains=G * n_chains, n_warmup=n_warmup,
                           n_sample=n_sample, thin=max(n_sample // 20, 1))
    nk = smp.shape[0]
    smp = smp.reshape(nk, G, n_chains, N, M)
    smp = np.moveaxis(smp, 1, 0).reshape(G, nk * n_chains, N, M)
    lp, _ = logpost_and_grad(smp, data, a, b, beta, H, gammas[:, None], 1.0)
    order = np.argsort(-lp, axis=1)[:, :n_seed]
    extra = np.take_along_axis(smp, order[:, :, None, None], axis=1)
    return map_estimate(data, a, b, beta, H, gammas, N, M, rng,
                        n_restarts=n_restarts, n_steps=n_steps, W_extra=extra)


def hmc_posterior(
    data,
    a,
    b,
    beta,
    H,
    gamma: float,
    eta: float,
    N: int,
    M: int,
    rng,
    n_chains: int = 32,
    n_warmup: int = 1500,
    n_sample: int = 3000,
    thin: int = 10,
    n_leapfrog: int = 12,
    step_size: float = 0.05,
    init_scale: float = 0.1,
    W_init=None,
):
    """Batched Metropolis-corrected HMC on the tempered posterior over ``W``.

    ``gamma`` and ``eta`` may be per-chain arrays of shape ``(n_chains,)``, so a
    whole ``(gamma, eta)`` grid runs in one sweep.  The leapfrog step size is
    adapted per chain during warmup towards a 0.70 acceptance rate, which
    matters here because the posterior stiffness varies by orders of magnitude
    across the ``gamma`` grid.

    Returns ``(samples, accept_rate)`` with ``samples`` of shape
    ``(n_kept, n_chains, N, M)`` -- stacked, not concatenated, so the caller can
    recover which chain (hence which ``(gamma, eta)``) each sample came from --
    and ``accept_rate`` of shape ``(n_chains,)``.

    The zero-field RBM has a discrete gauge group (hidden permutations and sign
    flips) of order ``2^M M!``; chains do not mix across its copies, and they do
    not need to -- every observable used downstream (``P_W(v)``, both KLs,
    ``J_eff``, ``u_r^2``) is gauge invariant.
    """
    if W_init is None:
        W = rng.normal(scale=init_scale, size=(n_chains, N, M))
    else:
        W = np.broadcast_to(W_init, (n_chains, N, M)).copy()
    eps = np.full(n_chains, float(step_size))
    accepted = np.zeros(n_chains)
    n_kept = 0
    out = []
    target_acc = 0.70

    for it in range(n_warmup + n_sample):
        p0 = rng.normal(size=W.shape)
        lp0, g0 = logpost_and_grad(W, data, a, b, beta, H, gamma, eta)
        Wn, p, g = W.copy(), p0.copy(), g0.copy()
        e = eps[:, None, None]
        p = p + 0.5 * e * g
        for k in range(n_leapfrog):
            Wn = Wn + e * p
            lp, g = logpost_and_grad(Wn, data, a, b, beta, H, gamma, eta)
            if k < n_leapfrog - 1:
                p = p + e * g
        p = p + 0.5 * e * g

        dH = (lp - 0.5 * np.sum(p**2, axis=(-1, -2))) - (
            lp0 - 0.5 * np.sum(p0**2, axis=(-1, -2))
        )
        dH = np.nan_to_num(dH, nan=-np.inf, posinf=0.0)
        acc = np.log(rng.uniform(size=n_chains)) < np.minimum(0.0, dH)
        W = np.where(acc[:, None, None], Wn, W)

        if it < n_warmup:
            rate = np.exp(np.minimum(0.0, dH))
            eps *= np.exp(0.05 * (rate - target_acc))
            eps = np.clip(eps, 1e-5, 1.0)
        else:
            accepted += acc
            n_kept += 1
            if (it - n_warmup) % thin == 0:
                out.append(W.copy())

    samples = np.stack(out, axis=0) if out else W[None]
    return samples, accepted / max(n_kept, 1)


# --------------------------------------------------------------------------
# persistent block-Gibbs sampling (finite nu)
# --------------------------------------------------------------------------


def gibbs_sweeps(v, W, a, b, beta, n_sweeps: int, rng) -> np.ndarray:
    """``n_sweeps`` alternating block-Gibbs updates ``h|v`` then ``v|h``.

    ``v`` is ``(n_chains, N)`` in ``{-1,+1}``.  For ``+-1`` spins the conditional
    is ``P(s=+1|x) = sigmoid(2 beta x)``.
    """
    for _ in range(n_sweeps):
        ph = 1.0 / (1.0 + np.exp(-2.0 * beta * (b + v @ W)))
        h = np.where(rng.uniform(size=ph.shape) < ph, 1.0, -1.0)
        pv = 1.0 / (1.0 + np.exp(-2.0 * beta * (a + h @ W.T)))
        v = np.where(rng.uniform(size=pv.shape) < pv, 1.0, -1.0)
    return v


def neg_phase_chains(v, W, a, b, beta) -> np.ndarray:
    """Rao-Blackwellised chain estimate of ``<v h^T>`` given visible states ``v``."""
    th = np.tanh(beta * (b + v @ W))
    return v.T @ th / v.shape[0]


# --------------------------------------------------------------------------
# order parameters referred to the DATA COVARIANCE eigenbasis
#
# The manuscript defines u_k as the overlap between the k-th eigenvector of the
# coupling and the k-th eigenvector of the data covariance C, and h_k as the
# condensation of *generated* samples along that same direction.  The routines
# above measure overlaps against the planted patterns xi_r instead, which is
# only the same thing when the empirical C happens to diagonalise in the planted
# basis.  These are the faithful versions.
# --------------------------------------------------------------------------


def cov_eigvecs(C, k: int = 3) -> np.ndarray:
    """Top-``k`` eigenvectors of ``C``, unit norm, columns in descending order."""
    _, vec = np.linalg.eigh(np.asarray(C, dtype=np.float64))
    return vec[:, ::-1][:, :k]


def mode_alignment(W, cvecs) -> np.ndarray:
    """``u_k^2``: squared overlap of the k-th eigenvector of ``J_eff = W W^T``
    with the k-th data-covariance eigenvector.

    ``W`` is ``(..., N, M)`` and ``cvecs`` is ``(N, R)``; returns ``(..., R)``.
    Both eigenvectors are defined up to a sign, so the square is the invariant.
    """
    W = np.asarray(W, dtype=np.float64)
    J = np.matmul(W, np.swapaxes(W, -1, -2))
    _, vw = np.linalg.eigh(J)
    vw = np.flip(vw, axis=-1)  # descending
    R = np.asarray(cvecs).shape[1]
    return np.einsum("...nr,nr->...r", vw[..., :R], np.asarray(cvecs)) ** 2


def condensation(logp, V, dirs) -> np.ndarray:
    """``h_k^2 = E_P[(v.u_k/sqrt(N))^2]`` for unit directions ``dirs`` ``(N, R)``.

    This is the *generation* side: whether samples drawn from the model actually
    line up with the direction, as opposed to the direction merely being present
    in the weights.  ``|v| = sqrt(N)`` on the hypercube, so ``h_k^2`` lies in
    ``[0, 1]`` and equals 1 for a model frozen along ``u_k``.
    """
    N = V.shape[1]
    m = (V @ np.asarray(dirs, dtype=np.float64)) / np.sqrt(N)  # (2^N, R)
    return np.einsum("...v,vr->...r", np.exp(logp), m**2)


def random_directions(N: int, R: int, rng) -> np.ndarray:
    """``R`` random unit directions in ``R^N``, as a control for ``condensation``."""
    d = rng.normal(size=(N, R))
    return d / np.linalg.norm(d, axis=0, keepdims=True)
