"""PCD-MAP trainer on GPU (PyTorch) for the OOE lattice-protein experiment.

Same observables as the numpy version (02_ooe_train.py):
    s_a(t) = chain · c_a / N     (chain-to-data-eigenvector overlap)
    U_ka(t) = full (K_eig × K) overlap of J's top-|λ| eigenvectors with c_a
    λ_k(J) = signed top-|λ| eigenvalues of J

Differences from numpy version:
- Chains, J, h live on CUDA
- gibbs_sweep uses batched gather + Gumbel-max categorical sampling
- Eigendecomposition of J done on CPU (small 540×540 matrix, GPU not faster)

Usage:
    uv run python scripts/02_ooe_train_gpu.py \
        --teacher ../lattice_proteins/data/teacher_beta1000_n10k.h5 \
        --spike results/spike_diagnostic.h5 \
        --k 1000 --gamma 0.01 --m-train 3000 --t-age 5000 --seed 0 \
        --out results/gpu/k1000.h5
"""

# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "numpy>=2.4",
#     "h5py>=3.12",
#     "torch>=2.5",
# ]
# ///

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--spike", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, required=True,
                   help="site-update attempts per gradient step (L=27 ≈ 1 sweep)")
    p.add_argument("--gamma", type=float, required=True)
    p.add_argument("--m-train", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--t-age", type=int, default=5000)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--n-chains", type=int, default=256)
    p.add_argument("--n-top", type=int, default=3)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eig-every", type=int, default=200)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def one_hot_np(msa: np.ndarray, Q: int) -> np.ndarray:
    M, L = msa.shape
    oh = np.zeros((M, L, Q), dtype=np.float64)
    oh[np.arange(M)[:, None], np.arange(L)[None, :], msa] = 1.0
    return oh


def one_hot_torch(X: torch.Tensor, Q: int) -> torch.Tensor:
    """(N, L) long tensor -> (N, L, Q) float tensor, one-hot on device."""
    N, L = X.shape
    oh = torch.zeros((N, L, Q), dtype=torch.float32, device=X.device)
    oh.scatter_(2, X.unsqueeze(-1), 1.0)
    return oh


def suff_stats_torch(X: torch.Tensor, Q: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute f1 (L, Q) and f2 (L, L, Q, Q) from (N, L) long tensor on device."""
    N, L = X.shape
    oh = one_hot_torch(X, Q)                           # (N, L, Q) float32
    f1 = oh.mean(dim=0)                                # (L, Q)
    # f2[i, j, a, b] = mean_n oh[n, i, a] * oh[n, j, b]
    f2 = torch.einsum("nia,njb->ijab", oh, oh) / N     # (L, L, Q, Q)
    idx = torch.arange(L, device=X.device)
    f2[idx, idx] = 0.0
    return f1, f2


def gibbs_site_update_gpu(h: torch.Tensor, J: torch.Tensor, X: torch.Tensor,
                          i: int, gen: torch.Generator) -> None:
    """Update a single site i across all N chains (in-place).

    h: (L, Q), J: (L, L, Q, Q), X: (N, L) long.
    """
    N, L = X.shape
    Q = h.shape[1]
    Ji = J[i]                                           # (L, Q, Q)
    idx = X.unsqueeze(-1).unsqueeze(-1).expand(N, L, Q, 1)
    coup = Ji.unsqueeze(0).expand(N, -1, -1, -1).gather(-1, idx).squeeze(-1)
    coup[:, i, :] = 0.0
    lf = h[i] + coup.sum(dim=1)                         # (N, Q)
    g_noise = -torch.log(-torch.log(torch.rand(N, Q, generator=gen,
                                                device=X.device) + 1e-30) + 1e-30)
    X[:, i] = (lf + g_noise).argmax(dim=1)


def gibbs_updates_gpu(h: torch.Tensor, J: torch.Tensor, X: torch.Tensor,
                      gen: torch.Generator, n_updates: int) -> torch.Tensor:
    """Do n_updates individual site-update attempts (batched over chains on GPU).

    Each attempt picks one random site and resamples it from its full conditional.
    n_updates=L corresponds to one "sweep" on average.

    h: (L, Q), J: (L, L, Q, Q), X: (N, L) long, modified in-place.
    """
    L = X.shape[1]
    sites = torch.randint(L, (n_updates,), generator=gen, device=X.device)
    for i_tensor in sites:
        gibbs_site_update_gpu(h, J, X, int(i_tensor.item()), gen)
    return X


def J_to_flat(J: torch.Tensor) -> torch.Tensor:
    """(L, L, Q, Q) -> (L*Q, L*Q) in the (site, letter) one-hot basis."""
    L, _, Q, _ = J.shape
    return J.permute(0, 2, 1, 3).reshape(L * Q, L * Q)


def eig_J_both_ends_cpu(J_gpu: torch.Tensor, K: int) -> tuple[np.ndarray, np.ndarray]:
    Jf = J_to_flat(J_gpu).cpu().numpy().astype(np.float64)
    Jf = 0.5 * (Jf + Jf.T)
    evals, evecs = np.linalg.eigh(Jf)
    order = np.argsort(np.abs(evals))[::-1][:K]
    return evals[order], evecs[:, order]


def main() -> None:
    args = parse_args()
    t0 = time.time()
    dev = torch.device(args.device)
    print(f"[ooe-gpu] device={dev}, torch={torch.__version__}")

    # --- Load teacher ---
    with h5py.File(args.teacher, "r") as f:
        msa_all = f["msa"][:]
        L = int(f.attrs["L"])
        Q = int(f.attrs["Q"])
    msa_all = msa_all - 1
    assert msa_all.shape[1] == L

    np_rng = np.random.default_rng(args.seed)
    perm = np_rng.permutation(msa_all.shape[0])
    train_idx = perm[: args.m_train]
    train_X = msa_all[train_idx].astype(np.int64)
    print(f"[ooe-gpu] teacher L={L} Q={Q} M_train={args.m_train}")

    # --- Data sufficient stats (on GPU) ---
    train_X_gpu = torch.from_numpy(train_X).to(dev)
    data_f1, data_f2 = suff_stats_torch(train_X_gpu, Q)

    # --- Spike eigenvectors ---
    with h5py.File(args.spike, "r") as f:
        C_vecs_all = f["top_eigenvectors"][:]
        C_evals_all = f["top_eigenvalues"][:]
    K = args.n_top
    C_vecs_np = C_vecs_all[:, :K].astype(np.float64)     # (L*Q, K)
    C_vecs_gpu = torch.from_numpy(C_vecs_np.astype(np.float32)).to(dev)
    print(f"[ooe-gpu] spike top-{K} eigenvalues: {C_evals_all[:K]}")

    # --- Init params + chains ---
    torch_gen = torch.Generator(device=dev).manual_seed(args.seed)
    h = torch.zeros((L, Q), dtype=torch.float32, device=dev)
    J = torch.zeros((L, L, Q, Q), dtype=torch.float32, device=dev)
    chains = torch.randint(0, Q, size=(args.n_chains, L), generator=torch_gen,
                           device=dev, dtype=torch.int64)

    # --- Training loop ---
    K_eig = max(6, 2 * K)
    n_log_max = args.t_age // args.log_every + 2
    t_arr = np.zeros(n_log_max, dtype=np.int64)
    s_arr = np.zeros((n_log_max, K))
    U_arr = np.zeros((n_log_max, K_eig, K))
    lam_eig_arr = np.zeros((n_log_max, K_eig))
    log_idx = 0
    J_evals_top, J_evecs_top = np.zeros(K_eig), np.zeros((L * Q, K_eig))

    idx_diag = torch.arange(L, device=dev)
    print(f"[ooe-gpu] k={args.k} site-updates/step (L={L}, 1 sweep≈{L}), γ={args.gamma}, "
          f"t_age={args.t_age}, M_chains={args.n_chains}")

    for step in range(args.t_age):
        # k site-update attempts on persistent chains
        chains = gibbs_updates_gpu(h, J, chains, torch_gen, args.k)
        model_f1, model_f2 = suff_stats_torch(chains, Q)
        grad_h = model_f1 - data_f1
        grad_J = model_f2 - data_f2
        h -= args.lr * (grad_h + args.gamma * h)
        J -= args.lr * (grad_J + args.gamma * J)
        # Symmetrize + zero diagonal
        J = 0.5 * (J + J.permute(1, 0, 3, 2))
        J[idx_diag, idx_diag] = 0.0

        # Periodically diagonalize J on CPU (540x540, fast)
        if step % args.eig_every == 0 or step == args.t_age - 1:
            J_evals_top, J_evecs_top = eig_J_both_ends_cpu(J, K_eig)

        # Log observables
        if step % args.log_every == 0 or step == args.t_age - 1:
            # σ²_a = chain VARIANCE in c_a direction (correct order parameter for Potts)
            oh = one_hot_torch(chains, Q)                         # (N, L, Q)
            oh_centered = oh - data_f1.unsqueeze(0)
            oh_flat = oh_centered.reshape(args.n_chains, L * Q)    # (N, L*Q)
            proj = oh_flat @ C_vecs_gpu                            # (N, K)
            sigma2_a = (proj ** 2).mean(dim=0).cpu().numpy().astype(np.float64)  # (K,)
            U_full = J_evecs_top.T @ C_vecs_np                     # (K_eig, K)
            t_arr[log_idx] = step
            s_arr[log_idx] = sigma2_a
            U_arr[log_idx] = U_full
            lam_eig_arr[log_idx] = J_evals_top
            log_idx += 1

            if step % (args.log_every * 10) == 0 or step == args.t_age - 1:
                sigma2_sum = float(sigma2_a.sum())
                u_sub = (U_full ** 2).sum(axis=0)
                max_u = np.abs(U_full).max(axis=0)
                print(f"  step {step:>6d}  Σσ²={sigma2_sum:.4f}  σ²={sigma2_a.round(4)} "
                      f"max|u|={max_u.round(3)}  ||P_J c||²={u_sub.round(3)}  "
                      f"λ={J_evals_top[:3].round(2)}  [t={time.time()-t0:.1f}s]")

    t_arr = t_arr[:log_idx]
    s_arr = s_arr[:log_idx]
    U_arr = U_arr[:log_idx]
    lam_eig_arr = lam_eig_arr[:log_idx]

    # --- Save ---
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.attrs["k"] = args.k
        f.attrs["gamma"] = args.gamma
        f.attrs["m_train"] = args.m_train
        f.attrs["seed"] = args.seed
        f.attrs["t_age"] = args.t_age
        f.attrs["lr"] = args.lr
        f.attrs["n_chains"] = args.n_chains
        f.attrs["n_top"] = K
        f.attrs["n_eig"] = K_eig
        f.attrs["log_every"] = args.log_every
        f.attrs["eig_every"] = args.eig_every
        f.attrs["L"] = L
        f.attrs["Q"] = Q
        f.attrs["device"] = args.device
        f.attrs["teacher"] = str(args.teacher)
        f.attrs["wall_time_s"] = time.time() - t0
        f["t"] = t_arr
        f["sigma2_a"] = s_arr                         # (T, K): chain variance in c_a dir
        f["U_ka"] = U_arr
        f["lam_eig"] = lam_eig_arr
        f["s_subspace_sq"] = (s_arr ** 2).sum(axis=1)
        f["u_subspace_sq"] = (U_arr ** 2).sum(axis=1)
        f["u_max_per_a"] = np.abs(U_arr).max(axis=1)
        f["h_final"] = h.cpu().numpy()
        f["J_final"] = J.cpu().numpy()
        f["chains_final"] = chains.cpu().numpy()
    print(f"[ooe-gpu] saved {out} [t={time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
