"""PCD-MAP OOE trainer for PF00018 real MSA (GPU).

Same control parameter (k = site-update attempts per gradient step) and the
same set of instrumented observables as ``02_ooe_train_gpu.py``. The only
difference is the data loader: instead of an HDF5 teacher file, we load the
PF00018 FASTA via ``adabmDCA.DatasetDCA`` and use its importance-reweighted
sufficient statistics. Spike eigenvectors c_a are read from the corresponding
``spike_diagnostic_pf00018.h5`` produced by ``01_spike_diagnostic_pf00018.py``.

Logged observables per ~log_every steps:
    σ²_a(t) = E_chains[(chain_oh - f1_data) · c_a]²        # Potts order parameter
    U_ka(t) = full (K_eig × K) overlap of top-|λ| J-eigenvectors with c_a
    λ_k(t) = signed top-|λ| eigenvalues of J

Gradients use WEIGHTED data sufficient statistics (reweighted f1, f2).

Usage:
    uv run python scripts/02_ooe_train_pf00018.py \
        --spike results/spike_diagnostic_pf00018.h5 \
        --weights results/PF00018_weights.pt \
        --k 48 --gamma 0.01 --t-age 10000 --seed 0 \
        --out results/pf00018/k48.h5
"""

# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "numpy>=2.4",
#     "h5py>=3.12",
#     "torch>=2.5",
#     "adabmDCA>=0.5",
# ]
# ///

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from adabmDCA.dataset import DatasetDCA

HERE = Path(__file__).resolve().parent.parent
REPO_ROOT = HERE.parent.parent
DEFAULT_FASTA = REPO_ROOT / "temperature_tuning" / "data" / "PF00018_full.fasta"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fasta", default=str(DEFAULT_FASTA))
    p.add_argument("--weights", default=str(HERE / ".." / "data" / "PF00018_weights.pt"),
                   help="torch-saved reweighting weights; produced by the spike diagnostic")
    p.add_argument("--spike", default=str(HERE / ".." / "data" / "spike_diagnostic_pf00018.h5"))
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, required=True,
                   help="site-update attempts per gradient step (L=48 ≈ 1 sweep)")
    p.add_argument("--gamma", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--t-age", type=int, default=10000)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--n-chains", type=int, default=256)
    p.add_argument("--n-top", type=int, default=3)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eig-every", type=int, default=200)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def one_hot_torch(X: torch.Tensor, Q: int) -> torch.Tensor:
    N, L = X.shape
    oh = torch.zeros((N, L, Q), dtype=torch.float32, device=X.device)
    oh.scatter_(2, X.unsqueeze(-1), 1.0)
    return oh


def model_suff_stats(X: torch.Tensor, Q: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Unweighted f1, f2 from the MCMC chains."""
    N, L = X.shape
    oh = one_hot_torch(X, Q)
    f1 = oh.mean(dim=0)
    f2 = torch.einsum("nia,njb->ijab", oh, oh) / N
    idx = torch.arange(L, device=X.device)
    f2[idx, idx] = 0.0
    return f1, f2


def data_suff_stats_weighted(X_int: torch.Tensor, Q: int,
                             w_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Importance-weighted data sufficient statistics (f1, f2) in float32."""
    M, L = X_int.shape
    oh = one_hot_torch(X_int, Q)                                 # (M, L, Q) float32
    wn = w_norm.to(oh.dtype)
    f1 = torch.einsum("m,mla->la", wn, oh)                       # (L, Q)
    oh_w = oh * wn.view(M, 1, 1)
    f2 = torch.einsum("mia,mjb->ijab", oh_w, oh)                 # (L, L, Q, Q)
    idx = torch.arange(L, device=X_int.device)
    f2[idx, idx] = 0.0
    return f1, f2


def gibbs_site_update_gpu(h: torch.Tensor, J: torch.Tensor, X: torch.Tensor,
                          i: int, gen: torch.Generator) -> None:
    N, L = X.shape
    Q = h.shape[1]
    Ji = J[i]
    idx = X.unsqueeze(-1).unsqueeze(-1).expand(N, L, Q, 1)
    coup = Ji.unsqueeze(0).expand(N, -1, -1, -1).gather(-1, idx).squeeze(-1)
    coup[:, i, :] = 0.0
    lf = h[i] + coup.sum(dim=1)
    g = -torch.log(-torch.log(torch.rand(N, Q, generator=gen, device=X.device) + 1e-30) + 1e-30)
    X[:, i] = (lf + g).argmax(dim=1)


def gibbs_updates_gpu(h: torch.Tensor, J: torch.Tensor, X: torch.Tensor,
                      gen: torch.Generator, n_updates: int) -> torch.Tensor:
    L = X.shape[1]
    sites = torch.randint(L, (n_updates,), generator=gen, device=X.device)
    for i_tensor in sites:
        gibbs_site_update_gpu(h, J, X, int(i_tensor.item()), gen)
    return X


def J_to_flat(J: torch.Tensor) -> torch.Tensor:
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
    print(f"[ooe-pf18] device={dev}, torch={torch.__version__}")

    # --- Load MSA + weights ---
    w_cache = Path(args.weights)
    if not w_cache.exists():
        raise FileNotFoundError(f"weights cache {w_cache} missing — run the spike diagnostic first")
    ds = DatasetDCA(args.fasta, alphabet="protein", no_reweighting=True,
                    device=dev, dtype=torch.float32, message=False)
    ds.weights = torch.load(w_cache, map_location=dev).to(device=dev, dtype=torch.float32)
    L = ds.get_num_residues()
    Q = ds.get_num_states()
    M = len(ds)
    M_eff = float(ds.get_effective_size())
    X_int = ds.data.to(torch.int64)                              # (M, L)
    w = ds.weights.to(torch.float64)
    w_norm = (w / w.sum()).to(torch.float32)
    print(f"[ooe-pf18] L={L} Q={Q} M={M} M_eff={M_eff:.1f}")

    # --- Weighted data sufficient statistics (computed once) ---
    data_f1, data_f2 = data_suff_stats_weighted(X_int, Q, w_norm)
    print(f"[ooe-pf18] data suff stats ready [t={time.time()-t0:.1f}s]")

    # --- Spike c_a ---
    with h5py.File(args.spike, "r") as f:
        C_vecs_all = f["top_eigenvectors"][:]
        C_evals_all = f["top_eigenvalues"][:]
        data_f1_saved = f["f1"][:]
    K = args.n_top
    C_vecs_np = C_vecs_all[:, :K].astype(np.float64)
    C_vecs_gpu = torch.from_numpy(C_vecs_np.astype(np.float32)).to(dev)
    print(f"[ooe-pf18] spike top-{K} eigenvalues: {C_evals_all[:K]}")
    f1_consistency = float(np.max(np.abs(data_f1.cpu().numpy() - data_f1_saved)))
    print(f"[ooe-pf18] f1 match between trainer and spike diag: max|Δ|={f1_consistency:.2e}")

    # --- Init params + chains ---
    torch_gen = torch.Generator(device=dev).manual_seed(args.seed)
    h = torch.zeros((L, Q), dtype=torch.float32, device=dev)
    J = torch.zeros((L, L, Q, Q), dtype=torch.float32, device=dev)
    chains = torch.randint(0, Q, size=(args.n_chains, L), generator=torch_gen,
                           device=dev, dtype=torch.int64)

    K_eig = max(6, 2 * K)
    n_log_max = args.t_age // args.log_every + 2
    t_arr = np.zeros(n_log_max, dtype=np.int64)
    s_arr = np.zeros((n_log_max, K))
    U_arr = np.zeros((n_log_max, K_eig, K))
    lam_eig_arr = np.zeros((n_log_max, K_eig))
    log_idx = 0
    J_evals_top, J_evecs_top = np.zeros(K_eig), np.zeros((L * Q, K_eig))

    idx_diag = torch.arange(L, device=dev)
    print(f"[ooe-pf18] k={args.k} site-updates/step (L={L}, 1 sweep≈{L}), γ={args.gamma}, "
          f"t_age={args.t_age}, M_chains={args.n_chains}")

    for step in range(args.t_age):
        chains = gibbs_updates_gpu(h, J, chains, torch_gen, args.k)
        model_f1, model_f2 = model_suff_stats(chains, Q)
        grad_h = model_f1 - data_f1
        grad_J = model_f2 - data_f2
        h -= args.lr * (grad_h + args.gamma * h)
        J -= args.lr * (grad_J + args.gamma * J)
        J = 0.5 * (J + J.permute(1, 0, 3, 2))
        J[idx_diag, idx_diag] = 0.0

        if step % args.eig_every == 0 or step == args.t_age - 1:
            J_evals_top, J_evecs_top = eig_J_both_ends_cpu(J, K_eig)

        if step % args.log_every == 0 or step == args.t_age - 1:
            oh = one_hot_torch(chains, Q)
            oh_centered = oh - data_f1.unsqueeze(0)
            oh_flat = oh_centered.reshape(args.n_chains, L * Q)
            proj = oh_flat @ C_vecs_gpu                         # (N, K)
            sigma2_a = (proj ** 2).mean(dim=0).cpu().numpy().astype(np.float64)
            U_full = J_evecs_top.T @ C_vecs_np                  # (K_eig, K)
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

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.attrs["k"] = args.k
        f.attrs["gamma"] = args.gamma
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
        f.attrs["M"] = M
        f.attrs["M_eff"] = M_eff
        f.attrs["device"] = args.device
        f.attrs["fasta"] = str(args.fasta)
        f.attrs["wall_time_s"] = time.time() - t0
        f["t"] = t_arr
        f["sigma2_a"] = s_arr
        f["U_ka"] = U_arr
        f["lam_eig"] = lam_eig_arr
        f["s_subspace_sq"] = (s_arr ** 2).sum(axis=1)
        f["u_subspace_sq"] = (U_arr ** 2).sum(axis=1)
        f["u_max_per_a"] = np.abs(U_arr).max(axis=1)
        f["h_final"] = h.cpu().numpy()
        f["J_final"] = J.cpu().numpy()
        f["chains_final"] = chains.cpu().numpy()
    print(f"[ooe-pf18] saved {out} [t={time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
