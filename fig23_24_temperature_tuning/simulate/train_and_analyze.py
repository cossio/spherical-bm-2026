"""
Train BM models at varying L2 regularization and analyze eigenvalue spectra.

Goal: find a regime where coupling eigenvalues are outliers (learned signal)
but the model is paramagnetic at β=1 (PMo regime), so that lowering the
sampling temperature triggers a PM→FM transition.
"""

import numpy as np
import torch
from pathlib import Path

from adabmDCA.dataset import DatasetDCA
from adabmDCA.training import train_graph
from adabmDCA.sampling import get_sampler
from adabmDCA.io import save_params, load_params

# ── Configuration ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = str(ROOT / "data/PF00014_full.fasta")
OUT_DIR = ROOT / "data"
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# Regularization values to scan
L2_REGS = [0.01, 0.05, 0.1, 0.2, 0.5]

# Training hyperparameters
LR = 0.01
NSWEEPS = 10
MAX_EPOCHS = 3000
TARGET_PEARSON = 0.90
NUM_CHAINS = 10000


def zero_sum_gauge(J, h):
    """
    Project coupling matrix J (L, q, L, q) and fields h (L, q)
    into zero-sum gauge.
    """
    L, q = h.shape
    J_zs = (
        J
        - J.mean(dim=1, keepdim=True)
        - J.mean(dim=3, keepdim=True)
        + J.mean(dim=(1, 3), keepdim=True)
    )
    h_zs = h - h.mean(dim=1, keepdim=True)
    return J_zs, h_zs


def coupling_eigenvalues(J_zs):
    """
    Reshape zero-sum-gauge J (L, q, L, q) into (L*q, L*q) symmetric matrix
    and compute eigenvalues.
    """
    L, q, _, _ = J_zs.shape
    M = J_zs.reshape(L * q, L * q)
    M = 0.5 * (M + M.T)
    eigvals = torch.linalg.eigvalsh(M)
    return eigvals.cpu().numpy()


def init_chains(num_chains, L, q, fi, device, dtype):
    """Initialize chains by sampling from single-site frequencies."""
    chains = torch.zeros(num_chains, L, q, device=device, dtype=dtype)
    for i in range(L):
        cats = torch.multinomial(fi[i], num_chains, replacement=True)
        chains[:, i, :] = torch.nn.functional.one_hot(cats, q).to(dtype)
    return chains


def train_one(dataset, l2_reg, device, dtype):
    """Train a single BM model and return parameters."""
    L = dataset.get_num_residues()
    q = dataset.get_num_states()
    M_eff = dataset.get_effective_size()

    fi, fij = dataset.get_frequencies(pseudocount=1.0 / M_eff)
    fi = fi.to(device=device, dtype=dtype)
    fij = fij.to(device=device, dtype=dtype)

    # Init parameters
    h = torch.log(fi + 1e-6)
    h = h - h.mean(dim=1, keepdim=True)
    J = torch.zeros(L, q, L, q, device=device, dtype=dtype)
    params = {"bias": h, "coupling_matrix": J}

    chains = init_chains(NUM_CHAINS, L, q, fi, device, dtype)

    # Mask: zero diagonal blocks (no self-coupling)
    mask = torch.ones(L, q, L, q, device=device, dtype=dtype)
    for i in range(L):
        mask[i, :, i, :] = 0.0

    sampler = get_sampler("gibbs")

    print(f"  Training with l2_reg={l2_reg} ...")
    chains, params, _, log = train_graph(
        sampler=sampler,
        chains=chains,
        mask=mask,
        fi_target=fi,
        fij_target=fij,
        params=params,
        nsweeps=NSWEEPS,
        lr=LR,
        max_epochs=MAX_EPOCHS,
        target_pearson=TARGET_PEARSON,
        l2_reg=l2_reg,
        progress_bar=True,
    )

    final_pearson = log["Pearson"][-1] if log["Pearson"] else None
    n_epochs = len(log["Pearson"])
    print(f"  Done. Final Pearson: {final_pearson:.4f}, epochs: {n_epochs}")

    return params, chains, log


def analyze_eigenvalues(params):
    """Extract eigenvalue spectrum from trained parameters."""
    J = params["coupling_matrix"].cpu()
    h = params["bias"].cpu()
    J_zs, _ = zero_sum_gauge(J, h)
    eigvals = coupling_eigenvalues(J_zs)
    return np.sort(eigvals)[::-1]  # descending


def main():
    print(f"Device: {DEVICE}")
    print(f"Loading dataset from {DATA_PATH} ...")
    dataset = DatasetDCA(
        DATA_PATH,
        alphabet="protein",
        device=DEVICE,
        dtype=DTYPE,
    )
    L = dataset.get_num_residues()
    q = dataset.get_num_states()
    M_eff = dataset.get_effective_size()
    print(f"L={L}, q={q}, M_eff={M_eff:.0f}, dim=L*q={L*q}")

    all_eigvals = {}

    for l2_reg in L2_REGS:
        tag = f"l2_{l2_reg:.4f}"
        param_path = OUT_DIR / f"params_{tag}.dat"

        if param_path.exists():
            print(f"\n[l2_reg={l2_reg}] Loading from {param_path}")
            params = load_params(str(param_path), tokens="protein", device=DEVICE, dtype=DTYPE)
        else:
            print(f"\n[l2_reg={l2_reg}] Training ...")
            params, chains, log = train_one(dataset, l2_reg, DEVICE, DTYPE)
            save_params(str(param_path), params, tokens="protein")
            print(f"  Saved to {param_path}")

        eigvals = analyze_eigenvalues(params)
        all_eigvals[l2_reg] = eigvals
        np.save(OUT_DIR / f"eigvals_{tag}.npy", eigvals)

        # Quick summary: identify bulk and outliers
        n = len(eigvals)
        # Estimate bulk from middle 80% of eigenvalues
        mid = eigvals[n // 10 : -n // 10]
        bulk_std = np.std(mid)
        bulk_mean = np.mean(mid)
        edge_upper = bulk_mean + 2 * bulk_std
        n_outliers = np.sum(eigvals > edge_upper * 1.1)
        print(f"  Top 10 eigvals: {eigvals[:10].round(4)}")
        print(f"  Bulk: mean={bulk_mean:.4f}, std={bulk_std:.4f}, edge≈{edge_upper:.4f}")
        print(f"  Outliers (>1.1×edge): {n_outliers}")

    # Save all
    np.savez(
        OUT_DIR / "all_eigvals.npz",
        **{f"l2_{r:.4f}": v for r, v in all_eigvals.items()},
    )
    print("\nAll done. Eigenvalue data saved to results/")


if __name__ == "__main__":
    main()
