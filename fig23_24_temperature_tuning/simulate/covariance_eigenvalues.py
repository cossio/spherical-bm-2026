"""
For each (l2_reg, β), compute the eigenvalue spectrum of the empirical covariance
matrix of generated samples, and compare to data covariance eigenvalues.

This directly probes the PM→FM transition: in the paramagnetic phase, the
generated covariance is ~identity (isotropic), while in the ferromagnetic phase,
it develops outlier eigenvalues aligned with the data directions.
"""

import numpy as np
import torch
from pathlib import Path

from adabmDCA.dataset import DatasetDCA
from adabmDCA.sampling import get_sampler
from adabmDCA.io import load_params

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = str(ROOT / "data/PF00014_full.fasta")
OUT_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

L2_REGS = [0.01, 0.05, 0.1, 0.2, 0.5]
BETAS = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0])

NUM_SAMPLES = 10000
NSWEEPS_EQUIL = 500
NSWEEPS_SAMPLE = 100


def covariance_eigenvalues(chains):
    """
    Compute eigenvalues of the covariance matrix of one-hot encoded sequences.
    chains: (B, L, q) one-hot
    Returns eigenvalues in descending order.
    """
    B, L, q = chains.shape
    # Flatten to (B, L*q) and compute covariance
    X = chains.reshape(B, L * q).float()
    X = X - X.mean(dim=0, keepdim=True)
    # Covariance (L*q, L*q)
    C = (X.T @ X) / (B - 1)
    eigvals = torch.linalg.eigvalsh(C)
    return eigvals.cpu().numpy()[::-1]  # descending


def sample_at_beta(params, beta, L, q, n_samples, device, dtype):
    """Sample sequences at inverse temperature β."""
    sampler = get_sampler("gibbs")
    fi_uniform = torch.ones(L, q, device=device, dtype=dtype) / q
    chains = torch.zeros(n_samples, L, q, device=device, dtype=dtype)
    for i in range(L):
        cats = torch.multinomial(fi_uniform[i], n_samples, replacement=True)
        chains[:, i, :] = torch.nn.functional.one_hot(cats, q).to(dtype)
    chains = sampler(chains, params, NSWEEPS_EQUIL, beta)
    chains = sampler(chains, params, NSWEEPS_SAMPLE, beta)
    return chains


def main():
    print(f"Device: {DEVICE}")

    dataset = DatasetDCA(DATA_PATH, alphabet="protein", device=DEVICE, dtype=DTYPE)
    L = dataset.get_num_residues()
    q = dataset.get_num_states()
    data_chains = dataset.to_one_hot().to(DEVICE, DTYPE)
    print(f"L={L}, q={q}, dim=L*q={L*q}")

    # Data covariance eigenvalues
    data_eigvals = covariance_eigenvalues(data_chains)
    np.save(OUT_DIR / "cov_eigvals_data.npy", data_eigvals)
    print(f"Data cov: top5 = {data_eigvals[:5].round(4)}")

    all_results = {}

    for l2_reg in L2_REGS:
        tag = f"l2_{l2_reg:.4f}"
        param_path = OUT_DIR / f"params_{tag}.dat"
        if not param_path.exists():
            print(f"Skipping l2_reg={l2_reg}")
            continue

        print(f"\n{'='*60}")
        print(f"l2_reg = {l2_reg}")
        print(f"{'='*60}")
        params = load_params(str(param_path), tokens="protein", device=DEVICE, dtype=DTYPE)

        for beta in BETAS:
            print(f"  β={beta:.2f} ... ", end="", flush=True)
            chains = sample_at_beta(params, beta, L, q, NUM_SAMPLES, DEVICE, DTYPE)
            ev = covariance_eigenvalues(chains)
            key = f"{tag}_beta_{beta:.2f}"
            all_results[key] = ev
            print(f"top5 = {ev[:5].round(4)}")

    # Save
    np.savez(OUT_DIR / "cov_eigvals_vs_beta.npz", **all_results)
    print(f"\nSaved to {OUT_DIR / 'cov_eigvals_vs_beta.npz'}")

    plot_results(all_results, data_eigvals)


def plot_results(all_results, data_eigvals):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(L2_REGS), 1, figsize=(10, 4 * len(L2_REGS)), sharex=True)
    if len(L2_REGS) == 1:
        axes = [axes]

    cmap = plt.cm.coolwarm
    beta_norm = plt.Normalize(vmin=np.log10(BETAS.min()), vmax=np.log10(BETAS.max()))

    for ax_idx, l2_reg in enumerate(L2_REGS):
        ax = axes[ax_idx]
        tag = f"l2_{l2_reg:.4f}"

        # Plot data eigenvalues
        ax.semilogy(range(1, 51), data_eigvals[:50], "k-", lw=2, alpha=0.5, label="data")

        for beta in BETAS:
            key = f"{tag}_beta_{beta:.2f}"
            if key not in all_results:
                continue
            ev = all_results[key]
            color = cmap(beta_norm(np.log10(beta)))
            ax.semilogy(range(1, 51), ev[:50], "-", color=color, alpha=0.7, label=f"β={beta:.1f}")

        ax.set_ylabel("eigenvalue")
        ax.set_title(f"$\\lambda_2 = {l2_reg}$")
        ax.legend(fontsize=6, ncol=3)

    axes[-1].set_xlabel("eigenvalue index")
    fig.suptitle("Covariance eigenvalues of generated samples vs β", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cov_eigenvalues_vs_beta.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "cov_eigenvalues_vs_beta.png", dpi=150, bbox_inches="tight")
    print(f"Saved to {FIG_DIR / 'cov_eigenvalues_vs_beta.pdf'}")
    plt.close(fig)

    # Also: summary plot of top eigenvalue vs beta for each l2
    fig, ax = plt.subplots(figsize=(8, 5))
    data_top = data_eigvals[0]
    ax.axhline(data_top, ls="--", color="black", alpha=0.5, label=f"data λ₁={data_top:.3f}")

    for l2_reg in L2_REGS:
        tag = f"l2_{l2_reg:.4f}"
        top1 = []
        betas_found = []
        for beta in BETAS:
            key = f"{tag}_beta_{beta:.2f}"
            if key in all_results:
                top1.append(all_results[key][0])
                betas_found.append(beta)
        if top1:
            ax.plot(betas_found, top1, "o-", label=f"$\\lambda_2={l2_reg}$", markersize=4)

    ax.set_xlabel("Inverse temperature β")
    ax.set_ylabel("Top eigenvalue of generated covariance")
    ax.set_xscale("log")
    ax.legend()
    ax.set_title("Top covariance eigenvalue vs sampling temperature")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "top_cov_eigval_vs_beta.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "top_cov_eigval_vs_beta.png", dpi=150, bbox_inches="tight")
    print(f"Saved to {FIG_DIR / 'top_cov_eigval_vs_beta.pdf'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
