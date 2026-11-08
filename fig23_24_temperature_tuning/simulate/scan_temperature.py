"""
Scan sampling temperature β for trained BM models.
For each (l2_reg, β), generate sequences and compute quality metrics:
- Pearson correlation of pairwise frequencies (model vs data)
- Mean Hamming distance to nearest natural sequence
- Entropy of generated sequences
- Energy statistics
"""

import numpy as np
import torch
from pathlib import Path

from adabmDCA.dataset import DatasetDCA
from adabmDCA.sampling import get_sampler
from adabmDCA.io import load_params
from adabmDCA.statmech import compute_energy

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = str(ROOT / "data/PF00014_full.fasta")
OUT_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# Which models to scan
L2_REGS = [0.01, 0.05, 0.1, 0.2, 0.5]

# Temperature scan
BETAS = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0])

# Sampling parameters
NUM_SAMPLES = 5000
NSWEEPS_EQUIL = 500  # equilibration sweeps
NSWEEPS_SAMPLE = 100  # between measurements


def compute_fi_fij(chains):
    """Compute single-site and pairwise frequencies from one-hot chains."""
    # chains: (B, L, q)
    B, L, q = chains.shape
    fi = chains.mean(dim=0)  # (L, q)
    # pairwise: fij[i,a,j,b] = <x_i^a x_j^b>
    # Use outer product
    flat = chains.reshape(B, L * q)  # (B, L*q)
    fij_flat = (flat.T @ flat) / B  # (L*q, L*q)
    fij = fij_flat.reshape(L, q, L, q)
    return fi, fij


def pearson_fij(fij_model, fij_data, fi_model, fi_data):
    """Pearson correlation of connected correlations c_ij = f_ij - f_i * f_j."""
    L, q, _, _ = fij_data.shape
    # Connected correlations
    cij_data = fij_data - fi_data[:, :, None, None] * fi_data[None, None, :, :]
    cij_model = fij_model - fi_model[:, :, None, None] * fi_model[None, None, :, :]

    # Extract upper triangle (i < j) elements
    mask = torch.triu(torch.ones(L, L, device=fij_data.device), diagonal=1)
    mask = mask[:, None, :, None].expand(L, q, L, q)

    cd = cij_data[mask.bool()].cpu().numpy()
    cm = cij_model[mask.bool()].cpu().numpy()

    r = np.corrcoef(cd, cm)[0, 1]
    return r


def sequence_entropy(chains):
    """Per-site entropy of generated sequences."""
    # chains: (B, L, q) one-hot
    fi = chains.mean(dim=0)  # (L, q)
    fi_np = fi.cpu().numpy()
    fi_np = np.clip(fi_np, 1e-10, 1.0)
    S = -np.sum(fi_np * np.log(fi_np), axis=1)  # (L,)
    return S.mean()  # average over sites


def hamming_to_data(gen_chains, data_chains, n_compare=1000):
    """Mean Hamming distance of generated sequences to nearest natural sequence."""
    # Convert one-hot to integer sequences
    gen_seq = gen_chains[:n_compare].argmax(dim=-1)  # (n, L)
    data_seq = data_chains.argmax(dim=-1)  # (M, L)

    # For each generated seq, find min Hamming to any natural seq
    # Do in batches to avoid OOM
    min_hamm = []
    for i in range(0, len(gen_seq), 100):
        batch = gen_seq[i : i + 100]  # (b, L)
        # Hamming distance: count mismatches
        diffs = (batch[:, None, :] != data_seq[None, :, :]).float().sum(dim=-1)  # (b, M)
        min_d = diffs.min(dim=1).values  # (b,)
        min_hamm.append(min_d)
    min_hamm = torch.cat(min_hamm)
    L = gen_seq.shape[1]
    return (min_hamm / L).mean().item()  # normalized


def sample_at_beta(params, beta, L, q, n_samples, device, dtype):
    """Sample sequences at inverse temperature β."""
    sampler = get_sampler("gibbs")

    # Initialize chains from uniform
    fi_uniform = torch.ones(L, q, device=device, dtype=dtype) / q
    chains = torch.zeros(n_samples, L, q, device=device, dtype=dtype)
    for i in range(L):
        cats = torch.multinomial(fi_uniform[i], n_samples, replacement=True)
        chains[:, i, :] = torch.nn.functional.one_hot(cats, q).to(dtype)

    # Equilibrate
    chains = sampler(chains, params, NSWEEPS_EQUIL, beta)
    # Sample
    chains = sampler(chains, params, NSWEEPS_SAMPLE, beta)

    return chains


def main():
    print(f"Device: {DEVICE}")

    # Load dataset
    dataset = DatasetDCA(DATA_PATH, alphabet="protein", device=DEVICE, dtype=DTYPE)
    L = dataset.get_num_residues()
    q = dataset.get_num_states()
    fi_data, fij_data = dataset.get_frequencies(pseudocount=0.0)
    fi_data = fi_data.to(DEVICE, DTYPE)
    fij_data = fij_data.to(DEVICE, DTYPE)
    data_chains = dataset.to_one_hot().to(DEVICE, DTYPE)
    print(f"L={L}, q={q}, data shape={data_chains.shape}")

    results = {}

    for l2_reg in L2_REGS:
        tag = f"l2_{l2_reg:.4f}"
        param_path = OUT_DIR / f"params_{tag}.dat"
        if not param_path.exists():
            print(f"Skipping l2_reg={l2_reg}: no params file")
            continue

        print(f"\n{'='*60}")
        print(f"l2_reg = {l2_reg}")
        print(f"{'='*60}")
        params = load_params(str(param_path), tokens="protein", device=DEVICE, dtype=DTYPE)

        res = {"beta": [], "pearson": [], "entropy": [], "hamming": [], "energy_mean": [], "energy_std": []}

        for beta in BETAS:
            print(f"  β={beta:.2f} ... ", end="", flush=True)

            chains = sample_at_beta(params, beta, L, q, NUM_SAMPLES, DEVICE, DTYPE)

            # Compute metrics
            fi_gen, fij_gen = compute_fi_fij(chains)
            r = pearson_fij(fij_gen, fij_data, fi_gen, fi_data)
            S = sequence_entropy(chains)
            H = hamming_to_data(chains, data_chains)
            E = compute_energy(chains, params)
            E_mean = E.mean().item()
            E_std = E.std().item()

            res["beta"].append(beta)
            res["pearson"].append(r)
            res["entropy"].append(S)
            res["hamming"].append(H)
            res["energy_mean"].append(E_mean)
            res["energy_std"].append(E_std)

            print(f"Pearson={r:.4f}, S={S:.3f}, Hamming={H:.3f}, E={E_mean:.1f}±{E_std:.1f}")

        results[l2_reg] = res

    # Save results
    np.savez(OUT_DIR / "temperature_scan.npz", **{
        f"l2_{r:.4f}_{k}": np.array(v)
        for r, res in results.items()
        for k, v in res.items()
    })
    print(f"\nSaved to {OUT_DIR / 'temperature_scan.npz'}")

    # Quick plot
    plot_results(results)


def plot_results(results):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for l2_reg, res in sorted(results.items()):
        betas = res["beta"]
        label = f"$\\lambda_2={l2_reg}$"

        axes[0, 0].plot(betas, res["pearson"], "o-", label=label, markersize=4)
        axes[0, 1].plot(betas, res["entropy"], "o-", label=label, markersize=4)
        axes[1, 0].plot(betas, res["hamming"], "o-", label=label, markersize=4)
        axes[1, 1].plot(betas, res["energy_mean"], "o-", label=label, markersize=4)

    axes[0, 0].set_ylabel("Pearson (connected corr.)")
    axes[0, 0].set_title("Pairwise correlation quality")
    axes[0, 0].axhline(1.0, ls="--", color="gray", alpha=0.5)

    axes[0, 1].set_ylabel("Per-site entropy")
    axes[0, 1].set_title("Diversity (entropy)")

    axes[1, 0].set_ylabel("Norm. Hamming to nearest natural")
    axes[1, 0].set_title("Distance to data")

    axes[1, 1].set_ylabel("Mean energy")
    axes[1, 1].set_title("Energy of generated sequences")

    for ax in axes.flat:
        ax.set_xlabel("Inverse temperature β")
        ax.set_xscale("log")
        ax.legend(fontsize=7)

    fig.suptitle("Temperature tuning: quality metrics vs sampling temperature", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "temperature_scan.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "temperature_scan.png", dpi=150, bbox_inches="tight")
    print(f"Saved to {FIG_DIR / 'temperature_scan.pdf'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
