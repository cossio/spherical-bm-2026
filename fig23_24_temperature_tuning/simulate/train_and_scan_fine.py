"""
Train new BM models at intermediate gamma values and run a fine temperature scan.
Designed to run training in parallel on CPU.
"""

import numpy as np
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from adabmDCA.dataset import DatasetDCA
from adabmDCA.training import train_graph
from adabmDCA.sampling import get_sampler
from adabmDCA.io import save_params, load_params
from adabmDCA.statmech import compute_energy

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = str(ROOT / "data/PF00014_full.fasta")
OUT_DIR = ROOT / "data"

# New gamma values to fill in between 0.1 and 0.4
NEW_GAMMAS = [0.12, 0.15, 0.18, 0.25, 0.30, 0.35]

# All gammas for the fine scan (existing + new)
ALL_GAMMAS = [0.01, 0.05, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.30, 0.35, 0.5]

# Fine beta grid focused around 1
BETAS_FINE = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0])

# Training parameters
LR = 0.01
NSWEEPS = 10
MAX_EPOCHS = 3000
TARGET_PEARSON = 0.90
NUM_CHAINS_TRAIN = 10000

# Sampling parameters
NUM_SAMPLES = 5000
NSWEEPS_EQUIL = 500
NSWEEPS_SAMPLE = 100


def zero_sum_gauge(J):
    return (
        J
        - J.mean(dim=1, keepdim=True)
        - J.mean(dim=3, keepdim=True)
        + J.mean(dim=(1, 3), keepdim=True)
    )


def coupling_eigenvalues(J_zs):
    L, q, _, _ = J_zs.shape
    M = J_zs.reshape(L * q, L * q).numpy()
    M = 0.5 * (M + M.T)
    eigvals = np.linalg.eigvalsh(M)
    return np.sort(eigvals)[::-1]


def train_one_model(gamma):
    """Train a single model at given gamma. Runs on CPU."""
    device = torch.device("cpu")
    dtype = torch.float32

    tag = f"l2_{gamma:.4f}"
    param_path = OUT_DIR / f"params_{tag}.dat"

    if param_path.exists():
        print(f"[gamma={gamma}] Already trained, skipping.", flush=True)
        return gamma, True

    print(f"[gamma={gamma}] Training...", flush=True)

    dataset = DatasetDCA(DATA_PATH, alphabet="protein", device=device, dtype=dtype)
    L = dataset.get_num_residues()
    q = dataset.get_num_states()
    M_eff = dataset.get_effective_size()

    fi, fij = dataset.get_frequencies(pseudocount=1.0 / M_eff)
    fi, fij = fi.to(device, dtype), fij.to(device, dtype)

    h = torch.log(fi + 1e-6)
    h = h - h.mean(dim=1, keepdim=True)
    J = torch.zeros(L, q, L, q, device=device, dtype=dtype)
    params = {"bias": h, "coupling_matrix": J}

    chains = torch.zeros(NUM_CHAINS_TRAIN, L, q, device=device, dtype=dtype)
    for i in range(L):
        cats = torch.multinomial(fi[i], NUM_CHAINS_TRAIN, replacement=True)
        chains[:, i, :] = torch.nn.functional.one_hot(cats, q).to(dtype)

    mask = torch.ones(L, q, L, q, device=device, dtype=dtype)
    for i in range(L):
        mask[i, :, i, :] = 0.0

    sampler = get_sampler("gibbs")

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
        l2_reg=gamma,
        progress_bar=False,
    )

    final_pearson = log["Pearson"][-1] if log["Pearson"] else None
    n_epochs = len(log["Pearson"])
    print(f"[gamma={gamma}] Done. Pearson={final_pearson:.4f}, epochs={n_epochs}", flush=True)

    save_params(str(param_path), params, tokens="protein")

    # Also save eigenvalues
    J_zs = zero_sum_gauge(params["coupling_matrix"].cpu())
    eigvals = coupling_eigenvalues(J_zs)
    np.save(OUT_DIR / f"eigvals_{tag}.npy", eigvals)

    return gamma, True


def compute_fi_fij(chains):
    B, L, q = chains.shape
    fi = chains.mean(dim=0)
    flat = chains.reshape(B, L * q)
    fij_flat = (flat.T @ flat) / B
    fij = fij_flat.reshape(L, q, L, q)
    return fi, fij


def pearson_fij(fij_model, fij_data, fi_model, fi_data):
    L, q, _, _ = fij_data.shape
    cij_data = fij_data - fi_data[:, :, None, None] * fi_data[None, None, :, :]
    cij_model = fij_model - fi_model[:, :, None, None] * fi_model[None, None, :, :]
    mask = torch.triu(torch.ones(L, L, device=fij_data.device), diagonal=1)
    mask = mask[:, None, :, None].expand(L, q, L, q)
    cd = cij_data[mask.bool()].cpu().numpy()
    cm = cij_model[mask.bool()].cpu().numpy()
    return np.corrcoef(cd, cm)[0, 1]


def sequence_entropy(chains):
    fi = chains.mean(dim=0).cpu().numpy()
    fi = np.clip(fi, 1e-10, 1.0)
    S = -np.sum(fi * np.log(fi), axis=1)
    return S.mean()


def hamming_to_data(gen_chains, data_chains, n_compare=1000):
    gen_seq = gen_chains[:n_compare].argmax(dim=-1)
    data_seq = data_chains.argmax(dim=-1)
    min_hamm = []
    for i in range(0, len(gen_seq), 100):
        batch = gen_seq[i : i + 100]
        diffs = (batch[:, None, :] != data_seq[None, :, :]).float().sum(dim=-1)
        min_d = diffs.min(dim=1).values
        min_hamm.append(min_d)
    min_hamm = torch.cat(min_hamm)
    L = gen_seq.shape[1]
    return (min_hamm / L).mean().item()


def run_fine_scan():
    """Run temperature scan on all gammas with fine beta grid."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"\nFine scan on device: {device}")

    dataset = DatasetDCA(DATA_PATH, alphabet="protein", device=device, dtype=dtype)
    L = dataset.get_num_residues()
    q = dataset.get_num_states()
    fi_data, fij_data = dataset.get_frequencies(pseudocount=0.0)
    fi_data, fij_data = fi_data.to(device, dtype), fij_data.to(device, dtype)
    data_chains = dataset.to_one_hot().to(device, dtype)

    sampler = get_sampler("gibbs")
    results = {}

    for gamma in ALL_GAMMAS:
        tag = f"l2_{gamma:.4f}"
        param_path = OUT_DIR / f"params_{tag}.dat"
        if not param_path.exists():
            print(f"Skipping gamma={gamma}: no params")
            continue

        print(f"\n--- gamma={gamma} ---")
        params = load_params(str(param_path), tokens="protein", device=device, dtype=dtype)

        res = {"beta": [], "pearson": [], "entropy": [], "hamming": [], "energy_mean": [], "energy_std": []}

        for beta in BETAS_FINE:
            print(f"  beta={beta:.2f} ... ", end="", flush=True)

            # Initialize chains
            fi_uniform = torch.ones(L, q, device=device, dtype=dtype) / q
            chains = torch.zeros(NUM_SAMPLES, L, q, device=device, dtype=dtype)
            for i in range(L):
                cats = torch.multinomial(fi_uniform[i], NUM_SAMPLES, replacement=True)
                chains[:, i, :] = torch.nn.functional.one_hot(cats, q).to(dtype)

            chains = sampler(chains, params, NSWEEPS_EQUIL, beta)
            chains = sampler(chains, params, NSWEEPS_SAMPLE, beta)

            fi_gen, fij_gen = compute_fi_fij(chains)
            r = pearson_fij(fij_gen, fij_data, fi_gen, fi_data)
            S = sequence_entropy(chains)
            H = hamming_to_data(chains, data_chains)
            E = compute_energy(chains, params)

            res["beta"].append(beta)
            res["pearson"].append(r)
            res["entropy"].append(S)
            res["hamming"].append(H)
            res["energy_mean"].append(E.mean().item())
            res["energy_std"].append(E.std().item())

            print(f"Pearson={r:.4f}", flush=True)

        results[gamma] = res

    # Save
    out = {}
    for gamma, res in results.items():
        tag = f"l2_{gamma:.4f}"
        for k, v in res.items():
            out[f"{tag}_{k}"] = np.array(v)
    np.savez(OUT_DIR / "temperature_scan_fine.npz", **out)
    print(f"\nSaved to {OUT_DIR / 'temperature_scan_fine.npz'}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # Phase 1: Train new models in parallel on CPU
    print("=" * 60)
    print("Phase 1: Training new models")
    print("=" * 60)

    # Use 4 parallel workers (each needs ~1-2 cores for torch + memory)
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(train_one_model, g): g for g in NEW_GAMMAS}
        for future in as_completed(futures):
            gamma = futures[future]
            try:
                result = future.result()
                print(f"[gamma={gamma}] completed successfully")
            except Exception as e:
                print(f"[gamma={gamma}] FAILED: {e}")

    # Phase 2: Fine temperature scan (uses GPU if available)
    print("\n" + "=" * 60)
    print("Phase 2: Fine temperature scan")
    print("=" * 60)
    run_fine_scan()
