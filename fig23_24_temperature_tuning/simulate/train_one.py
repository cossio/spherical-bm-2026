"""Train a single BM model at a given gamma value."""
import sys
import numpy as np
import torch
from pathlib import Path

from adabmDCA.dataset import DatasetDCA
from adabmDCA.training import train_graph
from adabmDCA.sampling import get_sampler
from adabmDCA.io import save_params

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = str(ROOT / "data/PF00014_full.fasta")
OUT_DIR = ROOT / "data"


def zero_sum_gauge(J):
    return (
        J
        - J.mean(dim=1, keepdim=True)
        - J.mean(dim=3, keepdim=True)
        + J.mean(dim=(1, 3), keepdim=True)
    )


def main(gamma):
    device = torch.device("cpu")
    dtype = torch.float32

    tag = f"l2_{gamma:.4f}"
    param_path = OUT_DIR / f"params_{tag}.dat"

    if param_path.exists():
        print(f"[gamma={gamma}] Already trained, skipping.")
        return

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

    chains = torch.zeros(10000, L, q, device=device, dtype=dtype)
    for i in range(L):
        cats = torch.multinomial(fi[i], 10000, replacement=True)
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
        nsweeps=10,
        lr=0.01,
        max_epochs=3000,
        target_pearson=0.90,
        l2_reg=gamma,
        progress_bar=True,
    )

    final_pearson = log["Pearson"][-1] if log["Pearson"] else None
    n_epochs = len(log["Pearson"])
    print(f"[gamma={gamma}] Done. Pearson={final_pearson:.4f}, epochs={n_epochs}", flush=True)

    save_params(str(param_path), params, tokens="protein")

    J_zs = zero_sum_gauge(params["coupling_matrix"].cpu())
    M = J_zs.reshape(L * q, L * q).numpy()
    M = 0.5 * (M + M.T)
    eigvals = np.sort(np.linalg.eigvalsh(M))[::-1]
    np.save(OUT_DIR / f"eigvals_{tag}.npy", eigvals)
    print(f"[gamma={gamma}] Saved params and eigenvalues.", flush=True)


if __name__ == "__main__":
    gamma = float(sys.argv[1])
    main(gamma)
