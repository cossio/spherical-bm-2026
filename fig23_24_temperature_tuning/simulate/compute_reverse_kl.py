"""Compute log-likelihood and generalization gap for DCA Boltzmann machines.

For each trained BM at regularization strength gamma:
  1. Estimate log Z via Annealed Importance Sampling (AIS),
     annealing from the profile model (bias-only) to the full Potts model.
  2. Evaluate log Q(x) = -E(x) - log Z on:
     - Model samples (drawn by Gibbs sampling from Q)
     - Held-out test sequences from P*
  3. Compute the generalization gap:
       Delta = <log Q>_Q - <log Q>_{P*} = -H(Q) - <log Q>_{P*}
     This measures the difference between the model's self-entropy and its
     cross-entropy on held-out data. It equals D_KL(Q || P*_emp) where P*_emp
     is the empirical distribution of the test set (uniform over test sequences).
  4. Also report the test log-likelihood per residue: <log Q>_{P*} / L

Note: The true D_KL(Q || P*) = <log Q - log P*>_Q requires log P*(x) which is
inaccessible. The generalization gap Delta is a computable proxy: it is >= 0
when Q concentrates on modes not well-represented in the test set, and < 0 when
Q is flatter than the test distribution (typical of regularized/paramagnetic models).

Usage:
    uv run python compute_reverse_kl.py
    uv run python compute_reverse_kl.py --family PF00014 --test-fraction 0.2
"""

import argparse
import json
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

import matplotlib.pyplot as plt
import numpy as np
import torch

from adabmDCA.dataset import DatasetDCA
from adabmDCA.io import load_params
from adabmDCA.sampling import gibbs_sampling
from adabmDCA.statmech import compute_energy

# ── Configuration ────────────────────────────────────────────────────────

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# AIS parameters
AIS_N_CHAINS = 5000       # number of AIS chains
AIS_N_STEPS = 200         # number of intermediate temperatures (beta values)
AIS_NSWEEPS = 10          # Gibbs sweeps per AIS step
AIS_N_REPEATS = 5         # number of independent AIS runs for error bars

# Sampling parameters for <log Q>_Q
SAMPLE_N_CHAINS = 10000   # samples from the model
SAMPLE_NSWEEPS_EQUIL = 500
SAMPLE_NSWEEPS_SAMPLE = 200


def profile_log_Z(h: torch.Tensor) -> float:
    """Log partition function of the profile (bias-only) model.

    The profile model factorizes: Z_0 = prod_i sum_a exp(h_{ia}).
    So log Z_0 = sum_i log(sum_a exp(h_{ia})) = sum_i logsumexp(h_i).
    """
    return torch.logsumexp(h, dim=1).sum().item()


def ais_log_Z(
    params: dict[str, torch.Tensor],
    n_chains: int,
    n_steps: int,
    nsweeps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float]:
    """Estimate log Z of the full Potts model via AIS.

    Anneals from the profile model (beta=0 for couplings) to the full model (beta=1).
    The intermediate distribution at inverse temperature beta_t is:

        p_t(x) ~ exp(h . x + beta_t * (1/2) x^T J x)

    i.e., the bias term is always at full strength, and the coupling term is
    annealed from 0 to 1. This ensures the reference distribution (t=0) is the
    tractable profile model.

    Returns:
        (log_Z_estimate, log_Z_std): mean and std of log Z from the AIS chains.
    """
    h = params["bias"]
    J = params["coupling_matrix"]
    L, q = h.shape

    # Temperature schedule: sigmoidal (concentrates near endpoints)
    # beta_k for k = 0, ..., n_steps
    k = torch.linspace(0, 1, n_steps + 1, device=device, dtype=dtype)
    betas = k  # linear schedule; sigmoidal can be: torch.sigmoid(12*(k-0.5))

    # Initialize chains from profile model
    logits = h.unsqueeze(0).expand(n_chains, -1, -1)  # (n_chains, L, q)
    probs = torch.softmax(logits, dim=-1)
    chains = torch.zeros(n_chains, L, q, device=device, dtype=dtype)
    for i in range(L):
        cats = torch.multinomial(probs[0, i], n_chains, replacement=True)
        chains[:, i, :] = torch.nn.functional.one_hot(cats, q).to(dtype)

    # Log weights (importance weights)
    log_w = torch.zeros(n_chains, device=device, dtype=dtype)

    # Reference log Z
    log_Z0 = profile_log_Z(h)

    # Precompute coupling contribution: E_coupling(x) = -(1/2) x^T J x
    # E_full(x) = -h.x - (1/2) x^T J x
    # E_profile(x) = -h.x
    # So E_full = E_profile + E_coupling  where E_coupling = -(1/2) x^T J x
    # At temperature beta: E_beta(x) = -h.x - beta * (1/2) x^T J x
    #                                 = E_profile(x) + beta * E_coupling(x)

    for t in range(n_steps):
        beta_prev = betas[t]
        beta_curr = betas[t + 1]
        delta_beta = beta_curr - beta_prev

        # Update weights: log w += -delta_beta * E_coupling(x)
        # E_coupling(x) = E_full(x) - E_profile(x) = -(1/2) x^T J x
        x_flat = chains.view(n_chains, -1)  # (n_chains, L*q)
        J_flat = J.reshape(L * q, L * q)
        coupling_energy = -0.5 * torch.sum(x_flat * (x_flat @ J_flat), dim=1)
        log_w += -delta_beta * coupling_energy

        # Gibbs sampling at temperature beta_curr
        # Build intermediate params: h stays same, J scaled by beta_curr
        params_t = {
            "bias": h,
            "coupling_matrix": beta_curr * J,
        }
        chains = gibbs_sampling(chains, params_t, nsweeps)

    # log Z = log Z_0 + log(mean(exp(log_w)))
    # Use logsumexp for numerical stability
    log_Z = log_Z0 + torch.logsumexp(log_w, dim=0).item() - np.log(n_chains)

    # Standard error estimate via jackknife
    # log Z_k = log Z_0 + logsumexp(log_w excluding k) - log(n-1)
    # This is expensive; instead use the ESS-based approximation
    log_w_np = log_w.cpu().numpy()
    # Effective sample size
    max_lw = log_w_np.max()
    w_normalized = np.exp(log_w_np - max_lw)
    ess = w_normalized.sum() ** 2 / (w_normalized**2).sum()

    return log_Z, ess


def sample_from_model(
    params: dict[str, torch.Tensor],
    n_samples: int,
    nsweeps_equil: int,
    nsweeps_sample: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate samples from the BM by Gibbs sampling."""
    L, q = params["bias"].shape

    # Initialize from profile model
    logits = params["bias"].unsqueeze(0).expand(n_samples, -1, -1)
    probs = torch.softmax(logits, dim=-1)
    chains = torch.zeros(n_samples, L, q, device=device, dtype=dtype)
    for i in range(L):
        cats = torch.multinomial(probs[0, i], n_samples, replacement=True)
        chains[:, i, :] = torch.nn.functional.one_hot(cats, q).to(dtype)

    # Equilibrate
    chains = gibbs_sampling(chains, params, nsweeps_equil)
    # Final sample
    chains = gibbs_sampling(chains, params, nsweeps_sample)

    return chains


def compute_reverse_kl(
    params: dict[str, torch.Tensor],
    test_chains: torch.Tensor,
    log_Z: float,
    model_chains: torch.Tensor | None = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Compute D_KL(Q || P*) = <log Q>_Q - <log Q>_{P*}.

    log Q(x) = -E(x) - log Z

    Args:
        params: trained model parameters (h, J).
        test_chains: one-hot encoded test sequences (from P*).
        log_Z: estimated log partition function.
        model_chains: samples from Q. If None, must be provided externally.

    Returns:
        dict with reverse KL, its components, and diagnostics.
    """
    # Compute -E(x) for test sequences
    E_test = compute_energy(test_chains, params)  # E = -h.x - (1/2) x^T J x
    log_Q_test = -E_test - log_Z  # log Q(x) = -E(x) - log Z

    # Compute -E(x) for model samples
    E_model = compute_energy(model_chains, params)
    log_Q_model = -E_model - log_Z

    # D_KL(Q || P*) = <log Q(x)>_Q - <log Q(x)>_{P*}
    mean_log_Q_model = log_Q_model.mean().item()
    mean_log_Q_test = log_Q_test.mean().item()

    rev_kl = mean_log_Q_model - mean_log_Q_test

    # Standard errors
    n_model = len(model_chains)
    n_test = len(test_chains)
    se_model = log_Q_model.std().item() / np.sqrt(n_model)
    se_test = log_Q_test.std().item() / np.sqrt(n_test)
    se_total = np.sqrt(se_model**2 + se_test**2)

    # Also report the negative entropy <-log Q>_Q = -mean_log_Q_model
    # and cross-entropy <-log Q>_{P*} = -mean_log_Q_test
    return {
        "rev_kl": rev_kl,
        "rev_kl_se": se_total,
        "mean_log_Q_model": mean_log_Q_model,
        "mean_log_Q_test": mean_log_Q_test,
        "neg_entropy_Q": -mean_log_Q_model,  # H(Q)
        "cross_entropy": -mean_log_Q_test,    # H(P*, Q)
        "mean_energy_model": E_model.mean().item(),
        "mean_energy_test": E_test.mean().item(),
        "log_Z": log_Z,
    }


def run_for_gamma(
    gamma: float,
    params: dict[str, torch.Tensor],
    test_chains: torch.Tensor,
    L: int,
    q: int,
    ais_n_chains: int = AIS_N_CHAINS,
    ais_n_steps: int = AIS_N_STEPS,
    ais_nsweeps: int = AIS_NSWEEPS,
    ais_n_repeats: int = AIS_N_REPEATS,
    sample_n_chains: int = SAMPLE_N_CHAINS,
    device: torch.device = DEVICE,
    dtype: torch.dtype = DTYPE,
) -> dict:
    """Run full reverse KL computation for one gamma value."""
    tag = f"l2_{gamma:.4f}"

    # 1. Estimate log Z via AIS (multiple runs for error bars)
    print(f"  [{tag}] Running AIS ({ais_n_repeats} repeats, "
          f"{ais_n_chains} chains, {ais_n_steps} steps) ...", flush=True)

    log_Z_estimates = []
    ess_values = []
    t0 = time.time()

    for rep in range(ais_n_repeats):
        log_Z_est, ess = ais_log_Z(
            params, ais_n_chains, ais_n_steps, ais_nsweeps, device, dtype
        )
        log_Z_estimates.append(log_Z_est)
        ess_values.append(ess)
        print(f"    rep {rep+1}: log_Z = {log_Z_est:.2f}, ESS = {ess:.0f}", flush=True)

    log_Z_mean = np.mean(log_Z_estimates)
    log_Z_std = np.std(log_Z_estimates) if ais_n_repeats > 1 else 0.0
    dt_ais = time.time() - t0
    print(f"    log_Z = {log_Z_mean:.2f} +/- {log_Z_std:.2f}  "
          f"(mean ESS={np.mean(ess_values):.0f}, {dt_ais:.1f}s)", flush=True)

    # 2. Sample from the model
    print(f"  [{tag}] Sampling {sample_n_chains} sequences ...", flush=True)
    t0 = time.time()
    model_chains = sample_from_model(
        params, sample_n_chains,
        SAMPLE_NSWEEPS_EQUIL, SAMPLE_NSWEEPS_SAMPLE,
        device, dtype,
    )
    dt_sample = time.time() - t0
    print(f"    Sampling done ({dt_sample:.1f}s)", flush=True)

    # 3. Compute reverse KL
    result = compute_reverse_kl(
        params, test_chains, log_Z_mean, model_chains, device
    )
    result["gamma"] = gamma
    result["log_Z_std"] = log_Z_std
    result["ais_ess_mean"] = float(np.mean(ess_values))
    result["ais_n_repeats"] = ais_n_repeats
    result["log_Z_estimates"] = [float(x) for x in log_Z_estimates]
    N = L * q  # effective dimension
    result["rev_kl_per_N"] = result["rev_kl"] / N
    result["rev_kl_per_L"] = result["rev_kl"] / L

    print(f"  [{tag}] D_KL(Q||P*) = {result['rev_kl']:.2f}  "
          f"= {result['rev_kl_per_N']:.4f} per N  "
          f"= {result['rev_kl_per_L']:.4f} per L", flush=True)
    print(f"    H(Q) = {result['neg_entropy_Q']:.2f}, "
          f"H(P*,Q) = {result['cross_entropy']:.2f}", flush=True)

    return result


def plot_results(results: list[dict], L: int, q: int, family: str, outdir: Path):
    """Plot generalization gap, test log-likelihood, and log Z vs gamma."""
    outdir.mkdir(parents=True, exist_ok=True)

    gammas = np.array([r["gamma"] for r in results])
    N = L * q

    # Per-residue quantities (divided by L, more interpretable for proteins)
    gap_per_L = np.array([r["rev_kl"] / L for r in results])
    gap_se_per_L = np.array([r["rev_kl_se"] / L for r in results])
    test_ll_per_L = np.array([r["mean_log_Q_test"] / L for r in results])
    model_ll_per_L = np.array([r["mean_log_Q_model"] / L for r in results])
    neg_entropy_per_L = np.array([r["neg_entropy_Q"] / L for r in results])
    cross_entropy_per_L = np.array([r["cross_entropy"] / L for r in results])

    # Figure 1: Three-panel summary
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: Test log-likelihood (the main metric for model quality)
    ax = axes[0]
    ax.plot(gammas, test_ll_per_L, "o-", ms=5, color="C0",
            label=r"$\langle \log Q \rangle_{P^*} / L$ (test)")
    ax.plot(gammas, model_ll_per_L, "s--", ms=4, color="C3", alpha=0.6,
            label=r"$\langle \log Q \rangle_Q / L$ (model)")
    ax.set_xlabel(r"$\gamma$ (L2 regularization)", fontsize=12)
    ax.set_ylabel("log-likelihood per residue", fontsize=12)
    ax.set_title(f"Log-likelihood ({family})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: Generalization gap
    ax = axes[1]
    ax.errorbar(gammas, gap_per_L, yerr=gap_se_per_L,
                fmt="o-", ms=5, capsize=3, color="C1")
    ax.axhline(0, ls="--", color="gray", alpha=0.5)
    ax.set_xlabel(r"$\gamma$ (L2 regularization)", fontsize=12)
    ax.set_ylabel(r"$\Delta / L = (\langle \log Q \rangle_Q - \langle \log Q \rangle_{P^*}) / L$",
                  fontsize=11)
    ax.set_title("Generalization gap")
    ax.grid(alpha=0.3)

    # Panel 3: Entropy decomposition
    ax = axes[2]
    ax.plot(gammas, neg_entropy_per_L, "s-", ms=4, color="C4",
            label=r"$H(Q) / L$")
    ax.plot(gammas, cross_entropy_per_L, "^-", ms=4, color="C2",
            label=r"$H_{cross}(P^*, Q) / L$")
    ax.set_xlabel(r"$\gamma$ (L2 regularization)", fontsize=12)
    ax.set_ylabel("per-residue quantity", fontsize=12)
    ax.set_title("Entropy decomposition")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(f"DCA Boltzmann machine: {family} (L={L}, q={q})", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / f"reverse_kl_vs_gamma_{family}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"reverse_kl_vs_gamma_{family}.png", dpi=200, bbox_inches="tight")
    print(f"Saved plots to {outdir}/reverse_kl_vs_gamma_{family}.{{pdf,png}}")
    plt.close(fig)

    # Figure 2: log Z vs gamma (diagnostic)
    fig, ax = plt.subplots(figsize=(7, 5))
    log_Zs = np.array([r["log_Z"] for r in results])
    log_Z_stds = np.array([r["log_Z_std"] for r in results])
    ax.errorbar(gammas, log_Zs / L, yerr=log_Z_stds / L,
                fmt="o-", ms=5, capsize=3)
    ax.set_xlabel(r"$\gamma$ (L2 regularization)", fontsize=12)
    ax.set_ylabel(r"$\log Z / L$", fontsize=12)
    ax.set_title(f"Log partition function ({family})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"logZ_vs_gamma_{family}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"logZ_vs_gamma_{family}.png", dpi=200, bbox_inches="tight")
    print(f"Saved log Z plot to {outdir}/logZ_vs_gamma_{family}.{{pdf,png}}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compute reverse KL D_KL(Q || P*) for DCA models"
    )
    parser.add_argument("--family", default="PF00014",
                        help="Protein family name")
    parser.add_argument("--test-fraction", type=float, default=0.2,
                        help="Fraction of data to hold out as test set")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ais-chains", type=int, default=AIS_N_CHAINS)
    parser.add_argument("--ais-steps", type=int, default=AIS_N_STEPS)
    parser.add_argument("--ais-sweeps", type=int, default=AIS_NSWEEPS)
    parser.add_argument("--ais-repeats", type=int, default=AIS_N_REPEATS)
    parser.add_argument("--sample-chains", type=int, default=SAMPLE_N_CHAINS)
    parser.add_argument("--gammas", type=float, nargs="+", default=None,
                        help="Specific gamma values to evaluate (default: all available)")
    parser.add_argument("--params-dir", type=str, default=None,
                        help="Directory to search for params files (default: results/ and results/<family>/)")
    parser.add_argument("--outdir", default=str(ROOT / "data"))
    parser.add_argument("--figdir", default="figures")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Family: {args.family}")

    # ── Load data ────────────────────────────────────────────────────────
    data_path = DATA_DIR / f"{args.family}_full.fasta"
    print(f"Loading dataset from {data_path} ...")

    dataset = DatasetDCA(
        str(data_path),
        alphabet="protein",
        device=DEVICE,
        dtype=DTYPE,
    )
    L = dataset.get_num_residues()
    q = dataset.get_num_states()
    M = len(dataset)
    N = L * q
    print(f"L={L}, q={q}, N=L*q={N}, M={M}")

    # Train/test split
    all_chains = dataset.to_one_hot()  # (M, L, q)
    perm = torch.randperm(M, generator=torch.Generator().manual_seed(args.seed))
    n_test = int(M * args.test_fraction)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    test_chains = all_chains[test_idx].to(DEVICE, DTYPE)
    print(f"Train/test split: {len(train_idx)} / {n_test} sequences")

    # ── Find available models ────────────────────────────────────────────
    if args.params_dir is not None:
        search_dirs = [Path(args.params_dir)]
    else:
        search_dirs = [RESULTS_DIR, RESULTS_DIR / args.family]

    param_files = {}
    for search_dir in search_dirs:
        for pf in sorted(search_dir.glob("params_l2_*.dat")):
            gamma_str = pf.stem.replace("params_l2_", "")
            gamma = float(gamma_str)
            if args.gammas is None or gamma in args.gammas:
                param_files[gamma] = pf

    gammas_available = sorted(param_files.keys())
    print(f"Found models at gamma = {gammas_available}")

    if not gammas_available:
        print("No trained models found! Run train_and_analyze.py first.")
        return

    # ── Main computation loop ────────────────────────────────────────────
    all_results = []
    for gamma in gammas_available:
        print(f"\n{'='*60}")
        print(f"gamma = {gamma}")
        print(f"{'='*60}")

        params = load_params(
            str(param_files[gamma]), tokens="protein",
            device=DEVICE, dtype=DTYPE,
        )

        result = run_for_gamma(
            gamma=gamma,
            params=params,
            test_chains=test_chains,
            L=L, q=q,
            ais_n_chains=args.ais_chains,
            ais_n_steps=args.ais_steps,
            ais_nsweeps=args.ais_sweeps,
            ais_n_repeats=args.ais_repeats,
            sample_n_chains=args.sample_chains,
            device=DEVICE,
            dtype=DTYPE,
        )
        all_results.append(result)

    # ── Save results ─────────────────────────────────────────────────────
    out_file = outdir / f"reverse_kl_{args.family}.json"
    with open(out_file, "w") as f:
        json.dump({
            "family": args.family,
            "L": L,
            "q": q,
            "N": N,
            "M": M,
            "n_test": n_test,
            "test_fraction": args.test_fraction,
            "seed": args.seed,
            "ais_n_chains": args.ais_chains,
            "ais_n_steps": args.ais_steps,
            "ais_nsweeps": args.ais_sweeps,
            "ais_n_repeats": args.ais_repeats,
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # ── Print summary table ──────────────────────────────────────────────
    print(f"\n{'gamma':>8s}  {'Gap/L':>10s}  {'<logQ>_P*/L':>12s}  "
          f"{'H(Q)/L':>10s}  {'H(P*,Q)/L':>10s}  {'logZ/L':>10s}  {'ESS':>6s}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['gamma']:8.4f}  {r['rev_kl']/L:10.4f}  "
              f"{r['mean_log_Q_test']/L:12.4f}  "
              f"{r['neg_entropy_Q']/L:10.4f}  "
              f"{r['cross_entropy']/L:10.4f}  "
              f"{r['log_Z']/L:10.4f}  "
              f"{r['ais_ess_mean']:6.0f}")

    # ── Plot ─────────────────────────────────────────────────────────────
    plot_results(all_results, L, q, args.family, figdir)


if __name__ == "__main__":
    main()
