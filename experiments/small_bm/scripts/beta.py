# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy>=2.4", "scipy>=1.17"]
# ///
"""Independent exact-state reconstruction of the Appendix generation-temperature BM.

Reconstructed prior: exp[-N gamma Tr(J²)/4], hence precision N gamma
for the independent entries j=(J_ij, i<j). MAP uses the SUM likelihood.
All KL values are total nats. Teacher and datasets are never conditioned.
The misaligned control transforms only the student, keeping the teacher fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp
from scipy.stats import t


def configurations(n):
    return (2 * ((np.arange(2**n)[:, None] >> np.arange(n)) & 1) - 1).astype(float)


def fit_map(features, counts, gamma_precision):
    k = counts.sum()
    sufficient = counts @ features

    def loss(j):
        logits = features @ j
        z = logsumexp(logits)
        p = np.exp(logits - z)
        return (k * z - sufficient @ j + gamma_precision * (j @ j) / 2,
                k * (p @ features) - sufficient + gamma_precision * j)

    fit = minimize(loss, np.zeros(features.shape[1]), jac=True, method="L-BFGS-B",
                   options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 2000,
                            "maxls": 40})
    residual = float(np.max(np.abs(loss(fit.x)[1])))
    if residual > 2e-5:
        raise RuntimeError(f"MAP did not converge: {fit.message}; residual={residual}")
    return fit.x, residual


def distribution_metrics(logq, logp, spins, direction):
    q, p = np.exp(logq), np.exp(logp)
    covariance = np.einsum("bs,si,sj->bij", q, spins, spins, optimize=True)
    eigvals = np.linalg.eigvalsh(covariance)
    return np.column_stack((
        np.sum(p * (logp - logq), axis=-1),
        np.sum(q * (logq - logp), axis=-1),
        q @ (spins @ direction)**2 / spins.shape[1],
        (eigvals[:, -1] - eigvals[:, -2]) / spins.shape[1],
    ))


def run(args):
    n, k, omega, gamma = 8, 4, 1.5, 6.0
    spins = configurations(n)
    rows, cols = np.triu_indices(n, 1)
    features = spins[:, rows] * spins[:, cols]
    teacher_logits = omega * spins.sum(axis=1)**2 / (2*n)
    logp = teacher_logits - logsumexp(teacher_logits)
    p = np.exp(logp)
    betas = np.unique(np.r_[np.geomspace(.2, 8, 65), 1., 3.38])
    gauge = np.r_[np.ones(n//2), -np.ones(n//2)]
    gauge_pairs = gauge[rows] * gauge[cols]
    data = np.zeros((args.datasets, k), dtype=int)
    weights = np.zeros((args.datasets, len(rows)))
    metrics = np.zeros((args.datasets, len(betas), 2, 4))
    residuals = np.zeros(args.datasets)
    teacher_variance = np.zeros(args.datasets)
    alignments = np.zeros((args.datasets, 2))
    spectral_errors = np.zeros(args.datasets)
    for seed in range(args.datasets):
        rng = np.random.default_rng(args.seed + seed)
        data[seed] = rng.choice(len(spins), size=k, p=p)
        counts = np.bincount(data[seed], minlength=len(spins))
        j, residuals[seed] = fit_map(features, counts, n*gamma)
        weights[seed] = j
        mat = np.zeros((n,n))
        mat[rows,cols] = mat[cols,rows] = j
        eigenvalues, eigenvectors = np.linalg.eigh(mat)
        v = eigenvectors[:,-1]
        teacher_variance[seed] = p @ (spins @ v)**2/n
        alignments[seed] = [(v.sum()/np.sqrt(n))**2,
                            ((gauge*v).sum()/np.sqrt(n))**2]
        spectral_errors[seed] = np.max(np.abs(eigenvalues-np.linalg.eigvalsh(gauge[:,None]*mat*gauge[None,:])))
        for control, jj in enumerate((j, gauge_pairs*j)):
            logits = betas[:, None]*(features @ jj)
            logq = logits - logsumexp(logits, axis=1, keepdims=True)
            metrics[seed,:,control] = distribution_metrics(logq, logp, spins,
                                                           v if control==0 else gauge*v)
        if (seed+1) % 25 == 0:
            print(f"beta: {seed+1}/{args.datasets} datasets", flush=True)

    logits = weights @ features.T
    def mean_kl(beta, reverse):
        logq = beta*logits-logsumexp(beta*logits,axis=1,keepdims=True)
        if reverse:
            return np.sum(np.exp(logq)*(logq-logp),axis=1).mean()
        return np.sum(p*(logp-logq),axis=1).mean()
    optima = [minimize_scalar(lambda b: mean_kl(b, r), bounds=(.2,8),
                             method="bounded").x for r in (False,True)]
    critical = float(t.ppf(.995, args.datasets-1)) if args.datasets>1 else None
    mean = metrics.mean(axis=0)
    ci = (critical*metrics.std(axis=0,ddof=1)/np.sqrt(args.datasets)
          if args.datasets>1 else np.zeros_like(mean))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output, betas=betas, metrics=metrics,
                        mean=mean, ci99=ci, datasets=data, couplings=weights,
                        map_gradient_residual=residuals,
                        teacher_variance=teacher_variance, alignments=alignments,
                        gauge=gauge, spectral_errors=spectral_errors,
                        continuous_optima=optima, spins=spins, teacher_logp=logp)
    metadata = {
        "experiment":"generation_temperature", "N":n,"K":k,"omega":omega,
        "gamma":gamma,"datasets":args.datasets,"seed_base":args.seed,
        "dataset_rng":"numpy.default_rng(seed_base + dataset_index)",
        "objective":"K logZ - sum_data pair_products dot j + N gamma ||j||²/2",
        "assumptions":["MAP fit to independent upper-triangle couplings",
                       "main-paper Gaussian prior normalization transfers to BM",
                       "fixed balanced gauge (four plus, four minus) applied to student only",
                       "independent reproducible dataset seeds"],
        "metric_axis":["forward_kl","reverse_kl","mode_variance_per_N","covariance_gap_per_N"],
        "metrics_shape":"[dataset,beta,control,metric]",
        "controls":["trained","misaligned"],"kl_units":"total nats",
        "ci":"pointwise 99% Student-t across iid datasets",
        "continuous_mean_kl_beta_optima":dict(zip(["forward","reverse"],optima)),
        "grid_mean_kl_beta_optima":dict(zip(["forward","reverse"],
            [float(betas[mean[:,0,j].argmin()]) for j in (0,1)])),
        "max_map_gradient_residual":float(residuals.max()),
        "max_gauge_spectrum_error":float(spectral_errors.max()),
        "mean_teacher_alignment_squared":alignments.mean(axis=0).tolist(),
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy":np.__version__,"scipy":scipy.__version__,"python":platform.python_version(),
        "output_sha256":hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(metadata,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets",type=int,default=200)
    parser.add_argument("--seed",type=int,default=1000)
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1]/"results/beta.npz")
    run(parser.parse_args())
