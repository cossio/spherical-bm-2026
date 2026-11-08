# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "torch",
#     "numpy",
# ]
# ///
"""Joint (sigma_prior, eta) sweep for the Bayesian GAN.

Manuscript prediction: stronger prior (smaller sigma_prior, equivalently
larger gamma) shifts the cold/warm optimum from warm (eta_* < 1) to
cold (eta_* > 1). Saves a single .npz with kl_pq[s, e, seed],
kl_qp[s, e, seed], modes[s, e, seed].
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bayes_gan import BGANConfig, train_bayes_gan  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--sigmas", type=float, nargs="+",
                   default=[0.3, 1.0, 3.0])
    p.add_argument("--etas", type=float, nargs="+",
                   default=[0.1, 0.3, 1.0, 3.0, 10.0])
    p.add_argument("--n_outer", type=int, default=3000)
    p.add_argument("--burn_in", type=int, default=1500)
    p.add_argument("--adam_warmup", type=int, default=500)
    p.add_argument("--sample_every", type=int, default=50)
    p.add_argument("--out_dir", type=str, default=str(__import__("pathlib").Path(__file__).resolve().parent.parent / "data"))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n_chains_g", type=int, default=None,
                   help="override BGANConfig.n_chains_g")
    p.add_argument("--out_name", type=str, default="sigma_eta_sweep.npz")
    p.add_argument("--target", choices=("ring", "gaussian"), default="ring",
                   help="ring (8-mode mixture) or gaussian (single mode)")
    args = p.parse_args()

    cfg = BGANConfig()
    from dataclasses import replace
    if args.n_chains_g is not None:
        cfg = replace(cfg, n_chains_g=args.n_chains_g)
    if args.target == "gaussian":
        cfg = replace(cfg, target_n_modes=1, target_radius=0.0,
                      target_sigma=0.5)
    n_s, n_e, n_seed = len(args.sigmas), len(args.etas), len(args.seeds)
    kl_pq = np.full((n_s, n_e, n_seed), np.nan)
    kl_qp = np.full((n_s, n_e, n_seed), np.nan)
    modes = np.full((n_s, n_e, n_seed), np.nan)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name

    for si, sigma in enumerate(args.sigmas):
        for ei, eta in enumerate(args.etas):
            for ki, seed in enumerate(args.seeds):
                t0 = time.time()
                out = train_bayes_gan(
                    cfg, eta=eta, n_outer=args.n_outer, seed=seed,
                    burn_in=args.burn_in, sample_every=args.sample_every,
                    adam_warmup=args.adam_warmup, device=args.device,
                    prior_sigma=sigma, verbose=False,
                )
                kl_pq[si, ei, ki] = float(out["kl_pq"])
                kl_qp[si, ei, ki] = float(out["kl_qp"])
                modes[si, ei, ki] = int(out["modes"])
                dt = time.time() - t0
                print(f"sigma={sigma:5.2f}  eta={eta:6.3f}  seed={seed}  "
                      f"KL(p|q)={float(out['kl_pq']):6.3f}  "
                      f"modes={int(out['modes'])}/8  ({dt:5.1f}s)",
                      flush=True)
                # checkpoint after every run
                np.savez(out_path,
                         sigmas=np.asarray(args.sigmas),
                         etas=np.asarray(args.etas),
                         seeds=np.asarray(args.seeds),
                         kl_pq=kl_pq, kl_qp=kl_qp, modes=modes)


if __name__ == "__main__":
    main()
