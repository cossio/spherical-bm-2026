# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Check release integrity, dataset intervals, and exact chosen-weight losses.

This never optimizes weights or samples posterior chains. The full candidate
search archive is not included, so candidate optimality is outside this audit.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('map_runner',ROOT/'map/run_map.py')
map_runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(map_runner)
def read(p):return json.loads(p.read_text())
def same(a,b,tol=2e-12):np.testing.assert_allclose(a,b,rtol=0,atol=tol)
def check_summary(row):
    a=np.asarray(row['dataset_values']);assert a.shape==(8,) and np.isfinite(a).all()
    mean=a.mean();half=t.ppf(.975,7)*a.std(ddof=1)/np.sqrt(8)
    same(row['mean'],mean);same(row['interval_95'],[mean-half,mean+half])
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-integrity',action='store_true',help='For development before the checksum manifest exists')
    args=parser.parse_args()
    if not args.skip_integrity:
        for filename,digest in read(ROOT/'data/checksums.json').items():
            assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==digest,filename
    eta=read(ROOT/'data/eta.json');assert len(eta['curves'])==2 and not eta['sampling_failures']
    for c in eta['curves']:
        assert len(c['points'])==8
        for point in c['points']:
            for direction in ['forward','reverse']:check_summary(point[direction])
    gamma=read(ROOT/'data/gamma.json');assert len(gamma['points'])==8 and not gamma['flagged_points']
    for point in gamma['points']:
        for key in ['reverse_kl','covariance_lambda1_over_N','teacher_q1']:check_summary(point[key])
    curve=read(ROOT/'map/curve.json'); assert len(curve['gamma'])==70
    same(np.asarray(curve['reverse_kl_by_dataset']).mean(axis=0),curve['reverse_kl_mean'])
    errors=[]
    for seed in range(8):
        data,teacher,V,lpstar=map_runner.make_data(seed)
        with np.load(ROOT/f'map/inputs/seed_{seed}.npz') as z:same(z['data'],data)
        assert map_runner.array_hash(data)==eta['data_sha256_by_seed'][str(seed)]==curve['data_sha256_by_seed'][str(seed)]
        summary=read(ROOT/f'map/results/seed_{seed}/summary.json')
        with np.load(ROOT/f'map/results/seed_{seed}/map_weights.npz') as z:
            same(z['gammas'],curve['gamma']);weights=z['weights']
        for j,(g,W) in enumerate(zip(curve['gamma'],weights)):
            lp=map_runner.visible_logp(W,V);p=np.exp(lp)
            reverse=float(p@(lp-lpstar)/16)
            same(reverse,curve['reverse_kl_by_dataset'][seed][j],2e-10)
            target=map_runner.Target(data,g);objective,gradient=target(W.ravel())
            row=summary['cells'][j]
            same(objective,-row['logpost'],2e-8)
            same(np.linalg.norm(gradient),row['gradient_norm'],2e-10)
            assert np.linalg.norm(gradient)<1e-4
            errors.append(abs(reverse-curve['reverse_kl_by_dataset'][seed][j]))
        print(f'MAP seed {seed}: all 70 exact losses and stationarity checks pass',flush=True)
    # Reevaluate held-fixed beta weights independently at both grid endpoints,
    # beta=1 and each displayed mean forward-KL minimum, for all datasets.
    with np.load(ROOT/'data/beta.npz') as z:a={k:z[k] for k in z.files}
    with np.load(ROOT/'data/beta_weights.npz') as z:weights=z['weights'];datasets=z['data']
    beta_count=0
    for seed in range(8):
        data,teacher,V,lpstar=map_runner.make_data(seed);same(data,datasets[seed]);pstar=np.exp(lpstar)
        _,evec=np.linalg.eigh(data.T@data/16);v=evec[:,-1]
        for gi in range(2):
            selected=int(a['forward_kl'][:,gi].mean(axis=0).argmin())
            indices=sorted({0,len(a['betas'])-1,int(np.flatnonzero(a['betas']==1)[0]),selected})
            for bi in indices:
                W=weights[seed,gi]*a['betas'][bi];lp=map_runner.visible_logp(W,V);p=np.exp(lp)
                same(pstar@(lpstar-lp)/16,a['forward_kl'][seed,gi,bi],2e-10)
                same(p@(lp-lpstar)/16,a['reverse_kl'][seed,gi,bi],2e-10)
                same(p@((V@v)**2)/16,a['q_data1'][seed,gi,bi],2e-10)
                beta_count+=1
    with np.load(ROOT/'data/nu.npz') as z:
        assert z['time'].shape==(499,);same(z['dataset_seeds'],np.arange(8,16))
        n_curves=0
        for key in z.files:
            if key.endswith('__replicas'):
                assert z[key].shape[0]==8 and z[key].shape[-1]==499
                same(z[key].mean(axis=1),z[key[:-10]])
                n_curves+=1
        assert n_curves==20
    print(json.dumps({'map_fits_checked':560,'max_map_loss_error':max(errors),'beta_exact_cells_checked':beta_count,'nu_curves':n_curves,'eta_precision_flags':len(eta['precision_flags']),'status':'passed'},indent=2))
if __name__=='__main__':main()
