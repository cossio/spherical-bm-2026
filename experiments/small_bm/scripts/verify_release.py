# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy>=2.4", "scipy>=1.17", "pymupdf>=1.26"]
# ///
"""Check released observations, uncertainty, MAP replay, and rendered figures."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from scipy.special import logsumexp
from scipy.stats import t
import pymupdf

ROOT=Path(__file__).resolve().parents[1]

def equal(actual,expected):
    np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-12,equal_nan=True)

def mean_ci(values,axis,probability,mean,halfwidth):
    n=values.shape[axis]
    equal(values.mean(axis=axis),mean)
    equal(t.ppf((1+probability)/2,n-1)*values.std(axis=axis,ddof=1)/np.sqrt(n),halfwidth)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map-replay',type=Path)
    p.add_argument('--gamma-replay',type=Path)
    p.add_argument('--figure-dir',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    checks={}
    for relative,digest in json.loads((ROOT/'data/sha256.json').read_text()).items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest, relative
    checks['released_file_hashes']=True
    with np.load(ROOT/'data/beta.npz') as d:
        mean_ci(d['metrics'],0,.99,d['mean'],d['ci99'])
        f=d['spins'][:,np.triu_indices(8,1)[0]]*d['spins'][:,np.triu_indices(8,1)[1]]
        logits=d['couplings']@f.T
        for bi,beta in enumerate(d['betas']):
            lq=beta*logits-logsumexp(beta*logits,axis=1,keepdims=True)
            equal(np.sum(np.exp(d['teacher_logp'])*(d['teacher_logp']-lq),axis=-1),d['metrics'][:,bi,0,0])
            equal(np.sum(np.exp(lq)*(lq-d['teacher_logp']),axis=-1),d['metrics'][:,bi,0,1])
        assert d['map_gradient_residual'].max()<3e-7
        checks['beta_exact_KL_and_dataset_intervals']=True
    with np.load(ROOT/'data/posterior.npz') as d:
        q=d['predictive']; teacher=d['teacher']
        equal(q.sum(axis=-1),1)
        equal(np.sum(teacher*(np.log(teacher)-np.log(q)),axis=-1),d['metrics'][... ,0])
        equal(np.sum(q*(np.log(q)-np.log(teacher)),axis=-1),d['metrics'][... ,1])
        mean_ci(d['metrics'],2,.99,d['mean'],d['ci99_halfwidth'])
        assert d['rank_split_rhat'].max()<1.005
        assert d['bulk_ess'].min()>2300 and not d['divergences'].any()
        checks['posterior_exact_predictive_KL_intervals_and_diagnostics']=True
    with np.load(ROOT/'data/dynamics.npz') as d:
        equal(d['metrics'].mean(axis=1),d['dataset_metrics'])
        mean_ci(d['dataset_metrics'],0,.95,d['mean'],d['ci95_halfwidth'])
        assert np.isnan(d['mean'][:,0,1]).all()  # Degenerate zero-coupling eigenvector.
        equal(d['map_couplings'],d['map_couplings'].swapaxes(-1,-2))
        checks['dynamics_replica_aggregation_and_dataset_intervals']=True
    with np.load(ROOT/'data/gamma_final.npz') as d:
        summary=json.loads((ROOT/'data/gamma_final.json').read_text())
        for cell in summary['cells']:
            ei=np.flatnonzero(d['eta']==cell['eta'])[0]; gi=np.flatnonzero(d['gamma']==cell['gamma'])[0]
            for mi,name in enumerate(d['metric_names']):
                if name in cell['statistics']:
                    equal(d['mean'][ei,gi,mi],cell['statistics'][name]['mean'])
                    equal(d['mc_se'][ei,gi,mi],cell['statistics'][name]['mc_se'])
        critical=np.where(d['method']=='adjusted_hmc',t.ppf(.975,7),1.96)
        equal(d['mc_ci95_halfwidth'],critical[...,None]*d['mc_se'])
        checks['gamma_source_summaries_and_method_specific_intervals']=True
        if args.gamma_replay:
            with np.load(args.gamma_replay/'gamma_final.npz') as replay:
                assert set(d.files)==set(replay.files)
                for k in d:
                    if d[k].dtype.kind in 'biufc': equal(d[k],replay[k])
                    else: np.testing.assert_array_equal(d[k],replay[k])
            checks['all_gamma_summary_replay_arrays']=True
    with np.load(ROOT/'data/bm_map_sweep.npz') as d:
        curve=json.loads((ROOT/'data/curve.json').read_text())
        equal(d['gamma'],curve['gamma']);equal(d['reverse_kl'],curve['reverse_kl_mean'])
        i=np.argmin(d['reverse_kl']);equal(d['gamma'][i],.006603955045704656)
        equal(d['reverse_kl'][i],.2845659548248616)
        if args.map_replay:
            with np.load(args.map_replay/'bm_map_sweep.npz') as replay:
                assert set(d.files)==set(replay.files)
                for k in d:
                    if d[k].dtype.kind in 'biufc': equal(d[k],replay[k])
                    else: np.testing.assert_array_equal(d[k],replay[k])
            checks['all_MAP_replay_arrays']=True
    if args.figure_dir:
        comparisons={}
        for published in sorted((ROOT/'figures').glob('*.pdf')):
            regenerated=args.figure_dir/published.name
            with pymupdf.open(published) as a,pymupdf.open(regenerated) as b:
                assert len(a)==len(b)==1
                ap=a[0].get_pixmap(dpi=144);bp=b[0].get_pixmap(dpi=144)
                assert (ap.width,ap.height)==(bp.width,bp.height)
                av=np.frombuffer(ap.samples,dtype=np.uint8);bv=np.frombuffer(bp.samples,dtype=np.uint8)
                delta=np.abs(av.astype(np.int16)-bv.astype(np.int16))
                comparisons[published.stem]={'identical_pixels':bool(np.array_equal(av,bv)),'max_channel_difference':int(delta.max()),'mean_channel_difference':float(delta.mean())}
                assert np.array_equal(av,bv),comparisons[published.stem]
        checks['rendered_figures']=comparisons
    result={'passed':True,'checks':checks}
    print(json.dumps(result,indent=2))
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
