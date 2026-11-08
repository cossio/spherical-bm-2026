# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Verify all production artifacts and emit paired eight-dataset MAP curves."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import t
from run_map import N, M, K, ANCHORS, make_data, array_hash, sha, states

def interval(x):
 x=np.asarray(x);m=x.mean(axis=0);h=t.ppf(.975,len(x)-1)*x.std(axis=0,ddof=1)/np.sqrt(len(x))
 return {'mean':m.tolist(),'interval95':np.stack([m-h,m+h],axis=-1).tolist()}

def verify(folder, expected_seed, inputs):
 manifest=json.loads((folder/'manifest.json').read_text())
 assert manifest['complete'] and manifest['validation_passed']
 for name,digest in manifest['sha256'].items():assert sha(folder/name)==digest,(folder,name)
 summary=json.loads((folder/'summary.json').read_text());cfg=summary['config']
 assert cfg['seed']==expected_seed and cfg['N']==N and cfg['M']==M and cfg['K']==K
 assert cfg['runner_sha256']==sha(Path(__file__).parent/'run_map.py')
 assert cfg['source_input_sha256']==sha(inputs/f'seed_{expected_seed}.npz')
 assert cfg['source_provenance_sha256']==sha(inputs/f'seed_{expected_seed}.json')
 data,teacher,V,lpteacher=make_data(expected_seed)
 assert array_hash(data)==cfg['data_sha256']
 with np.load(folder/'data.npz') as z:assert np.array_equal(data,z['data'])
 with np.load(folder/'map_weights.npz') as z:gammas=z['gammas'];chosen=z['weights']
 with np.load(folder/'candidate_weights.npz') as z:allweights=z['weights'];allstarts=z['starts']
 records=json.loads((folder/'candidate_records.json').read_text())
 assert len(records)==len(allweights)==len(allstarts)
 assert len(gammas)==len(chosen)==len(summary['cells'])
 assert np.array_equal(gammas,np.array(cfg['gammas']))
 errs=[];Hs=states(M)
 for gi,(gamma,W,cell) in enumerate(zip(gammas,chosen,summary['cells'])):
  j=cell['best_record'];assert np.array_equal(W,allweights[j]);assert cell['gamma']==gamma
  contenders=[r for r in records if r['gamma_index']==gi]
  assert max(r['logpost'] for r in contenders)-cell['logpost']<=1.01e-10
  # Direct full joint enumeration uses exp(v^T W h), no log-cosh formulas.
  joint=V@W@Hs.T
  logz=logsumexp(joint)
  lp=logsumexp(joint,axis=1)-logz
  p=np.exp(lp)
  ids=((data>0)*(1<<np.arange(N))).sum(axis=1)
  objective=float(lp[ids].sum()-.5*gamma*np.sum(W*W))
  forward=float(np.exp(lpteacher)@(lpteacher-lp)/N)
  reverse=float(p@(lp-lpteacher)/N)
  error=max(abs(objective-cell['logpost']),abs(forward-cell['forward']),abs(reverse-cell['reverse']))
  assert error<1e-8,(expected_seed,gamma,error)
  errs.append(error)
 logpost=np.array([c['logpost'] for c in summary['cells']])
 norm2=np.sum(chosen*chosen,axis=(1,2))
 loglik=logpost+gammas*norm2/2
 candidate_norm2=np.sum(allweights*allweights,axis=(1,2))
 candidate_gammas=np.array([r['gamma'] for r in records])
 candidate_loglik=np.array([r['logpost'] for r in records])+candidate_gammas*candidate_norm2/2
 candidate_envelope=(candidate_loglik[:,None]-candidate_norm2[:,None]*gammas[None,:]/2).max(axis=0)
 envelope_gap=float(np.max(candidate_envelope-logpost))
 monotone_objective_violation=float(max(0.,np.diff(logpost).max()))
 monotone_norm_violation=float(max(0.,np.diff(norm2).max()))
 assert envelope_gap<=1e-7,('cross-grid candidate envelope failure',expected_seed,envelope_gap)
 assert monotone_objective_violation<=1e-7,('objective monotonicity failure',expected_seed,monotone_objective_violation)
 assert monotone_norm_violation<=1e-5,('weight norm monotonicity requires review',expected_seed,monotone_norm_violation)
 return summary,{'seed':expected_seed,'files_verified':len(manifest['sha256']),
    'candidate_count':len(records),'full_joint_max_error':max(errs),'manifest_sha256':sha(folder/'manifest.json'),
    'cross_grid_candidate_envelope_gap':envelope_gap,'monotone_objective_violation':monotone_objective_violation,
    'monotone_norm_violation':monotone_norm_violation,'final_continuation_gain':summary['continuation'][-1]['max_logpost_gain']}

def main(args):
 out=args.results/'curve.json'
 if out.exists():raise FileExistsError(out)
 summaries=[];audits=[]
 exact=ANCHORS+[.01,.1,3.4,10.]
 expected_grid=np.array(sorted(set([g for g in np.geomspace(1e-4,1e4,65) if not any(np.isclose(g,a,rtol=1e-12,atol=0) for a in exact)]+exact)))
 for seed in range(8):
  s,a=verify(args.results/f'seed_{seed}',seed,args.inputs);summaries.append(s);audits.append(a)
  cfg=s['config']
  assert cfg['random_starts']==96 and cfg['source_starts']==24 and cfg['maxiter']==5000 and cfg['continuation_roundtrips']==4,('not production protocol',seed)
  assert np.array_equal(np.array(cfg['gammas']),expected_grid),('not production grid',seed)
  print(f'verified seed {seed} candidates {a["candidate_count"]} full joint error {a["full_joint_max_error"]:.3g}',flush=True)
 gammas=np.array(summaries[0]['config']['gammas'])
 for s in summaries:assert np.array_equal(gammas,s['config']['gammas'])
 reverse=np.array([[c['reverse'] for c in s['cells']] for s in summaries])
 forward=np.array([[c['forward'] for c in s['cells']] for s in summaries])
 rev=interval(reverse);fwd=interval(forward)
 uniform=summaries[0]['uniform_reverse']
 compact=json.loads(args.finite_temperature.read_text())
 identities=json.loads((Path(__file__).resolve().parents[1]/'data/eta.json').read_text())['data_sha256_by_seed']
 historical={'data_sha256_by_seed':identities,'groups':{
   str(row['gamma']):{'runs':[{'dataset_seed':seed,'typical_kl':{'reverse':{'mean':value}}} for seed,value in enumerate(row['reverse_kl']['dataset_values'])]}
   for row in compact['points']}}
 for s in summaries:assert historical['data_sha256_by_seed'][str(s['config']['seed'])]==s['config']['data_sha256']
 contrasts={};finite=[]
 for gamma in ANCHORS:
  gi=np.flatnonzero(np.isclose(gammas,gamma,rtol=1e-13,atol=0));assert len(gi)==1;gi=int(gi[0])
  group=historical['groups'][str(gamma)]
  hot=np.array([r['typical_kl']['reverse']['mean'] for r in sorted(group['runs'],key=lambda r:r['dataset_seed'])])
  contrasts[str(gamma)]={'finite_eta_minus_map':interval(hot-reverse[:,gi]),'map_minus_uniform':interval(reverse[:,gi]-uniform)}
  finite.append({'gamma':gamma,'reverse_by_dataset':hot.tolist(),'reverse':interval(hot)})
 selected=int(np.argmin(reverse.mean(axis=0)))
 adjacent=np.diff(reverse,axis=1)
 decrease_after_min=adjacent[:,selected:].mean(axis=0)
 selected_tail_max=max(0.,float(decrease_after_min.max())) if decrease_after_min.size else 0.
 selected_tail_min=min(0.,float(decrease_after_min.min())) if decrease_after_min.size else 0.
 result={'schema':'rbm_map_curve_20261107_v1','gamma':gammas.tolist(),'dataset_seeds':list(range(8)),
   'reverse_kl_mean':rev['mean'],'reverse_kl_interval95':rev['interval95'],
   'forward_kl_mean':fwd['mean'],'forward_kl_interval95':fwd['interval95'],
   'reverse_kl_by_dataset':reverse.tolist(),'forward_kl_by_dataset':forward.tolist(),
   'data_sha256_by_seed':{str(s['config']['seed']):s['config']['data_sha256'] for s in summaries},
   'uniform_reverse':uniform,'uniform_forward':summaries[0]['uniform_forward'],
   'finite_temperature':finite,'anchor_contrasts':contrasts,
   'descriptive_shape':{'selected_mean_minimum_gamma':float(gammas[selected]),'selected_mean_minimum_reverse':float(reverse.mean(axis=0)[selected]),
      'largest_negative_adjacent_change_after_selected_minimum':selected_tail_min,
      'largest_positive_adjacent_change_after_selected_minimum':selected_tail_max,
      'largest_mean_reverse_minus_uniform':float((reverse.mean(axis=0)-uniform).max()),
      'largest_individual_reverse_minus_uniform':float((reverse-uniform).max()),
      'adjacent_difference_mean':adjacent.mean(axis=0).tolist(),
      'monotone_after_minimum_tolerance_1e-8':bool(selected_tail_min>=-1e-8)},
   'validation':{'passed':True,'datasets':audits,
      'max_gradient_norm':max(s['validation']['max_gradient_norm'] for s in summaries),
      'min_hessian_eigenvalue':min(s['validation']['min_hessian_eigenvalue'] for s in summaries),
      'max_full_joint_error':max(a['full_joint_max_error'] for a in audits),
      'max_cross_grid_candidate_envelope_gap':max(a['cross_grid_candidate_envelope_gap'] for a in audits),
      'max_monotone_objective_violation':max(a['monotone_objective_violation'] for a in audits),
      'max_monotone_norm_violation':max(a['monotone_norm_violation'] for a in audits),
      'max_final_continuation_gain':max(a['final_continuation_gain'] for a in audits),
      'total_optimized_candidates':sum(a['candidate_count'] for a in audits)},
   'global_uniform_certified_by_dataset':[[c['uniform_global_certificate'] for c in s['cells']] for s in summaries],
   'finite_temperature_report_sha256':sha(args.finite_temperature),'collector_sha256':sha(__file__),
   'status':'best-found MAP; global optimization unproved except flagged uniform sufficient-bound cells',
   'interval_scope':'pointwise descriptive Student-t across same eight datasets; no multiplicity correction'}
 out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 report=['# RBM MAP regularization results','','All eight production datasets passed artifact, dataset-identity and independent full-joint enumeration checks.',
    f"Saved {result['validation']['total_optimized_candidates']} optimized starts across {len(gammas)} gamma values per dataset.",
    f"Maximum gradient norm: {result['validation']['max_gradient_norm']:.3g}; minimum numerical Hessian eigenvalue: {result['validation']['min_hessian_eigenvalue']:.3g}.",
    f"Maximum full-joint objective/KL discrepancy: {result['validation']['max_full_joint_error']:.3g}.",'',
    f"Largest cross-grid candidate-envelope gap: {result['validation']['max_cross_grid_candidate_envelope_gap']:.3g}; largest objective monotonicity violation: {result['validation']['max_monotone_objective_violation']:.3g}.",
    f"Largest final continuation-roundtrip gain: {result['validation']['max_final_continuation_gain']:.3g}.",'',
    f"Selected mean reverse-KL minimum: {reverse.mean(axis=0)[selected]:.9f} nats/spin at gamma {gammas[selected]:.9g}.",
    f"Uniform reverse KL: {uniform:.9f}. Largest mean excess above uniform: {(reverse.mean(axis=0)-uniform).max():.3g}.",
    f"Largest downward adjacent change after the selected mean minimum: {selected_tail_min:.3g}.",
    'These are grid descriptions, not proofs of a globally optimal fit or continuous-parameter shape.','',
    '| Gamma | Mean best-found MAP reverse KL | Finite eta=.02 minus MAP, paired 95% interval |','|---:|---:|---:|']
 for gamma in ANCHORS:
  gi=int(np.flatnonzero(np.isclose(gammas,gamma,rtol=1e-13,atol=0))[0]);c=contrasts[str(gamma)]['finite_eta_minus_map'];lo,hi=c['interval95']
  report.append(f"| {gamma:.8g} | {reverse.mean(axis=0)[gi]:.8f} | {c['mean']:.8f} [{lo:.8f}, {hi:.8f}] |")
 report+=['','Both curves use the same eight datasets. The finite-temperature curve is the existing posterior mean individual-model reverse KL; the MAP curve is one objective-selected fitted model per dataset. Dataset intervals are pointwise and descriptive. Existing finite-temperature sampling limitations are unchanged.','',
 'For gamma >= lambda_max(D^T D), the uniform weights are certified global maximizers using log cosh(x) <= x^2/2 and log Z(W) >= (N+M)*log(2). All other points remain best-found local optima, irrespective of passed gradient/Hessian checks.']
 (args.results/'REPORT.md').write_text('\n'.join(report)+'\n')
 print(json.dumps(result['descriptive_shape'],indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=Path(__file__).parent/'results');ap.add_argument('--inputs',type=Path,default=Path(__file__).parent/'inputs')
 ap.add_argument('--finite-temperature',type=Path,default=Path(__file__).resolve().parents[1]/'data/gamma.json')
 main(ap.parse_args())
