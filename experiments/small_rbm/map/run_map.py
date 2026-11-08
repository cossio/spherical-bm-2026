# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Exact finite RBM best-found MAP continuation; all choices use training objective."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.linalg import hadamard
from scipy.optimize import minimize
from scipy.special import logsumexp

N, M, K = 16, 3, 16
ANCHORS = [1., 3., 6., 10.92600861117378, 30., 100., 1000., 10000.]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()

def states(n):
    return 2.*((np.arange(2**n)[:, None] >> np.arange(n)) & 1)-1.

def visible_logp(W, V):
    logits = np.logaddexp(V@W, -V@W).sum(axis=1)
    return logits-logsumexp(logits)

def make_data(seed):
    V = states(N)
    teacher = hadamard(N)[:, [1, 2]]*np.array([1., .5])
    lp = visible_logp(teacher, V)
    ids = np.random.default_rng(1000+seed).choice(2**N, size=K, p=np.exp(lp))
    return V[ids].copy(), teacher, V, lp

class Target:
    def __init__(self, data, gamma):
        self.data, self.gamma, self.H = data, float(gamma), states(M)
    def __call__(self, x):
        W = np.asarray(x).reshape(N, M)
        fields = W@self.H.T
        lh = np.logaddexp(fields, -fields).sum(axis=0)
        lz = logsumexp(lh)
        ph = np.exp(lh-lz)
        positive = self.data@W
        lp = np.logaddexp(positive, -positive).sum()-K*lz-self.gamma*.5*np.sum(W*W)
        grad = self.data.T@np.tanh(positive)-K*(np.tanh(fields)*ph)@self.H-self.gamma*W
        return -float(lp), -grad.ravel()

def hessian(target, x):
    out = np.empty((x.size, x.size))
    for j in range(x.size):
        delta = 2e-5*max(1., abs(x[j]))
        plus, minus = x.copy(), x.copy()
        plus[j] += delta
        minus[j] -= delta
        out[:, j] = (target(plus)[1]-target(minus)[1])/(2*delta)
    return .5*(out+out.T)

def canonical(W):
    A = W.copy()
    for j in range(M):
        if A[np.argmax(np.abs(A[:, j])), j] < 0:
            A[:, j] *= -1
    return A[:, sorted(range(M), key=lambda j: tuple(np.round(A[:, j], 7)))]

def diverse_indices(weights, values, count):
    selected, canons = [], []
    for j in np.argsort(values):
        c = canonical(weights[j])
        if not any(np.linalg.norm(c-a)<1e-3 for a in canons):
            selected.append(int(j)); canons.append(c)
        if len(selected) >= count:
            break
    return selected

def atomic_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)

def run(args):
    start_time = time.monotonic()
    out = args.outdir / f'seed_{args.seed}'
    out.mkdir(parents=True, exist_ok=False)
    data, teacher, V, lp_teacher = make_data(args.seed)
    source_path = args.inputs / f'seed_{args.seed}.npz'
    source_info_path = args.inputs / f'seed_{args.seed}.json'
    with np.load(source_path) as z:
        assert np.array_equal(data, z['data']), 'dataset mismatch'
        source_W = z['weights'].copy()
    source_info = json.loads(source_info_path.read_text())
    exact = ANCHORS+[.01,.1,3.4,10.]
    default_grid = [g for g in np.geomspace(1e-4,1e4,65) if not any(np.isclose(g,a,rtol=1e-12,atol=0) for a in exact)] + exact
    gammas = np.array(sorted(set(args.gammas or default_grid)))
    config = {'schema':'rbm_map_20261107_v1', 'seed':args.seed, 'N':N, 'M':M, 'K':K,
        'gammas':gammas.tolist(), 'data_rng_seed':1000+args.seed, 'data_sha256':array_hash(data),
        'source_input_sha256':sha(source_path), 'source_provenance_sha256':sha(source_info_path),
        'runner_sha256':sha(__file__), 'python':platform.python_version(),
        'packages':{k:importlib.metadata.version(k) for k in ['numpy','scipy']},
        'created_utc':datetime.now(timezone.utc).isoformat(), 'random_starts':args.random_starts,
        'source_starts':args.source_starts, 'maxiter':args.maxiter, 'continuation_roundtrips':args.roundtrips,
        'objective':'sum_k log P_W(v_k) - gamma ||W||_F^2/2', 'kl_units':'nats per visible spin',
        'selection':'maximum training objective; gradient norm breaks numerical ties within 1e-10; no test KL selection',
        'status':'best found local optimum; no general global optimality guarantee'}
    atomic_json(out/'config.json', config)
    np.savez_compressed(out/'data.npz', data=data, teacher=teacher)
    rng = np.random.default_rng([20261107, args.seed])
    pools = [[] for _ in gammas]
    records, weights, starts = [], [], []
    def optimize(gi, W0, provenance):
        target = Target(data, gammas[gi])
        initial_obj = target(W0)[0]
        result = minimize(target, np.asarray(W0).ravel(), jac=True, method='L-BFGS-B',
            options={'maxiter':args.maxiter,'ftol':2e-15,'gtol':1e-8,'maxls':50,'maxcor':30})
        W = result.x.reshape(N,M)
        objective, grad = target(W)
        retained_initial = not np.isfinite(objective) or objective > initial_obj+1e-10
        if retained_initial:
            W = np.asarray(W0).copy(); objective, grad = target(W)
        record = {'gamma_index':int(gi),'gamma':float(gammas[gi]),'provenance':provenance,
            'logpost':-objective,'initial_logpost':-initial_obj,'gradient_max_abs':float(abs(grad).max()),
            'gradient_norm':float(np.linalg.norm(grad)), 'success':bool(result.success),
            'message':str(result.message),'iterations':int(result.nit),'nfev':int(result.nfev),
            'retained_initial':retained_initial}
        idx = len(records); records.append(record); weights.append(W.copy()); starts.append(np.asarray(W0).copy())
        pools[gi].append(idx)
        return idx
    def best(gi):
        maximum = max(records[j]['logpost'] for j in pools[gi])
        return min((j for j in pools[gi] if records[j]['logpost']>=maximum-1e-10),key=lambda j:records[j]['gradient_norm'])
    for gi,gamma in enumerate(gammas):
        target = Target(data,gamma)
        values = np.array([target(W)[0] for W in source_W])
        idxs = diverse_indices(source_W,values,args.source_starts)
        for j in idxs:
            optimize(gi,source_W[j],{'kind':'saved_candidate','input_index':int(j),'source':source_info['candidates'][j]})
        scales = np.geomspace(.015, max(3., min(30.,2/np.sqrt(gamma))),args.random_starts)
        for j,scale in enumerate(scales):
            optimize(gi,rng.normal(size=(N,M))*scale,{'kind':'independent_random','index':j,'scale':float(scale)})
        eigval,eigvec = np.linalg.eigh(data.T@data)
        for scale in [.1,.5,1.,2.]:
            optimize(gi,eigvec[:,-M:]*scale,{'kind':'empirical_eigenvectors','scale':scale})
        for j in range(4):
            selected = rng.choice(K,size=M,replace=False)
            optimize(gi,data[selected].T*(.2+.4*j),{'kind':'data_patterns','indices':selected.tolist()})
        optimize(gi,np.zeros((N,M)),{'kind':'uniform'})
        b=best(gi)
        print(f'seed={args.seed} initial gamma={gamma:.8g} logpost={records[b]["logpost"]:.10g} starts={len(pools[gi])}',flush=True)
    continuation = []
    for trip in range(args.roundtrips):
        max_gain = 0.
        for direction, order in [('ascending',range(1,len(gammas))),('descending',range(len(gammas)-2,-1,-1))]:
            for gi in order:
                other=gi-1 if direction=='ascending' else gi+1
                old=records[best(gi)]['logpost']
                candidates=pools[other]
                local=diverse_indices(np.array([weights[j] for j in candidates]),np.array([-records[j]['logpost'] for j in candidates]),4)
                for k in local:
                    j=candidates[k]
                    optimize(gi,weights[j],{'kind':'continuation','roundtrip':trip,'direction':direction,'parent_record':j})
                max_gain=max(max_gain,records[best(gi)]['logpost']-old)
        continuation.append({'roundtrip':trip,'max_logpost_gain':max_gain})
        print(f'seed={args.seed} roundtrip={trip} gain={max_gain:.7g}',flush=True)
        if trip>=1 and max_gain<1e-8:
            break
    results=[]
    np.savez_compressed(out/'candidate_weights.npz',weights=np.array(weights),starts=np.array(starts))
    for gi,gamma in enumerate(gammas):
        target=Target(data,gamma)
        for attempt in range(5):
            j=best(gi); W=weights[j]; H=hessian(target,W.ravel()); ev,evec=np.linalg.eigh(H)
            if ev[0]>=-1e-6:
                break
            before=records[j]['logpost']
            for sign in [-1.,1.]:
                optimize(gi,W+sign*.1*evec[:,0].reshape(N,M),{'kind':'negative_curvature_escape','parent_record':j,'attempt':attempt})
            if records[best(gi)]['logpost']<=before+1e-10:
                break
        j=best(gi); W=weights[j].copy()
        parent=j; before=W.copy(); f,g=target(W)
        newton_iterations=0
        for _ in range(8):
            if np.linalg.norm(g)<1e-8:
                break
            H=hessian(target,W.ravel()); ev,evec=np.linalg.eigh(H)
            if ev[0]<-1e-6:
                break
            step=evec@((evec.T@g)/np.maximum(ev,1e-9))
            accepted=False
            for power in range(20):
                trial=W-step.reshape(N,M)*2.**(-power)
                f2,g2=target(trial)
                if f2<=f+1e-11 and np.linalg.norm(g2)<np.linalg.norm(g):
                    W,f,g=trial,f2,g2; accepted=True; newton_iterations+=1; break
            if not accepted:
                break
        if newton_iterations:
            idx=len(records); records.append({'gamma_index':int(gi),'gamma':float(gamma),
                'provenance':{'kind':'newton_stationarity_polish','parent_record':parent},
                'logpost':-f,'initial_logpost':-target(before)[0],
                'gradient_max_abs':float(abs(g).max()),'gradient_norm':float(np.linalg.norm(g)),
                'success':bool(np.linalg.norm(g)<1e-7),'message':'finite-difference Hessian Newton refinement',
                'iterations':newton_iterations,'retained_initial':False})
            weights.append(W.copy()); starts.append(before); pools[gi].append(idx)
        j=best(gi); W=weights[j]; H=hessian(target,W.ravel()); ev=np.linalg.eigvalsh(H)
        lp=visible_logp(W,V); p=np.exp(lp); pt=np.exp(lp_teacher)
        # Independent visible-state summation, separate from hidden-state fit.
        data_logits=np.logaddexp(data@W,-data@W).sum()
        v_logits=np.logaddexp(V@W,-V@W).sum(axis=1)
        independent_lp=data_logits-K*logsumexp(v_logits)-gamma*.5*np.sum(W*W)
        independent_grad=data.T@np.tanh(data@W)-K*V.T@(p[:,None]*np.tanh(V@W))-gamma*W
        objective,negative_gradient=target(W)
        C=(V.T*p)@V
        norm2=float(np.sum(W*W))
        eigenvalues=np.linalg.eigvalsh(W.T@W)[::-1]
        results.append({'gamma':float(gamma),'best_record':j,'logpost':-objective,
            'forward':float(pt@(lp_teacher-lp)/N),'reverse':float(p@(lp-lp_teacher)/N),
            'norm_squared':norm2,'raw_weight_eigenvalues':eigenvalues.tolist(),
            'covariance_largest_over_N':float(np.linalg.eigvalsh(C)[-1]/N),
            'teacher_q1':float(teacher[:,0]@C@teacher[:,0]/N**2),
            'gradient_max_abs':float(abs(negative_gradient).max()),'gradient_norm':float(np.linalg.norm(negative_gradient)),
            'winning_optimizer_success':records[j]['success'],
            'hessian_min_eigenvalue':float(ev[0]),'hessian_max_eigenvalue':float(ev[-1]),
            'visible_hidden_objective_error':abs(float(independent_lp+objective)),
            'visible_hidden_gradient_max_error':float(abs(independent_grad.ravel()+negative_gradient).max()),
            'visible_probability_sum_error':abs(float(p.sum())-1),
            'uniform_global_bound_gamma':float(np.linalg.eigvalsh(data.T@data)[-1]),
            'uniform_global_certificate':bool(gamma>=np.linalg.eigvalsh(data.T@data)[-1] and norm2<1e-12),
            'n_candidates':len(pools[gi]),
            'n_objective_ties_1e-6':sum(records[t]['logpost']>=-objective-1e-6 for t in pools[gi])})
    np.savez_compressed(out/'candidate_weights.npz',weights=np.array(weights),starts=np.array(starts))
    np.savez_compressed(out/'map_weights.npz',gammas=gammas,weights=np.array([weights[r['best_record']] for r in results]))
    atomic_json(out/'candidate_records.json',records)
    summary={'schema':'rbm_map_20261107_v1','config':config,'cells':results,'continuation':continuation,
        'uniform_reverse':float(np.mean(-N*np.log(2)-lp_teacher)/N),
        'uniform_forward':float(np.exp(lp_teacher)@(lp_teacher+N*np.log(2))/N),
        'elapsed_seconds':time.monotonic()-start_time}
    summary['validation']={'max_objective_error':max(r['visible_hidden_objective_error'] for r in results),
        'max_gradient_error':max(r['visible_hidden_gradient_max_error'] for r in results),
        'max_gradient_norm':max(r['gradient_norm'] for r in results),
        'min_hessian_eigenvalue':min(r['hessian_min_eigenvalue'] for r in results),
        'all_winning_optimizer_success':all(r['winning_optimizer_success'] for r in results)}
    validation=summary['validation']
    validation['thresholds']={'objective_error':1e-8,'gradient_error':1e-8,'gradient_norm':1e-4,'min_hessian_eigenvalue':-1e-6}
    validation['passed']=bool(validation['max_objective_error']<1e-8 and validation['max_gradient_error']<1e-8 and validation['max_gradient_norm']<1e-4 and validation['min_hessian_eigenvalue']>=-1e-6 and validation['all_winning_optimizer_success'])
    atomic_json(out/'summary.json',summary)
    atomic_json(out/'manifest.json',{'complete':True,'validation_passed':validation['passed'],'sha256':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}})
    print(json.dumps(summary['validation']),flush=True)
    print(f'COMPLETE seed={args.seed} seconds={summary["elapsed_seconds"]:.1f}',flush=True)
    if not validation['passed']:
        raise RuntimeError('Validation failed; outputs retained with failure status')

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--inputs',type=Path,default=Path(__file__).parent/'inputs')
    ap.add_argument('--outdir',type=Path,required=True)
    ap.add_argument('--random-starts',type=int,default=96)
    ap.add_argument('--source-starts',type=int,default=24)
    ap.add_argument('--maxiter',type=int,default=5000)
    ap.add_argument('--roundtrips',type=int,default=4)
    ap.add_argument('--gammas',type=float,nargs='+')
    run(ap.parse_args())
