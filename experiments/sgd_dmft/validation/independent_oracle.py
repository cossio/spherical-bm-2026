# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy", "torch"]
# ///
import argparse
import hashlib
import importlib.util
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch

source = Path(__file__).resolve().parents[1] / 'scripts' / 'run_sgd.py'
spec = importlib.util.spec_from_file_location('production_sgd', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
outdir = Path(tempfile.mkdtemp(prefix='independent-sgd-oracle-'))
results = {}
for label, c, initial, gamma in [
    ('A', [1.5,.5], [.3,.25], .5),
    ('B', [1.8,.2], [.3,.15], .5),
    ('C', [1.], [.3], .5),
    ('D', [1.], [.1], 1.5),
]:
    n, r, dt, nu, seed, steps = 6, 3, .01, .3, 9276, 4
    k = len(c)
    args = argparse.Namespace(case=label, N=n, replicas=r, dt=dt, Tmax=dt*steps,
        save_dt=dt, nu=nu, seed=seed, device='cpu', threads=1,
        output=str(outdir/f'{label}.npz'), selftest=False)
    module.simulate(args)
    # Same random inputs, independently implemented scalar-replica NumPy dynamics.
    g = torch.Generator(device='cpu').manual_seed(seed)
    gd = torch.Generator(device='cpu').manual_seed(seed+1000003)
    initial_x = torch.randn((r,n), generator=g, dtype=torch.float64).numpy()
    for j in range(r):
        initial_x[j,:k] = np.sqrt(n)*np.array(initial)
        initial_x[j,k:] *= np.sqrt(n*(1-np.dot(initial,initial))/np.dot(initial_x[j,k:],initial_x[j,k:]))
    x = np.array([initial_x.copy(), initial_x.copy()])
    w = np.zeros((2,r,n,n))
    data = np.zeros((k,n))
    if k == 1:
        data[0,0] = np.sqrt(n)
    else:
        data[0,:2] = np.sqrt(n/2)*np.sqrt(c)
        data[1,:2] = np.sqrt(n/2)*np.sqrt(c)*np.array([1.,-1.])
    covariance = sum(np.outer(y,y) for y in data)/n
    counts = np.zeros((r,k),dtype=np.int64)
    saved = {name:[] for name in ['s','kappa','trace','norm_squared_over_N','weight_signal_block']}
    for step in range(steps+1):
        saved['s'].append(x[:,:,:k].copy()/np.sqrt(n))
        saved['kappa'].append(np.array([[nu*(x[a,j]@w[a,j]@x[a,j]/n+(n-1)/n) for j in range(r)] for a in range(2)]))
        saved['trace'].append(np.trace(w,axis1=-2,axis2=-1))
        saved['norm_squared_over_N'].append(np.sum(x*x,axis=-1)/n)
        saved['weight_signal_block'].append(w[:,:,:k,:k].copy())
        if step == steps:
            break
        idx = torch.randint(k,(r,),generator=gd).numpy()
        z = torch.randn((r,n),generator=g,dtype=torch.float64).numpy()
        oldw, oldx = w.copy(),x.copy()
        for j in range(r):
            counts[j,idx[j]] += 1
            for a in range(2):
                v = oldx[a,j]
                positive = k*np.outer(data[idx[j]],data[idx[j]])/n if a == 0 else covariance
                negative = k*np.outer(v,v)/n
                w[a,j] = oldw[a,j]+dt*(positive-negative-gamma*oldw[a,j])/2
                projector = np.eye(n)-np.outer(v,v)/n
                trial = v+dt*nu*projector@oldw[a,j]@v+np.sqrt(2*nu*dt)*projector@z[j]
                x[a,j] = np.sqrt(n)*trial/np.linalg.norm(trial)
    errors = {}
    with np.load(outdir/f'{label}.npz') as actual:
        for name, values in saved.items():
            expected = np.moveaxis(np.array(values),0,2)
            assert expected.shape == actual[name].shape,(name,expected.shape,actual[name].shape)
            errors[name] = float(np.max(np.abs(expected-actual[name])))
        errors['counts'] = float(np.max(np.abs(counts-actual['data_counts'])))
        assert np.array_equal(actual['t'],np.arange(steps+1)*dt)
        assert np.max(list(errors.values())) < 1e-13,errors
    results[label] = errors
report = {'passed': True, 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'output_directory':str(outdir), 'torch':torch.__version__, 'maximum_errors':results}
(outdir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
