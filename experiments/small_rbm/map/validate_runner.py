# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy==2.5.4", "scipy==1.18.1"]
# ///
"""Independent visible enumeration and finite-difference objective checks."""
import json
import sys
from pathlib import Path
import numpy as np
from run_map import Target, make_data, states, visible_logp, hessian, sha
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rbm_exact import logpost_and_grad, log_prob_visible, Teacher
rng=np.random.default_rng(87643)
checks=[]
for seed in [0,3,7]:
 data,teacher,V,lp=make_data(seed)
 oldteacher=Teacher.hadamard(16,[1.,.5])
 oldlp=log_prob_visible(oldteacher.W,np.zeros(16),np.zeros(2),1.,V)
 olddata=oldteacher.sample(16,oldlp,V,np.random.default_rng(1000+seed))
 assert np.array_equal(data,olddata)
 for scale,gamma in [(.05,.0001),(.3,3.4),(1.5,1000.)]:
  W=rng.normal(size=(16,3))*scale
  target=Target(data,gamma)
  f,g=target(W)
  oldf,oldg=logpost_and_grad(W,data,np.zeros(16),np.zeros(3),1.,states(3),gamma)
  d=rng.normal(size=W.shape);d/=np.linalg.norm(d)
  eps=1e-5
  fd=(target(W+eps*d)[0]-target(W-eps*d)[0])/(2*eps)
  directional=float(g@d.ravel())
  H=hessian(target,W.ravel())
  fdg=(target(W+eps*d)[1]-target(W-eps*d)[1])/(2*eps)
  errors={'objective':abs(f+float(oldf)),'gradient':float(abs(g+oldg.ravel()).max()),
      'directional':abs(fd-directional),'hessian_product':float(abs(fdg-H@d.ravel()).max()),
      'visible_logp':float(abs(visible_logp(W,V)-log_prob_visible(W,np.zeros(16),np.zeros(3),1.,V)).max())}
  assert errors['objective']<1e-8 and errors['gradient']<1e-8 and errors['directional']<1e-5 and errors['hessian_product']<1e-5 and errors['visible_logp']<1e-10, errors
  checks.append({'seed':seed,'scale':scale,'gamma':gamma,'errors':errors})
source_paths=[Path(__file__),Path(__file__).parent/'run_map.py',Path(__file__).resolve().parents[1]/'scripts/rbm_exact.py']
print(json.dumps({'passed':True,'n_checks':len(checks),'checks':checks,'source_sha256':{p.name:sha(p) for p in source_paths}},indent=2))
