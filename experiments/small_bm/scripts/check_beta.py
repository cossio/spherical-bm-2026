# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy>=2.4", "scipy>=1.17"]
# ///
"""Independent numerical identities for the enumerated fully visible BM."""
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from beta import configurations, distribution_metrics, fit_map


def main():
    rng = np.random.default_rng(12345)
    n, k, gamma, omega = 8, 4, 6., 1.5
    s = configurations(n)
    rows, cols = np.triu_indices(n, 1)
    f = s[:, rows]*s[:, cols]
    logz = logsumexp(omega*s.sum(1)**2/(2*n))
    independent_logz = logsumexp([math.log(math.comb(n,l))+omega*(2*l-n)**2/(2*n)
                                 for l in range(n+1)])
    np.testing.assert_allclose(logz,independent_logz,atol=1e-13)
    j = rng.normal(size=len(rows))*.1
    m = np.zeros((n,n)); m[rows,cols]=m[cols,rows]=j
    np.testing.assert_allclose(f@j, np.einsum("si,ij,sj->s",s,m,s)/2,atol=1e-13)
    data = rng.integers(len(s),size=k)
    counts = np.bincount(data,minlength=len(s))
    def objective(jj):
        return k*logsumexp(f@jj)-counts@(f@jj)+n*gamma*(jj@jj)/2
    logits=f@j; p=np.exp(logits-logsumexp(logits))
    grad=k*(p@f)-counts@f+n*gamma*j
    eps=1e-6
    fd=np.array([(objective(j+eps*e)-objective(j-eps*e))/(2*eps)
                 for e in np.eye(len(j))])
    np.testing.assert_allclose(grad,fd,atol=2e-8)
    fit,residual=fit_map(f,counts,n*gamma)
    logp=omega*s.sum(1)**2/(2*n)-logz
    logq=f@fit-logsumexp(f@fit)
    direction=np.linalg.eigh(m)[1][:,-1]
    metrics=distribution_metrics(logq[None,:],logp,s,direction)[0]
    q=np.exp(logq)
    covariance=sum(qi*np.outer(si,si) for qi,si in zip(q,s))
    np.testing.assert_allclose(metrics[2],direction@covariance@direction/n,atol=1e-14)
    shell=s.sum(1)**2
    qm=np.array([q[shell==v].sum() for v in np.unique(shell)])
    pt=np.exp(logp)
    pm=np.array([pt[shell==v].sum() for v in np.unique(shell)])
    rev_parallel=np.sum(qm*np.log(qm/pm))
    fwd_parallel=np.sum(pm*np.log(pm/qm))
    rev_perp=fwd_perp=0.
    for a,v in enumerate(np.unique(shell)):
        mask=shell==v; conditional=q[mask]/qm[a]
        rev_perp+=qm[a]*np.sum(conditional*np.log(conditional*mask.sum()))
        fwd_perp+=pm[a]*np.mean(-np.log(conditional*mask.sum()))
    np.testing.assert_allclose(metrics[:2],[fwd_parallel+fwd_perp,rev_parallel+rev_perp],atol=1e-13)
    d=Path(__file__).resolve().parents[1]/"results"
    d.mkdir(parents=True,exist_ok=True)
    result={"passed":True,"checks":["Curie-Weiss binomial partition sum",
       "upper triangle versus quadratic energy","posterior gradient finite differences",
       "MAP stationarity","generated covariance independent loop",
       "forward and reverse shell chain rules"],
       "max_gradient_fd_error":float(np.max(np.abs(grad-fd))),"map_gradient_residual":residual}
    (d/"beta_checks.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
