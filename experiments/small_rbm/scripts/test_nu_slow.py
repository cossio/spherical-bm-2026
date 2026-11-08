#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy>=2.4"]
# ///
"""Independent tiny-kernel and frozen-runner tests of lazy negative phases."""
import numpy as np
from rbm_exact import Teacher, log_prob_visible, spin_states, neg_phase_chains, neg_phase_exact
from nu_controls import train_arm, update_gradient
from nu_slow import advance_visible, expected_negative_phase, mixed_time_grid, train_slow_arm


def main():
    rng=np.random.default_rng(202611031)
    N,M=4,2
    V,H=spin_states(N),spin_states(M)
    W=rng.normal(scale=.4,size=(N,M))
    beta=.8
    joint=np.exp(beta*np.einsum('vi,ij,hj->vh',V,W,H))
    joint/=joint.sum()
    pv,ph=joint.sum(1),joint.sum(0)
    A,B=joint.T/ph[:,None],joint/pv[:,None]
    P=B@A
    f=np.array([np.outer(v,np.tanh(beta*v@W)).ravel() for v in V])
    retained=V[[0,3,9]]
    for p in (0.,.25,1.):
        for kind in ('gibbs','refresh'):
            fresh=P if kind=='gibbs' else np.ones((len(V),1))*pv
            lazy=(1-p)*np.eye(len(V))+p*fresh
            np.testing.assert_allclose(pv@lazy,pv,atol=3e-15)
            actual=expected_negative_phase(retained,W,beta=beta,H=H,V=V,kind=kind,p=p)
            target=(lazy[[0,3,9]]@f).mean(0).reshape(N,M)
            np.testing.assert_allclose(actual,target,atol=3e-15,rtol=2e-13)
    np.testing.assert_allclose(expected_negative_phase(retained,W,beta=beta,H=H,V=V,kind='iid',p=1), (pv@f).reshape(N,M),atol=3e-15)
    # Direct sampled kernels preserve stationary marginals; held entries remain exact.
    initial=V[rng.choice(len(V),size=100_000,p=pv)]
    mask=rng.random(len(initial))<.25
    for kind in ('gibbs','refresh'):
        sampled=advance_visible(initial,W,beta=beta,H=H,kind=kind,refresh_mask=mask,rng=np.random.default_rng(918))
        np.testing.assert_array_equal(sampled[~mask],initial[~mask])
        index=((sampled>0)*(1<<np.arange(N))).sum(1)
        frequencies=np.bincount(index,minlength=len(V))/len(sampled)
        assert np.all(np.abs(frequencies-pv)<6*np.sqrt(pv*(1-pv)/len(sampled))+.0005)
    teacher=Teacher.hadamard(N,[1,.5])
    teacher_lp=log_prob_visible(teacher.W,np.zeros(N),np.zeros(2),1,V)
    data=teacher.sample(N,teacher_lp,V,np.random.default_rng(1000))
    W0=np.random.default_rng(20000).normal(scale=.08,size=(N,M))
    _,vec=np.linalg.eigh(data.T@data/N)
    v0=V[[0,4]]
    common=dict(W0=W0,data=data,V=V,teacher_lp=teacher_lp,xis=teacher.xis,cvecs=vec[:,::-1][:,:M],H=H,beta=1.,gamma=3.4,lr=.1,horizon=.4)
    for kind,old_kind,nu in (('gibbs','persistent',1),('refresh','iid',None),('exact','exact',None)):
        exact=kind=='exact'
        initial_visible=np.empty((0,N)) if exact else v0
        frozen=train_arm(**common,kind=old_kind,nu=nu,chains=0 if exact else 2,
                         rng=None if exact else np.random.default_rng(7123),initial_visible=initial_visible,
                         eval_interval=.1,save_weights=True)
        current=train_slow_arm(**common,kind=kind,p=None if exact else 1.,chains=0 if exact else 2,
                              rng=None if exact else np.random.default_rng(7123),schedule_rng=np.random.default_rng(99),
                              initial_visible=initial_visible,dense_until=.4,dense_interval=.1,coarse_interval=1)
        np.testing.assert_array_equal(current['final_W'],frozen['final_W'])
        for a,b in zip(current['trajectory'],frozen['trajectory']):
            for name in ('W','time','step','fwd','rev','lam','u_teacher','u_data','q_teacher','q_data'):
                np.testing.assert_array_equal(a[name],b[name],err_msg=f'{kind}/{name}')
    class NeverRefresh:
        def random(self,size):
            return np.ones(size)
    held=train_slow_arm(**common,kind='gibbs',p=.25,chains=2,rng=np.random.default_rng(7123),
                        schedule_rng=NeverRefresh(),initial_visible=v0,
                        dense_until=.4,dense_interval=.1,coarse_interval=1)
    current=W0.copy()
    moments=[]
    for row in held['trajectory']:
        np.testing.assert_array_equal(row['W'],current)
        item=row['sampler_at_current_W']
        np.testing.assert_array_equal(item['retained_visible_before_transition'],v0)
        moment=neg_phase_chains(v0,current,np.zeros(N),np.zeros(M),1.)
        exact=neg_phase_exact(current,np.zeros(N),np.zeros(M),1.,H)
        moments.append(moment)
        if item['has_next_update']:
            np.testing.assert_allclose(item['realized_residual'],moment-exact,atol=1e-15)
            np.testing.assert_array_equal(item['visible_after_transition'],v0)
            current+=.1*update_gradient(current,data,moment,1.,3.4)
        else:
            assert item['visible_after_transition'] is None and item['realized_residual'] is None
    assert np.linalg.norm(moments[1]-moments[0])>1e-5
    np.testing.assert_array_equal(held['final_W'],current)
    np.testing.assert_array_equal(held['refresh_counts'],[0,0])
    for lr in (.1,.02):
        updates,steps,times=mixed_time_grid(4.,lr,.4,.1,1.)
        np.testing.assert_allclose(times,[0,.1,.2,.3,.4,1,2,3,4],atol=1e-15)
        np.testing.assert_array_equal(steps,np.rint(times/lr).astype(int))
        assert updates==round(4/lr)
    for horizon in (1.,2.):
        run=train_slow_arm(**(common|{'horizon':horizon}),kind='gibbs',p=.25,chains=2,
                           rng=np.random.default_rng(41),schedule_rng=np.random.default_rng(42),initial_visible=v0,
                           dense_until=.4,dense_interval=.1,coarse_interval=.5)
        if horizon==1:
            short=run
        else:
            long=run
    long_rows={row['step']:row for row in long['trajectory']}
    for row in short['trajectory']:
        match=long_rows[row['step']]
        np.testing.assert_array_equal(row['W'],match['W'])
        np.testing.assert_array_equal(row['sampler_at_current_W']['retained_visible_before_transition'],match['sampler_at_current_W']['retained_visible_before_transition'])
        assert row['schedule_prefix_sha256']==match['schedule_prefix_sha256']
    print('PASS: tiny full-kernel conditional moments, stationary sampling, p1 frozen exact/iid/Gibbs identity, changing-W held states, dense/coarse grids and same-stream prefixes.')


if __name__=='__main__':
    main()
