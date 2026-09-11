"""Executable mathematical sanity checks; no empirical performance assertions."""
import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp
from evaluation.open_set_methods.mprisk_evidence import (centered_partition,log_partition,log_bayes_factors,
                    log_posterior,risk_components,PartitionTable)


def run_audits():
    rows=[]
    def check(name,error,tolerance,**kw):
        rows.append(dict(check=name,error=float(error),tolerance=tolerance,passed=bool(error<=tolerance),**kw))
        if error>tolerance:raise AssertionError(f'{name}: {error} > {tolerance}')
    rng=np.random.default_rng(151)
    for d in [3,16,128,512,768]:
        c=rng.uniform(-1,1,(12,5));k=np.r_[0.,1e-6,.1,1.,10.,50.,100.,500.,1000.,10000.,100000.,1e6]
        B=log_bayes_factors(c,k,1000.,d);lp=log_posterior(B,.3)
        check('posterior_normalization',np.max(np.abs(np.exp(lp).sum(1)-1)),1e-12,d=d)
        check('uninformative_prior',np.max(np.abs(np.exp(lp[0])-np.r_[.3,np.full(5,.7/5)])),1e-12,d=d)
        a=rng.integers(0,6,12);r=risk_components(lp,a)
        check('fixed_action_risk_identity',np.max(np.abs(r['risk']-(r['r_fa']+r['r_id']+r['r_fr']))),1e-10,d=d)
        tab=PartitionTable(d);interp=log_bayes_factors(c,k,1000.,d,tab)
        check('exact_vs_interpolated_log_evidence',np.max(np.abs(B-interp)),1e-6,d=d)
        check('zero_gallery_concentration',np.max(np.abs(log_bayes_factors(c,k,0.,d))),1e-12,d=d)
    # Independent spherical integral in d=3, aligned means.
    for k,g in [(0.,20.),(.1,20.),(1.,2.),(10.,20.)]:
        A=float(log_partition(k,3)+log_partition(g,3))
        numeric=.5*quad(lambda t:np.exp((k+g)*t-A),-1,1,epsabs=1e-10)[0]
        exact=float(np.exp(log_bayes_factors([[1.]],[k],g,3)[0,0]))
        check('quadrature_d3',abs(numeric-exact)/(1+abs(exact)),1e-9,kappa=k,gallery_kappa=g)
    # High-precision independent reference for high-order/small-k cases.
    import mpmath as mp
    with mp.workdps(60):
        for d in [3,512,1536]:
            for k in [1e-4,.1,10.,100.,1000.,100000.]:
                ref=float(mp.log(mp.hyp0f1(mp.mpf(d)/2,mp.mpf(k)**2/4))-k)
                value=float(centered_partition(k,d))
                check('mpmath_log_partition',abs(value-ref),2e-8,d=d,kappa=k)
    return rows


def probe_limit_rows(d,kg,beta,K=2):
    rows=[]
    for cosine in [-.5,0.,.5,.9]:
        for k in [0.,.01,.1,1.,10.,50.,100.,500.,1000.,1e4,1e5,1e6]:
            # This is a scalar equal-similarity kernel audit, not a realizable
            # arbitrary gallery geometry or raw-input corruption experiment.
            lp=log_posterior(log_bayes_factors(np.full((1,K),cosine),[k],kg,d),beta)
            p=np.exp(lp[0]);rows.append(dict(d=d,gallery_kappa=kg,probe_kappa=k,cosine=cosine,
                  p_unknown=p[0],reject_risk=1-p[0],accept_class_1_risk=1-p[1],
                  kind='controlled representation-distribution sensitivity; not raw input degradation'))
    return rows
