import numpy as np
import pytest
from scipy.special import logsumexp
from numpy.polynomial.legendre import leggauss
from evaluation.open_set_methods.mprisk_evidence import (
    Parameters, PartitionTable, centered_partition, log_partition,
    log_bayes_factors, log_posterior, risk_components, evaluate, fit_parameters,
)
from experiments.mprisk_evidence.audits import run_audits


def test_independent_numerical_audits():
    rows=run_audits()
    assert len(rows)>=47
    assert all(row['passed'] for row in rows)


@pytest.mark.parametrize('d',[2,3,16,128,512,1536])
def test_uniform_probe_recovers_nonuniform_prior(d):
    c=np.random.default_rng(1).uniform(-1,1,(3,4))
    b=log_bayes_factors(c,np.zeros(3),1000,d)
    lp=log_posterior(b,.3,[.1,.2,.3,.4])
    np.testing.assert_allclose(np.exp(lp),np.tile([.3,.07,.14,.21,.28],(3,1)),rtol=1e-13)


@pytest.mark.parametrize('d',[3,512,1536])
def test_exact_interpolation_and_no_extrapolation(d):
    table=PartitionTable(d)
    k=np.r_[0.,1e-8,np.geomspace(.001,1e6,111),3e6]
    np.testing.assert_allclose(table(k),centered_partition(k,d),atol=3e-7,rtol=1e-10)
    assert table.nodes>0


@pytest.mark.parametrize('c,k,g',[(-.9,1.,5.),(.3,10.,20.),(1.,3.,7.),(-1.,20.,20.)])
def test_evidence_independent_spherical_quadrature(c,k,g):
    t,w=leggauss(160)
    phi=np.linspace(0,2*np.pi,360,endpoint=False)
    u=np.sqrt(1-t[:,None]**2)*np.cos(phi)[None,:]
    dot=c*t[:,None]+np.sqrt(1-c*c)*u
    integrand=np.exp(k*t[:,None]+g*dot-log_partition(k,3)-log_partition(g,3))
    numeric=np.sum(w*integrand.mean(1))/2
    exact=np.exp(log_bayes_factors([[c]],[k],g,3)[0,0])
    np.testing.assert_allclose(exact,numeric,rtol=2e-10,atol=1e-12)


def test_product_symmetry():
    a=log_bayes_factors([[.25]],[30.],70.,32)
    b=log_bayes_factors([[.25]],[70.],30.,32)
    np.testing.assert_allclose(a,b,atol=1e-12)


def test_high_concentration_point_limit():
    c=np.array([[-.5,.5,.99]])
    b=log_bayes_factors(c,[1e8],20.,3)
    np.testing.assert_allclose(b,20*c-log_partition(20.,3),atol=3e-6,rtol=1e-7)


def test_equal_class_split_conserves_evidence():
    b=np.array([[2.,-1.]])
    p=np.exp(log_posterior(b,.4,[.3,.7]))
    dup=np.exp(log_posterior(np.array([[2.,2.,-1.]]),.4,[.15,.15,.7]))
    np.testing.assert_allclose([p[0,0],p[0,1],p[0,2]],[dup[0,0],dup[0,1]+dup[0,2],dup[0,3]])


def test_fixed_actions_not_posterior_argmax():
    p=np.array([[.1,.8,.1],[.1,.8,.1],[.7,.1,.2]])
    a=np.array([0,2,1]);r=risk_components(np.log(p),a)
    np.testing.assert_allclose(r['risk'],[.9,.9,.9])
    np.testing.assert_allclose(r['r_fa'],[0,.1,.7])
    np.testing.assert_allclose(r['r_id'],[0,.8,.2])
    np.testing.assert_allclose(r['r_fr'],[.9,0,0])
    np.testing.assert_array_equal(r['counterfactual_action'],[1,1,0])
    np.testing.assert_array_equal(a,[0,2,1])


def test_low_quality_acceptance_and_rejection():
    p=Parameters(1.,20.,.2)
    out=evaluate(np.array([[-.5,-.5],[.98,.5]]),np.zeros(2),3,p,np.array([0,1]))
    np.testing.assert_allclose(out['risk'],[.8,.6],atol=1e-12)
    assert 'r_ns' not in out


def test_batching_equivalence():
    rng=np.random.default_rng(5);c=rng.uniform(-1,1,(29,8));k=rng.uniform(0,100,29)
    actions=rng.integers(0,9,29);y=rng.integers(0,9,29)
    a=evaluate(c,k,512,Parameters(1.,300.,.5),actions,y,batch=3,dense=True)
    b=evaluate(c,k,512,Parameters(1.,300.,.5),actions,y,batch=50,dense=True)
    for name in a:np.testing.assert_allclose(a[name],b[name],atol=1e-12)


@pytest.mark.parametrize('k',[-1,np.inf,np.nan])
def test_invalid_concentration_raises(k):
    with pytest.raises(ValueError):log_bayes_factors([[.3]],[k],20,3)


@pytest.mark.parametrize('beta',[0,1,-.1,np.nan])
def test_invalid_prior_raises(beta):
    with pytest.raises(ValueError):log_posterior([[0]],beta)


def test_invalid_risk_inputs_raise():
    with pytest.raises(ValueError):risk_components(np.log([[.2,.8]]),np.array([.5]))
    with pytest.raises(ValueError):risk_components(np.log([[.2,.7]]),np.array([0]))
    with pytest.raises(ValueError):evaluate([[.5]],[1.],3,Parameters(),[0.5])
    with pytest.raises(ValueError):evaluate([[.5]],[1.],3,Parameters(),[0],batch=0)


def test_fit_and_selection_must_be_disjoint():
    with pytest.raises(ValueError):
        fit_parameters(np.zeros((10,2)),np.ones(10),3,np.zeros(10,int),np.arange(6),np.arange(4,10),.5)
