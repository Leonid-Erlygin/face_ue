import numpy as np

from evaluation.open_set_methods.mprisk_evidence import Parameters, evaluate
from experiments.evirisk_suite.worker import galue_uncertainty
from experiments.evirisk_suite.protocol import METHODS


def _toy():
    c=np.array([[0.8,0.1,-0.2],[0.2,0.7,0.1],[-0.1,0.0,0.2]],dtype=float)
    k=np.array([3.0,5.0,1.5])
    y=np.array([1,2,0],dtype=int)
    pars=Parameters(probe_scale=1.0,gallery_kappa=7.0,beta=0.4,point=True)
    return c,k,y,pars


def test_galue_vmf_is_one_minus_max_posterior():
    c,k,y,pars=_toy()
    actions=np.array([1,2,0],dtype=int)
    out=evaluate(c,k,5,pars,actions,y,table=None,dense=True)
    expected=1.0-np.exp(out['log_posterior']).max(axis=1)
    np.testing.assert_allclose(galue_uncertainty(out),expected,rtol=0,atol=2e-15)


def test_galue_score_does_not_depend_on_fixed_recognizer_action():
    c,k,y,pars=_toy()
    a1=np.array([1,2,0],dtype=int)
    a2=np.array([0,1,3],dtype=int)
    s1=galue_uncertainty(evaluate(c,k,5,pars,a1,y,table=None))
    s2=galue_uncertainty(evaluate(c,k,5,pars,a2,y,table=None))
    np.testing.assert_allclose(s1,s2,rtol=0,atol=0)


def test_method_registry_has_single_vmf_galue_name():
    assert 'GalUE' in METHODS
    assert 'vMF' in METHODS['GalUE']
    assert not any(name.startswith('GalUE-vMF') for name in METHODS)
