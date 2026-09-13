import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.special import expit,logsumexp
from evaluation.open_set_methods.mprisk_evidence import (risk_from_log_weights,risk_components,selected_log_odds,
    cost_log_odds,fit_parameters,Parameters,evaluate)
from experiments.mprisk_evidence.metrics import (Metrics,auc,ap,fit_positive_score_calibration,
    calibrated_log_odds,fit_linear,apply_linear,fit_risk_weights,apply_risk_weights)
from experiments.mprisk_evidence.recheck import (empirical_threshold,actions_at_threshold,fit_reference,
    reference_log_ratio,PreviousRun,final_review_gates,support_diagnostics)
from experiments.mprisk_evidence.score_updates import augmented_events,fit_ns_increment
from experiments.mprisk_evidence.artifacts import Tables,write_json,archive


@pytest.mark.parametrize('action',[0,1])
def test_no_saturation_of_rank_score(action):
    z=np.array([[0.,-1000.],[0.,-500.],[0.,-100.],[0.,-50.],[0.,0.],[0.,50.],[0.,100.],[0.,500.],[0.,1000.]])
    a=np.full(len(z),action,dtype=int);r=risk_from_log_weights(z,a)
    expected=(1-2*action)*z[:,1]
    np.testing.assert_allclose(r['score_log_odds'],expected,atol=1e-12)
    np.testing.assert_array_equal(np.argsort(r['score_log_odds']),np.argsort(expected))
    assert len(np.unique(r['score_log_odds']))==9
    assert len(np.unique(r['risk']))<9
    np.testing.assert_allclose(cost_log_odds(r['log_event_probabilities'],np.ones(3)),expected,atol=1e-12)


def test_three_class_alternatives_not_subtracted_from_one():
    z=np.array([[50.,0.,-30.],[100.,0.,-30.],[200.,0.,-30.]])
    lp=z-logsumexp(z,axis=1,keepdims=True)
    assert np.all(lp[:,0]==0)
    r=risk_components(lp,np.zeros(3,dtype=int))
    assert np.all(r['risk']>0) and np.all(np.diff(r['risk'])<0)
    np.testing.assert_allclose(r['score_log_odds'],logsumexp(z[:,1:],axis=1)-z[:,0])


@pytest.mark.parametrize('costs',[[1,1,1],[0,0,1],[1,0,0],[1,.2,3],[.001,1,.01]])
def test_cost_sum_matches_reference_in_ordinary_regime(costs):
    rng=np.random.default_rng(10);z=rng.normal(size=(40,5));a=rng.integers(0,5,40)
    r=risk_from_log_weights(z,a);c=np.asarray(costs);prob=np.exp(r['log_event_probabilities'][:,1:])@(c/c.max())
    score=cost_log_odds(r['log_event_probabilities'],costs)
    np.testing.assert_allclose(expit(score),prob,atol=1e-14)
    # Exact zero-cost events can have -inf score, which metrics support as a tie.
    m=Metrics(a,rng.integers(0,5,40));m.evaluate(score)


def test_stable_binary_nll_not_clipped_to_34():
    score=np.array([-1000.,1000.]);a=np.array([0,0]);y=np.array([1,0]);m=Metrics(a,y)
    r=m.evaluate(score,probabilities=expit(score),log_odds=score)
    assert r['error_nll']==1000.
    assert r['error_brier']==1.
    assert not r['is_probability'] and r['probability_reported']


def test_calibration_receives_original_log_odds():
    score=np.linspace(-1200.,-10.,100);y=np.arange(100)%4==0
    p=fit_positive_score_calibration(score,y)
    assert p['identifiable'] and p['success'] and p['slope']>0
    assert len(np.unique(calibrated_log_odds(score,p)))==100
    assert p['mean']==float(score.mean())


def test_calibration_one_class_is_reported_not_claimed_success():
    p=fit_positive_score_calibration(np.arange(12.),np.zeros(12))
    assert not p['success'] and not p['identifiable'] and p['slope']==0.
    assert p['positives']==0


@pytest.mark.parametrize('scores,far,expected',[([1,2,3,4],.5,2),([1,2,2,2],.5,0),([1,1,2,3],.5,2),([1,1],0,0),([1,1],1,2)])
def test_empirical_threshold_handles_ties(scores,far,expected):
    tau,r=empirical_threshold(scores,far)
    assert np.sum(np.array(scores)>=tau)==expected==r['actual_acceptances']
    assert r['achieved_fpir']<=far


def test_dbpedia_like_matching_reaches_missing_high_branch():
    # Boundary and dimension taken from the review only for a numerical regression,
    # NOT as a new fit or empirical benchmark result.
    tau=.997803;c=np.array([[tau-.0001],[tau],[tau+.0001]])
    c=np.repeat(c,8,axis=1);targets=np.zeros(3,dtype=int)
    r=fit_reference(c,targets,2/3,.5,768,'power')
    assert r['exact_kappa_root_found'] and r['gallery_kappa']>1e6
    assert abs(r['matching_log_margin'])<1e-5
    assert r['root_selection'].startswith('largest exact root')
    assert np.sum(actions_at_threshold(c,r['tau'])>0)==2


def test_power_ratio_against_high_precision():
    import mpmath as mp
    with mp.workdps(70):
        d=768;k=mp.mpf('2990608.3');c=mp.mpf('.997803')
        logS=mp.log(2)+mp.mpf(d)/2*mp.log(mp.pi)-mp.loggamma(mp.mpf(d)/2)
        norm=mp.loggamma(d-1+k)+mp.loggamma(mp.mpf(d)/2+k)+(k-1)*mp.log(2)-mp.mpf(d)/2*mp.log(mp.pi)-mp.loggamma(d-1+2*k)
        ref=float(logS+norm+k*mp.log1p(c))
    assert abs(float(reference_log_ratio(float(c),float(k),d,'power'))-ref)<1e-7


def test_nested_linear_candidate_exact_selection_and_application():
    rng=np.random.default_rng(5);X=rng.normal(size=(80,5));a=rng.integers(0,3,80);y=rng.integers(0,3,80)
    small=fit_linear(X[:,:2],a,y,budget=15)
    large=fit_linear(X,a,y,budget=0,nested=[('small',[0,1],small)])
    assert large['validation_prr']>=small['validation_prr']
    assert Metrics(a,y).prr(apply_linear(X,large))==large['validation_prr']
    assert len(large['nested_candidates'])==1


def test_ns_zero_coefficient_contains_exact_fitted_core():
    rng=np.random.default_rng(35);z=rng.normal(size=(80,5));a=rng.integers(0,5,80);y=rng.integers(0,5,80)
    e=risk_from_log_weights(z,a)['log_event_probabilities'];model=fit_risk_weights(e,a,y,budget=3,seed=778)
    aug=augmented_events(e,a,np.linspace(-1000.,0.,80))
    p=fit_ns_increment(aug,a,y,model['raw_weights'],core_events=e,seed=778)
    assert p['validation_prr']>=model['validation_prr']
    assert p['coefficient_grid'][0]['selection_prr']==model['validation_prr']
    np.testing.assert_allclose(expit(cost_log_odds(aug,[1,1,1,0])),expit(cost_log_odds(e,[1,1,1])))


def test_nonconverged_fit_never_selected_and_report_saved(tmp_path,monkeypatch):
    import evaluation.open_set_methods.mprisk_evidence as mod
    def failed(fun,x,**kwargs):
        return SimpleNamespace(x=np.array(x),fun=fun(x),success=False,message='iteration limit',nit=1)
    monkeypatch.setattr(mod,'minimize',failed)
    path=tmp_path/'attempts.json'
    with pytest.raises(RuntimeError,match='No converged'):
        fit_parameters(np.zeros((12,2)),np.ones(12),3,np.arange(12)%3,np.arange(6),np.arange(6,12),.5,
                       point=True,maxiter=1,report_path=path)
    report=json.loads(path.read_text());assert not report['selected_converged']
    assert all(len(c['attempts'])==2 for c in report['candidates'])
    assert report['selected_index'] is None


def test_failed_best_start_cannot_win(monkeypatch):
    import evaluation.open_set_methods.mprisk_evidence as mod
    calls=[0]
    def optimizer(fun,x,**kwargs):
        calls[0]+=1;ok=calls[0]>2
        return SimpleNamespace(x=np.array(x),fun=fun(x),success=ok,message='ok' if ok else 'limit',nit=1)
    monkeypatch.setattr(mod,'minimize',optimizer)
    _,report=fit_parameters(np.zeros((12,2)),np.ones(12),3,np.arange(12)%3,np.arange(6),np.arange(6,12),.5,point=True)
    assert report['selected_index']!=0 and report['selected_converged']


def test_inadequate_validation_is_flagged_without_fabrication(tmp_path):
    t=Tables(tmp_path)
    t.add('validation_support',dict(dataset='X',split='audit',false_accept=20,false_reject=0,misidentification=0))
    gate=final_review_gates(tmp_path,{'completed_datasets':[]},t)
    assert gate['status']=='hold' and not gate['blockers']
    assert any('false_reject:only_0' in s for s in gate['review_warnings'])
    assert not gate['raw_probability_calibration_certified']


def test_previous_zip_is_read_only_and_checks_inventory(tmp_path):
    root=tmp_path/'old';root.mkdir()
    write_json(root/'manifest.json',dict(status='complete',stage='sanity',domain='text',synthetic=True,model_version='reference-prior-evidence-1.0',arguments=dict(seed=777,beta=.5)))
    write_json(root/'data.json',{'value':4});z=archive(root)
    r=PreviousRun(z,'text',777,.5,True);assert r.read_json('data.json')['value']==4;r.close()
    (root/'data.json').write_text('{}')
    r=PreviousRun(root,'text',777,.5,True)
    with pytest.raises(ValueError,match='checksum'):r.read('data.json')
    r.close()


def test_known_only_ablation_preserves_extreme_order():
    from evaluation.open_set_methods.mprisk_evidence import Parameters,evaluate
    c=np.array([[1.,0.],[1.,.1],[1.,.2]])
    out=evaluate(c,np.ones(3),3,Parameters(1.,1000.,.5,True),np.array([1,1,0]))
    np.testing.assert_allclose(out['known_only_error_log_odds'][:2],[-1000.,-900.])
    assert np.isposinf(out['known_only_error_log_odds'][2])
    one=evaluate(c[:,:1],np.ones(3),3,Parameters(1.,1000.,.5,True),np.array([1,1,0]))
    assert np.isneginf(one['known_only_error_log_odds'][:2]).all()
    assert np.isposinf(one['known_only_error_log_odds'][2])
