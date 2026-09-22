from dataclasses import asdict
import inspect
import numpy as np
import pytest

from evaluation.open_set_methods.mprisk_evidence import Parameters
from experiments.mprisk_evidence import risk_model_selection as selection
from experiments.mprisk_evidence.metrics import Metrics


def report():
    return dict(selected_index=0, fit_indices=list(range(4)), candidates=[
        dict(parameters=asdict(Parameters(.01, 100000., .5)),
             success=True, eligible=True, at_boundary=True),
        dict(parameters=asdict(Parameters(1.8, 1530., .5)),
             success=True, eligible=True, at_boundary=False)])


def toy_evaluate(c, k, d, p, a, y, table):
    # Rank-oriented fit is deliberately worse in multiclass NLL here.
    assert np.all(np.isfinite(c)) and np.all(np.isfinite(k))
    assert np.all(y >= 0) and np.all(a >= 0)
    error = a != y
    errp = np.where(error, .9, .1)
    if p.probe_scale < 1:
        errp = 1 - errp
    events = np.full((len(a), 4), -np.inf)
    events[:, 0] = np.log1p(-errp)
    events[a > 0, 1] = np.log(errp[a > 0])
    events[a == 0, 3] = np.log(errp[a == 0])
    return dict(log_event_probabilities=events,
                score_log_odds=np.log(errp)-np.log1p(-errp),
                true_log_probability=np.full(len(a), -.01 if p.probe_scale < 1 else .0-1.))


def inputs():
    n = 80
    return np.zeros((n, 2)), np.ones(n), np.resize([1, 1, 0, 0], n), np.resize([1, 0, 0, 1], n)


def test_selects_by_ranking_not_smaller_nll(monkeypatch):
    monkeypatch.setattr(selection, 'evaluate', toy_evaluate)
    c, k, y, a = inputs()
    p, w, out = selection.select_evidence_model(c, k, 3, y, a, np.arange(4, 60), report(), .5, budget=8)
    assert p.probe_scale == 1.8
    assert out['selected_index'] == 1 and out['nll_selected_index'] == 0
    assert out['candidates'][1]['class_select_nll'] > out['candidates'][0]['class_select_nll']
    assert out['candidates'][1]['weighted_validation_prr'] >= out['candidates'][1]['unit_validation_prr']
    assert out['candidates'][1]['weighted_validation_prr'] > out['candidates'][0]['weighted_validation_prr']
    assert not out['point_fallback'] and not out['prediction_formula_changed']


def test_does_not_read_audit_or_test_outcomes(monkeypatch):
    monkeypatch.setattr(selection, 'evaluate', toy_evaluate)
    c, k, y, a = inputs(); ix = np.arange(4, 60)
    _, w1, r1 = selection.select_evidence_model(c, k, 3, y, a, ix, report(), .5, budget=8)
    excluded = np.setdiff1d(np.arange(len(k)), ix)
    c[excluded] = np.nan; k[excluded] = np.nan; y[excluded] = -999; a[excluded] = -999
    _, w2, r2 = selection.select_evidence_model(c, k, 3, y, a, ix, report(), .5, budget=8)
    assert r1 == r2 and w1 == w2
    assert 'test' not in inspect.signature(selection.select_evidence_model).parameters


def test_rejects_fit_selection_overlap():
    c, k, y, a = inputs()
    with pytest.raises(ValueError, match='disjoint'):
        selection.select_evidence_model(c, k, 3, y, a, np.arange(2, 20), report(), .5, budget=2)


def test_filters_unconverged_candidates():
    r = report(); r['candidates'][1]['success'] = False
    assert [i for i, _ in selection.finite_candidates(r, .5)] == [0]


def test_rejects_point_substitution():
    r = report(); r['candidates'][1]['parameters']['point'] = True
    with pytest.raises(ValueError, match='Point'):
        selection.finite_candidates(r, .5)


def test_rejects_prior_change():
    r = report(); r['candidates'][1]['parameters']['beta'] = .1
    with pytest.raises(ValueError, match='incompatible'):
        selection.finite_candidates(r, .5)


def test_original_prr_random_draws():
    _, _, y, a = inputs(); m = Metrics(a, y, 777); rng = np.random.default_rng(777)
    np.testing.assert_array_equal(m.random, rng.random(len(y)))
    np.testing.assert_array_equal(m.oracle, (a != y).astype(float)+1e-9*rng.random(len(y)))


def test_native_four_component_includes_three_component_solution():
    rng = np.random.default_rng(4)
    a=np.resize([1, 0, 0, 1], 80); y=np.resize([1, 1, 0, 0],80)
    x=rng.random((80,3)); z=np.column_stack((x,rng.random(80)))
    w3=selection.fit_native_component_weights(x,a,y,budget=12)
    w4=selection.fit_native_component_weights(z,a,y,budget=12,include=[w3['raw_weights']+[0.]])
    assert w4['validation_prr'] >= w3['validation_prr'] - 1e-12
    assert w4['component_count'] == 4


def test_omitted_tail_bounds_enclose_full_gallery():
    from experiments.whale_verification.preview_evirisk_candidate import unit_score_bounds
    from evaluation.open_set_methods.mprisk_evidence import log_bayes_factors, selected_log_odds, PartitionTable
    rng = np.random.default_rng(13); n=60; K=11; d=5
    c=np.sort(rng.uniform(-1,1,(n,K)),axis=1)[:,::-1]
    k=np.exp(rng.uniform(-2,4,n)); p=Parameters(1.3,20.,.4); table=PartitionTable(d)
    a=np.arange(n)%2
    low,high=unit_score_bounds(c[:,:5],k,K,d,p,a>0,table)
    b=log_bayes_factors(c,k*p.probe_scale,p.gallery_kappa,d,table)
    z=np.column_stack((np.zeros(n),b+np.log((1-p.beta)/(p.beta*K))))
    exact=selected_log_odds(z,a)
    assert np.all(low<=exact+1e-12) and np.all(exact<=high+1e-12)


def test_prr_interval_encloses_random_allowed_orderings():
    from experiments.whale_verification.preview_evirisk_candidate import interval_prr
    rng=np.random.default_rng(130); _,_,y,a=inputs(); m=Metrics(a,y)
    low=rng.normal(size=len(a)); high=low+rng.random(len(a))*2
    bound=interval_prr(m,low,high)
    for _ in range(100):
        value=m.prr(low+rng.random(len(a))*(high-low))
        assert bound['prr_lower']-1e-12<=value<=bound['prr_upper']+1e-12
