import numpy as np
import pytest
from scipy.special import logsumexp
from evaluation.open_set_methods.mprisk_evidence import risk_from_log_weights,cost_log_odds,fit_parameters,Parameters,evaluate
from experiments.mprisk_evidence.data import _pool,Data,fresh_protocol
from experiments.evirisk_precision import precise_cost_score,fit_precise_weights,_mp_keys
from experiments.mprisk_evidence.metrics import Metrics,auc


def test_float32_logk_decoded_in_float64():
    raw=np.array([[1.,0.],[1.,0.]],np.float32)
    _,k,_,_=_pool(raw,np.array([90.,90.],np.float32),[1,1],[1,2],[1],[9])
    assert np.isfinite(k).all()
    np.testing.assert_allclose(k,[np.exp(90.)],rtol=1e-14)


def test_float64_logk_out_of_range_rejected():
    with pytest.raises(ValueError):_pool([[1.,0.]],[1000.],[1],[1],[1],[9])


def test_data_gallery_length_rejected():
    x=Data(np.array([[1.,0.]]),np.ones(1),np.array([[1.,0.],[0.,1.]]),np.ones(1),np.array([9]),np.array([9,10]),np.array([1]),{})
    with pytest.raises(ValueError):x.validate()


def test_nonunit_rejection_logit_plateau_fixed():
    z=np.column_stack([np.zeros(3),[50.,100.,500.]])
    e=risk_from_log_weights(z,np.zeros(3,int))['log_event_probabilities']
    old=cost_log_odds(e,[1.,.5,.2]);new=precise_cost_score(e,[1.,.5,.2])
    assert len(np.unique(old))==1
    assert np.all(np.diff(new)>0)


def test_weighted_order_matches_high_precision_random():
    rng=np.random.default_rng(24)
    for w in [[1.,.5,.2],[1.,1.,1.],[.1,1.,.3],[0.,1.,0.]]:
        z=rng.normal(size=(80,5))*rng.choice([1.,10.,100.,500.],size=(80,1))
        e=risk_from_log_weights(z,rng.integers(0,5,80))['log_event_probabilities']
        s=precise_cost_score(e,w);mp=_mp_keys(e,np.r_[0.,np.asarray(w)/max(w)])
        order=np.argsort(s,kind='stable')
        assert all(mp[a]<=mp[b] for a,b in zip(order,order[1:]))


def test_rejection_auc_independent_of_positive_weights():
    rng=np.random.default_rng(1);z=rng.normal(size=(50,4))*50
    v=risk_from_log_weights(z,np.zeros(50,int));e=v['log_event_probabilities'];y=rng.integers(0,2,50)
    np.testing.assert_allclose(auc(y,v['score_log_odds']),auc(y,precise_cost_score(e,[1.,.5,.2])))


def test_unit_order_matches_original_log_odds():
    rng=np.random.default_rng(2);z=rng.normal(size=(40,4))*100
    v=risk_from_log_weights(z,rng.integers(0,4,40));s=precise_cost_score(v['log_event_probabilities'],[1,1,1])
    np.testing.assert_array_equal(np.argsort(s),np.argsort(v['score_log_odds']))


def test_precise_tuning_retains_unit_candidate():
    rng=np.random.default_rng(3);z=rng.normal(size=(60,4))*50;a=rng.integers(0,4,60);y=rng.integers(0,4,60)
    e=risk_from_log_weights(z,a)['log_event_probabilities'];fit=fit_precise_weights(e,a,y,8,11)
    assert fit['validation_prr']>=Metrics(a,y,11).prr(precise_cost_score(e,[1,1,1]))-1e-12


def test_invalid_fit_bounds():
    with pytest.raises(ValueError):fit_parameters(np.zeros((10,2)),np.ones(10),3,np.zeros(10,int),np.arange(5),np.arange(5,10),.5,gallery_kappa_bounds=(1.,0.))


def test_prior_density_bound_on_finite_space():
    rng=np.random.default_rng(99)
    for _ in range(100):
        r=rng.dirichlet(np.ones(8));q=rng.dirichlet(np.ones(8));f=rng.dirichlet(np.ones(8),3);h=rng.dirichlet(np.ones(8),3)
        pi=rng.dirichlet(np.ones(4));ph=rng.dirichlet(np.ones(4));b=f@(q/r);bh=h@(q/r)
        mass=pi*np.r_[1.,b];mh=ph*np.r_[1.,bh];D=mass.sum()
        tv=.5*np.abs(mass/D-mh/mh.sum()).sum()
        eps=np.abs(h-f)@(q/r)
        A=abs(ph[0]-pi[0])+np.sum(b*abs(ph[1:]-pi[1:])+ph[1:]*eps)
        assert tv<=A/D+1e-12


def test_read_original_metadata_not_backups(tmp_path):
    meta=tmp_path/'meta';meta.mkdir()
    (meta/'whale_face_tid_mid.txt').write_text('a.jpg 1 1\nb.jpg 2 2\n')
    (meta/'whale_1N_gallery_G1.csv').write_text('template,subject\n1,7\n')
    (meta/'whale_1N_probe_mixed.csv').write_text('template,subject\n2,7\n')
    (tmp_path/'backup.npz').write_bytes(b'invalid stale cache')
    d=fresh_protocol({'_target_':'evaluation.test_datasets.FaceRecogntionDataset','dataset_path':str(tmp_path),'dataset_name':'whale'})
    np.testing.assert_array_equal(d.probe_ids,[7])
