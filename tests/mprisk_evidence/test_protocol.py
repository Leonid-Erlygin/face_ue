from pathlib import Path
import json
import numpy as np
import pytest
from experiments.mprisk_evidence.data import _pool,synthetic_pair,split_validation,Data,instantiate_config
from experiments.mprisk_evidence.metrics import Metrics,fit_linear,apply_linear,fit_monotone,apply_monotone,fit_mlp,apply_mlp
from experiments.mprisk_evidence.artifacts import archive,write_json


def test_pooling_matches_repository_default():
    from evaluation.template_pooling_strategies import PoolingDefault
    rng=np.random.default_rng(9);raw=rng.normal(size=(8,5));k=rng.uniform(1,8,8)
    templates=np.array([2,1,2,1,1,3,3,2]);media=np.array([4,2,4,2,3,5,6,7])
    chosen=np.array([3,1,2]);subject=np.array([33,11,22])
    mu,kap,ids,t=_pool(raw,np.log(k),templates,media,chosen,subject)
    expected=PoolingDefault()(raw,k[:,None],templates,media)
    np.testing.assert_allclose(mu,expected[0],atol=1e-14)
    np.testing.assert_allclose(kap,expected[1][:,0],atol=1e-14)
    np.testing.assert_array_equal(t,[1,2,3]);np.testing.assert_array_equal(ids,[11,22,33])


def test_template_validation_split_disjoint():
    v,_=synthetic_pair(4,500)
    p,unit=split_validation(v,8)
    assert 'template-disjoint' in unit
    assert len(np.unique(np.concatenate(list(p.values()))))==v.n
    for a,b in [('fit','select'),('fit','audit'),('select','audit')]:assert not np.intersect1d(p[a],p[b]).size


def test_group_validation_split_disjoint():
    rng=np.random.default_rng(11);ids=np.repeat(np.arange(20),5)
    v=Data(rng.normal(size=(100,3)),np.ones(100),rng.normal(size=(10,3)),np.ones(10),ids,np.arange(10),np.arange(100),{}).validate()
    p,unit=split_validation(v)
    assert unit=='identity-disjoint'
    for a,b in [('fit','select'),('fit','audit'),('select','audit')]:assert not np.intersect1d(ids[p[a]],ids[p[b]]).size


def test_duplicate_gallery_identity_rejected():
    v,_=synthetic_pair(4,100);v.gallery_ids=np.array([1,1,2,3])
    with pytest.raises(ValueError):v.validate()


def test_simple_recursive_source_config_factory():
    # Instantiate a genuine source target without needing a Hydra working directory.
    x=instantiate_config({'_target_':'evaluation.open_set_methods.mprisk_evidence.Parameters','probe_scale':2.,'gallery_kappa':5.,'beta':.2})
    assert x.probe_scale==2.
    with pytest.raises(ValueError):instantiate_config({'_target_':'builtins.dict','_partial_':True})


def test_all_methods_same_recognition_metrics():
    a=np.array([1,0,2,2,0,1]);y=np.array([1,0,1,0,2,1]);m=Metrics(a,y)
    one=m.evaluate(np.arange(6.));two=m.evaluate(-np.arange(6.))
    for k in ['f1','fpir','fnir','error_rate','errors']:assert one[k]==two[k]
    assert one['errors']==3
    assert np.isclose(one['f1'],4/7)


def test_malformed_scores_raise():
    m=Metrics([0,1],[0,1])
    with pytest.raises(ValueError):m.evaluate([np.nan,0.])
    with pytest.raises(ValueError):m.evaluate([0.])
    with pytest.raises(ValueError):m.evaluate([0.,2.],True)


def test_positive_monotone_calibrator_preserves_order():
    p=np.linspace(.05,.95,50);y=(np.arange(50)%3==0)
    c=fit_monotone(p,y);q=apply_monotone(p,c)
    assert c['slope']>0
    assert np.all(np.diff(q)>=0)


def test_linear_search_contains_primary_risk_candidate():
    rng=np.random.default_rng(15);X=rng.random((100,3));a=rng.integers(0,3,100);y=rng.integers(0,3,100)
    fit=fit_linear(X,a,y,budget=5,positive=True,include=[np.ones(3)])
    assert fit['validation_prr']>=Metrics(a,y).prr(X.sum(1))-1e-12
    assert np.all(np.array(fit['raw_weights'])>=0)


def test_mlp_serializable_predictions():
    rng=np.random.default_rng(8);X=rng.normal(size=(80,5));y=(X[:,0]>0)
    p=fit_mlp(X,y);q=apply_mlp(X,p)
    assert q.shape==(80,) and np.all((q>=0)&(q<=1))
    json.dumps(p)


def test_zip_inventory_and_scratch_exclusion(tmp_path):
    root=tmp_path/'run.with.dot';root.mkdir();(root/'_cache').mkdir()
    (root/'_cache'/'derived.npy').write_bytes(b'scratch')
    (root/'result.csv').write_text('x\n1\n')
    write_json(root/'manifest.json',dict(status='failed',failure='test failure'))
    path=archive(root)
    assert path.name=='run.with.dot.zip'
    import zipfile
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        assert not any('_cache' in n for n in z.namelist())
        manifest=json.loads(z.read('run.with.dot/manifest.json'))
        assert manifest['status']=='failed'
        inventory=json.loads(z.read('run.with.dot/file_inventory.json'))
        assert 'result.csv' in inventory
