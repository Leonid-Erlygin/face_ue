"""No external data required. Source defaults and generated fixtures only."""
from pathlib import Path
from types import SimpleNamespace
import copy, json
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from experiments.mprisk_evidence.metrics import Metrics,fit_linear
from experiments.mprisk_evidence.data import synthetic_pair,split_validation,load_native
from experiments.evirisk_suite.protocol import make_plan,DOMAINS,FPIRS,assert_input_signatures,file_signature
from experiments.evirisk_suite.supplementary import feature_transform,transform,fit_fusion,predict_fusion
from experiments.evirisk_suite.input_audit import audit_inputs
from experiments.evirisk_full_suite import complete_inventory,verify_completed
from experiments.mprisk_evidence.artifacts import write_json
ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def restore_metric_default():
    old=Metrics.default_tie_policy
    yield
    Metrics.default_tie_policy=old


def args(tmp_path,**updates):
    a=dict(smoke=False,datasets=None,seeds=[777],paths_json=None,out=str(tmp_path/'out'),
        bio_config=str(ROOT/'configs/uncertainty_benchmark/mprisk_core_bio_complete.yaml'),
        text_config=str(ROOT/'configs/uncertainty_benchmark/mprisk_core_text_complete.yaml'),beta=.5)
    a.update(updates);return SimpleNamespace(**a)


def test_plan_nine_datasets_41_points(tmp_path):
    plan=make_plan(args(tmp_path))
    assert len(plan)==9 and sum(len(j['fpirs']) for j in plan)==41
    assert {j['dataset'] for j in plan}==set(sum((list(n) for n in DOMAINS.values()),[]))
    for j in plan: assert j['fpirs']==FPIRS[j['domain']]


def test_multiple_seeds_not_selected_by_results(tmp_path):
    p=make_plan(args(tmp_path,seeds=[777,778,779]))
    assert len(p)==27 and sum(len(j['fpirs']) for j in p)==123


def test_optional_path_override(tmp_path):
    path=tmp_path/'paths.json';path.write_text(json.dumps({'whale':{'test_root':'/different/test','validation_root':'/different/val'}}))
    j=make_plan(args(tmp_path,datasets=['whale'],paths_json=str(path)))[0]
    assert any('different/test' in s for s in j['missing'])
    assert j['core']['dataset_name_to_calibration_set']['whale']['dataset_path']=='/different/val'


def test_same_validation_test_root_rejected(tmp_path):
    path=tmp_path/'paths.json';path.write_text(json.dumps({'whale':{'test_root':'/same','validation_root':'/same'}}))
    with pytest.raises(ValueError,match='must differ'):make_plan(args(tmp_path,datasets=['whale'],paths_json=str(path)))


def test_unknown_dataset_never_silently_skipped(tmp_path):
    with pytest.raises(ValueError,match='Unknown'):make_plan(args(tmp_path,datasets=['made_up']))


def test_synthetic_separated_from_real(tmp_path):
    p=make_plan(args(tmp_path,smoke=True))
    assert len(p)==2 and all(j['synthetic'] and j['dataset'].startswith('synthetic-') for j in p)


def test_input_changed_blocks_resume(tmp_path):
    p=tmp_path/'input';p.write_text('old');j={'inputs':[file_signature(p)]}
    p.write_text('new data')
    with pytest.raises(ValueError,match='Input changed'):assert_input_signatures(j)


def test_seeded_ties_preserve_original_reference_draws():
    a=np.resize([0,0,1,2],80);y=np.resize([0,1,1,1],80);rng=np.random.default_rng(777)
    m=Metrics(a,y,777,tie_policy='seeded-permutation')
    np.testing.assert_array_equal(m.random,rng.random(80))
    np.testing.assert_array_equal(m.oracle,(a!=y).astype(float)+1e-9*rng.random(80))
    np.testing.assert_array_equal(m.tie,rng.permutation(80))


def test_tie_order_does_not_depend_on_labels():
    a=np.resize([0,1,2],70);y=np.resize([0,1,1],70);s=np.zeros(70)
    m=Metrics(a,y,17,tie_policy='seeded-permutation');n=Metrics(a,y[::-1],17,tie_policy='seeded-permutation')
    np.testing.assert_array_equal(m.order(s),n.order(s))
    p=Metrics(a,y,18,tie_policy='seeded-permutation')
    assert not np.array_equal(m.order(s),p.order(s))


def test_untied_metric_unchanged():
    a=np.resize([0,1,2],70);y=np.resize([0,1,1],70);s=np.random.default_rng(2).random(70)
    assert Metrics(a,y,tie_policy='legacy').prr(s)==Metrics(a,y,tie_policy='seeded-permutation').prr(s)


def test_invalid_tie_policy_rejected():
    with pytest.raises(ValueError):Metrics([0,1],[0,1],tie_policy='test-labels')


def test_shared_metric_policy_reaches_fitting_helpers():
    Metrics.default_tie_policy='seeded-permutation'
    a=np.resize([0,1,1,0],60);y=np.resize([0,0,1,1],60);x=np.ones((60,2))
    p=fit_linear(x,a,y,budget=2,seed=7)
    assert p['validation_prr']==Metrics(a,y,7).prr(np.zeros(60))


def test_fusion_preprocess_fit_only_on_allowed_rows():
    x=np.array([[1.,2.],[2.,3.],[1e90,-1e90],[np.inf,-np.inf]])
    p=feature_transform(x,[0,1]);q=feature_transform(x[:2],[0,1])
    assert p==q and np.isfinite(transform(x,p)).all()


def test_fusion_fits_ignore_excluded_outcomes():
    rng=np.random.default_rng(7);x=rng.normal(size=(100,3));a=np.resize([0,1,1,2],100);y=np.resize([0,0,1,1],100)
    ix=np.arange(60);p=fit_fusion(x,a,y,ix,5,9)
    y2=y.copy();y2[60:]=20;a2=a.copy();a2[60:]=0
    assert p==fit_fusion(x,a2,y2,ix,5,9)


def test_exact_primary_candidate_survives_linear_search():
    rng=np.random.default_rng(7);x=rng.normal(size=(100,3));a=np.resize([0,1,1,2],100);y=np.resize([0,0,1,1],100)
    score=(a!=y).astype(float);ix=np.arange(60)
    p=fit_fusion(x,a,y,ix,3,9,{'EviRisk':score})
    assert p['validation_prr']>=Metrics(a[ix],y[ix],9).prr(score[ix])-1e-12
    if p['exact_selected']: np.testing.assert_array_equal(predict_fusion(x,p,{'EviRisk':score}),score)


def test_completed_inventory_detects_tampering(tmp_path):
    write_json(tmp_path/'manifest.json',{'status':'complete'});(tmp_path/'scores.csv').write_text('a\n1\n')
    write_json(tmp_path/'completion_inventory.json',complete_inventory(tmp_path));assert verify_completed(tmp_path)
    (tmp_path/'scores.csv').write_text('a\n2\n')
    with pytest.raises(ValueError,match='changed'):verify_completed(tmp_path)


def write_native_fixture(root,name,data):
    root.mkdir();(root/'meta').mkdir();(root/'embeddings').mkdir()
    g=len(data.gallery);probe_templates=np.arange(data.n)+100;gallery_templates=np.arange(g)+1
    means=np.concatenate((data.gallery,data.mu));ts=np.concatenate((gallery_templates,probe_templates))
    raw=np.repeat(means,2,axis=0).astype('float32');templates=np.repeat(ts,2);medias=np.tile([1,2],len(ts))
    k=np.concatenate((data.gallery_kappa,data.kappa));lk=np.repeat(np.log(k),2)
    names=np.array([f'{i}.jpg' for i in range(len(raw))]);stem=name.lower()
    np.savez(root/'embeddings'/f'scf_embs_{name}.npz',embs=raw,unc=lk[:,None],image_ids=names)
    pd.DataFrame({0:names,1:templates,2:medias}).to_csv(root/'meta'/f'{stem}_face_tid_mid.txt',sep=' ',header=False,index=False)
    pd.DataFrame({'template':gallery_templates,'subject':data.gallery_ids}).to_csv(root/'meta'/f'{stem}_1N_gallery_G1.csv',index=False)
    pd.DataFrame({'template':probe_templates,'subject':data.probe_ids}).to_csv(root/'meta'/f'{stem}_1N_probe_mixed.csv',index=False)
    (root/'meta'/f'{stem}_name_5pts_score.txt').write_text('\n'.join(n+' 0 0' for n in names))
    return {'_target_':'evaluation.test_datasets.FaceRecogntionDataset','dataset_path':str(root),'dataset_name':name}


def test_native_format_loader_and_sampled_pool_audit(tmp_path):
    v,_=synthetic_pair(19,140)
    config=write_native_fixture(tmp_path/'native','IJBC',v)
    pooled=load_native(config,fresh_metadata=True)
    report=audit_inputs(config,pooled,tmp_path/'audit.json')
    assert not report['blockers']
    assert report['checks']['native_pooling_sample']['passed']
    assert report['checks']['export_row_identifiers']['match']
    assert report['checks']['SCF_geometry']['sample_n']==2*report['checks']['SCF_geometry']['template_n']


def test_wrong_export_row_ids_stop_experiment(tmp_path):
    v,_=synthetic_pair(19,140);config=write_native_fixture(tmp_path/'native','whale',v)
    path=tmp_path/'native/embeddings/scf_embs_whale.npz'
    with np.load(path) as f:items={k:f[k] for k in f.files}
    items['image_ids']=items['image_ids'][::-1];np.savez(path,**items)
    data=load_native(config,fresh_metadata=True)
    with pytest.raises(ValueError,match='row identifiers'):audit_inputs(config,data,tmp_path/'audit.json')
    assert json.loads((tmp_path/'audit.json').read_text())['status']=='failed'


def test_missing_export_ids_is_warning_not_false_certificate(tmp_path):
    v,_=synthetic_pair(19,140);config=write_native_fixture(tmp_path/'native','whale',v)
    path=tmp_path/'native/embeddings/scf_embs_whale.npz'
    with np.load(path) as f:items={k:f[k] for k in f.files if k!='image_ids'}
    np.savez(path,**items);data=load_native(config,fresh_metadata=True)
    report=audit_inputs(config,data,tmp_path/'audit.json')
    assert not report['blockers'] and any('conditional' in w for w in report['warnings'])
