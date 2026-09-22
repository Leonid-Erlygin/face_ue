"""Regressions for dummy MS1M names, local underflow policy and guarded resume."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import json
import shutil
import warnings

import numpy as np
import pandas as pd
import pytest

from experiments.evirisk_suite.input_audit import audit_inputs, source_separation
from experiments.evirisk_suite import audit_fix_resume as migration
from experiments.evirisk_suite.protocol import code_digest
from experiments.evirisk_full_suite import complete_inventory, verify_completed
from experiments.mprisk_evidence.artifacts import write_json
from experiments.mprisk_evidence.data import load_native
from evaluation.open_set_methods.mprisk_evidence import (
    _quiet_probability_underflow, risk_from_log_weights, cost_log_odds,
    selected_log_odds, log_posterior, evaluate, Parameters)
from experiments.mprisk_evidence.audits import run_audits

ROOT = Path(__file__).resolve().parents[2]


def ms1m_fixture(root):
    # Execute the actual supplied writer method without importing the training stack.
    file = ROOT/'training/dataset_classes/lightning_datasets.py'
    tree = ast.parse(file.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='MXFaceDataset')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name=='create_identification_meta')
    module = ast.Module(body=[method], type_ignores=[])
    scope = dict(np=np,pd=pd,Path=Path)
    exec(compile(ast.fix_missing_locations(module),str(file),'exec'),scope)
    labels = np.repeat(np.arange(8),35)
    scope['create_identification_meta'](SimpleNamespace(labels=labels),root,4)
    rng = np.random.default_rng(291)
    mu = rng.normal(size=(8,16));mu/=np.linalg.norm(mu,axis=1,keepdims=True)
    raw = mu[labels] + rng.normal(size=(len(labels),16))*.1
    np.savez(root/'embeddings/scf_embs_ms1m.npz',embs=raw.astype('float32'),
             unc=np.log(rng.uniform(10,100,len(labels)))[:,None].astype('float32'))
    config = dict(_target_='evaluation.test_datasets.FaceRecogntionDataset',
                  dataset_name='ms1m',dataset_path=str(root))
    return config


def test_actual_ms1m_generator_no_false_source_overlap(tmp_path):
    config=ms1m_fixture(tmp_path/'ms1m');data=load_native(config,fresh_metadata=True)
    report=audit_inputs(config,data,tmp_path/'report.json')
    assert report['checks']['probe_gallery_name_overlap']==1
    assert report['checks']['probe_gallery_name_overlap_examples']==['0']
    assert report['checks']['probe_gallery_real_source_name_overlap'] is None
    assert report['checks']['probe_gallery_template_overlap']==0
    assert report['checks']['probe_gallery_metadata_row_overlap']==0
    assert report['checks']['probe_gallery_media_index_overlap']==0
    assert report['checks']['ms1m_generator_layout_matches']
    assert report['checks']['native_pooling_sample']['passed']
    assert not report['blockers']
    assert any('not certified' in x for x in report['warnings'])
    assert any('No per-row identifiers' in x for x in report['warnings'])
    assert json.loads((tmp_path/'report.json').read_text())==report


def frame(names=('a.jpg','b.jpg','c.jpg','d.jpg')):
    return pd.DataFrame({0:names,1:[10,20,30,40],2:np.arange(4),3:[1,1,2,2]})


@pytest.mark.parametrize('dataset',['IJBC','IJBB','whale','ms1m','yahoo'])
def test_real_source_leaks_still_block(dataset):
    f=frame(('same.jpg','other.jpg','same.jpg','different.jpg'))
    c,w,b=source_separation(f,[30,40],[10,20],dataset)
    assert c['probe_gallery_real_source_name_overlap']==1
    assert any('share source names' in x for x in b)


def test_do_not_strip_directory_namespace():
    f=frame(('gallery/a.jpg','gallery/b.jpg','probe/a.jpg','probe/b.jpg'))
    c,w,b=source_separation(f,[30,40],[10,20],'IJBC')
    assert c['probe_gallery_name_overlap']==0 and not b


def test_detect_equivalent_loose_crop_names():
    f=frame(('/data/loose_crop/a.jpg','b.jpg','./a.jpg','c.jpg'))
    c,w,b=source_separation(f,[30,40],[10,20],'IJBC')
    assert c['probe_gallery_name_overlap']==1 and b


@pytest.mark.parametrize('change',['wrong_media','three_columns','other_dataset','mixed_names'])
def test_sentinel_exception_is_narrow(change):
    f=frame(('0',)*4);name='ms1m'
    if change=='wrong_media': f.iloc[1,2]=0
    if change=='three_columns': f=f.iloc[:,:3]
    if change=='other_dataset': name='whale'
    if change=='mixed_names': f.iloc[1,0]='real.jpg'
    c,w,b=source_separation(f,[30,40],[10,20],name)
    assert b


def test_shared_template_and_row_still_block_with_placeholder_names():
    c,w,b=source_separation(frame(('0',)*4),[20,30],[10,20],'ms1m')
    assert c['probe_gallery_template_overlap']==1
    assert c['probe_gallery_metadata_row_overlap']==1
    assert c['probe_gallery_media_index_overlap']==1
    assert any('template' in x for x in b)


def test_raw_concentration_error_not_suppressed(tmp_path):
    config=ms1m_fixture(tmp_path/'ms1m')
    path=Path(config['dataset_path'])/'embeddings/scf_embs_ms1m.npz'
    with np.load(path) as z: a={k:z[k] for k in z.files}
    a['unc'][0,0]=1000.;np.savez(path,**a)
    with pytest.raises(ValueError,match='float64 range'):
        load_native(config,fresh_metadata=True)


def test_extreme_probability_logs_quiet_and_identical():
    z=np.array([[0.,-1000.,-2000.],[0.,1000.,-1000.],[-1000.,0.,1000.]])
    a=np.array([0,1,1])
    with np.errstate(all='warn'):
        before=np.geterr().copy()
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter('always')
            out=risk_from_log_weights(z,a)
            odds=selected_log_odds(z,a)
            lp=log_posterior(z[:,1:],.5)
            weighted=cost_log_odds(out['log_event_probabilities'],[1.,.5,.2])
        assert not [w for w in warning_list if 'underflow' in str(w.message)]
        assert np.geterr()==before
        with np.errstate(under='ignore'):
            expected=risk_from_log_weights.__wrapped__(z,a)
        for k in out:
            np.testing.assert_array_equal(out[k],expected[k])
        np.testing.assert_array_equal(odds,out['score_log_odds'])
        assert np.isfinite(lp).all() and np.isfinite(weighted).all()


@pytest.mark.parametrize('operation',[lambda:np.divide(1.,0.),lambda:np.exp(1000.),lambda:np.sqrt(-1.)])
def test_non_underflow_exceptions_remain_enabled(operation):
    wrapped=_quiet_probability_underflow(operation)
    with np.errstate(all='raise'):
        before=np.geterr().copy()
        with pytest.raises(FloatingPointError): wrapped()
        assert np.geterr()==before


def test_numerical_preflight_keeps_checks_and_silences_only_tails():
    with np.errstate(all='warn'), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');rows=run_audits()
    assert len(rows)>=47 and all(r['passed'] for r in rows)
    assert not [w for w in caught if 'underflow' in str(w.message)]


@pytest.fixture
def migration_case(tmp_path,monkeypatch):
    oldroot=tmp_path/'out';source=oldroot/'source';live=tmp_path/'live'
    rel='experiments/evirisk_suite/input_audit.py'
    for root,contents in [(source,'# old audit\n'),(live,'# fixed audit\n')]:
        p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(contents)
        p=root/'evaluation/model.py';p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# unchanged arithmetic\n')
    p=live/migration.ADDED;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# migration helper\n')
    monkeypatch.setattr(migration,'PATCH_PAIR',{rel:{'old':migration.canonical_hash(source/rel),
                                                   'new':migration.canonical_hash(live/rel)}})
    common=dict(protocol='same-protocol',settings={'seed':777},versions={'numpy':'same'},
                jobs=[dict(seed=777,dataset='whale')])
    old=dict(common,code_sha256=code_digest(source));new=dict(common,code_sha256=code_digest(live))
    dest=oldroot/'runs/seed_777/whale'
    write_json(dest/'manifest.json',{'status':'complete'})
    (dest/'scores.csv').write_text('score\n1\n')
    write_json(dest/'completion_inventory.json',complete_inventory(dest))
    write_json(oldroot/'protocol.json',old)
    return oldroot,old,new,live


def test_migration_preserves_completed_job_and_code_history(migration_case):
    root,old,new,live=migration_case
    report=migration.validate_migration(root,old,new,live)
    assert report['reusable_jobs']==['seed_777/whale']
    assert not (root/'provenance_history').exists() # validation is read-only
    (root/'.suite.lock').write_text('123')
    dest=migration.record_migration(root,old,new,report)
    assert json.loads((dest/'protocol.json').read_text())==old
    assert code_digest(dest/'source_before')==old['code_sha256']
    assert verify_completed(root/'runs/seed_777/whale')
    assert json.loads((root/'protocol.json').read_text())==old # caller performs commit


@pytest.mark.parametrize('field',['settings','jobs','versions','protocol'])
def test_migration_rejects_protocol_changes(migration_case,field):
    root,old,new,live=migration_case
    new=copy.deepcopy(new);new[field]='unexpected change'
    with pytest.raises(ValueError,match='settings, jobs/config, inputs or library'):
        migration.validate_migration(root,old,new,live)


def test_migration_rejects_other_source_changes(migration_case):
    root,old,new,live=migration_case
    (live/'evaluation/model.py').write_text('# changed model\n')
    new['code_sha256']=code_digest(live)
    with pytest.raises(ValueError,match='changed source files'):
        migration.validate_migration(root,old,new,live)


def test_migration_rejects_archived_source_tampering(migration_case):
    root,old,new,live=migration_case
    (root/'source/evaluation/model.py').write_text('# tampered snapshot\n')
    with pytest.raises(ValueError,match='archived source'):
        migration.validate_migration(root,old,new,live)


def test_migration_rejects_modified_completed_results(migration_case):
    root,old,new,live=migration_case
    (root/'runs/seed_777/whale/scores.csv').write_text('score\n2\n')
    with pytest.raises(ValueError,match='Completed result changed'):
        migration.validate_migration(root,old,new,live)


def test_migration_rejects_generic_audit_bypass(migration_case):
    root,old,new,live=migration_case
    (live/'experiments/evirisk_suite/input_audit.py').write_text('# ignore all leakage\n')
    new['code_sha256']=code_digest(live)
    with pytest.raises(ValueError,match='unsupported old/new'):
        migration.validate_migration(root,old,new,live)


def test_migration_records_only_while_locked(migration_case):
    root,old,new,live=migration_case
    with pytest.raises(ValueError,match='suite lock'):
        migration.record_migration(root,old,new,{})
