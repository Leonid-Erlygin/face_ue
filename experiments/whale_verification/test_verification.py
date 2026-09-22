"""Regression checks. The input-parity tests use SYNTHETIC fixtures, not Whale."""
from __future__ import annotations
import json, tempfile, importlib.util,sys,os
from pathlib import Path
import unittest
import numpy as np,pandas as pd
from typing import Dict,Tuple
from scipy.stats import spearmanr
if __package__:
    from .source_loader import definitions
else:
    from source_loader import definitions
if __package__:
    from . import whale_input_preflight as pre
else:
    import whale_input_preflight as pre

REPO=Path(os.environ.get('WHALE_AUDIT_REPO', str(Path(__file__).resolve().parents[2]))).resolve()
RESULTS=Path(os.environ.get('WHALE_AUDIT_RESULTS','outputs/whale_evirisk_v10')).resolve()

def load_model():
 p=RESULTS/'source/evaluation/open_set_methods/mprisk_evidence.py';sp=importlib.util.spec_from_file_location('audit_test_model',p);m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m);return m

class Checks(unittest.TestCase):
 def test_corrected_metrics_match_original(self):
  original=definitions(REPO/'experiments/mprisk_core_experiments.py',['np_trapz','compute_osr_error_masks','f1_classic','fnir_fpir','rejection_curve_for_score','self_normalized_prr'],dict(np=np,pd=pd,Dict=Dict,Tuple=Tuple))
  sp=importlib.util.spec_from_file_location('corrected_metrics',REPO/'experiments/mprisk_evidence/metrics.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
  for fp in [.01,.05,.1,.2]:
   with np.load(RESULTS/f'fpir_{fp:g}/test.npz') as f:
    a=f['actions'];y=f['targets'];g=f['gallery_ids'].astype(int);ids=f['subject_ids'].astype(int);mym=m.Metrics(a,y,777)
    for col in range(f['scores'].shape[1]):
     s=f['scores'][:,col];gold=original['self_normalized_prr'](s,np.maximum(a-1,0),a==0,g,ids,np.linspace(0,.5,20),seed=777)[0]
     self.assertAlmostEqual(mym.prr(s),gold,places=12)
 def make_fixture(self,base,bad_order=False):
  root=base/'whale';meta=root/'meta';meta.mkdir(parents=True);(root/'embeddings').mkdir();res=base/'results';case=res/'fpir_0.1';case.mkdir(parents=True)
  rng=np.random.default_rng(88);d=512;ts=np.repeat([1,2,3,100,101,102,103,104,105],3);md=np.tile([1,1,2],9);raw=rng.normal(size=(27,d));raw/=np.linalg.norm(raw,axis=1,keepdims=True);raw=raw.astype('float32');lk=np.log(rng.uniform(400,2000,27)).astype('float32');names=[f'{i}.jpg' for i in range(27)]
  pd.DataFrame({0:names,1:ts,2:md}).to_csv(meta/'whale_face_tid_mid.txt',header=False,index=False,sep=' ')
  source_names=names[::-1] if bad_order else names
  (meta/'whale_name_5pts_score.txt').write_text('\n'.join(n+' '+' '.join(['0']*10)+' 1' for n in source_names))
  np.savez(root/'embeddings/scf_embs_whale.npz',embs=raw,unc=lk[:,None])
  chg=ts[:9];ig=np.repeat([0,1,2],3);chp=ts[9:];ip=np.repeat(np.arange(6),3)
  for tail,t,i in [('gallery_G1',chg,ig),('probe_mixed',chp,ip)]:pd.DataFrame({'template':t,'subject':i}).to_csv(meta/f'whale_1N_{tail}.csv',index=False)
  pure=definitions(REPO/'experiments/mprisk_evidence/data.py',['normalized','_pool'],dict(np=np));pool=pure['_pool'];norm=pure['normalized']
  pm,pk,pi,pt=pool(raw,lk,ts,md,chp,ip);gm,gk,gi,gt=pool(raw,lk,ts,md,chg,ig);pm=norm(pm);gm=norm(gm)
  c=pm@gm.T;tau=.04;act=np.where(c.max(1)>=tau,c.argmax(1)+1,0);y=np.r_[np.arange(1,4),[0,0,0]]
  m=load_model();pars=m.Parameters(.01,1000,.5,False);ev=m.evaluate(c,pk,d,pars,act,y)
  np.savez(case/'test.npz',actions=act,targets=y,template_ids=pt.astype(str),gallery_ids=gi.astype(str),kappa=pk,**ev)
  (case/'reference_decisions.json').write_text(json.dumps({'test':{'tau':tau}}));(res/'probability_model_fit.json').write_text(json.dumps({'parameters':{'probe_scale':.01,'gallery_kappa':1000,'beta':.5}}))
  (res/'data_manifest.json').write_text(json.dumps({'test':dict(representations_sha256=pre.digest(pm),concentrations_sha256=pre.digest(pk),gallery_sha256=pre.digest(gm))}))
  return root,res
 def test_synthetic_preflight(self):
  with tempfile.TemporaryDirectory() as t:
   b=Path(t);root,res=self.make_fixture(b);report=pre.one_dataset(root,'whale','test',REPO,res,b/'out',b/'empty_cache')
   self.assertEqual(report['blockers'],[]);self.assertTrue(all(report['checks']['archived_hash_matches'].values()))
   self.assertLess(report['checks']['fpir_0.1:independent_probability_replay']['max_error_log_odds'],1e-8)
   self.assertEqual(report['checks']['geometry']['sample_n'],9);self.assertEqual(report['checks']['geometry']['template_n'],3)
 def test_misaligned_image_name_list_blocked(self):
  with tempfile.TemporaryDirectory() as t:
   b=Path(t);root,res=self.make_fixture(b,True);report=pre.one_dataset(root,'whale','test',REPO,res,b/'out',b/'empty_cache')
   self.assertTrue(any('Image-name list' in x for x in report['blockers']))
 def test_no_false_row_provenance_certification(self):
  with tempfile.TemporaryDirectory() as t:
   b=Path(t);root,res=self.make_fixture(b);report=pre.one_dataset(root,'whale','test',REPO,res,b/'out',b/'empty_cache')
   self.assertTrue(any('no image/sample IDs' in x for x in report['warnings']))

if __name__=='__main__':unittest.main(verbosity=2)
