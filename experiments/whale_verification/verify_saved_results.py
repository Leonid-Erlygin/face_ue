#!/usr/bin/env python3
"""Compare returned predictions with ORIGINAL metrics; optionally verify vMF numerics.

Does not fit or select any parameters. Does not deserialize saved torch models.
"""
from __future__ import annotations
import argparse, importlib.util, sys, json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
if __package__:
    from .source_loader import definitions
else:
    from source_loader import definitions

def run(repo, results, out, gold=False):
    out.mkdir(parents=True, exist_ok=True)
    original = definitions(repo/'experiments/mprisk_core_experiments.py',
        ['np_trapz','compute_osr_error_masks','f1_classic','fnir_fpir','rejection_curve_for_score','self_normalized_prr'],
        dict(np=np,pd=pd,Dict=Dict,Tuple=Tuple))
    records=[]; masks=[]
    published=pd.read_csv(results/'tables/main_comparison.csv')
    for case in sorted(results.glob('fpir_*')):
        if not (case/'test.npz').exists(): continue
        fp=float(case.name[5:])
        with np.load(case/'test.npz',allow_pickle=False) as f:
            a={k:f[k] for k in f.files}
        act=a['actions']; y=a['targets']; g=a['gallery_ids'].astype(np.int64); ids=a['subject_ids'].astype(np.int64)
        mapping={x:i+1 for i,x in enumerate(g)}
        independent_y=np.array([mapping.get(x,0) for x in ids])
        mm=original['compute_osr_error_masks'](np.maximum(act-1,0),act==0,g,ids)
        masks.append(dict(fpir=fp,n=len(act),target_mismatch=int(np.sum(y!=independent_y)),
            error_mask_mismatch=int(np.sum(mm['any_error']!=(act!=y))),
            false_accept=int(mm['false_accept'].sum()),false_reject=int(mm['false_reject'].sum()),
            misidentification=int(mm['misidentification'].sum()),
            accepted_class_disagrees_with_top=int(np.sum((act>0)&(act!=a['top_classes'][:,0]))),
            risk_sum_max_error=float(np.max(abs(a['risk']-a['r_fa']-a['r_id']-a['r_fr'])))))
        for j,name in enumerate(a['score_names']):
            prr,curve,_,_=original['self_normalized_prr'](a['scores'][:,j],np.maximum(act-1,0),act==0,g,ids,np.linspace(0,.5,20),seed=777)
            row=published[(published.split=='test')&(published.target_fpir==fp)&(published.method==name)].iloc[0]
            records.append(dict(fpir=fp,method=name,archived_PRR=float(row.prr_f1),original_PRR=float(prr),
                delta=float(prr-row.prr_f1),original_F1=float(curve.f1_class.iloc[0]),archived_F1=float(row.f1)))
    pd.DataFrame(records).to_csv(out/'original_metric_comparison.csv',index=False)
    pd.DataFrame(masks).to_csv(out/'independent_saved_checks.csv',index=False)
    summary=dict(no_refitting=True,rows_compared=len(records),
        note='PRR is recomputed with the original runner, NOT with the metric implementation in the results archive.',
        target_mismatches=sum(x['target_mismatch'] for x in masks),
        error_mask_mismatches=sum(x['error_mask_mismatch'] for x in masks))
    if gold:
        import mpmath as mp
        p=results/'source/evaluation/open_set_methods/mprisk_evidence.py'
        spec=importlib.util.spec_from_file_location('saved_evidence_numerics',p)
        model=importlib.util.module_from_spec(spec);sys.modules[spec.name]=model;spec.loader.exec_module(model)
        mp.mp.dps=70;nu=mp.mpf(255)
        def A(k):
            return mp.mpf(0) if not k else mp.log(mp.gamma(256))+nu*mp.log(2/k)+mp.log(mp.besseli(nu,k))
        rr=[]
        for k in [0,1e-4,.1,1,4.09,9.5947,14.9441,33.22,90.7,272.16,512,907.2257,1530.85,1e4,1e5,1e6]:
            expected=A(mp.mpf(float(k)));actual=float(model.log_partition(k,512))
            rr.append(dict(kind='log_partition',k=k,g=None,c=None,actual=actual,gold=float(expected),abs_error=float(abs(mp.mpf(actual)-expected))))
        for k in [4.09,9.5947,14.9441,33.22,1000,5000]:
            for g in [907.2257,1530.85,1e5]:
                for c in [-1,-.3,0,.4,.8,1]:
                    kk,gg,cc=map(mp.mpf,map(str,[k,g,c]));h=mp.sqrt(kk*kk+gg*gg+2*kk*gg*cc)
                    expected=A(h)-A(kk)-A(gg)
                    actual=float(model.log_bayes_factors(np.array([[c]]),np.array([k]),g,512)[0,0])
                    rr.append(dict(kind='log_B',k=k,g=g,c=c,actual=actual,gold=float(expected),abs_error=float(abs(mp.mpf(actual)-expected))))
        df=pd.DataFrame(rr);df.to_csv(out/'arbitrary_precision_check.csv',index=False)
        summary['numerical_check_max_abs_error']=df.groupby('kind').abs_error.max().to_dict()
        summary['numerical_check_scope']='16 normalizers and 108 log Bayes factors; 70 decimal digits; not every raw-data row'
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--results',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--gold',action='store_true')
    a=p.parse_args();run(a.repo.resolve(),a.results.resolve(),a.out.resolve(),a.gold)
