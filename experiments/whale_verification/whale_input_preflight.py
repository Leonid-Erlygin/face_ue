#!/usr/bin/env python3
"""Verify Whale inputs and poolers without fitting models or changing data.

Requires the original repository and raw embs/unc NPZ + protocol metadata.
Reports remaining provenance limitations rather than declaring a blanket pass.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ive, hyp0f1, gammaln, logsumexp
from scipy.stats import spearmanr
from sklearn.preprocessing import normalize
if __package__:
    from .source_loader import definitions
else:
    from source_loader import definitions

def digest(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256()
    h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes());return h.hexdigest()

def file_sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

def namekey(x):
    s=str(x).replace('\\','/').removeprefix('./')
    return s.split('/loose_crop/',1)[-1]

def difference(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape:return dict(equal_shape=False,left_shape=list(a.shape),right_shape=list(b.shape))
    if a.dtype.kind in 'OUS' or b.dtype.kind in 'OUS':return dict(equal_shape=True,mismatches=int(np.sum(a.astype(str)!=b.astype(str))))
    return dict(equal_shape=True,max_abs=float(np.max(abs(a-b))) if a.size else 0,
                close=bool(np.allclose(a,b,rtol=2e-6,atol=2e-6)))

def logZ(k,d):
    """Independent direct log normalizer, not the archived centered-partition code."""
    k=np.asarray(k,dtype=np.float64); ans=np.zeros_like(k);nu=d/2-1
    small=(k>0)&(k<50);large=k>=50
    ans[small]=np.log(hyp0f1(d/2,k[small]**2/4))
    ans[large]=gammaln(d/2)+nu*np.log(2/k[large])+np.log(ive(nu,k[large]))+k[large]
    if not np.all(np.isfinite(ans)):raise ValueError('Independent normalizer outside supported numerical range')
    return ans

def one_dataset(root,name,role,repo,results,out,cache_root):
    meta=root/'meta';emb=root/'embeddings'/f'scf_embs_{name}.npz';media_path=meta/f'{name}_face_tid_mid.txt'
    frame=pd.read_csv(media_path,sep=r'\s+',header=None)
    images=np.array([namekey(x) for x in frame.iloc[:,0]])
    templates=frame.iloc[:,1].to_numpy();medias=frame.iloc[:,2].to_numpy()
    report=dict(role=role,name=name,checks={},blockers=[],warnings=[],source_files={})
    for p in [emb,media_path]:report['source_files'][str(p)]=dict(bytes=p.stat().st_size,sha256=file_sha(p))
    with np.load(emb,allow_pickle=False) as f:
        raw=f['embs']; logk=np.asarray(f['unc']).reshape(-1)
        rowids=next((key for key in ['image_ids','sample_ids','img_names','filenames'] if key in f),None)
        if rowids:
            found=np.array([namekey(x) for x in f[rowids].reshape(-1)])
            same=np.array_equal(found,images)
            report['checks']['export_row_ids']=dict(key=rowids,match=bool(same))
            if not same:report['blockers'].append('Exported row IDs and template metadata have different order; do not reorder automatically.')
        else:report['warnings'].append('Export contains no image/sample IDs: actual export row order remains conditional on the source pipeline.')
    if raw.ndim!=2 or len(raw)!=len(frame) or len(logk)!=len(raw):raise ValueError('Raw array and metadata lengths disagree')
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(logk)):raise ValueError('Nonfinite raw inputs')
    namefile=meta/f'{name}_name_5pts_score.txt'
    if namefile.exists():
        export_names=np.array([namekey(line.split()[0]) for line in namefile.read_text().splitlines() if line.strip()])
        match=np.array_equal(export_names,images)
        report['checks']['export_name_list_vs_template_metadata']=dict(equal_order=bool(match),export_count=len(export_names),metadata_count=len(images))
        report['source_files'][str(namefile)]=dict(bytes=namefile.stat().st_size,sha256=file_sha(namefile))
        if not match:report['blockers'].append('Image-name list used by the source dataset differs from face_tid_mid row order.')
    else:report['warnings'].append('No source image-name list; metadata row-order check unavailable.')
    # Cache records are inspected only, never overwritten. Object arrays are not unpickled.
    for file,keys in [('backup.npz',{'templates':templates,'medias':medias})]:
        p=root/file
        if p.exists():
            with np.load(p,allow_pickle=False) as f:
                for key,expected in keys.items():
                    if key in f:
                        result=difference(f[key],expected);report['checks'][f'{file}:{key}']=result
                        if not result.get('close',False):report['warnings'].append(f'{file}:{key} differs from fresh metadata. Historical cached runs are not directly comparable.')
                if 'img_names' in f:
                    try:
                        found=np.array([namekey(x) for x in f['img_names'].reshape(-1)])
                        report['checks']['backup_image_order']=dict(match=bool(np.array_equal(found,images)))
                        if not np.array_equal(found,images):report['warnings'].append('backup image names differ from fresh template metadata.')
                    except ValueError:report['warnings'].append('backup img_names is an object array; intentionally not unpickled.')
    pure=definitions(repo/'experiments/mprisk_evidence/data.py',['normalized','_pool'],dict(np=np))
    native=definitions(repo/'evaluation/template_pooling_strategies.py',['PoolingDefault'],
                       dict(np=np,normalize=normalize,AbstractTemplatePooling=object))['PoolingDefault']()
    subset=definitions(repo/'evaluation/embedding_utils.py',['get_template_subsets'],dict(np=np,tqdm=lambda x,*a,**k:x))['get_template_subsets']
    pooled={};protocols={}
    for group,tail in [('probe','probe_mixed'),('g1','gallery_G1')]:
        p=meta/f'{name}_1N_{tail}.csv';f=pd.read_csv(p);ch=f.iloc[:,0].to_numpy();ids=f.iloc[:,1].to_numpy();protocols[group]=(ch,ids)
        report['source_files'][str(p)]=dict(bytes=p.stat().st_size,sha256=file_sha(p))
        arr=pure['_pool'](raw,logk,templates,medias,ch,ids)
        # Match Data.validate's second normalization in the supplied adapter.
        arr=(pure['normalized'](arr[0]),arr[1],arr[2],arr[3]);pooled[group]=arr
        try:
            xf,lk,m,ts,ys=subset(raw,logk[:,None],templates,medias,ids,ch)
            if len(xf)!=len(ts):raise ValueError(f'Original get_template_subsets returns {len(xf)} feature rows but {len(ts)} template labels')
            nm,nk=native(xf,np.exp(lk),ts,m)
            comp=dict(direction=difference(arr[0],nm),concentration=difference(arr[1],nk.reshape(-1)),identities=difference(arr[2].astype(str),ys.astype(str)))
            report['checks'][f'native_pooling:{group}']=comp
            if not comp['direction'].get('close',False) or not comp['concentration'].get('close',False) or comp['identities'].get('mismatches',1):report['blockers'].append('Original and adapter poolers disagree for '+group)
        except Exception as e:
            report['checks'][f'native_pooling:{group}']=dict(error=str(e));report['blockers'].append('Original pooling contract could not be reproduced for '+group)
        cp=cache_root/'scf'/f'template_pool_gallery-PoolingDefault_probe-PoolingDefault_{name}'/('probe_g1.npz' if group=='probe' else 'gallery_g1.npz')
        if cp.exists():
            with np.load(cp,allow_pickle=False) as f:
                cr=dict(direction=difference(arr[0],f['template_pooled_features']),concentration=difference(arr[1],f['template_pooled_data_unc'].reshape(-1)))
                report['checks'][f'cached_pooling:{group}']=cr
                if not cr['direction'].get('close',False) or not cr['concentration'].get('close',False):report['warnings'].append('Cached native pooling differs from fresh inputs for '+group)
    pm,pk,pi,pt=pooled['probe'];gm,gk,gi,gt=pooled['g1']
    if len(np.unique(gi))!=len(gi):report['blockers'].append('Repeated gallery identities')
    if len(np.intersect1d(pt,gt)):report['blockers'].append('Gallery/probe template overlap')
    pimages=images[np.isin(templates,pt)];gimages=images[np.isin(templates,gt)]
    ni=len(np.intersect1d(pimages,gimages));report['checks']['probe_gallery_image_name_overlap']=ni
    if ni:report['blockers'].append('Image names shared between probe and gallery templates')
    index={str(x):i+1 for i,x in enumerate(gi)};y=np.array([index.get(str(x),0) for x in pi]);mx=np.empty(len(pm));best=np.empty(len(pm),int);truecos=np.full(len(pm),np.nan)
    for start in range(0,len(pm),128):
        end=min(start+128,len(pm));c=pm[start:end]@gm.T;mx[start:end]=c.max(1);best[start:end]=c.argmax(1)+1
        jj=np.flatnonzero(y[start:end]>0);truecos[start+jj]=c[jj,y[start+jj]-1]
    # Recompute individual-image diagnostic, separately from pooled templates.
    subject_by_template={t:s for t,s in zip(pt,pi)};raw_y=np.array([index.get(str(subject_by_template.get(t,'__absent__')),0) for t in templates]);ix=np.flatnonzero(raw_y>0)
    raw_c=np.sum(normalize(raw[ix].astype(float))*gm[raw_y[ix]-1],axis=1);raw_k=np.exp(logk[ix].astype(float))
    report['checks']['geometry']=dict(sample_n=len(ix),sample_spearman=float(spearmanr(raw_k,raw_c).statistic),
        template_n=int((y>0).sum()),template_spearman=float(spearmanr(pk[y>0],truecos[y>0]).statistic))
    report['checks']['fresh_hashes']=dict(representations_sha256=digest(pm),concentrations_sha256=digest(pk),gallery_sha256=digest(gm))
    old=json.loads((results/'data_manifest.json').read_text())[role]
    report['checks']['archived_hash_matches']={k:(v==old.get(k)) for k,v in report['checks']['fresh_hashes'].items()}
    if not all(report['checks']['archived_hash_matches'].values()):report['warnings'].append('Fresh byte hashes differ from the run; inspect floating tolerances and data changes before comparison.')
    default_pars=json.loads((results/'probability_model_fit.json').read_text())['parameters']
    for case in sorted(results.glob('fpir_*')):
        if not (case/f'{role}.npz').exists():continue
        selected_model=case/'selected_probability_model.json'
        pars=json.loads(selected_model.read_text())['parameters'] if selected_model.exists() else default_pars
        with np.load(case/f'{role}.npz',allow_pickle=False) as f:saved={k:f[k] for k in f.files}
        if not np.array_equal(pt.astype(str),saved['template_ids']):report['blockers'].append(case.name+': probe order/IDs differ');continue
        if not np.array_equal(gi.astype(str),saved['gallery_ids']):report['blockers'].append(case.name+': gallery order/IDs differ');continue
        if not np.array_equal(y,saved['targets']):report['blockers'].append(case.name+': independently reconstructed targets differ');continue
        dec=json.loads((case/'reference_decisions.json').read_text());tau=dec[role]['tau'];actions=np.where(mx>=tau,best,0)
        near=abs(mx-tau)<=1e-10;changed=actions!=saved['actions'];n=int(changed.sum());off=int(np.sum(changed&~near))
        report['checks'][case.name+':actions']=dict(mismatches=n,away_from_boundary=off,tau=tau,concentration=difference(pk,saved['kappa']))
        if off:report['blockers'].append(case.name+': reference decisions differ away from threshold')
        if n and not off:report['warnings'].append(case.name+': floating-point action difference exactly at threshold; inspect before claiming equality')
        chosen=np.unique(np.r_[np.random.default_rng(123).choice(len(pm),min(64,len(pm)),replace=False),np.flatnonzero((actions==0)&(y>0))[:64]])
        c=pm[chosen]@gm.T;k=pk[chosen]*pars['probe_scale'];g=pars['gallery_kappa'];h=np.sqrt(k[:,None]**2+g*g+2*k[:,None]*g*c)
        b=logZ(h,pm.shape[1])-logZ(k,pm.shape[1])[:,None]-logZ(np.array(g),pm.shape[1]);beta=pars['beta']
        z=np.column_stack([np.full(len(chosen),np.log(beta)),b+np.log((1-beta)/len(gm))]);den=logsumexp(z,axis=1)
        lp=z-den[:,None];ac=saved['actions'][chosen];wrong=z.copy();wrong[np.arange(len(chosen)),ac]=-np.inf
        odds=logsumexp(wrong,axis=1)-z[np.arange(len(chosen)),ac]
        err=float(np.max(abs(odds-saved['score_log_odds'][chosen])))
        report['checks'][case.name+':independent_probability_replay']=dict(rows=len(chosen),max_error_log_odds=err,max_error_true_log_probability=float(np.max(abs(lp[np.arange(len(chosen)),y[chosen]]-saved['true_log_probability'][chosen]))))
        if err>2e-6:report['blockers'].append(case.name+': direct probability formula differs from saved result')
    report['raw_n']=len(raw);report['probe_n']=len(pm);report['gallery_n']=len(gm)
    report['status']='failed_checks' if report['blockers'] else 'checks_passed_subject_to_provenance_limitations'
    out.mkdir(parents=True,exist_ok=True);(out/f'{role}.json').write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--results',type=Path,required=True)
    p.add_argument('--whale-root',type=Path,required=True);p.add_argument('--val-root',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,default=Path('/app/cache/template_cache_new_v2'));p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();reports={}
    for role,root,name in [('validation',a.val_root,'whale_val'),('test',a.whale_root,'whale')]:
        try:reports[role]=one_dataset(root,name,role,a.repo,a.results,a.out,a.cache_root)
        except Exception as e:reports[role]=dict(status='failed',error=str(e));a.out.mkdir(parents=True,exist_ok=True);(a.out/f'{role}.json').write_text(json.dumps(reports[role],indent=2))
    summary={k:{n:v.get(n) for n in ['status','blockers','warnings','error']} for k,v in reports.items()}
    summary['scope']='No model fitting. Matching local numeric identity IDs across datasets is not treated as leakage: IDs may be independently renumbered. Cross-dataset identity provenance still requires the original mapping.'
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    sys.exit(1 if any(v['status'] in ['failed','failed_checks'] for v in reports.values()) else 0)
