"""Read-only replay and protocol diagnostics for the September sanity revision.

Archive contents are data, never imported code. No labels from the test split
are used for fitting or for selecting an evidence parameter.
"""
from __future__ import annotations
from pathlib import Path, PurePosixPath
import hashlib, io, json, zipfile
import numpy as np
from scipy.special import logsumexp
from evaluation.open_set_methods.mprisk_evidence import Parameters, evaluate, cost_log_odds
from experiments.mprisk_evidence.artifacts import sha256, write_json
from experiments.mprisk_evidence.metrics import Metrics, masks, auc, ap


class PreviousRun:
    def __init__(self,path,domain,seed,beta,synthetic=False):
        self.path=Path(path).resolve();self.zip=None
        if self.path.is_dir():
            self.prefix='';manifest=self.read_json('manifest.json');self.fingerprint=sha256(self.path/'manifest.json')
        else:
            self.zip=zipfile.ZipFile(self.path)
            names=self.zip.namelist()
            for name in names:
                p=PurePosixPath(name)
                if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe previous-run ZIP path')
            if len(set(names))!=len(names):raise ValueError('Duplicate ZIP paths')
            roots=[n for n in names if n.endswith('/manifest.json') and n.count('/')==1]
            if len(roots)!=1:raise ValueError('Expected one root study manifest in previous-run ZIP')
            self.prefix=roots[0][:-len('manifest.json')]
            manifest=self.read_json('manifest.json');self.fingerprint=sha256(self.path)
        self.manifest=manifest
        if manifest.get('status')!='complete' or manifest.get('stage')!='sanity':raise ValueError('Previous run must be a completed sanity run')
        if manifest.get('model_version')!='reference-prior-evidence-1.0':raise ValueError('Replay requires original version 1.0 probability-score output; do not use a corrected 1.1 run as PREVIOUS_RUN')
        if manifest.get('domain')!=domain or bool(manifest.get('synthetic'))!=bool(synthetic):raise ValueError('Previous domain/synthetic flag mismatch')
        if int(manifest['arguments']['seed'])!=seed or float(manifest['arguments']['beta'])!=beta:raise ValueError('Previous seed/prior mismatch')
        self.inventory=self.read_json('file_inventory.json')
    def read(self,name):
        if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:raise ValueError('Unsafe member')
        raw=self.zip.read(self.prefix+name) if self.zip else (self.path/name).read_bytes()
        if hasattr(self,'inventory'):
            info=self.inventory.get(name)
            if info is None:raise ValueError('Previous file not listed in inventory: '+name)
            if len(raw)!=info['bytes'] or hashlib.sha256(raw).hexdigest()!=info['sha256']:raise ValueError('Previous checksum mismatch: '+name)
        return raw
    def read_json(self,name):return json.loads(self.read(name))
    def arrays(self,name):
        with np.load(io.BytesIO(self.read(name)),allow_pickle=False) as f:return {k:f[k] for k in f.files}
    def close(self):
        if self.zip:self.zip.close()


def reuse_split(previous,name,val,test):
    from experiments.mprisk_evidence.study import slug
    base='datasets/'+slug(name)+'/'
    info=previous.read_json(base+'data_manifest.json')
    if [val.n,test.n]!=info['original_n']:raise ValueError('Original dataset size differs from previous run')
    vi=np.asarray(info['validation_original_indices'],dtype=int);ti=np.asarray(info['test_original_indices'],dtype=int)
    if len(np.unique(vi))!=len(vi) or len(np.unique(ti))!=len(ti):raise ValueError('Duplicate previous sample indices')
    val=val if np.array_equal(vi,np.arange(val.n)) else val.take(vi)
    test=test if np.array_equal(ti,np.arange(test.n)) else test.take(ti)
    for label,data in [('validation',val),('test',test)]:
        actual=data.metadata()
        for field in ['n','K','d','representations_sha256','concentrations_sha256','gallery_sha256','templates_sha256']:
            if actual[field]!=info[label][field]:raise ValueError(f'{name}/{label}: data fingerprint changed: {field}')
    split=previous.read_json(base+'validation_split.json');parts={k:np.asarray(split[k],int) for k in ['fit','select','audit']}
    if not np.array_equal(np.sort(np.concatenate(list(parts.values()))),np.arange(val.n)):raise ValueError('Invalid saved split')
    return val,test,vi,ti,parts,split['unit']


def replay_dataset(previous,name,val,test,cval,ctest,parts,table,root,tables,seed):
    """Old fit + old decisions; recompute only numerical representation of risk."""
    from experiments.mprisk_evidence.study import slug
    base='datasets/'+slug(name)+'/'
    fit=previous.read_json(base+'probability_model_fit.json')
    dest=Path(root)/'numerical_replay'/'datasets'/slug(name);dest.mkdir(parents=True,exist_ok=True)
    for filename in ['probability_model_fit.json','data_manifest.json','validation_split.json']:
        write_json(dest/('previous_'+filename),previous.read_json(base+filename))
    old_case=base+'fpir_0.1/'
    write_json(dest/'previous_reference_decision_fit.json',previous.read_json(old_case+'reference_decision_fit.json'))
    for label,data,c in [('validation',val,cval),('test',test,ctest)]:
        saved=previous.arrays(old_case+'per_example/'+label+'.npz')
        for key,current in [('targets',data.targets),('subject_ids',data.probe_ids.astype(str)),('template_ids',data.template_ids.astype(str)),('gallery_ids',data.gallery_ids.astype(str))]:
            if not np.array_equal(saved[key],current):raise ValueError(f'Previous array alignment failed: {name}/{label}/{key}')
        rows={}
        for method,param_key in [('MPRisk','parameters'),('Point-vMF (NLL fitted)','point_parameters')]:
            p=Parameters(**fit[param_key]);v=evaluate(c,data.kappa,data.d,p,saved['actions'],data.targets,table)
            if method=='MPRisk':
                if not np.allclose(v['log_known_bayes_factor'],saved['log_known_bayes_factor'],rtol=1e-9,atol=1e-6):
                    raise ValueError('Replay evidence differs: refuse to label as a numerical-only comparison')
                rows.update(v)
            oldscore=saved['scores'][:,list(saved['score_names']).index(method)]
            ix=np.arange(data.n) if label=='test' else parts['audit']
            m=Metrics(saved['actions'][ix],data.targets[ix],seed)
            for variant,score,prob,odds in [('archived_probability_score',oldscore,oldscore,None),('stable_log_odds_same_fit',v['score_log_odds'],v['risk'],v['score_log_odds'])]:
                context=dict(dataset=name,split='test' if label=='test' else 'audit',method=method,variant=variant,seed=seed,target_fpir=.1)
                tables.add('numerical_replay_comparison',m.evaluate(score[ix],probabilities=prob[ix],log_odds=None if odds is None else odds[ix]),**context)
                for subset in ['accepted','rejected']:
                    j=(saved['actions'][ix]>0) if subset=='accepted' else (saved['actions'][ix]==0)
                    if j.any():
                        mm=Metrics(saved['actions'][ix][j],data.targets[ix][j],seed)
                        tables.add('numerical_replay_conditional',mm.evaluate(score[ix][j],probabilities=prob[ix][j],log_odds=None if odds is None else odds[ix][j]),subset=subset,**context)
        np.savez_compressed(dest/(label+'.npz'),actions=saved['actions'],targets=data.targets,
            archived_score_names=saved['score_names'],archived_scores=saved['scores'],
            archived_risk=saved['risk'],template_ids=data.template_ids.astype(str),**rows)
    write_json(dest/'status.json',dict(status='complete',fitted_parameters_unchanged=True,recognition_actions_unchanged=True,
              source_fingerprint=previous.fingerprint,calibrators_refitted=False,
              note='Replay does not repair posterior calibration; historical other scores remain historical'))


def empirical_threshold(scores,far):
    """Largest attainable number of acceptances <= floor(far*N), with ties.

    Acceptance is score >= tau. All tied boundary scores are either kept or
    excluded together; ties are not split using labels or array order.
    """
    s=np.asarray(scores,dtype=float).reshape(-1)
    if not len(s) or np.any(~np.isfinite(s)) or not 0<=far<=1:raise ValueError('Invalid threshold inputs')
    n=len(s);budget=int(np.floor(float(far)*n))
    if budget>=n:tau=float(np.nextafter(s.min(),-np.inf))
    elif budget<=0:tau=float(np.nextafter(s.max(),np.inf))
    else:
        boundary=float(np.partition(s,n-budget)[n-budget]);tau=boundary
        if np.sum(s>=tau)>budget:tau=float(np.nextafter(boundary,np.inf))
    accepted=int(np.sum(s>=tau))
    return tau,dict(target_fpir=float(far),achieved_fpir=accepted/n,unknown_count=n,
                    requested_acceptances=budget,actual_acceptances=accepted,tie_policy='conservative whole-score-group',
                    unavoidable_tie_shortfall=budget-accepted,tau=tau)


def actions_at_threshold(c,tau):
    c=np.asarray(c)
    return np.where(np.max(c,axis=1)>=tau,np.argmax(c,axis=1)+1,0).astype(int)


def support_diagnostics(tables,context,val,test,parts,av,at):
    rows=[]
    for split,data,a,ix in [('fit',val,av,parts['fit']),('select',val,av,parts['select']),('audit',val,av,parts['audit']),('test',test,at,np.arange(test.n))]:
        m=masks(a[ix],data.targets[ix]);counts={k:int(m[k].sum()) for k in ['false_accept','false_reject','misidentification','true_reject','tp']}
        row=dict(split=split,n=len(ix),known=int(m['known'].sum()),unknown=int((~m['known']).sum()),**counts)
        rows.append(row);tables.add('validation_support',row,**context)
    return rows


def final_review_gates(root,manifest,tables):
    """Evidence of validity vs. study completion. Never a performance-win gate."""
    blockers=[];warnings=[]
    for dataset in manifest.get('completed_datasets',[]):
        report=Path(root)/'datasets'/dataset/'probability_model_fit.json'
        if not report.exists():continue
        report=json.loads(report.read_text())
        for model,key in [('evidence','fit'),('point','point_fit')]:
            r=report[key]
            if not r.get('selected_converged'):blockers.append(f'{dataset}:{model}:fit_not_converged')
            if r.get('selected_at_boundary'):warnings.append(f'{dataset}:{model}:boundary_requires_review')
    for row in tables.data.get('validation_support',[]):
        if row['split'] not in ['select','audit']:continue
        for event in ['false_accept','false_reject','misidentification']:
            if row[event]<10:warnings.append(f'{row["dataset"]}:{row["split"]}:{event}:only_{row[event]}')
    for row in tables.data.get('reference_operating_point',[]):
        if row.get('unavoidable_tie_shortfall',0)>0:warnings.append(f'{row["dataset"]}:{row["split"]}:FPIR_tie_shortfall')
        if row.get('exact_kappa_root_found') is False:warnings.append(f'{row["dataset"]}:{row["split"]}:no_matching_probability_kappa')
    for row in tables.data.get('calibration_fit',[]):
        if not row.get('identifiable',True):warnings.append(f'{row["dataset"]}:{row["method"]}:calibration_not_identifiable')
    gate=dict(status='hold' if blockers or warnings else 'requires_human_review',
              blockers=sorted(set(blockers)),review_warnings=sorted(set(warnings)),
              raw_probability_calibration_certified=False,test_results_used_to_select_parameters=False,
              note='No fresh validation data or input corruption was generated. Sparse validation support is NOT fixed by code.')
    write_json(Path(root)/'review_gates.json',gate)
    return gate


def reference_log_ratio(cosines,kappa,d,kind):
    """Class density / uniform background for the historical M=0 model.

    The beta-function form avoids subtracting two huge lgamma values for power
    spherical densities when the matching concentration exceeds one million.
    """
    from scipy.special import gammaln, betaln
    from evaluation.open_set_methods.mprisk_evidence import log_surface,centered_partition
    c=np.clip(np.asarray(cosines,dtype=float),-1+1e-9,1-1e-9);k=float(kappa)
    if kind=='power':
        b=(d-1)/2
        peak=gammaln(b)-betaln(k+b,b)-(d-1)*np.log(2)-b*np.log(np.pi)+log_surface(d)
        return peak+k*np.log1p((c-1)/2)
    if kind=='vMF':return k*(c-1)-centered_partition(k,d)
    raise ValueError('Unknown reference density: '+str(kind))


def fit_reference(c,targets,far,beta,d,kind):
    """Order-statistic operating point, with deterministic high-branch matching.

    Unknown labels are used only on the declared operating-point fit split.
    Concentration matching affects historical probability baselines, not the
    threshold decisions: those are always evaluated directly in cosine space.
    """
    from scipy.optimize import brentq, minimize_scalar
    tau,report=empirical_threshold(np.max(c,axis=1)[np.asarray(targets)==0],far)
    K=c.shape[1];offset=np.log((1-beta)/(beta*K))
    def margin(z):return float(reference_log_ratio(tau,np.exp(z),d,kind)+offset)
    grid=np.linspace(0,np.log(1e8),1024);values=np.array([margin(v) for v in grid]);roots=[]
    for j in range(len(grid)-1):
        if values[j]==0:roots.append(float(np.exp(grid[j])))
        elif values[j]*values[j+1]<0:roots.append(float(np.exp(brentq(margin,grid[j],grid[j+1],xtol=1e-13))))
    if values[-1]==0:roots.append(float(np.exp(grid[-1])))
    found=bool(roots)
    if found:k=max(roots)
    else:
        ix=int(np.argmin(abs(values)));lo=grid[max(0,ix-1)];hi=grid[min(len(grid)-1,ix+1)]
        res=minimize_scalar(lambda z:abs(margin(z)),bounds=(lo,hi),method='bounded')
        k=float(np.exp(res.x))
    report.update(gallery_kappa=k,exact_kappa_root_found=found,matching_roots=roots,
                  matching_log_margin=margin(np.log(k)),matching_bracket=[1.,1e8],
                  root_selection='largest exact root in fixed bracket; no PRR or test accuracy selection',
                  actions_source='empirical cosine threshold, not a rounded posterior argmax',
                  probability_baseline_approximation=not found)
    return report


def model_limit_diagnostics(tables,name,c,k,d,targets,parts,pars,table,seed):
    """Validation-only comparison at shared rows; no limit is selected by test."""
    from dataclasses import replace
    from evaluation.open_set_methods.mprisk_evidence import class_log_weights,selected_log_odds,log_partition
    for split,ids in parts.items():
        for label,p in [('selected_finite',pars),('uniform_probe',replace(pars,probe_scale=0.)),('point_probe',replace(pars,point=True))]:
            result=evaluate(c[ids],k[ids],d,p,np.zeros(len(ids),int),targets[ids],table)
            tables.add('validation_model_limits',dict(limit=label,split=split,n=len(ids),diagnostic_only=True,
                class_nll=float(-result['true_log_probability'].mean())),dataset=name,seed=seed)
        # As kappa_g -> infinity, B_i = q_x(g_i)/u(g_i), with probe held fixed.
        nll=0.
        for lo in range(0,len(ids),128):
            ix=ids[lo:lo+128];effective=k[ix]*pars.probe_scale
            logB=effective[:,None]*np.asarray(c[ix])-log_partition(effective,d)[:,None]
            odds=selected_log_odds(class_log_weights(logB,pars.beta),targets[ix])
            nll+=np.logaddexp(0.,odds).sum()
        tables.add('validation_model_limits',dict(limit='point_gallery',split=split,n=len(ids),diagnostic_only=True,
            class_nll=float(nll/len(ids))),dataset=name,seed=seed)


def quality_distance_grid(tables,context,val,test,cval,ctest,parts,vold,told,sv,st,pv,pt):
    """Predefined quartiles from fit covariates, including distant rejected probes."""
    edges_k=np.quantile(val.kappa[parts['fit']],[.25,.5,.75])
    maxv=np.max(cval,axis=1);maxt=np.max(ctest,axis=1)
    edges_s=np.quantile(maxv[parts['fit']],[.25,.5,.75])
    for split,data,sim,a,scores,p,ix in [('audit',val,maxv,vold['actions'],sv,pv,parts['audit']),('test',test,maxt,told['actions'],st,pt,np.arange(test.n))]:
        q=np.searchsorted(edges_k,data.kappa,side='right');g=np.searchsorted(edges_s,sim,side='right')
        for qi in range(4):
            for gi in range(4):
                j=ix[(q[ix]==qi)&(g[ix]==gi)];m=masks(a[j],data.targets[j]);rejected=a[j]==0
                row=dict(split=split,quality_bin=qi,similarity_bin=gi,n=len(j),n_rejected=int(rejected.sum()),
                    false_rejections=int(m['false_reject'].sum()),true_rejections=int(m['true_reject'].sum()),
                    observed_error=float(m['any_error'].mean()) if len(j) else np.nan,
                    mean_risk=float(p['risk'][j].mean()) if len(j) else np.nan,
                    mean_log_odds=float(p['score_log_odds'][j].mean()) if len(j) else np.nan,
                    quality_edges=edges_k.tolist(),similarity_edges=edges_s.tolist(),bin_source='validation fit covariates')
                tables.add('quality_distance_grid',row,**context)
                for method in ['MPRisk','MPRisk-paper retuned','HolUE (refit fixed decisions)']:
                    yy=m['false_reject'][rejected];sc=scores[method][j][rejected]
                    tables.add('quality_distance_fr_detection',dict(split=split,quality_bin=qi,similarity_bin=gi,
                        n=len(yy),positives=int(yy.sum()),auroc=auc(yy,sc),auprc=ap(yy,sc)),method=method,**context)
