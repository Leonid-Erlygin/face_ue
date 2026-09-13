"""Cross-domain sanity and full, fixed-recognizer evidence-risk experiments."""
from __future__ import annotations
from dataclasses import asdict,replace
from pathlib import Path
import hashlib,json,time
import numpy as np
from scipy.special import logsumexp,gammaln
from scipy.stats import ks_2samp
from evaluation.open_set_methods.mprisk_evidence import (MODEL_VERSION,Parameters,PartitionTable,evaluate,fit_parameters,
                 log_bayes_factors,log_posterior,log_nonspecificity,log_surface,cost_log_odds)
from experiments.mprisk_evidence.data import (load_native,similarities,split_validation,run_legacy,native_method,
                 synthetic_pair,synthetic_legacy,digest_array,instantiate_config)
from experiments.mprisk_evidence.metrics import (Metrics,masks,auc,ap,bins,fit_linear,apply_linear,fit_monotone,
                 apply_monotone,fit_logistic,apply_logistic,ranks,bootstrap,fit_mlp,apply_mlp,fit_positive_score_calibration,apply_positive_score_calibration)
from experiments.mprisk_evidence.artifacts import write_json,Tables
from experiments.mprisk_evidence.audits import probe_limit_rows
from experiments.mprisk_evidence.recheck import (PreviousRun,reuse_split,replay_dataset,fit_reference,actions_at_threshold,support_diagnostics,model_limit_diagnostics,quality_distance_grid)
from experiments.mprisk_evidence.metrics import fit_risk_weights,apply_risk_weights


def slug(s):
    import re
    return re.sub(r'[^A-Za-z0-9_.-]+','_',str(s))


def subsample(data,limit,seed):
    if not limit or data.n<=limit:return data,np.arange(data.n)
    ix=np.sort(np.random.default_rng(seed).choice(data.n,limit,replace=False));return data.take(ix),ix


def threshold_from_actions(c,actions):
    s=np.max(c,axis=1);acc=actions>0
    if not acc.any():return float(np.nextafter(s.max(),np.inf))
    if acc.all():return float(np.nextafter(s.min(),-np.inf))
    lo,hi=float(s[~acc].max()),float(s[acc].min())
    if lo>hi+1e-7:raise AssertionError('Legacy M=0 actions are not a single cosine-threshold rule')
    return (lo+hi)/2


def simple_features(c,kappa,tau,T=1.,gal=None,beta=.5,d=512):
    """Same augmented-score definitions as the repository; scores, not error probabilities."""
    result={k:[] for k in ['SCF','AccScr','MSP','Entropy','Margin','GalUE']}
    for lo in range(0,len(c),128):
        cc=np.asarray(c[lo:lo+128]);aug=np.column_stack((cc,np.full(len(cc),tau)))
        lp=aug/T;lp-=logsumexp(lp,axis=1,keepdims=True);p=np.exp(lp)
        top=np.partition(aug,-2,axis=1)[:,-2:]
        if gal is None:gp=lp
        else:
            kg=gal['gallery_kappa'];temp=gal.get('galue_T',gal['temperature']);K=cc.shape[1]
            from experiments.mprisk_evidence.recheck import reference_log_ratio
            loglike=reference_log_ratio(cc,kg,d,gal['gallery_prior'])-log_surface(d)
            gp=np.column_stack((loglike+np.log((1-beta)/K),np.full(len(cc),np.log(beta)-log_surface(d))))/temp
            gp-=logsumexp(gp,axis=1,keepdims=True)
        vals=dict(SCF=-np.asarray(kappa)[lo:lo+len(cc)],AccScr=-np.abs(cc.max(1)-tau),
                  MSP=1-p.max(1),Entropy=-(p*lp).sum(1),Margin=-(top.max(1)-top.min(1)),GalUE=1-np.exp(gp).max(1))
        for key,value in vals.items():result[key].append(value)
    return {key:np.concatenate(value) for key,value in result.items()}


def fit_holue(core,val,test,vlegacy,tlegacy,select,far,out,seed,synthetic=False):
    if synthetic:
        v,t=vlegacy,tlegacy
        pars=fit_logistic(np.column_stack((v['kl1'],v['kl2']))[select],(vlegacy['actions']!=val.targets)[select],seed)
        return (apply_logistic(np.column_stack((v['kl1'],v['kl2'])),pars),
                apply_logistic(np.column_stack((t['kl1'],t['kl2'])),pars),v,t,dict(synthetic_standin=True,**pars))
    import torch
    from omegaconf import open_dict,OmegaConf
    from evaluation.metrics import FrrFarIdent
    v=run_legacy(core,val,far,'HolUE',vlegacy['gallery_kappa']);t=run_legacy(core,test,far,'HolUE',tlegacy['gallery_kappa'])
    conf=native_method(core,'HolUE').calibration_transform
    conf=OmegaConf.create(OmegaConf.to_container(conf,resolve=True))
    with open_dict(conf):
        conf.log_dir=str(out/'holue_calibrator');conf.normalize_kl_by_test=False
    torch.manual_seed(seed);np.random.seed(seed)
    cal=instantiate_config(conf);cal.return_mode='error_prob';cal.vis=False
    err=FrrFarIdent();a=vlegacy['actions'][select]
    # For rejected probes the selected known index is immaterial for correctness.
    err(np.maximum(a-1,0),a==0,val.gallery_ids,val.probe_ids[select])
    cal.train_calibration_parameters(v['kl1'][select],v['kl2'][select],err,
                                    dataset_name=val.source['dataset_name'],far=far)
    pv=cal.apply_calibration_transform(v['kl1'],v['kl2'],None,dataset_name='validation',far=far)
    pt=cal.apply_calibration_transform(t['kl1'],t['kl2'],None,dataset_name='test',far=far)
    dest=out/'holue_calibrator';dest.mkdir(parents=True,exist_ok=True)
    torch.save(cal.model.state_dict(),dest/'state_dict.pt')
    norm={k:v.detach().cpu().numpy().tolist() if torch.is_tensor(v) else v for k,v in vars(cal).items()
          if k in ['X_mean_val','X_std_val','x_mean','x_std','mean','std','return_mode','use_norm','normalize_kl_by_test']}
    write_json(dest/'normalization.json',norm)
    return np.asarray(pv),np.asarray(pt),v,t,dict(config=OmegaConf.to_container(conf,resolve=True),training_indices=select.tolist())


from experiments.mprisk_evidence.score_updates import score_variants


def record_scores(tables,context,split,data,legacy,score_dict,probability,idx=None):
    ix=np.arange(data.n) if idx is None else np.asarray(idx,dtype=int)
    a=legacy['actions'][ix];y=data.targets[ix];m=Metrics(a,y,context['seed']);common=dict(context,split=split)
    for name,whole_score in score_dict.items():
        score=whole_score[ix]
        pp=probability.get(name);side='test' if split=='test' else 'validation'
        probs=None if pp is None else pp[side][ix]
        logits=None if pp is None or pp[side+'_log_odds'] is None else pp[side+'_log_odds'][ix]
        row=m.evaluate(score,probabilities=probs,log_odds=logits)
        tables.add('main_mprisk_core_comparison',row,method=name,**common)
        curves,rc=m.curves(score);tables.add('rejection_curves',curves,method=name,**common);tables.add('risk_coverage_curves',rc,method=name,**common)
        for target in ['any_error','false_accept','false_reject','misidentification']:
            yy=m.m[target];tables.add('error_type_detection',dict(target=target,positives=int(yy.sum()),n=len(yy),auroc=auc(yy,score),auprc=ap(yy,score)),method=name,**common)
        # Conditional FR/FA evaluation avoids rewarding the trivial accepted/rejected gate.
        for subset,target in [('rejected','false_reject'),('accepted','false_accept'),('accepted','misidentification')]:
            keep=(a==0) if subset=='rejected' else (a>0);yy=m.m[target][keep]
            tables.add('error_type_detection_conditional',dict(subset=subset,target=target,n=int(keep.sum()),positives=int(yy.sum()),
                   auroc=auc(yy,score[keep]) if keep.any() else np.nan,auprc=ap(yy,score[keep]) if keep.any() else np.nan),method=name,**common)
        if probs is not None:
            for subset,keep in [('accepted',a>0),('rejected',a==0)]:
                if keep.any():
                    tables.add('conditional_calibration',Metrics(a[keep],y[keep],context['seed']).evaluate(
                        score[keep],probabilities=probs[keep],log_odds=None if logits is None else logits[keep]),
                        subset=subset,method=name,**common)
        if name in probability:
            br=bins(probs,m.m['any_error'])
            ece=sum(b['count']*abs(b['predicted']-b['observed']) for b in br if b['count'])/len(score)
            tables.add('reliability_bins',br,method=name,**common)
            tables.add('reliability_metrics',dict(ece=ece,**row),method=name,**common)
        if name.startswith(('MPRisk','Linear','Hybrid','Supervised','HolUE')):
            tables.add('fair_tuning_comparison',row,method=name,**common)


def quality_diagnostics(tables,context,val,test,vold,told,pv,pt,scores):
    edges=np.quantile(val.kappa,[0,.25,.5,.75,1]);group=np.searchsorted(edges[1:-1],test.kappa,side='right')
    tables.add('quality_shift',dict(validation_mean=float(val.kappa.mean()),test_mean=float(test.kappa.mean()),
               validation_median=float(np.median(val.kappa)),test_median=float(np.median(test.kappa)),
               ks_distance=float(ks_2samp(val.kappa,test.kappa).statistic),validation_quantile_edges=edges.tolist()),**context)
    a,y=told['actions'],test.targets;m=masks(a,y)
    for q in range(4):
        ix=np.flatnonzero(group==q)
        if not len(ix):continue
        for name in ['MPRisk','MPRisk-paper retuned','MPRisk-paper no NS','HolUE (refit fixed decisions)']:
            tables.add('quality_stratified',Metrics(a[ix],y[ix],context['seed']).evaluate(scores[name][ix],probabilities=pt['risk'][ix] if name=='MPRisk' else None,log_odds=pt['score_log_odds'][ix] if name=='MPRisk' else None),method=name,quality_bin=q,**context)
        for state in ['false_reject','true_reject','false_accept','misidentification','tp']:
            j=ix[m[state][ix]]
            tables.add('quality_error_groups',dict(quality_bin=q,state=state,n=len(j),
               mean_risk=float(pt['risk'][j].mean()) if len(j) else np.nan,
               mean_p_unknown=float(pt['p0'][j].mean()) if len(j) else np.nan,
               mean_kappa=float(test.kappa[j].mean()) if len(j) else np.nan),**context)
    # Representative examples include both confident mistakes and uncertain correct decisions.
    for label,keep,descending in [('confident_errors',m['any_error'],False),('uncertain_correct',~m['any_error'],True),
                                  ('false_rejections',m['false_reject'],True),('true_rejections',m['true_reject'],True)]:
        ix=np.flatnonzero(keep);ix=ix[np.argsort(pt['score_log_odds'][ix])];ix=ix[::-1] if descending else ix
        for j in ix[:10]:tables.add('qualitative_examples',dict(group=label,index=int(j),template=str(test.template_ids[j]),
              subject=str(test.probe_ids[j]),target=int(y[j]),action=int(a[j]),kappa=float(test.kappa[j]),
              risk=float(pt['risk'][j]),score_log_odds=float(pt['score_log_odds'][j]),p_unknown=float(pt['p0'][j]),selected_probability=float(pt['selected_probability'][j])),**context)


def full_extras(tables,context,args,val,test,cval,ctest,parts,pars,table,vold,told,pv,pt,sv,st,learned,features,root):
    a,y=told['actions'],test.targets;av,yv=vold['actions'],val.targets;select=parts['select'];cv=features['validation_components'];ct=features['test_components']
    base=Metrics(a,y,args.seed)
    # All nonempty subsets of the THREE risk components; no meaningless zero-NS ablations.
    for code in range(1,8):
        cols=[j for j in range(3) if code&(1<<j)];name='+'.join(['FA','ID','FR'][j] for j in cols)
        tables.add('mprisk_component_ablation',base.evaluate(cost_log_odds(ct,np.array([float(j in cols) for j in range(3)]))),variant=name,**context)
    for fraction in [.05,.1,.25,.5,1.]:
        for repeat in range(3):
            n=max(8,int(len(select)*fraction));ix=np.random.default_rng(args.seed+repeat).choice(select,min(n,len(select)),replace=False)
            p=fit_risk_weights(cv[ix],av[ix],yv[ix],args.search_budget,args.seed+repeat)
            tables.add('validation_size_ablation',base.evaluate(apply_linear(ct,p)),fraction=fraction,repeat=repeat,**context)
            tables.add('lambda_stability',dict(fraction=fraction,repeat=repeat,weights=p['raw_weights'],selection_count=len(ix)),**context)
    # Actual probability-model refitting on subsets, separate from score-weight sample-size tests.
    for fraction in [.25,.5,1.]:
        ix=parts['fit'];n=max(8,int(len(ix)*fraction));sub=np.sort(np.random.default_rng(args.seed).choice(ix,n,replace=False))
        pp,report=fit_parameters(cval,val.kappa,val.d,yv,sub,select,args.beta,table,maxiter=args.fit_iterations,max_fit=args.max_fit,seed=args.seed)
        v=evaluate(ctest,test.kappa,test.d,pp,a,y,table)
        tables.add('probability_fit_sample_size',dict(fraction=fraction,fit_n=len(sub),probe_scale=pp.probe_scale,gallery_kappa=pp.gallery_kappa,
                   class_nll=float(-v['true_log_probability'].mean()),**base.evaluate(v['score_log_odds'],probabilities=v['risk'],log_odds=v['score_log_odds'])),**context)
        write_json(root/f'probability_refit_{fraction}.json',report)
    # Post-hoc parameter perturbations are diagnostics, NEVER selected on test.
    for parameter,factors in [('probe_scale',[0.,.1,.3,1.,3.,10.]),('gallery_kappa',[.5,1.,2.]),('beta',[.1,.25,.5,.75,.9])]:
        for factor in factors:
            value=factor if parameter=='beta' else getattr(pars,parameter)*factor
            pp=replace(pars,**{parameter:value});v=evaluate(ctest,test.kappa,test.d,pp,a,y,table)
            tables.add('hyperparameter_sensitivity',dict(parameter=parameter,value=value,diagnostic_only=True,
                       class_nll=float(-v['true_log_probability'].mean()),**base.evaluate(v['score_log_odds'],probabilities=v['risk'],log_odds=v['score_log_odds'])),**context)
    from scipy.special import expit
    prior_error=np.where(a==0,1-args.beta,1-(1-args.beta)/len(test.gallery))
    prior_odds=np.log(prior_error)-np.log1p(-prior_error)
    for name,odds in [('uniform_probe',prior_odds),('no_unknown_event',pt['known_only_error_log_odds'])]:
        tables.add('background_model_ablation',base.evaluate(odds,probabilities=expit(odds),log_odds=odds),variant=name,
                   note='tests unknown-event evidence, NOT necessity of an explicit continuum',**context)
    bootnames=['MPRisk','MPRisk-paper retuned','HolUE (refit fixed decisions)','Linear all baseline scores','Linear all + MPRisk']
    tables.add('bootstrap_prr_differences',bootstrap(a,y,{n:st[n] for n in bootnames},'MPRisk',test.probe_ids,args.bootstrap,args.seed),**context)
    # Risk bound uses independent audit only, never the score-selection subset.
    ix=parts['audit'];pred=pv['risk'][ix];err=(av[ix]!=yv[ix]).astype(float);B=min(1.,np.mean((pred-err)**2)+np.sqrt(np.log(20)/(2*len(ix))))
    for coverage in [.25,.5,.75,.9]:
        tables.add('audit_risk_bound',dict(coverage=coverage,n=len(ix),delta=.05,brier=float(np.mean((pred-err)**2)),
             excess_bound=min(1.,np.sqrt(2*min(coverage,1-coverage)*B)/coverage),
             certified=False,assumptions='Conditional calculation only: iid audit probes and exact population coverage required; dependence not certified; ECE is not TV'),**context)


def run_dataset(args,core,name,root,tables,transfer):
    out=root/'datasets'/slug(name);out.mkdir(parents=True,exist_ok=True)
    if args.synthetic:val,test=synthetic_pair(args.seed,args.synthetic_n);fars=[.1] if args.stage=='sanity' else [.05,.1]
    else:
        config=next((d for d in core.test_datasets if str(d.dataset_name)==name),None)
        if config is None:raise KeyError(name)
        test=load_native(config);val=load_native(core.dataset_name_to_calibration_set[name])
        fars=[.1] if args.stage=='sanity' else list(map(float,core.far_list))
    original_n=(val.n,test.n)
    previous=None
    if getattr(args,'previous_run',None):
        previous=PreviousRun(args.previous_run,args.domain,args.seed,args.beta,args.synthetic)
        val,test,vi,ti,parts,unit=reuse_split(previous,name,val,test)
    else:
        if args.stage=='sanity':val,vi=subsample(val,args.sanity_val,args.seed);test,ti=subsample(test,args.sanity_test,args.seed+1)
        else:vi=np.arange(val.n);ti=np.arange(test.n)
        parts,unit=split_validation(val,args.seed)
    if val.d!=test.d:raise ValueError('Calibration and test representation dimensions differ')
    write_json(out/'data_manifest.json',dict(validation=val.metadata(),test=test.metadata(),original_n=original_n,
                 validation_original_indices=vi,test_original_indices=ti,
                 beta=args.beta,beta_is_not_fpir=True,synthetic=args.synthetic,
                 previous_run_fingerprint=None if previous is None else previous.fingerprint))
    write_json(out/'validation_split.json',dict(unit=unit,**parts))
    cval,backend=similarities(val,out/'_cache'/'validation_cosines.npy',args.device)
    ctest,_=similarities(test,out/'_cache'/'test_cosines.npy',args.device)
    table=PartitionTable(val.d);write_json(out/'partition_numerics.json',table.manifest())
    if previous is not None:
        try:replay_dataset(previous,name,val,test,cval,ctest,parts,table,root,tables,args.seed)
        finally:previous.close()
        tables.save()
    if getattr(args,'replay_only',False):
        write_json(out/'status.json',dict(status='complete',replay_only=True,fars=[]))
        return
    pars,report=fit_parameters(cval,val.kappa,val.d,val.targets,parts['fit'],parts['select'],args.beta,table,
                     maxiter=args.fit_iterations,max_fit=args.max_fit,seed=args.seed,report_path=out/'evidence_fit_attempts.json')
    point,pr=fit_parameters(cval,val.kappa,val.d,val.targets,parts['fit'],parts['select'],args.beta,table,point=True,
                     maxiter=args.fit_iterations,max_fit=args.max_fit,seed=args.seed,report_path=out/'point_fit_attempts.json')
    write_json(out/'probability_model_fit.json',dict(parameters=asdict(pars),fit=report,point_parameters=asdict(point),point_fit=pr,
                 note='Shared vMF gallery dispersion is FIT ON VALIDATION, not inherited from power-spherical FAR matching'))
    for model,rep in [('evidence',report),('point',pr)]:
        for row in rep.get('coordinate_slices',[]):tables.add('validation_parameter_slices',row,dataset=name,model=model,seed=args.seed)
    model_limit_diagnostics(tables,name,cval,val.kappa,val.d,val.targets,parts,pars,table,args.seed)
    tables.add('probe_distribution_sanity',probe_limit_rows(val.d,pars.gallery_kappa,args.beta),dataset=name,seed=args.seed)
    tables.add('model_parameters',dict(**asdict(pars),validation_partition_unit=unit,n_validation=val.n,n_test=test.n,backend=backend),dataset=name,seed=args.seed)
    transfer[name]=dict(pars=pars,d=val.d,fars={})
    for far in fars:
        case=out/f'fpir_{far:g}';case.mkdir(exist_ok=True)
        context=dict(dataset=name,target_fpir=far,beta=args.beta,seed=args.seed)
        print(f'[{name} FPIR={far}] freeze common recognition decisions',flush=True)
        start=time.perf_counter()
        protocol=getattr(args,'reference_protocol','empirical')
        if protocol=='empirical':
            kind='vMF' if args.synthetic else str(native_method(core,'MPRisk raw').get('gallery_prior','power'))
            vr=fit_reference(cval[parts['fit']],val.targets[parts['fit']],far,args.beta,val.d,kind)
            tr=fit_reference(ctest,test.targets,far,args.beta,test.d,kind)
            av=actions_at_threshold(cval,vr['tau']);at=actions_at_threshold(ctest,tr['tau'])
            if args.synthetic:
                vold=synthetic_legacy(val,cval,far,args.beta,tau=vr['tau']);told=synthetic_legacy(test,ctest,far,args.beta,tau=tr['tau'])
            else:
                vold=run_legacy(core,val,far,gallery_kappa=vr['gallery_kappa'],actions_override=av,cosines=cval)
                told=run_legacy(core,test,far,gallery_kappa=tr['gallery_kappa'],actions_override=at,cosines=ctest)
            vold['tau']=vr['tau'];told['tau']=tr['tau']
            for split,rec in [('validation_fit',vr),('test_benchmark',tr)]:
                tables.add('reference_operating_point',dict(split=split,**rec),**context)
            if np.mean(told['actions'][test.targets==0]>0)!=tr['achieved_fpir']:raise AssertionError('Reference FPIR mismatch')
        else:
            if args.synthetic:
                fitted_reference=synthetic_legacy(val.take(parts['fit']),cval[parts['fit']],far,args.beta)
                vold=synthetic_legacy(val,cval,far,args.beta,tau=fitted_reference['tau']);told=synthetic_legacy(test,ctest,far,args.beta)
            else:
                fitted_reference=run_legacy(core,val.take(parts['fit']),far,cosines=cval[parts['fit']])
                vold=run_legacy(core,val,far,gallery_kappa=fitted_reference['gallery_kappa'],cosines=cval)
                told=run_legacy(core,test,far,cosines=ctest)
            vr=tr=None
        write_json(case/'reference_decision_fit.json',dict(protocol=protocol,
            validation_gallery_kappa=vold['gallery_kappa'],test_gallery_kappa=told['gallery_kappa'],
            validation_fit_indices=parts['fit'],validation_rule_uses_audit_labels=False,
            validation_matching=vr,test_matching=tr,
            test_rule='test-unknown-label benchmark operating point, frozen for all scores; NOT prospective threshold calibration'))
        support_diagnostics(tables,context,val,test,parts,vold['actions'],told['actions'])
        legacy_seconds=time.perf_counter()-start
        av,at=vold['actions'],told['actions'];vh=digest_array(av);th=digest_array(at)
        start=time.perf_counter();pv=evaluate(cval,val.kappa,val.d,pars,av,val.targets,table);pt=evaluate(ctest,test.kappa,test.d,pars,at,test.targets,table)
        evidence_seconds=time.perf_counter()-start
        pointv=evaluate(cval,val.kappa,val.d,point,av,val.targets,table);pointt=evaluate(ctest,test.kappa,test.d,point,at,test.targets,table)
        # Independent exact numerical check at selected actual data rows.
        ix=np.arange(min(test.n,32));exact=evaluate(ctest[ix],test.kappa[ix],test.d,pars,at[ix],test.targets[ix],None)
        err=float(np.max(np.abs(exact['risk']-pt['risk'][ix])))
        if err>2e-6:raise AssertionError(f'Exact/interpolated risk mismatch: {err}')
        logerr=float(np.max(np.abs(exact['score_log_odds']-pt['score_log_odds'][ix])))
        if logerr>2e-6:raise AssertionError(f'Exact/interpolated log-odds mismatch: {logerr}')
        delta=exact['score_log_odds'][:,None]-exact['score_log_odds'][None,:]
        approx=pt['score_log_odds'][ix];delta_approx=approx[:,None]-approx[None,:]
        resolved=np.abs(delta)>2*logerr+1e-10
        inversions=int(np.sum(resolved & (delta*delta_approx<0)))
        if inversions:raise AssertionError('Numerically resolvable ranking inversions')
        tables.add('normalization_audit',dict(max_mass_error=float(np.max(np.abs(pt['posterior_mass']-1))),
                  exact_risk_error=err,exact_log_odds_error=logerr,resolved_ranking_inversions=inversions,
                  probability_zero_count=int(np.sum(pt['risk']==0)),probability_one_count=int(np.sum(pt['risk']==1)),
                  distinct_log_odds_count=int(len(np.unique(pt['score_log_odds']))),risk_min=float(pt['risk'].min()),risk_max=float(pt['risk'].max()),
                  counterfactual_disagreement=float(np.mean(pt['counterfactual_action']!=at)),
                  scored_action_hash=th,reference_action_hash=th,passed=True),**context)
        tables.add('posterior_scoring',dict(class_nll=float(-pt['true_log_probability'].mean()),class_brier=float(pt['class_brier'].mean()),
                  known_prevalence=float(np.mean(test.targets>0)),prior_known=1-args.beta,
                  note='Calibration metrics refer to this observed test mixture; prior/prevalence mismatch is not hidden'),split='test',**context)
        ai=parts['audit']
        tables.add('posterior_scoring',dict(class_nll=float(-pv['true_log_probability'][ai].mean()),class_brier=float(pv['class_brier'][ai].mean()),known_prevalence=float(np.mean(val.targets[ai]>0)),prior_known=1-args.beta),split='audit',**context)
        sv,st,probability,learned,feat=score_variants(val,test,cval,ctest,pv,pt,pointv,pointt,vold,told,parts['select'],core,far,case,args,pars.probe_scale)
        for fit_name,fit_value in learned.items():
            if 'identifiable' in fit_value and ('calibrat' in fit_name):
                tables.add('calibration_fit',dict(identifiable=fit_value['identifiable'],success=fit_value.get('success'),
                    input_score=fit_value.get('input_score'),n=fit_value.get('n'),positives=fit_value.get('positives')),method=fit_name,**context)
        # Exact zero-NS candidate must equal the fitted core in the selection objective.
        for ns_name in ['MPRisk + NS diagnostic','MPRisk + NS original-scale diagnostic']:
            if learned[ns_name]['validation_prr']+1e-12<learned['MPRisk PRR-weighted']['validation_prr']:
                raise AssertionError('NS diagnostic lost its nested no-NS candidate')
        record_scores(tables,context,'audit',val,vold,sv,probability,parts['audit']);record_scores(tables,context,'test',test,told,st,probability)
        for fit_name,fit_value in learned.items():
            if 'raw_weights' in fit_value:
                tables.add('fair_tuning_weights',dict(weights=fit_value['raw_weights'],features=fit_value.get('feature_names',[]),selection_prr=fit_value['validation_prr'],identifiable=fit_value['identifiable']),method=fit_name,**context)
        quality_diagnostics(tables,context,val,test,vold,told,pv,pt,st)
        quality_distance_grid(tables,context,val,test,cval,ctest,parts,vold,told,sv,st,pv,pt)
        for other in ['MPRisk-paper raw','MPRisk-paper retuned','HolUE (refit fixed decisions)','Linear all baseline scores']:
            tables.add('ranking_comparison',ranks(st['MPRisk'],st[other],args.seed),reference='MPRisk',comparison=other,**context)
        tables.add('runtime_overhead',dict(legacy_including_FAR_matching_seconds=legacy_seconds,
             evidence_validation_plus_test_seconds=evidence_seconds,n=val.n+test.n,d=test.d,K=len(test.gallery),
             evidence_per_probe_ms=1000*evidence_seconds/(val.n+test.n),
             includes_encoder=False,includes_similarity=False,backend='CPU float64 evidence; '+backend+' cosine'),**context)
        if args.stage=='full':full_extras(tables,context,args,val,test,cval,ctest,parts,pars,table,vold,told,pv,pt,sv,st,learned,feat,case)
        if vh!=digest_array(av) or th!=digest_array(at):raise AssertionError('Recognition actions were modified')
        for label,data,legacy,p,scores in [('validation',val,vold,pv,sv),('test',test,told,pt,st)]:
            path=case/'per_example'/f'{label}.npz';path.parent.mkdir(exist_ok=True)
            names=list(scores);save=dict(actions=legacy['actions'],targets=data.targets,subject_ids=np.asarray(data.probe_ids).astype(str),
                 template_ids=np.asarray(data.template_ids).astype(str),gallery_ids=np.asarray(data.gallery_ids).astype(str),
                 kappa=data.kappa,effective_kappa=data.kappa*pars.probe_scale,score_names=np.array(names),scores=np.column_stack([scores[n] for n in names]),
                 r_ns_legacy=legacy['r_ns'],log_nonspecificity=legacy['log_nonspecificity'],
                 max_cosine=np.max(cval if label=='validation' else ctest,axis=1),**p)
            prob_names=list(probability)
            save['probability_names']=np.array(prob_names)
            save['probabilities']=np.column_stack([probability[n][label] for n in prob_names])
            save['probability_log_odds']=np.column_stack([probability[n][label+'_log_odds'] if probability[n][label+'_log_odds'] is not None else np.full(data.n,np.nan) for n in prob_names])
            save['schema_version']=np.array('risk-score-separation-1.1')
            save['historical_probability_argmax_actions']=legacy.get('native_actions',legacy['actions'])
            if label=='validation':
                role=np.full(data.n,'',dtype='U8')
                for key,indices in parts.items():role[indices]=key
                save['partition']=role
            np.savez_compressed(path,**save)
        transfer[name]['fars'][far]=dict(weights=learned['MPRisk PRR-weighted'],components=feat['test_components'],actions=at.copy(),targets=test.targets.copy())
        write_json(case/'transfer_weights.json',learned['MPRisk PRR-weighted'])
        np.savez_compressed(case/'transfer_arrays.npz',components=feat['test_components'],actions=at,targets=test.targets)
        # Retain reference calibrated scores/features for all subsequent audits.
        np.savez_compressed(case/'fusion_features.npz',validation=feat['validation_features'],test=feat['test_features'],names=np.array(feat['feature_names']),
                            validation_kl=feat['validation_kl'],test_kl=feat['test_kl'],validation_augmented=feat['validation_augmented_features'],test_augmented=feat['test_augmented_features'],augmented_names=np.array(feat['augmented_feature_names']))
        tables.save();write_json(case/'status.json',dict(status='complete',reference_actions_sha256=th,model_version=MODEL_VERSION))
    write_json(out/'status.json',dict(status='complete',fars=fars,parameters=asdict(pars)))
    if args.stage=='full':
        # Parameter transfer between same-dimensional datasets is optional and expensive;
        # score-weight transfer is the direct counterpart of the original study.
        current=transfer[name]
        for source_far,src in current['fars'].items():
            for target_far,dest in current['fars'].items():
                if source_far==target_far:continue
                u=apply_linear(dest['components'],src['weights']);m=Metrics(dest['actions'],dest['targets'],args.seed)
                tables.add('operating_point_transfer',m.evaluate(u),dataset=name,source_fpir=source_far,target_fpir=target_far,seed=args.seed)
    tables.save()


def cross_dataset_transfer(transfer,tables,args):
    for source,s in transfer.items():
        for target,t in transfer.items():
            if source==target:continue
            for far in sorted(set(s['fars'])&set(t['fars'])):
                src,dst=s['fars'][far],t['fars'][far]
                u=apply_linear(dst['components'],src['weights']);m=Metrics(dst['actions'],dst['targets'],args.seed)
                tables.add('cross_dataset_transfer',m.evaluate(u),source_dataset=source,target_dataset=target,target_fpir=far,seed=args.seed,
                    transferred='three component score weights and source standardization; destination posterior remains locally fitted')
    tables.save()


def reload_transfer(root,names):
    out={}
    for name in names:
        base=Path(root)/'datasets'/slug(name)
        if not (base/'status.json').exists():continue
        status=json.loads((base/'status.json').read_text())
        if status.get('status')!='complete':continue
        out[name]={'fars':{}}
        for far in status['fars']:
            case=base/f'fpir_{far:g}'
            with np.load(case/'transfer_arrays.npz',allow_pickle=False) as f:arr={k:f[k] for k in f.files}
            arr['weights']=json.loads((case/'transfer_weights.json').read_text())
            out[name]['fars'][far]=arr
    return out
