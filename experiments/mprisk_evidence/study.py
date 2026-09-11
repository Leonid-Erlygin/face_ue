"""Cross-domain sanity and full, fixed-recognizer evidence-risk experiments."""
from __future__ import annotations
from dataclasses import asdict,replace
from pathlib import Path
import hashlib,json,time
import numpy as np
from scipy.special import logsumexp,gammaln
from scipy.stats import ks_2samp
from evaluation.open_set_methods.mprisk_evidence import (MODEL_VERSION,Parameters,PartitionTable,evaluate,fit_parameters,
                 log_bayes_factors,log_posterior,log_nonspecificity,log_surface)
from experiments.mprisk_evidence.data import (load_native,similarities,split_validation,run_legacy,native_method,
                 synthetic_pair,synthetic_legacy,digest_array,instantiate_config)
from experiments.mprisk_evidence.metrics import (Metrics,masks,auc,ap,bins,fit_linear,apply_linear,fit_monotone,
                 apply_monotone,fit_logistic,apply_logistic,ranks,bootstrap,fit_mlp,apply_mlp,fit_positive_score_calibration,apply_positive_score_calibration)
from experiments.mprisk_evidence.artifacts import write_json,Tables
from experiments.mprisk_evidence.audits import probe_limit_rows


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
            if gal['gallery_prior']=='power':
                ln=gammaln(d-1+kg)+gammaln(d/2+kg)+(kg-1)*np.log(2)-d/2*np.log(np.pi)-gammaln(d-1+2*kg)
                loglike=ln+kg*np.log1p(np.clip(cc,-1+1e-9,1-1e-9))
            else:
                from evaluation.open_set_methods.mprisk_evidence import log_partition
                loglike=-log_surface(d)-log_partition(kg,d)+kg*cc
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


def score_variants(val,test,cval,ctest,pv,pt,pointv,pointt,vold,told,select,core,far,root,args,probe_scale=1.):
    av,at=vold['actions'],told['actions'];yv,yt=val.targets,test.targets
    cv=np.column_stack([pv[k] for k in ['r_fa','r_id','r_fr']]);ct=np.column_stack([pt[k] for k in ['r_fa','r_id','r_fr']])
    ov=np.column_stack([vold['components'][k] for k in ['r_fa','r_id','r_fr']]+[vold['r_ns']])
    ot=np.column_stack([told['components'][k] for k in ['r_fa','r_id','r_fr']]+[told['r_ns']])
    sv={'MPRisk':pv['risk'],'Point-vMF (NLL fitted)':pointv['risk'],'MPRisk-paper raw':ov.sum(1),'MPRisk-paper no NS':ov[:,:3].sum(1)}
    st={'MPRisk':pt['risk'],'Point-vMF (NLL fitted)':pointt['risk'],'MPRisk-paper raw':ot.sum(1),'MPRisk-paper no NS':ot[:,:3].sum(1)}
    probability={'MPRisk','Point-vMF (NLL fitted)','MPRisk-paper no NS'}
    tv,tt=threshold_from_actions(cval,av),threshold_from_actions(ctest,at)
    beta=float(args.beta)
    # Match the repository MSP objective, but only on our selection subset and
    # against the SAME frozen labels used for every method.
    candidates=np.geomspace(.001,50,25 if args.stage=='full' else 10);best=(-np.inf,1.)
    for temp in candidates:
        u=simple_features(cval[select],val.kappa[select],tv,float(temp))['MSP']
        sc=auc(av[select]!=yv[select],u)
        if np.isfinite(sc) and sc>best[0]:best=(sc,float(temp))
    gv=dict(vold);gt=dict(told)
    if core is not None and 'dataset_name_to_T_scale' in core:
        gv['galue_T']=gt['galue_T']=float(core.dataset_name_to_T_scale[test.source['dataset_name']])
    bv=simple_features(cval,val.kappa,tv,best[1],gv,beta,val.d);bt=simple_features(ctest,test.kappa,tt,best[1],gt,beta,test.d)
    sv.update(bv);st.update(bt)
    hv,ht,vh,th,hpars=fit_holue(core,val,test,vold,told,select,far,root,args.seed,args.synthetic)
    sv['HolUE (refit fixed decisions)']=hv;st['HolUE (refit fixed decisions)']=ht
    # A supervised error-probability output, not a theorem about model correctness.
    probability.add('HolUE (refit fixed decisions)')
    sv['HolUE raw']=-(vh['kl1']+vh['kl2']);st['HolUE raw']=-(th['kl1']+th['kl2'])
    simple_names=['SCF','AccScr','MSP','Margin'];all_names=['SCF','AccScr','MSP','Entropy','Margin','GalUE','HolUE (refit fixed decisions)','HolUE raw']
    Xv=np.column_stack([sv[n] for n in all_names]);Xt=np.column_stack([st[n] for n in all_names])
    learned={}
    nsv=(av==0)*pv['p0']*np.exp(log_nonspecificity(val.kappa*probe_scale,val.d))
    nst=(at==0)*pt['p0']*np.exp(log_nonspecificity(test.kappa*probe_scale,test.d))
    kv=np.column_stack((vh['kl1'],vh['kl2']));kt=np.column_stack((th['kl1'],th['kl2']))
    specifications=[('MPRisk PRR-weighted',cv,ct,True,[np.ones(3)]),
                    ('MPRisk-paper retuned',ov,ot,True,[np.ones(4)]),
                    ('Linear simple',np.column_stack([sv[n] for n in simple_names]),np.column_stack([st[n] for n in simple_names]),False,None),
                    ('Linear all baseline scores',Xv,Xt,False,None),
                    ('Linear all + MPRisk',np.column_stack((Xv,cv)),np.column_stack((Xt,ct)),False,[np.r_[np.zeros(len(all_names)),np.ones(3)]]),
                    ('MPRisk + NS diagnostic',np.column_stack((cv,nsv)),np.column_stack((ct,nst)),True,[np.array([1,1,1,0])]),
                    ('Linear HolUE KL',kv,kt,False,None),
                    ('Hybrid KL + MPRisk',np.column_stack((kv,cv)),np.column_stack((kt,ct)),False,[np.r_[np.zeros(2),np.ones(3)]])]
    feature_sets={
       'MPRisk PRR-weighted':['r_FA','r_ID','r_FR'],
       'MPRisk-paper retuned':['paper_r_FA','paper_r_ID','paper_r_FR','paper_r_NS'],
       'Linear simple':simple_names, 'Linear all baseline scores':all_names,
       'Linear all + MPRisk':all_names+['r_FA','r_ID','r_FR'],
       'MPRisk + NS diagnostic':['r_FA','r_ID','r_FR','posterior_P0_times_N_of_scaled_probe'],
       'Linear HolUE KL':['KL1','KL2'],
       'Hybrid KL + MPRisk':['KL1','KL2','r_FA','r_ID','r_FR']}
    for name,v,t,positive,include in specifications:
        pars=fit_linear(v[select],av[select],yv[select],args.search_budget,args.seed,positive,include)
        pars['feature_names']=feature_sets[name]
        sv[name]=apply_linear(v,pars);st[name]=apply_linear(t,pars);learned[name]=pars
    oldcal=fit_positive_score_calibration(sv['MPRisk-paper retuned'][select],av[select]!=yv[select])
    learned['MPRisk-paper calibrated']=oldcal
    sv['MPRisk-paper calibrated']=apply_positive_score_calibration(sv['MPRisk-paper retuned'],oldcal)
    st['MPRisk-paper calibrated']=apply_positive_score_calibration(st['MPRisk-paper retuned'],oldcal)
    probability.add('MPRisk-paper calibrated')
    mon=fit_monotone(pv['risk'][select],av[select]!=yv[select]);learned['MPRisk calibrated']=mon
    sv['MPRisk calibrated']=apply_monotone(pv['risk'],mon);st['MPRisk calibrated']=apply_monotone(pt['risk'],mon);probability.add('MPRisk calibrated')
    logit=fit_logistic(np.column_stack((Xv,cv))[select],av[select]!=yv[select],args.seed);learned['Supervised logistic']=logit
    sv['Supervised logistic']=apply_logistic(np.column_stack((Xv,cv)),logit);st['Supervised logistic']=apply_logistic(np.column_stack((Xt,ct)),logit);probability.add('Supervised logistic')
    if args.stage=='full':
        net=fit_mlp(np.column_stack((Xv,cv))[select],av[select]!=yv[select],args.seed)
        learned['Supervised MLP']=net
        sv['Supervised MLP']=apply_mlp(np.column_stack((Xv,cv)),net)
        st['Supervised MLP']=apply_mlp(np.column_stack((Xt,ct)),net)
        probability.add('Supervised MLP')
    write_json(root/'score_fits.json',dict(fits=learned,MSP_temperature=best[1],MSP_selection_auroc=best[0],HolUE=hpars,
                                        baseline_features=all_names,selection_indices=select,
                                        note='PRR-tuned coefficients are score weights, not measured error costs'))
    return sv,st,probability,learned,dict(validation_components=cv,test_components=ct,validation_features=Xv,test_features=Xt,
                                        validation_old_components=ov,test_old_components=ot,feature_names=all_names,
                                        validation_kl=np.column_stack((vh['kl1'],vh['kl2'])),test_kl=np.column_stack((th['kl1'],th['kl2'])))


def record_scores(tables,context,split,data,legacy,score_dict,probability,idx=None):
    ix=np.arange(data.n) if idx is None else np.asarray(idx,dtype=int)
    a=legacy['actions'][ix];y=data.targets[ix];m=Metrics(a,y,context['seed']);common=dict(context,split=split)
    for name,whole_score in score_dict.items():
        score=whole_score[ix];row=m.evaluate(score,name in probability);tables.add('main_mprisk_core_comparison',row,method=name,**common)
        curves,rc=m.curves(score);tables.add('rejection_curves',curves,method=name,**common);tables.add('risk_coverage_curves',rc,method=name,**common)
        for target in ['any_error','false_accept','false_reject','misidentification']:
            yy=m.m[target];tables.add('error_type_detection',dict(target=target,positives=int(yy.sum()),n=len(yy),auroc=auc(yy,score),auprc=ap(yy,score)),method=name,**common)
        # Conditional FR/FA evaluation avoids rewarding the trivial accepted/rejected gate.
        for subset,target in [('rejected','false_reject'),('accepted','false_accept'),('accepted','misidentification')]:
            keep=(a==0) if subset=='rejected' else (a>0);yy=m.m[target][keep]
            tables.add('error_type_detection_conditional',dict(subset=subset,target=target,n=int(keep.sum()),positives=int(yy.sum()),
                   auroc=auc(yy,score[keep]) if keep.any() else np.nan,auprc=ap(yy,score[keep]) if keep.any() else np.nan),method=name,**common)
        if name in probability:
            br=bins(score,m.m['any_error'])
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
            tables.add('quality_stratified',Metrics(a[ix],y[ix],context['seed']).evaluate(scores[name][ix],name=='MPRisk'),method=name,quality_bin=q,**context)
        for state in ['false_reject','true_reject','false_accept','misidentification','tp']:
            j=ix[m[state][ix]]
            tables.add('quality_error_groups',dict(quality_bin=q,state=state,n=len(j),
               mean_risk=float(pt['risk'][j].mean()) if len(j) else np.nan,
               mean_p_unknown=float(pt['p0'][j].mean()) if len(j) else np.nan,
               mean_kappa=float(test.kappa[j].mean()) if len(j) else np.nan),**context)
    # Representative examples include both confident mistakes and uncertain correct decisions.
    for label,keep,descending in [('confident_errors',m['any_error'],False),('uncertain_correct',~m['any_error'],True),
                                  ('false_rejections',m['false_reject'],True),('true_rejections',m['true_reject'],True)]:
        ix=np.flatnonzero(keep);ix=ix[np.argsort(pt['risk'][ix])];ix=ix[::-1] if descending else ix
        for j in ix[:10]:tables.add('qualitative_examples',dict(group=label,index=int(j),template=str(test.template_ids[j]),
              subject=str(test.probe_ids[j]),target=int(y[j]),action=int(a[j]),kappa=float(test.kappa[j]),
              risk=float(pt['risk'][j]),p_unknown=float(pt['p0'][j]),selected_probability=float(pt['selected_probability'][j])),**context)


def full_extras(tables,context,args,val,test,cval,ctest,parts,pars,table,vold,told,pv,pt,sv,st,learned,features,root):
    a,y=told['actions'],test.targets;av,yv=vold['actions'],val.targets;select=parts['select'];cv=features['validation_components'];ct=features['test_components']
    base=Metrics(a,y,args.seed)
    # All nonempty subsets of the THREE risk components; no meaningless zero-NS ablations.
    for code in range(1,8):
        cols=[j for j in range(3) if code&(1<<j)];name='+'.join(['FA','ID','FR'][j] for j in cols)
        tables.add('mprisk_component_ablation',base.evaluate(ct[:,cols].sum(1)),variant=name,**context)
    for fraction in [.05,.1,.25,.5,1.]:
        for repeat in range(3):
            n=max(8,int(len(select)*fraction));ix=np.random.default_rng(args.seed+repeat).choice(select,min(n,len(select)),replace=False)
            p=fit_linear(cv[ix],av[ix],yv[ix],args.search_budget,args.seed+repeat,True,[np.ones(3)])
            tables.add('validation_size_ablation',base.evaluate(apply_linear(ct,p)),fraction=fraction,repeat=repeat,**context)
            tables.add('lambda_stability',dict(fraction=fraction,repeat=repeat,weights=p['raw_weights'],selection_count=len(ix)),**context)
    # Actual probability-model refitting on subsets, separate from score-weight sample-size tests.
    for fraction in [.25,.5,1.]:
        ix=parts['fit'];n=max(8,int(len(ix)*fraction));sub=np.sort(np.random.default_rng(args.seed).choice(ix,n,replace=False))
        pp,report=fit_parameters(cval,val.kappa,val.d,yv,sub,select,args.beta,table,maxiter=args.fit_iterations,max_fit=args.max_fit,seed=args.seed)
        v=evaluate(ctest,test.kappa,test.d,pp,a,y,table)
        tables.add('probability_fit_sample_size',dict(fraction=fraction,fit_n=len(sub),probe_scale=pp.probe_scale,gallery_kappa=pp.gallery_kappa,
                   class_nll=float(-v['true_log_probability'].mean()),**base.evaluate(v['risk'],True)),**context)
        write_json(root/f'probability_refit_{fraction}.json',report)
    # Post-hoc parameter perturbations are diagnostics, NEVER selected on test.
    for parameter,factors in [('probe_scale',[0.,.1,.3,1.,3.,10.]),('gallery_kappa',[.5,1.,2.]),('beta',[.1,.25,.5,.75,.9])]:
        for factor in factors:
            value=factor if parameter=='beta' else getattr(pars,parameter)*factor
            pp=replace(pars,**{parameter:value});v=evaluate(ctest,test.kappa,test.d,pp,a,y,table)
            tables.add('hyperparameter_sensitivity',dict(parameter=parameter,value=value,diagnostic_only=True,
                       class_nll=float(-v['true_log_probability'].mean()),**base.evaluate(v['risk'],True)),**context)
    for name,score in [('uniform_probe',np.where(a==0,1-args.beta,1-(1-args.beta)/len(test.gallery))),
                       ('no_unknown_event',np.where(a==0,1.,1-pt['selected_probability']/np.maximum(1-pt['p0'],1e-15)))]:
        tables.add('background_model_ablation',base.evaluate(np.clip(score,0,1),True),variant=name,
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
    if args.stage=='sanity':val,vi=subsample(val,args.sanity_val,args.seed);test,ti=subsample(test,args.sanity_test,args.seed+1)
    else:vi=np.arange(val.n);ti=np.arange(test.n)
    if val.d!=test.d:raise ValueError('Calibration and test representation dimensions differ')
    write_json(out/'data_manifest.json',dict(validation=val.metadata(),test=test.metadata(),original_n=original_n,
                 validation_original_indices=vi,test_original_indices=ti,
                 beta=args.beta,beta_is_not_fpir=True,synthetic=args.synthetic))
    parts,unit=split_validation(val,args.seed);write_json(out/'validation_split.json',dict(unit=unit,**parts))
    cval,backend=similarities(val,out/'_cache'/'validation_cosines.npy',args.device)
    ctest,_=similarities(test,out/'_cache'/'test_cosines.npy',args.device)
    table=PartitionTable(val.d);write_json(out/'partition_numerics.json',table.manifest())
    pars,report=fit_parameters(cval,val.kappa,val.d,val.targets,parts['fit'],parts['select'],args.beta,table,
                     maxiter=args.fit_iterations,max_fit=args.max_fit,seed=args.seed)
    point,pr=fit_parameters(cval,val.kappa,val.d,val.targets,parts['fit'],parts['select'],args.beta,table,point=True,
                     maxiter=args.fit_iterations,max_fit=args.max_fit,seed=args.seed)
    write_json(out/'probability_model_fit.json',dict(parameters=asdict(pars),fit=report,point_parameters=asdict(point),point_fit=pr,
                 note='Shared vMF gallery dispersion is FIT ON VALIDATION, not inherited from power-spherical FAR matching'))
    tables.add('probe_distribution_sanity',probe_limit_rows(val.d,pars.gallery_kappa,args.beta),dataset=name,seed=args.seed)
    tables.add('model_parameters',dict(**asdict(pars),validation_partition_unit=unit,n_validation=val.n,n_test=test.n,backend=backend),dataset=name,seed=args.seed)
    transfer[name]=dict(pars=pars,d=val.d,fars={})
    for far in fars:
        case=out/f'fpir_{far:g}';case.mkdir(exist_ok=True)
        context=dict(dataset=name,target_fpir=far,beta=args.beta,seed=args.seed)
        print(f'[{name} FPIR={far}] freeze original M=0 recognition decisions',flush=True)
        start=time.perf_counter()
        if args.synthetic:
            fitted_reference=synthetic_legacy(val.take(parts['fit']),cval[parts['fit']],far,args.beta)
            vold=synthetic_legacy(val,cval,far,args.beta,tau=fitted_reference['tau'])
            told=synthetic_legacy(test,ctest,far,args.beta)
        else:
            # The audit labels must not choose the validation decision rule.
            fitted_reference=run_legacy(core,val.take(parts['fit']),far)
            vold=run_legacy(core,val,far,gallery_kappa=fitted_reference['gallery_kappa'])
            told=run_legacy(core,test,far)
        write_json(case/'reference_decision_fit.json',dict(
            validation_gallery_kappa=vold['gallery_kappa'],test_gallery_kappa=told['gallery_kappa'],
            validation_fit_indices=parts['fit'],validation_rule_uses_audit_labels=False,
            test_rule='source target-FPIR benchmark construction, frozen for all uncertainty methods'))
        legacy_seconds=time.perf_counter()-start
        av,at=vold['actions'],told['actions'];vh=digest_array(av);th=digest_array(at)
        start=time.perf_counter();pv=evaluate(cval,val.kappa,val.d,pars,av,val.targets,table);pt=evaluate(ctest,test.kappa,test.d,pars,at,test.targets,table)
        evidence_seconds=time.perf_counter()-start
        pointv=evaluate(cval,val.kappa,val.d,point,av,val.targets,table);pointt=evaluate(ctest,test.kappa,test.d,point,at,test.targets,table)
        # Independent exact numerical check at selected actual data rows.
        ix=np.arange(min(test.n,32));exact=evaluate(ctest[ix],test.kappa[ix],test.d,pars,at[ix],test.targets[ix],None)
        err=float(np.max(np.abs(exact['risk']-pt['risk'][ix])))
        if err>2e-6:raise AssertionError(f'Exact/interpolated risk mismatch: {err}')
        tables.add('normalization_audit',dict(max_mass_error=float(np.max(np.abs(pt['posterior_mass']-1))),
                  exact_risk_error=err,risk_min=float(pt['risk'].min()),risk_max=float(pt['risk'].max()),
                  counterfactual_disagreement=float(np.mean(pt['counterfactual_action']!=at)),
                  scored_action_hash=th,reference_action_hash=th,passed=True),**context)
        tables.add('posterior_scoring',dict(class_nll=float(-pt['true_log_probability'].mean()),class_brier=float(pt['class_brier'].mean()),
                  known_prevalence=float(np.mean(test.targets>0)),prior_known=1-args.beta,
                  note='Calibration metrics refer to this observed test mixture; prior/prevalence mismatch is not hidden'),split='test',**context)
        ai=parts['audit']
        tables.add('posterior_scoring',dict(class_nll=float(-pv['true_log_probability'][ai].mean()),class_brier=float(pv['class_brier'][ai].mean()),known_prevalence=float(np.mean(val.targets[ai]>0)),prior_known=1-args.beta),split='audit',**context)
        sv,st,probability,learned,feat=score_variants(val,test,cval,ctest,pv,pt,pointv,pointt,vold,told,parts['select'],core,far,case,args,pars.probe_scale)
        record_scores(tables,context,'audit',val,vold,sv,probability,parts['audit']);record_scores(tables,context,'test',test,told,st,probability)
        for fit_name,fit_value in learned.items():
            if 'raw_weights' in fit_value:
                tables.add('fair_tuning_weights',dict(weights=fit_value['raw_weights'],features=fit_value.get('feature_names',[]),selection_prr=fit_value['validation_prr'],identifiable=fit_value['identifiable']),method=fit_name,**context)
        quality_diagnostics(tables,context,val,test,vold,told,pv,pt,st)
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
                            validation_kl=feat['validation_kl'],test_kl=feat['test_kl'])
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
