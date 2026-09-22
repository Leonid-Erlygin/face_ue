#!/usr/bin/env python3
"""Whale rerun: validation-weighted EviRisk, unit ablation, and diagnostics.

Run from the repository root. Real outputs use all probes by default. Raw input
images/audio are never read; native SCF representations and IJB-format metadata
are required. Synthetic mode tests the pipeline, not Whale performance.
"""
from __future__ import annotations
import argparse,datetime,json,sys,traceback,platform,shutil
from pathlib import Path
from dataclasses import asdict,replace
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr
from evaluation.open_set_methods.mprisk_evidence import Parameters,PartitionTable,evaluate,fit_parameters,cost_log_odds
from experiments.mprisk_evidence.data import load_native,similarities,split_validation,synthetic_pair,digest_array,native_method,run_legacy
from experiments.mprisk_evidence.metrics import Metrics,masks,auc,bootstrap,fit_risk_weights,fit_linear,apply_linear
from experiments.mprisk_evidence.recheck import empirical_threshold,actions_at_threshold,fit_reference
from experiments.mprisk_evidence.artifacts import Tables,write_json,archive,Tee,sha256
from experiments.evirisk_precision import precise_cost_score,fit_precise_weights
from experiments.mprisk_evidence.study import simple_features
from experiments.mprisk_evidence.risk_model_selection import (
    select_evidence_model, reuse_probability_fit, fit_native_component_weights)


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/uncertainty_benchmark/mprisk_core_bio_complete.yaml'))
    p.add_argument('--whale-root',default=None);p.add_argument('--val-root',default=None)
    p.add_argument('--out',required=True);p.add_argument('--fpirs',nargs='+',type=float,default=[.05,.1,.2])
    p.add_argument('--seed',type=int,default=777);p.add_argument('--beta',type=float,default=.5)
    p.add_argument('--device',default='cpu');p.add_argument('--max-fit',type=int,default=3000)
    p.add_argument('--fit-iterations',type=int,default=250);p.add_argument('--search-budget',type=int,default=1024)
    p.add_argument('--bootstrap',type=int,default=200);p.add_argument('--gallery-kappa-max',type=float,default=1e5)
    p.add_argument('--probe-scale-max',type=float,default=100.)
    p.add_argument('--probability-selection',choices=['validation-prr','multiclass-nll'],default='validation-prr',
                   help='Default: retain all converged finite-q fits and select jointly with costs by validation PRR. NLL reproduces the old selection rule.')
    p.add_argument('--reuse-probability-fit',default=None,
                   help='Extracted earlier run directory. Reuses converged likelihood fits after hash/split checks; no repeated likelihood optimization.')
    p.add_argument('--include-mprisk',action='store_true',
                   help='Retune original power-spherical MPRisk three- and four-component comparators on the same selection rows.')
    p.add_argument('--include-holue',action='store_true',help='Use native power-spherical GalUE and refit HolUE on the same validation decisions; needs original repository dependencies')
    p.add_argument('--synthetic',action='store_true');p.add_argument('--synthetic-n',type=int,default=300)
    p.add_argument('--legacy-weight-tuning',action='store_true',help='Keep historical float64 weight selection; still report corrected ordering alongside the raw score')
    a=p.parse_args()
    if not 0<a.beta<1 or any(not 0<f<1 for f in a.fpirs):p.error('beta and FPIR must be in (0,1)')
    if not a.gallery_kappa_max>.1 or not a.probe_scale_max>1e-4:p.error('Invalid fit bounds')
    if min(a.fit_iterations,a.search_budget)<1 or a.bootstrap<0 or a.max_fit<0:p.error('Invalid budgets')
    if a.synthetic and (a.include_holue or a.include_mprisk):p.error('Synthetic smoke has no native paper baselines')
    return a


def model_fit(c,data,parts,args,table,point=False):
    return fit_parameters(c,data.kappa,data.d,data.targets,parts['fit'],parts['select'],args.beta,table,
             point=point,maxiter=args.fit_iterations,max_fit=args.max_fit,seed=args.seed,
             probe_scale_bounds=(1e-4,args.probe_scale_max),gallery_kappa_bounds=(.1,args.gallery_kappa_max))


def error_counts(a,y):
    return {k:int(v.sum()) for k,v in masks(a,y).items()}


def geometry(tables,data,c,parts=None):
    subsets={'all':np.arange(data.n)} if parts is None else parts
    for name,ix in subsets.items():
        y=data.targets[ix];known=y>0
        truecos=np.asarray(c[ix[known],y[known]-1])
        wrong=np.array(c[ix[known]],copy=True)
        wrong[np.arange(len(wrong)),y[known]-1]=-np.inf
        imp=wrong.max(1) if len(wrong) else np.empty(0)
        kval=data.kappa[ix[known]]
        cor=float(spearmanr(kval,truecos).statistic) if len(kval)>2 and np.std(kval)>0 and np.std(truecos)>0 else np.nan
        tables.add('geometry',dict(part=name,n=len(ix),K=len(data.gallery),d=data.d,known=int(known.sum()),
               kappa_q=np.quantile(data.kappa[ix],[0,.1,.5,.9,1]).tolist(),true_cosine_q=np.quantile(truecos,[0,.1,.5,.9,1]).tolist() if len(truecos) else [],
               impostor_max_q=np.quantile(imp,[0,.1,.5,.9,1]).tolist() if len(imp) else [],kappa_true_cosine_spearman=cor),split=data.source['audit_split'])


def record(tables,split,a,y,scores,unit,context):
    metric=Metrics(a,y,context['seed'])
    for method,score in scores.items():
        probability=unit['risk'] if method=='EviRisk-1' else None
        odds=unit['score_log_odds'] if method=='EviRisk-1' else None
        tables.add('main_comparison',metric.evaluate(score,probabilities=probability,log_odds=odds),split=split,method=method,**context)
        for gate in ['accepted','rejected']:
            ix=a>0 if gate=='accepted' else a==0
            if not ix.any():continue
            yy=a[ix]!=y[ix]
            tables.add('conditional_errors',dict(gate=gate,n=int(ix.sum()),errors=int(yy.sum()),observed_error=float(yy.mean()),
                       error_auroc=auc(yy,score[ix]),distinct_score=int(len(np.unique(score[ix]))),
                       mean_error_probability=float(unit['risk'][ix].mean()) if method=='EviRisk-1' else np.nan),
                       split=split,method=method,**context)
        curves,rc=metric.curves(score)
        tables.add('rejection_curves',curves,split=split,method=method,**context)
        tables.add('risk_coverage',rc,split=split,method=method,**context)


def galue_uncertainty(point_eval):
    p0=np.asarray(point_eval['p0'],dtype=np.float64)
    top=np.asarray(point_eval['top_probabilities'],dtype=np.float64)
    if top.ndim!=2 or top.shape[0]!=len(p0) or top.shape[1]<1:
        raise ValueError('Point posterior must contain at least one known class')
    score=1.0-np.maximum(p0,top[:,0])
    if np.any(~np.isfinite(score)) or np.any((score< -1e-12)|(score>1+1e-12)):
        raise FloatingPointError('Invalid GalUE uncertainty')
    return np.clip(score,0.0,1.0)


def run(args,out,tables,state):
    if args.synthetic:
        val,test=synthetic_pair(args.seed,args.synthetic_n);core=None
    else:
        from omegaconf import OmegaConf
        core=OmegaConf.load(args.config);core.exp_dir=str(out/'native_logs')
        # Keep the same prior for evidence, point and native baselines.
        for method in core.open_set_identification_methods:
            if 'beta' in method.recognition_method:
                method.recognition_method.beta=float(args.beta)
        testcfg=next((x for x in core.test_datasets if str(x.dataset_name)=='whale'),None)
        if testcfg is None:raise ValueError('Config contains no whale dataset')
        valcfg=core.dataset_name_to_calibration_set['whale']
        if args.whale_root:testcfg.dataset_path=args.whale_root
        if args.val_root:valcfg.dataset_path=args.val_root
        OmegaConf.save(core,out/'resolved_config.yaml',resolve=True)
        # Read source metadata, not potentially stale backup.npz caches.
        val=load_native(valcfg,fresh_metadata=True);test=load_native(testcfg,fresh_metadata=True)
    if val.d!=test.d:raise ValueError('Validation/test embedding dimensions differ')
    val.source['audit_split']='validation';test.source['audit_split']='test'
    write_json(out/'data_manifest.json',dict(validation=val.metadata(),test=test.metadata(),
               raw_row_order_verification='The embs/unc-only export has no image IDs; row order cannot be independently proved from this file. Source metadata order is required.',
               fresh_metadata=not args.synthetic))
    parts,unit=split_validation(val,args.seed);write_json(out/'validation_split.json',dict(unit=unit,**parts))
    cv,backend=similarities(val,out/'_cache'/'validation_cosines.npy',args.device)
    ct,_=similarities(test,out/'_cache'/'test_cosines.npy',args.device)
    geometry(tables,val,cv,parts);geometry(tables,test,ct)
    table=PartitionTable(val.d);write_json(out/'partition_numerics.json',table.manifest())
    reused_from=None
    if args.reuse_probability_fit:
        pars_nll,fit,point,pfit,reused_from=reuse_probability_fit(args.reuse_probability_fit,val,test,parts,args)
        print('[probability fit] reused checked converged candidates from '+reused_from,flush=True)
    else:
        pars_nll,fit=model_fit(cv,val,parts,args,table);point,pfit=model_fit(cv,val,parts,args,table,True)
    write_json(out/'probability_model_fit.json',dict(parameters=asdict(pars_nll),fit=fit,point_parameters=asdict(point),point_fit=pfit,
        parameters_role='NLL incumbent/candidate generation only; final parameters are in fpir_*/selected_probability_model.json',
        reused_from=reused_from,probability_selection=args.probability_selection))
    for model,rep in [('NLL candidate-generation incumbent',fit),('GalUE',pfit)]:
        if rep.get('selected_at_boundary'):state['warnings'].append(model+': selected likelihood fit is at a configured boundary')
        for row in rep.get('coordinate_slices',[]):tables.add('validation_parameter_slices',row,model=model)
    median=float(np.median(val.kappa[parts['fit']]))
    state.update(backend=backend,main_method='EviRisk',point_method='GalUE',
                 probability_selection=args.probability_selection,probability_fit_reused_from=reused_from,
                 recognition_protocol='empirical FPIR on test unknowns for benchmark only; no deployment calibration guarantee',
                 score_tuning='historical logit' if args.legacy_weight_tuning else 'validation PRR; label-free precise weighted-loss ordering at inference')
    for far in args.fpirs:
        print(f'[Whale diagnostics] FPIR={far:g}',flush=True)
        case=out/f'fpir_{far:g}';case.mkdir();context=dict(dataset='synthetic' if args.synthetic else 'whale',target_fpir=far,seed=args.seed)
        tv,rv=empirical_threshold(np.asarray(cv).max(1)[parts['fit']][val.targets[parts['fit']]==0],far)
        tt,rt=empirical_threshold(np.asarray(ct).max(1)[test.targets==0],far)
        av=actions_at_threshold(cv,tv);at=actions_at_threshold(ct,tt)
        hashes={'validation':digest_array(av),'test':digest_array(at)}
        write_json(case/'reference_decisions.json',dict(validation=rv,test=rt,action_hashes=hashes,test_threshold_uses_test_unknown_labels=True))
        for label,data,a,ix in [('fit',val,av,parts['fit']),('select',val,av,parts['select']),('audit',val,av,parts['audit']),('test',test,at,np.arange(test.n))]:
            counts=error_counts(a[ix],data.targets[ix]);tables.add('support',dict(split=label,n=len(ix),**counts),**context)
            if label in ['select','audit']:
                for e in ['false_accept','false_reject','misidentification']:
                    if counts[e]<10:state['warnings'].append(f'FPIR={far:g} {label}: {e} count={counts[e]}')
        ix=parts['select'];fitter=fit_risk_weights if args.legacy_weight_tuning else fit_precise_weights
        if args.probability_selection=='validation-prr':
            pars,wfit,selection=select_evidence_model(cv,val.kappa,val.d,val.targets,av,ix,
                fit,args.beta,table,args.search_budget,args.seed,args.legacy_weight_tuning)
        else:
            pars=pars_nll
            pv_fit=evaluate(cv[ix],val.kappa[ix],val.d,pars,av[ix],val.targets[ix],table)
            wfit=fitter(pv_fit['log_event_probabilities'],av[ix],val.targets[ix],args.search_budget,args.seed)
            selection=dict(objective='validation multiclass NLL (historical model selection)',
                selected_index=fit['selected_index'],selected_parameters=asdict(pars),weight_fit=wfit,
                selection_indices=ix.tolist(),candidates=[],uses_test_outcomes=False)
        # Persist the final specification BEFORE evaluating its test predictions.
        write_json(case/'evidence_selection.json',selection)
        write_json(case/'selected_probability_model.json',dict(parameters=asdict(pars),
            candidate_index=selection['selected_index'],selection_rule=args.probability_selection))
        for candidate in selection['candidates']:
            tables.add('probability_selection',dict(candidate_index=candidate['candidate_index'],
                probe_scale=candidate['parameters']['probe_scale'],gallery_kappa=candidate['parameters']['gallery_kappa'],
                class_select_nll=candidate['class_select_nll'],unit_validation_prr=candidate['unit_validation_prr'],
                weighted_validation_prr=candidate['weighted_validation_prr'],
                selected=candidate['candidate_index']==selection['selected_index']),**context)
        pv=evaluate(cv,val.kappa,val.d,pars,av,val.targets,table)
        pt=evaluate(ct,test.kappa,test.d,pars,at,test.targets,table)
        ppointv=evaluate(cv,val.kappa,val.d,point,av,val.targets,table)
        ppointt=evaluate(ct,test.kappa,test.d,point,at,test.targets,table)
        ev,et=pv['log_event_probabilities'],pt['log_event_probabilities']
        write_json(case/'weights.json',dict(EviRisk=wfit,training_indices=ix))
        write_json(case/'GalUE_fit.json',dict(parameters=asdict(point),fit=pfit,
            score='1 - max posterior probability over known and unknown classes',density='vMF'))
        sv,va=precise_cost_score(ev,wfit['raw_weights'],True);st,ta=precise_cost_score(et,wfit['raw_weights'],True)
        write_json(case/'rank_precision.json',dict(validation=va,test=ta))
        scoresv={'EviRisk':sv,'EviRisk-1':pv['score_log_odds'],
                  'EviRisk float64-logit diagnostic':cost_log_odds(ev,wfit['raw_weights']),
                  'GalUE':galue_uncertainty(ppointv),
                  'SCF':-val.kappa,'AccScr':-np.abs(np.asarray(cv).max(1)-tv)}
        scorest={'EviRisk':st,'EviRisk-1':pt['score_log_odds'],
                  'EviRisk float64-logit diagnostic':cost_log_odds(et,wfit['raw_weights']),
                  'GalUE':galue_uncertainty(ppointt),
                  'SCF':-test.kappa,'AccScr':-np.abs(np.asarray(ct).max(1)-tt)}
        # Keep the previous lowest-NLL model as an explicit, equally retuned control.
        if args.probability_selection=='validation-prr':
            oldrow=next(r for r in selection['candidates'] if r['selected_by_nll'])
            oldw=oldrow['weight_fit']['raw_weights']
            oldv=evaluate(cv,val.kappa,val.d,pars_nll,av,val.targets,table)
            oldt=evaluate(ct,test.kappa,test.d,pars_nll,at,test.targets,table)
            scoresv['EviRisk NLL-selected control']=precise_cost_score(oldv['log_event_probabilities'],oldw)
            scorest['EviRisk NLL-selected control']=precise_cost_score(oldt['log_event_probabilities'],oldw)
            scoresv['EviRisk NLL-selected unit control']=oldv['score_log_odds']
            scorest['EviRisk NLL-selected unit control']=oldt['score_log_odds']
        # Reuse the exact augmented-score convention of the supplied study.
        bv=simple_features(cv,val.kappa,tv,beta=args.beta,d=val.d)
        bt=simple_features(ct,test.kappa,tt,beta=args.beta,d=test.d)
        for key in ['MSP','Entropy','Margin']:
            scoresv[key]=bv[key];scorest[key]=bt[key]
        cols=['SCF','AccScr','MSP','Margin']
        xv=np.column_stack([bv[k] for k in cols]);xt=np.column_stack([bt[k] for k in cols])
        linear=fit_linear(xv[ix],av[ix],val.targets[ix],args.search_budget,args.seed)
        write_json(case/'Lin4_fit.json',dict(columns=cols,fit=linear,training_indices=ix))
        scoresv['Lin-4']=apply_linear(xv,linear);scorest['Lin-4']=apply_linear(xt,linear)
        # These are diagnostics at FIXED fitted model/score parameters, not fair
        # retrained competitors and not automatically selected by test quality.
        for label,data,c,a,p in [('validation',val,cv,av,pv),('test',test,ct,at,pt)]:
            check_ix=np.sort(np.random.default_rng(args.seed).choice(data.n,min(64,data.n),replace=False))
            exact=evaluate(c[check_ix],data.kappa[check_ix],data.d,pars,a[check_ix],data.targets[check_ix],None)
            err=float(np.max(np.abs(exact['score_log_odds']-p['score_log_odds'][check_ix])))
            tables.add('numerical_audit',dict(split=label,exact_interpolation_log_odds_error=err,
                       posterior_mass_error=float(abs(p['posterior_mass']-1).max()),input_indices=check_ix.tolist(),passed=err<2e-6),**context)
            if err>=2e-6:raise FloatingPointError('Exact/interpolated log odds differ beyond audit tolerance')
            scores=scoresv if label=='validation' else scorest
            variants={'constant-kappa':np.full(data.n,median),
                      'shuffled-kappa':np.random.default_rng(args.seed+(0 if label=='validation' else 1)).permutation(data.kappa),
                      'uninformative-q':np.zeros(data.n)}
            for variant,k in variants.items():
                vv=evaluate(c,k,data.d,pars,a,data.targets,table)
                sc=precise_cost_score(vv['log_event_probabilities'],wfit['raw_weights'])
                scores['Diagnostic '+variant]=sc
                jj=parts['audit'] if label=='validation' else np.arange(data.n)
                tables.add('fixed_parameter_controls',Metrics(a[jj],data.targets[jj],args.seed).evaluate(sc[jj]),variant=variant,
                           split='audit' if label=='validation' else 'test',refitted=False,**context)
            reject=a==0
            if reject.any() and wfit['raw_weights'][2]>0:
                au=auc(data.targets[reject]>0,p['score_log_odds'][reject]);aw=auc(data.targets[reject]>0,scores['EviRisk'][reject])
                tables.add('rejection_invariance',dict(split=label,unit_auroc=au,weighted_auroc=aw,
                         absolute_difference=abs(au-aw),lambda_fr=wfit['raw_weights'][2],expected='same ordering for strictly positive FR weight'),**context)
            for event,keep in masks(a,data.targets).items():
                if event not in ['false_reject','true_reject','false_accept','misidentification','tp']:continue
                j=np.flatnonzero(keep)
                tables.add('event_geometry',dict(split=label,event=event,n=len(j),
                     median_kappa=float(np.median(data.kappa[j])) if len(j) else np.nan,
                     median_effective_kappa=float(np.median(data.kappa[j]*pars.probe_scale)) if len(j) else np.nan,
                     mean_max_cosine=float(np.mean(np.asarray(c[j]).max(1))) if len(j) else np.nan,
                     mean_p_unknown=float(p['p0'][j].mean()) if len(j) else np.nan,
                     mean_unit_risk=float(p['risk'][j].mean()) if len(j) else np.nan),**context)
        if args.include_holue or args.include_mprisk:
            from experiments.mprisk_evidence.study import fit_holue
            kind=str(native_method(core,'MPRisk raw').get('gallery_prior','power'))
            vref=fit_reference(cv[parts['fit']],val.targets[parts['fit']],far,args.beta,val.d,kind)
            tref=fit_reference(ct,test.targets,far,args.beta,test.d,kind)
            vl=run_legacy(core,val,far,gallery_kappa=vref['gallery_kappa'],actions_override=av,cosines=cv)
            tl=run_legacy(core,test,far,gallery_kappa=tref['gallery_kappa'],actions_override=at,cosines=ct)
            vl['tau']=tv;tl['tau']=tt
            if args.include_holue:
                hv,ht,_,_,hp=fit_holue(core,val,test,vl,tl,ix,far,case,args.seed,False)
                write_json(case/'HolUE_fit.json',hp)
                scoresv['HolUE']=hv;scorest['HolUE']=ht
            if args.include_mprisk:
                native_v=np.column_stack([vl['components'][key] for key in ['r_fa','r_id','r_fr']])
                native_t=np.column_stack([tl['components'][key] for key in ['r_fa','r_id','r_fr']])
                nw3=fit_native_component_weights(native_v[ix],av[ix],val.targets[ix],args.search_budget,args.seed)
                nv4=np.column_stack((native_v,vl['r_ns']));nt4=np.column_stack((native_t,tl['r_ns']))
                nw4=fit_native_component_weights(nv4[ix],av[ix],val.targets[ix],args.search_budget,args.seed,
                                                 include=[nw3['raw_weights']+[0.]])
                scoresv['MPRisk reference (retuned 3)']=native_v@nw3['raw_weights']
                scorest['MPRisk reference (retuned 3)']=native_t@nw3['raw_weights']
                scoresv['MPRisk reference (retuned 4)']=nv4@nw4['raw_weights']
                scorest['MPRisk reference (retuned 4)']=nt4@nw4['raw_weights']
                write_json(case/'MPRisk_reference_fit.json',dict(three=nw3,four=nw4,training_indices=ix.tolist(),
                    gallery_density=kind,temperature=vl['temperature'],validation_gallery_kappa=vl['gallery_kappa'],
                    test_gallery_kappa=tl['gallery_kappa'],same_reference_actions=True,
                    note='Original M=0 components, freshly tuned on the common validation subset; not a new copy of paper table values.'))
        audit=parts['audit']
        record(tables,'audit',av[audit],val.targets[audit],{k:s[audit] for k,s in scoresv.items()},
               {k:v[audit] for k,v in pv.items()},context)
        record(tables,'test',at,test.targets,scorest,pt,context)
        if args.bootstrap:
            chosen={k:s for k,s in scorest.items() if k in ['EviRisk','EviRisk-1','EviRisk NLL-selected control','HolUE','GalUE','Lin-4',
                      'MPRisk reference (retuned 3)','MPRisk reference (retuned 4)']}
            tables.add('bootstrap',bootstrap(at,test.targets,chosen,'EviRisk',test.probe_ids,args.bootstrap,args.seed),**context)
        for label,data,a,p,scores in [('validation',val,av,pv,scoresv),('test',test,at,pt,scorest)]:
            if digest_array(a)!=hashes[label]:raise AssertionError('Recognizer actions changed')
            role=np.full(data.n,'test',dtype='U8')
            if label=='validation':
                for key,ids in parts.items():role[ids]=key
            np.savez_compressed(case/(label+'.npz'),actions=a,targets=data.targets,template_ids=data.template_ids.astype(str),
               subject_ids=data.probe_ids.astype(str),gallery_ids=data.gallery_ids.astype(str),kappa=data.kappa,
               role=role,score_names=np.array(list(scores)),scores=np.column_stack(list(scores.values())),**p)
        tables.save()
    state['warnings']=sorted(set(state['warnings']))
    state['status']='complete';state['claim_whale_bug_fixed']=False
    state['new_test_results_must_not_be_used_to_choose_model']=True


def main():
    args=arguments();out=Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):raise SystemExit('Output directory must be empty: '+str(out))
    out.mkdir(parents=True,exist_ok=True);state=dict(status='running',synthetic=args.synthetic,warnings=[],arguments=vars(args),
          started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),python=platform.python_version())
    tables=Tables(out);oldout,olderr=sys.stdout,sys.stderr;log=(out/'run.log').open('w',buffering=1)
    sys.stdout=Tee(oldout,log);sys.stderr=Tee(olderr,log);exitcode=0
    try:
        write_json(out/'method_registry.json',{'EviRisk':'finite evidence model and three weights selected on validation PRR; ordinal ranking, not error probability',
            'EviRisk NLL-selected control':'previous model-selection rule retained; weights independently fitted on the same validation rows',
            'EviRisk-1':'unit weights for the PRR-selected probability model; conditional error probability reported separately',
            'MPRisk reference (retuned 3)':'original power-spherical M=0 posterior and FA/ID/FR, freshly tuned on common validation rows',
            'MPRisk reference (retuned 4)':'same native posterior plus original NS component, freshly tuned; includes best three-component candidate',
            'GalUE':'point-representation GalUE with vMF class densities; NLL-fitted gallery concentration; uncertainty is one minus the largest posterior class probability',
            'old_artifact_aliases':{'MPRisk PRR-weighted':'EviRisk','MPRisk':'EviRisk-1'}})
        snapshot=out/'source';snapshot.mkdir()
        files=['evaluation/open_set_methods/mprisk_evidence.py','experiments/mprisk_evidence/data.py',
               'experiments/mprisk_evidence/metrics.py','experiments/mprisk_evidence/recheck.py',
               'experiments/evirisk_precision.py','experiments/evirisk_whale_diagnostics.py',
               'experiments/mprisk_evidence/risk_model_selection.py']
        state['source_sha256']={}
        for name in files:
            path=ROOT/name;dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
            state['source_sha256'][name]=sha256(path)
        run(args,out,tables,state)
    except BaseException as exc:
        state['status']='failed';state['failure']=str(exc);exitcode=1
        (out/'failure.txt').write_text(traceback.format_exc());traceback.print_exc()
    finally:
        state['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        tables.save();write_json(out/'manifest.json',state)
        sys.stdout.flush();sys.stderr.flush();sys.stdout,sys.stderr=oldout,olderr;log.close()
        z=archive(out);print(f'STATUS: {state["status"]}\nRESULTS: {out}\nRESULT ZIP: {z}')
    return exitcode

if __name__=='__main__':raise SystemExit(main())
