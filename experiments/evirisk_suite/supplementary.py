"""Matched comparisons, probability diagnostics and frozen supplementary studies.

Everything called a fit uses explicit validation indices. Test arrays are used
only by predict/report code. Weighted ordinal ranks are never calibration inputs
or numerical features: use the transportable weighted-loss logit instead.
"""
from __future__ import annotations
from dataclasses import asdict, replace
from pathlib import Path
import json, time
import numpy as np
from scipy.special import expit
from evaluation.open_set_methods.mprisk_evidence import evaluate, fit_parameters, cost_log_odds
from experiments.evirisk_precision import precise_cost_score, fit_precise_weights
from experiments.mprisk_evidence.metrics import (
    Metrics, masks, auc, ap, bins, fit_linear, apply_linear,
    fit_positive_score_calibration, calibrated_log_odds, fit_logistic,
    logistic_log_odds, fit_mlp, apply_mlp)
from experiments.mprisk_evidence.risk_model_selection import select_evidence_model
from experiments.mprisk_evidence.artifacts import write_json
from experiments.mprisk_evidence.audits import probe_limit_rows


def feature_transform(x, indices):
    """Fit finite clipping bounds on selection only (needed for saturated logits)."""
    x = np.asarray(x, float); chosen = x[np.asarray(indices, int)]
    if np.isnan(x).any(): raise ValueError('NaN fusion input')
    lower, upper = [], []
    for j in range(x.shape[1]):
        a = chosen[np.isfinite(chosen[:,j]), j]
        if not len(a): lower.append(-1.); upper.append(1.)
        else:
            span = max(1., float(a.max()-a.min()))
            lower.append(float(a.min()-span)); upper.append(float(a.max()+span))
    return dict(lower=lower, upper=upper,
                rule='finite bounds fitted on selection; replace infinities and clip extrapolations only for numerical fusion/calibration')


def transform(x, p):
    x = np.asarray(x, float)
    if np.isnan(x).any(): raise ValueError('NaN fusion input')
    return np.clip(x, p['lower'], p['upper'])


def fit_fusion(x, actions, targets, indices, budget, seed, exact_scores=None):
    ix = np.asarray(indices,int)
    prep = feature_transform(x,ix); z = transform(x,prep)
    fitted = fit_linear(z[ix],actions[ix],targets[ix],budget,seed)
    if not fitted['identifiable']: raise ValueError('Linear selection PRR is undefined')
    objective = Metrics(actions[ix],targets[ix],seed)
    best = float(objective.prr(apply_linear(z[ix],fitted)))
    selected = None; tested = []
    for name, score in (exact_scores or {}).items():
        value = float(objective.prr(np.asarray(score)[ix]))
        tested.append(dict(name=name,selection_prr=value))
        if np.isfinite(value) and value > best:
            best, selected = value, name
    return dict(preprocess=prep,linear=fitted,exact_selected=selected,
                validation_prr=best,exact_candidates=tested,training_indices=ix.tolist(),
                random_candidates=budget,objective='validation selection PRR')


def predict_fusion(x, fitted, scores):
    name = fitted['exact_selected']
    return np.asarray(scores[name]).copy() if name else apply_linear(transform(x,fitted['preprocess']),fitted['linear'])


def probability_reports(tables,ctx,split,a,y,score,prob,odds,name):
    for group,keep in [('all',np.ones(len(a),bool)),('accepted',a>0),('rejected',a==0)]:
        if not keep.any(): continue
        m=Metrics(a[keep],y[keep],ctx['seed'])
        row=m.evaluate(score[keep],probabilities=prob[keep],log_odds=odds[keep] if odds is not None else None)
        br=bins(prob[keep],a[keep]!=y[keep])
        ece=sum(r['count']*abs(r['predicted']-r['observed']) for r in br if r['count'])/keep.sum()
        tables.add('probability_calibration',dict(row,ece=float(ece)),split=split,subset=group,method=name,**ctx)
        tables.add('reliability_bins',br,split=split,subset=group,method=name,**ctx)


def score_reports(tables,ctx,split,a,y,scores):
    mm=masks(a,y)
    for name,s in scores.items():
        for target in ['any_error','false_accept','misidentification','false_reject']:
            positive=mm[target]
            tables.add('error_types',dict(n=len(a),positives=int(positive.sum()),
                auroc=auc(positive,s),auprc=ap(positive,s)),method=name,target=target,split=split,**ctx)
        for gate,target in [('rejected','false_reject'),('accepted','false_accept'),('accepted','misidentification')]:
            keep=a==0 if gate=='rejected' else a>0
            if not keep.any(): continue
            positive=mm[target][keep]
            tables.add('error_types_conditional',dict(n=int(keep.sum()),positives=int(positive.sum()),
                auroc=auc(positive,s[keep]),auprc=ap(positive,s[keep])),
                method=name,target=target,subset=gate,split=split,**ctx)


def enrich(args,ctx,case,tables,val,test,cv,ct,parts,pars,pars_nll,fit,table,
           av,at,pv,pt,wfit,sv,st,vl,tl,vh=None,th=None):
    ix=parts['select']; ai=parts['audit']; yv,yt=val.targets,test.targets
    ev,et=pv['log_event_probabilities'],pt['log_event_probabilities']
    w=np.asarray(wfit['raw_weights'],float)
    lc_v,lc_t=cost_log_odds(ev,w),cost_log_odds(et,w)
    # Train calibration on a transportable numerical score, never ordinal ranks.
    clip=feature_transform(lc_v[:,None],ix)
    c_v=transform(lc_v[:,None],clip).ravel(); c_t=transform(lc_t[:,None],clip).ravel()
    calibration=fit_positive_score_calibration(c_v[ix],av[ix]!=yv[ix])
    calv,calt=calibrated_log_odds(c_v,calibration),calibrated_log_odds(c_t,calibration)
    sv['EviRisk-Cal']=sv['EviRisk'].copy(); st['EviRisk-Cal']=st['EviRisk'].copy()
    probability={
        'EviRisk-1': (pv['risk'],pt['risk'],pv['score_log_odds'],pt['score_log_odds']),
        'EviRisk-Cal': (expit(calv),expit(calt),calv,calt),
    }
    write_json(case/'probability_calibrator.json',dict(fit=calibration,preprocess=clip,
        training_indices=ix,ranking_score='unchanged EviRisk ordinal order',
        input='cost_log_odds; NOT the per-batch rank vector'))

    # Registered baseline features: no recursive inclusion of Lin-All in itself.
    baseline=['SCF','AccScr','MSP','Entropy','Margin','GalUE']
    features_v={n:sv[n] for n in baseline}; features_t={n:st[n] for n in baseline}
    holue='HolUE' if 'HolUE' in sv else 'Synthetic HolUE stand-in'
    if holue in sv:
        features_v['HolUE probability']=sv[holue];features_t['HolUE probability']=st[holue]
        baseline.append('HolUE probability')
        if np.all((sv[holue]>=0)&(sv[holue]<=1)) and np.all((st[holue]>=0)&(st[holue]<=1)):
            probability[holue]=(sv[holue],st[holue],None,None)
    if vl is not None:
        # HolUE uses the same feature construction with its OWN configured
        # temperature, not necessarily the temperature of MPRisk (text differs).
        hv=vl if vh is None else vh; ht=tl if th is None else th
        for name,field in [('HolUE KL1','kl1'),('HolUE KL2','kl2')]:
            features_v[name]=hv[field];features_t[name]=ht[field];baseline.append(name)
        for field in ['r_fa','r_id','r_fr']:
            name='MPRisk '+field
            features_v[name]=vl['components'][field];features_t[name]=tl['components'][field];baseline.append(name)
        features_v['MPRisk r_ns']=vl['r_ns'];features_t['MPRisk r_ns']=tl['r_ns'];baseline.append('MPRisk r_ns')
        for name in ['MPRisk reference (retuned 3)','MPRisk reference (retuned 4)']:
            if name in sv:
                features_v[name]=sv[name];features_t[name]=st[name];baseline.append(name)
    fits={}
    if vl is not None:
        cols=['HolUE KL1','HolUE KL2']
        xv=np.column_stack([features_v[n] for n in cols]);xt=np.column_stack([features_t[n] for n in cols])
        p=fit_fusion(xv,av,yv,ix,args.search_budget,args.seed)
        p['feature_names']=cols;fits['Lin-KL']=p
        sv['Lin-KL']=predict_fusion(xv,p,sv);st['Lin-KL']=predict_fusion(xt,p,st)
    xv=np.column_stack([features_v[n] for n in baseline]); xt=np.column_stack([features_t[n] for n in baseline])
    exact={n:sv[n] for n in ['Lin-4','Lin-KL',holue,'GalUE',
           'MPRisk reference (retuned 3)','MPRisk reference (retuned 4)'] if n in sv}
    p=fit_fusion(xv,av,yv,ix,args.search_budget,args.seed,exact)
    p['feature_names']=baseline;fits['Lin-All']=p
    sv['Lin-All']=predict_fusion(xv,p,sv);st['Lin-All']=predict_fusion(xt,p,st)
    extra=['EviRisk unit logit','EviRisk weighted loss logit','EviRisk r_fa','EviRisk r_id','EviRisk r_fr']
    xev=np.column_stack((xv,pv['score_log_odds'],lc_v,pv['r_fa'],pv['r_id'],pv['r_fr']))
    xet=np.column_stack((xt,pt['score_log_odds'],lc_t,pt['r_fa'],pt['r_id'],pt['r_fr']))
    exact={n:sv[n] for n in ['Lin-All','EviRisk','EviRisk-1']}
    p=fit_fusion(xev,av,yv,ix,args.search_budget,args.seed,exact)
    p['feature_names']=baseline+extra;fits['Lin-All+E']=p
    sv['Lin-All+E']=predict_fusion(xev,p,sv);st['Lin-All+E']=predict_fusion(xet,p,st)
    if p['validation_prr']+1e-12 < max(fits['Lin-All']['validation_prr'],Metrics(av[ix],yv[ix],args.seed).prr(sv['EviRisk'][ix])):
        raise AssertionError('Expanded fusion lost an exact nested candidate')
    prep=feature_transform(xev,ix); zv=transform(xev,prep);zt=transform(xet,prep)
    logistic=fit_logistic(zv[ix],av[ix]!=yv[ix],args.seed)
    lv,lt=logistic_log_odds(zv,logistic),logistic_log_odds(zt,logistic)
    sv['Supervised logistic']=lv;st['Supervised logistic']=lt
    probability['Supervised logistic']=(expit(lv),expit(lt),lv,lt)
    if args.supplementary=='full':
        mlp=fit_mlp(zv[ix],av[ix]!=yv[ix],args.seed)
        sv['Supervised MLP']=apply_mlp(zv,mlp);st['Supervised MLP']=apply_mlp(zt,mlp)
        probability['Supervised MLP']=(sv['Supervised MLP'],st['Supervised MLP'],None,None)
    else: mlp=None
    write_json(case/'fusion_fit.json',dict(fits=fits,baseline_registry=baseline,expanded_registry=baseline+extra,
        supervised_preprocess=prep,logistic=logistic,mlp=mlp,training_indices=ix,
        no_ordinal_rank_features=True,means_and_scales_fit_on='selection only',
        KL_note='HolUE KL features use the actual HolUE model/temperature; MPRisk uses its separate native posterior.'))
    np.savez_compressed(case/'fusion_features.npz',names=np.array(baseline+extra),validation=xev,test=xet)
    for name,p in fits.items():
        tables.add('fusion_selection',dict(method=name,selection_prr=p['validation_prr'],
            exact_selected=p['exact_selected'],feature_count=len(p['feature_names']),budget=args.search_budget),**ctx)
    score_reports(tables,ctx,'audit',av[ai],yv[ai],{n:s[ai] for n,s in sv.items()})
    score_reports(tables,ctx,'test',at,yt,st)
    for name,(vp,tp,vo,to) in probability.items():
        probability_reports(tables,ctx,'audit',av[ai],yv[ai],sv[name][ai],vp[ai],None if vo is None else vo[ai],name)
        probability_reports(tables,ctx,'test',at,yt,st[name],tp,to,name)
    for split,ids,pred,data in [('audit',ai,pv,val),('test',np.arange(test.n),pt,test)]:
        tables.add('posterior_scoring',dict(class_nll=float(-pred['true_log_probability'][ids].mean()),
            class_brier=float(pred['class_brier'][ids].mean()),n=len(ids),known_fraction=float(np.mean(data.targets[ids]>0)),
            prior_known=1-pars.beta),split=split,method='EviRisk probability model',**ctx)
    for split,pp,ind in [('validation',0,None),('test',1,None)]:
        arrays={};names=list(probability)
        arrays['names']=np.array(names)
        arrays['probabilities']=np.column_stack([probability[n][pp] for n in names])
        arrays['log_odds']=np.column_stack([probability[n][pp+2] if probability[n][pp+2] is not None else np.full(len(probability[n][pp]),np.nan) for n in names])
        np.savez_compressed(case/(split+'_probabilities.npz'),**arrays)
    # Seven nonempty component subsets, both fixed learned costs and unit costs.
    for code in range(1,8):
        mask=np.array([bool(code&(1<<j)) for j in range(3)],float)
        label='+'.join(n for n,b in zip(['FA','ID','FR'],mask) if b)
        for scheme,cost in [('unit',mask),('selected-costs',w*mask)]:
            score=precise_cost_score(et,cost)
            tables.add('component_ablation',Metrics(at,yt,args.seed).evaluate(score),
                components=label,weights=cost.tolist(),scheme=scheme,refitted=False,**ctx)
    # Conditional finite-sample calculation for weighted loss, NOT probability Brier.
    loss=(w[0]*masks(av[ai],yv[ai])['false_accept']+
          w[1]*masks(av[ai],yv[ai])['misidentification']+w[2]*masks(av[ai],yv[ai])['false_reject'])
    predicted=np.exp(ev[ai,1:])@w
    squared=float(np.mean((predicted-loss)**2));upper=min(1.,squared+np.sqrt(np.log(20)/(2*len(ai))))
    for gamma in [.25,.5,.75,.9]:
        tables.add('audit_risk_bound',dict(coverage=gamma,n=len(ai),confidence_delta=.05,
            squared_loss_error=squared,excess_bound=min(1.,np.sqrt(2*min(gamma,1-gamma)*upper)/gamma),
            normalized_costs=w.tolist(),certified=False,
            assumptions='Conditional on fitted models/costs; iid audit probes and exact population coverage required. Identity dependence is not certified.'),**ctx)
    # Predeclared single FPIR avoids repeating the expensive full refit ablations.
    failures=[]
    if args.supplementary=='full' and abs(ctx['target_fpir']-.1)<1e-12:
        failures=full_diagnostics(args,ctx,case,tables,val,test,cv,ct,parts,pars,fit,table,av,at,pv,pt,wfit,sv,st)
    return dict(probability_methods=list(probability),fusion_methods=list(fits),
                supplementary_failures=failures,full_supplementary_fpir=.1,
                calibration_identifiable=calibration['identifiable'])


def full_diagnostics(args,ctx,case,tables,val,test,cv,ct,parts,pars,fit,table,av,at,pv,pt,wfit,sv,st):
    select=parts['select']; yv,yt=val.targets,test.targets; ev=pv['log_event_probabilities'];et=pt['log_event_probabilities']
    base=Metrics(at,yt,args.seed);w=np.asarray(wfit['raw_weights']);failures=[]
    fractions=[.1,.5,1.] if args.synthetic else [.05,.1,.25,.5,1.]
    for fraction in fractions:
        for repeat in range(args.validation_repeats):
            n=min(len(select),max(8,int(len(select)*fraction)))
            sub=np.sort(np.random.default_rng(args.seed+repeat).choice(select,n,replace=False))
            fitw=fit_precise_weights(ev[sub],av[sub],yv[sub],args.search_budget,args.seed+repeat)
            # Undefined sub-sample PRR is reported, not interpreted as successful fitting.
            score=precise_cost_score(et,fitw['raw_weights'])
            tables.add('validation_size_weights',dict(base.evaluate(score),fraction=fraction,repeat=repeat,
                selection_n=n,weights=fitw['raw_weights'],identifiable=fitw['identifiable'],
                posterior_fixed=True,note='Only weights refitted; posterior selected on full selection data, NOT an end-to-end data-efficiency claim'),**ctx)
            write_json(case/'weight_size_fits'/f'{fraction:g}_{repeat}.json',dict(indices=sub,fit=fitw))
    # Regenerate ALL likelihood starts, then select by PRR again. Never fall back
    # to NLL-only selection in a supplementary probability-fit experiment.
    for fraction in ([.5] if args.synthetic else [.25,.5]):
        source=parts['fit']; n=min(len(source),max(8,int(len(source)*fraction)))
        sub=np.sort(np.random.default_rng(args.seed).choice(source,n,replace=False))
        path=case/'sample_size_refits'/f'{fraction:g}';path.mkdir(parents=True,exist_ok=True)
        try:
            _,rep=fit_parameters(cv,val.kappa,val.d,yv,sub,select,args.beta,table,
                maxiter=args.fit_iterations,max_fit=args.max_fit,seed=args.seed,report_path=path/'candidates.json',
                probe_scale_bounds=(1e-4,args.probe_scale_max),gallery_kappa_bounds=(.1,args.gallery_kappa_max))
            p,ww,sel=select_evidence_model(cv,val.kappa,val.d,yv,av,select,rep,args.beta,table,args.search_budget,args.seed)
            write_json(path/'selection.json',sel)
            pred=evaluate(ct,test.kappa,test.d,p,at,yt,table)
            score=precise_cost_score(pred['log_event_probabilities'],ww['raw_weights'])
            tables.add('probability_fit_size',dict(base.evaluate(score),fraction=fraction,requested_fit_n=n,
                actual_fit_n=len(rep['fit_indices']),parameters=asdict(p),weights=ww['raw_weights'],
                status='complete',class_nll=float(-pred['true_log_probability'].mean()),
                selection_rule='validation PRR over regenerated converged fits and weights'),**ctx)
        except (ValueError,RuntimeError,FloatingPointError) as exc:
            # Keep a failed diagnostic visible; never insert an old fitted result.
            failures.append(str(exc));write_json(path/'failure.json',dict(error=str(exc)))
            tables.add('probability_fit_size',dict(fraction=fraction,status='failed',error=str(exc)),**ctx)
    for parameter,factors in [('probe_scale',[0.,.1,.3,1.,3.,10.]),('gallery_kappa',[.5,1.,2.]),('beta',[.1,.25,.5,.75,.9])]:
        for f in factors:
            value=f if parameter=='beta' else getattr(pars,parameter)*f
            p=replace(pars,**{parameter:value}); pred=evaluate(ct,test.kappa,test.d,p,at,yt,table)
            score=precise_cost_score(pred['log_event_probabilities'],w)
            tables.add('parameter_sensitivity',dict(base.evaluate(score),parameter=parameter,value=value,
                parameter_multiplier=f,weights_fixed=True,recognition_fixed=True,diagnostic_only=True,
                class_nll=float(-pred['true_log_probability'].mean())),**ctx)
    # No-unknown event control: rejected decision has probability zero of being
    # correct in this deliberately closed-set model; keep +inf scores as ties.
    score=pt['known_only_error_log_odds']
    tables.add('background_ablation',base.evaluate(score),variant='remove unknown class, unit risk',
        diagnostic_only=True,**ctx)
    tables.add('distribution_limits',probe_limit_rows(test.d,pars.gallery_kappa,args.beta),**ctx)
    edges=np.quantile(val.kappa[parts['fit']],[0,.25,.5,.75,1.]); groups=np.searchsorted(edges[1:-1],test.kappa,side='right')
    for q in range(4):
        keep=groups==q
        if not keep.any(): continue
        for name in ['EviRisk','EviRisk-1','HolUE','MPRisk reference (retuned 3)','Lin-All+E']:
            if name in st:
                tables.add('quality_strata',Metrics(at[keep],yt[keep],args.seed).evaluate(st[name][keep]),
                    method=name,quality_bin=q,fit_quantiles=edges.tolist(),**ctx)
    for tag,mask in [('confident_errors',at!=yt),('uncertain_correct',at==yt),('false_rejections',(at==0)&(yt>0))]:
        ids=np.flatnonzero(mask);order=np.argsort(st['EviRisk'][ids],kind='stable')
        if tag!='confident_errors': order=order[::-1]
        for j in ids[order[:10]]:
            tables.add('examples',dict(group=tag,template=str(test.template_ids[j]),subject=str(test.probe_ids[j]),
                row=int(j),action=int(at[j]),target=int(yt[j]),unit_risk=float(pt['risk'][j]),
                ordinal_score=float(st['EviRisk'][j]),kappa=float(test.kappa[j])),**ctx)
    return failures


def within_dataset_transfer(root,tables,args):
    for source in args.fpirs:
        weights=json.loads((root/f'fpir_{source:g}'/'weights.json').read_text())['EviRisk']['raw_weights']
        for target in args.fpirs:
            if source==target: continue
            with np.load(root/f'fpir_{target:g}'/'test.npz',allow_pickle=False) as data:
                score=precise_cost_score(data['log_event_probabilities'],weights)
                tables.add('operating_point_transfer',Metrics(data['actions'],data['targets'],args.seed).evaluate(score),
                    dataset=args.dataset,seed=args.seed,source_fpir=source,target_fpir=target,
                    transferred='three weights ONLY; target posterior remains locally fitted',diagnostic_only=True)
