"""Score registry with separate probability outputs and log-domain cost tuning."""
import numpy as np
from scipy.special import expit
from evaluation.open_set_methods.mprisk_evidence import log_nonspecificity,cost_log_odds
from experiments.mprisk_evidence.metrics import (Metrics,auc,fit_linear,apply_linear,fit_risk_weights,apply_risk_weights,
    fit_positive_score_calibration,calibrated_log_odds,fit_logistic,logistic_log_odds,fit_mlp,apply_mlp)
from experiments.mprisk_evidence.artifacts import write_json


def augmented_events(events,actions,log_ns):
    """Algebra for risk+NS scoring only, NOT a fourth recognition-error event."""
    events=np.array(events,copy=True);rejected=np.asarray(actions)==0
    ln=np.asarray(log_ns,dtype=float)
    if np.any(ln>1e-8) or np.any(~np.isfinite(ln)):raise ValueError('Invalid log nonspecificity')
    ln=np.minimum(ln,0.)
    extra=np.full(len(events),-np.inf)
    extra[rejected]=events[rejected,0]+ln[rejected]
    with np.errstate(divide='ignore',invalid='raise'):
        events[rejected,0]+=np.log(-np.expm1(ln[rejected]))
    return np.column_stack((events,extra))


def fit_ns_increment(events,actions,targets,core_weights,core_events=None,seed=777):
    """Only the NS coefficient varies; zero includes the EXACT fitted core."""
    m=Metrics(actions,targets,seed);best=None;grid=[]
    for coefficient in [0.,.01,.03,.1,.3,1.,3.,10.]:
        w=np.r_[core_weights,coefficient]
        score=cost_log_odds(core_events,core_weights) if coefficient==0 and core_events is not None else cost_log_odds(events,w)
        v=m.prr(score)
        grid.append(dict(ns_coefficient=coefficient,selection_prr=float(v)))
        if np.isfinite(v) and (best is None or v>best[0]):best=(float(v),w)
    if best is None:best=(np.nan,np.r_[core_weights,0.])
    return dict(kind='cost_weights_log_events',raw_weights=best[1].tolist(),weights=best[1].tolist(),
                validation_prr=best[0],identifiable=bool(np.isfinite(best[0])),coefficient_grid=grid,
                core_weights_fixed=True,no_NS_seed_included=True,extra_term_is_error_probability=False)


def score_variants(val,test,cval,ctest,pv,pt,pointv,pointt,vold,told,select,core,far,root,args,probe_scale=1.):
    from experiments.mprisk_evidence.study import simple_features,threshold_from_actions,fit_holue
    av,at=vold['actions'],told['actions'];yv,yt=val.targets,test.targets
    ev,et=pv['log_event_probabilities'],pt['log_event_probabilities']
    cv=np.column_stack([pv[k] for k in ['r_fa','r_id','r_fr']]);ct=np.column_stack([pt[k] for k in ['r_fa','r_id','r_fr']])
    ov=np.column_stack([vold['components'][k] for k in ['r_fa','r_id','r_fr']]+[vold['r_ns']])
    ot=np.column_stack([told['components'][k] for k in ['r_fa','r_id','r_fr']]+[told['r_ns']])
    oldev=augmented_events(vold['components']['log_event_probabilities'],av,vold['log_nonspecificity'])
    oldet=augmented_events(told['components']['log_event_probabilities'],at,told['log_nonspecificity'])
    sv={};st={};probability={};learned={}
    def add_probability(name,vp,tp,vl=None,tl=None,vr=None,tr=None):
        sv[name]=np.asarray(vp if vr is None and vl is None else (vl if vr is None else vr))
        st[name]=np.asarray(tp if tr is None and tl is None else (tl if tr is None else tr))
        probability[name]=dict(validation=np.asarray(vp),test=np.asarray(tp),validation_log_odds=vl,test_log_odds=tl)
    add_probability('MPRisk',pv['risk'],pt['risk'],pv['score_log_odds'],pt['score_log_odds'])
    add_probability('Point-vMF (NLL fitted)',pointv['risk'],pointt['risk'],pointv['score_log_odds'],pointt['score_log_odds'])
    add_probability('MPRisk-paper no NS',vold['components']['risk'],told['components']['risk'],
                    vold['components']['score_log_odds'],told['components']['score_log_odds'])
    sv['MPRisk-paper raw']=cost_log_odds(oldev,np.ones(4));st['MPRisk-paper raw']=cost_log_odds(oldet,np.ones(4))
    tv=float(vold['tau']) if 'tau' in vold else threshold_from_actions(cval,av)
    tt=float(told['tau']) if 'tau' in told else threshold_from_actions(ctest,at)
    candidates=np.geomspace(.001,50,25 if args.stage=='full' else 10);best=(-np.inf,1.)
    for temp in candidates:
        u=simple_features(cval[select],val.kappa[select],tv,float(temp))['MSP']
        sc=auc(av[select]!=yv[select],u)
        if np.isfinite(sc) and sc>best[0]:best=(sc,float(temp))
    gv=dict(vold);gt=dict(told)
    if core is not None and 'dataset_name_to_T_scale' in core:
        gv['galue_T']=gt['galue_T']=float(core.dataset_name_to_T_scale[test.source['dataset_name']])
    sv.update(simple_features(cval,val.kappa,tv,best[1],gv,args.beta,val.d))
    st.update(simple_features(ctest,test.kappa,tt,best[1],gt,args.beta,test.d))
    hv,ht,vh,th,hpars=fit_holue(core,val,test,vold,told,select,far,root,args.seed,args.synthetic)
    add_probability('HolUE (refit fixed decisions)',hv,ht)
    sv['HolUE raw']=-(vh['kl1']+vh['kl2']);st['HolUE raw']=-(th['kl1']+th['kl2'])
    for label,valuev,valuet in [('HolUE KL1',vh['kl1'],th['kl1']),('HolUE KL2',vh['kl2'],th['kl2'])]:
        sv[label]=valuev;st[label]=valuet
    for i,key in enumerate(['paper_r_FA','paper_r_ID','paper_r_FR','paper_r_NS']):sv[key]=ov[:,i];st[key]=ot[:,i]
    # Fit genuine nonnegative cost combinations directly from logarithmic masses.
    for name,v,t in [('MPRisk PRR-weighted',ev,et),('MPRisk-paper retuned',oldev,oldet)]:
        p=fit_risk_weights(v[select],av[select],yv[select],args.search_budget,args.seed)
        p['feature_names']=['r_FA','r_ID','r_FR']+(['paper_r_NS'] if v.shape[1]==5 else [])
        learned[name]=p;sv[name]=apply_risk_weights(v,p);st[name]=apply_risk_weights(t,p)
    for name,source in [('MPRisk calibrated','MPRisk'),('MPRisk-paper calibrated','MPRisk-paper retuned')]:
        p=fit_positive_score_calibration(sv[source][select],av[select]!=yv[select]);p['input_score']=source
        learned[name]=p;lv=calibrated_log_odds(sv[source],p);lt=calibrated_log_odds(st[source],p)
        # Preserve raw ordering even if calibrated sigmoid saturates or one-class fit is constant.
        add_probability(name,expit(lv),expit(lt),lv,lt,sv[source],st[source])
    kv=np.column_stack((vh['kl1'],vh['kl2']));kt=np.column_stack((th['kl1'],th['kl2']))
    # Matched incremental-feature test: core weights stay fixed; only lambda_NS varies.
    for suffix,scale in [('',probe_scale),(' original-scale',1.)]:
        vv=augmented_events(ev,av,log_nonspecificity(val.kappa*scale,val.d))
        vt=augmented_events(et,at,log_nonspecificity(test.kappa*scale,test.d))
        name='MPRisk + NS'+suffix+' diagnostic'
        p=fit_ns_increment(vv[select],av[select],yv[select],learned['MPRisk PRR-weighted']['raw_weights'],core_events=ev[select],seed=args.seed)
        p.update(ns_probe_scale=float(scale),compared_to='MPRisk PRR-weighted',feature_names=['r_FA','r_ID','r_FR','NS'])
        learned[name]=p
        sv[name]=sv['MPRisk PRR-weighted'].copy() if p['raw_weights'][-1]==0 else apply_risk_weights(vv,p)
        st[name]=st['MPRisk PRR-weighted'].copy() if p['raw_weights'][-1]==0 else apply_risk_weights(vt,p)
    simple_names=['SCF','AccScr','MSP','Margin']
    all_names=['SCF','AccScr','MSP','Entropy','Margin','GalUE','HolUE (refit fixed decisions)','HolUE raw',
               'HolUE KL1','HolUE KL2','paper_r_FA','paper_r_ID','paper_r_FR','paper_r_NS',
               'Point-vMF (NLL fitted)','MPRisk-paper raw','MPRisk-paper no NS','MPRisk-paper retuned','MPRisk-paper calibrated']
    Xv=np.column_stack([sv[n] for n in all_names]);Xt=np.column_stack([st[n] for n in all_names])
    augmented_names=all_names+['MPRisk','MPRisk PRR-weighted','MPRisk calibrated',
                             'MPRisk + NS diagnostic','MPRisk + NS original-scale diagnostic','r_FA','r_ID','r_FR']
    Av=np.column_stack((Xv,sv['MPRisk'],sv['MPRisk PRR-weighted'],sv['MPRisk calibrated'],
                        sv['MPRisk + NS diagnostic'],sv['MPRisk + NS original-scale diagnostic'],cv))
    At=np.column_stack((Xt,st['MPRisk'],st['MPRisk PRR-weighted'],st['MPRisk calibrated'],
                        st['MPRisk + NS diagnostic'],st['MPRisk + NS original-scale diagnostic'],ct))
    def linear(name,v,t,names,nested=None):
        p=fit_linear(v[select],av[select],yv[select],args.search_budget,args.seed,nested=nested)
        p['feature_names']=names;learned[name]=p;sv[name]=apply_linear(v,p);st[name]=apply_linear(t,p)
    simple_cols=[all_names.index(n) for n in simple_names]
    linear('Linear simple',Xv[:,simple_cols],Xt[:,simple_cols],simple_names)
    linear('Linear HolUE KL',kv,kt,['HolUE KL1','HolUE KL2'])
    kl_cols=[all_names.index(n) for n in ['HolUE KL1','HolUE KL2']]
    linear('Linear all baseline scores',Xv,Xt,all_names,[('Linear simple',simple_cols,learned['Linear simple']),
           ('Linear HolUE KL',kl_cols,learned['Linear HolUE KL'])])
    hybrid_names=['HolUE KL1','HolUE KL2','MPRisk','MPRisk PRR-weighted','r_FA','r_ID','r_FR']
    hycols=[augmented_names.index(n) for n in hybrid_names]
    linear('Hybrid KL + MPRisk',Av[:,hycols],At[:,hycols],hybrid_names,[('Linear HolUE KL',[0,1],learned['Linear HolUE KL'])])
    linear('Linear all + MPRisk',Av,At,augmented_names,[('Linear all baseline scores',list(range(len(all_names))),learned['Linear all baseline scores']),
           ('Hybrid KL + MPRisk',hycols,learned['Hybrid KL + MPRisk'])])
    logit=fit_logistic(Av[select],av[select]!=yv[select],args.seed);learned['Supervised logistic']=logit
    lv=logistic_log_odds(Av,logit);lt=logistic_log_odds(At,logit)
    add_probability('Supervised logistic',expit(lv),expit(lt),lv,lt)
    if args.stage=='full':
        net=fit_mlp(Av[select],av[select]!=yv[select],args.seed);learned['Supervised MLP']=net
        add_probability('Supervised MLP',apply_mlp(Av,net),apply_mlp(At,net))
    write_json(root/'score_fits.json',dict(fits=learned,MSP_temperature=best[1],MSP_selection_auroc=best[0],HolUE=hpars,
        baseline_features=all_names,augmented_features=augmented_names,selection_indices=select,
        all_means='explicit finite registry of standalone and pre-fitted scores; excludes recursive larger fusions',
        ranking_encodings=dict(primary='error log odds',cost_weighted='log odds of loss divided by maximum cost',historical_raw='monotone log-odds encoding of paper-form score; NOT error probability'),
        note='PRR-tuned coefficients are ranking weights, not measured error costs. Log-odds ranking is distinct from probability output.'))
    return sv,st,probability,learned,dict(validation_components=ev,test_components=et,
        validation_probability_components=cv,test_probability_components=ct,
        validation_features=Xv,test_features=Xt,validation_augmented_features=Av,test_augmented_features=At,
        validation_old_components=ov,test_old_components=ot,feature_names=all_names,augmented_feature_names=augmented_names,
        validation_kl=kv,test_kl=kt)
