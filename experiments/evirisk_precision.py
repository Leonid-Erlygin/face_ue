"""Label-free ordering of a weighted expected loss near float64 plateaus.

The output is an ordinal score, NOT a risk probability and not a logit.
Ordinary separated values use the existing log-domain formula. Close groups are
ordered by the signed residual around a shared event cost. Cancellation or
indistinguishable residuals use mpmath on the saved log-event masses. No targets,
classifications or validation metrics enter this computation.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.special import logsumexp
from evaluation.open_set_methods.mprisk_evidence import cost_log_odds


def _mp_keys(events, weights):
    import mpmath as mp
    finite=events[np.isfinite(events)]
    span=float(np.max(finite)-np.min(finite)) if finite.size else 0.
    digits=max(80,int(math.ceil(span/math.log(10)))+45)
    if digits>10000:
        raise FloatingPointError('Risk tie needs >10000 decimal digits; inspect saved log masses')
    with mp.workdps(digits):
        w=list(map(lambda x:mp.mpf(float(x)),weights));keys=[]
        for row in events:
            shift=float(np.max(row));p=[mp.mpf(0) if np.isneginf(x) else mp.exp(mp.mpf(float(x))-mp.mpf(shift)) for x in row]
            keys.append(mp.fsum(a*b for a,b in zip(w,p))/mp.fsum(p))
    return keys


def precise_cost_score(log_events, costs, return_audit=False):
    e=np.asarray(log_events,dtype=np.float64);w=np.asarray(costs,dtype=np.float64)
    raw=cost_log_odds(e,w)  # validates finite costs, sizes, and log-event domain
    if not np.all(np.isfinite(logsumexp(e,axis=1))):raise ValueError('An event row has zero total mass')
    ww=np.r_[0.,w/w.max()]
    n=len(e);order=np.argsort(raw,kind='stable');groups=[];start=0
    eps=32*np.finfo(float).eps
    for j in range(1,n+1):
        separate=(j==n)
        if j<n:
            a,b=raw[order[j-1]],raw[order[j]]
            separate=(a!=b) and (not np.isfinite(a+b) or b-a>eps*(1+abs(a)+abs(b)))
        if separate:
            groups.append(order[start:j]);start=j
    result=np.empty(n,dtype=float);rank=0;repaired=0;mp_rows=0
    for ix in groups:
        if len(ix)==1:
            result[ix[0]]=rank;rank+=1;continue
        ee=e[ix];anchors=ww[np.argmax(ee,axis=1)]
        keys=None
        if np.all(anchors==anchors[0]):
            delta=ww-anchors[0];use=delta!=0
            terms=ee[:,use]+np.log(abs(delta[use]))
            la,sign=logsumexp(terms,b=np.sign(delta[use]),axis=1,return_sign=True)
            la=la-logsumexp(ee,axis=1)
            keys=[(int(s),float(-v if s<0 else v) if s else 0.) for s,v in zip(sign,la)]
            # Distinct input rows that collapse to an identical residual, or
            # nearly cancelled opposite-sign terms, need higher precision.
            collision=len(set(keys))<len(keys) and len(np.unique(ee,axis=0))>1
            pos=delta[use]>0;neg=delta[use]<0
            cancel=False
            if np.any(pos) and np.any(neg):
                cancel=np.any(np.abs(logsumexp(terms[:,pos],axis=1)-logsumexp(terms[:,neg],axis=1))<1e-10)
            if collision or cancel:keys=None
        if keys is None:
            keys=_mp_keys(ee,ww);mp_rows+=len(ix)
        local=sorted(range(len(ix)),key=lambda j:keys[j])
        previous=None;assigned=rank
        for j in local:
            key=keys[j]
            if previous is not None and key!=previous:assigned+=1
            result[ix[j]]=assigned;previous=key
        if len(set(raw[ix].tolist()))<len(set(result[ix].tolist())):repaired+=1
        rank+=len(ix)
    audit=dict(n=n,close_groups=sum(len(g)>1 for g in groups),repaired_plateau_groups=repaired,
               high_precision_rows=mp_rows,raw_distinct=int(len(np.unique(raw))),rank_distinct=int(len(np.unique(result))),
               score_kind='ordinal weighted-loss score; not an error probability',uses_labels=False)
    return (result,audit) if return_audit else result


def fit_precise_weights(log_events,actions,targets,budget=1024,seed=777):
    """Same candidate family as fit_risk_weights, label-free precise ordering."""
    from experiments.mprisk_evidence.metrics import Metrics
    e=np.asarray(log_events);d=e.shape[1]-1;rng=np.random.default_rng(seed)
    candidates=[np.ones(d),*np.maximum(np.eye(d),1e-12)]
    candidates.extend(np.exp(rng.uniform(-8,8,(budget,d))))
    metric=Metrics(actions,targets,seed);best=None;seed_scores=[]
    for j,w in enumerate(candidates):
        value=metric.prr(precise_cost_score(e,w))
        if j<=d:seed_scores.append(float(value))
        if np.isfinite(value) and (best is None or value>best[0]):best=(float(value),w/w.max())
    if best is None:best=(float('nan'),np.ones(d))
    return dict(kind='precise_cost_order',raw_weights=best[1].tolist(),validation_prr=best[0],
                identifiable=bool(np.isfinite(best[0])),budget=budget,seed=seed,seed_selection_prr=seed_scores,
                cost_floor=1e-12,scale_convention='max weight = 1',primary_variant='EviRisk',
                objective='validation selection PRR; weights are learned ranking parameters, not elicited application costs')
