"""OSR risk from reference-prior evidence (no temperature and no NS penalty).

q_x(z)=p_ref(z|x), reference prior u=uniform(S^(d-1)),
f_i=vMF(g_i,k_i), unknown f_0=u.  B_i=integral q_x f_i/u.
Class 0 denotes unknown, class i+1 denotes row i of the supplied gallery.
The action scored is supplied externally, never replaced by a posterior argmax.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from functools import wraps
import math
import numpy as np
from scipy.special import ive, gammaln, logsumexp, expit
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

MODEL_VERSION = 'reference-prior-evidence-1.2-whale-audit'


def _quiet_probability_underflow(function):
    """Allow vanishing probability tails locally; preserve other error policies.

    Log scores remain available when exp(log_probability) rounds to zero.
    This is not a global numpy.seterr change or blanket warning suppression.
    A fresh errstate is created per call (also for nested calls / NumPy 2).
    """
    @wraps(function)
    def wrapped(*args, **kwargs):
        with np.errstate(under='ignore'):
            return function(*args, **kwargs)
    return wrapped


def check_kappa(k):
    a=np.asarray(k,dtype=np.float64)
    if np.any(~np.isfinite(a)) or np.any(a<0):
        raise ValueError('Expected finite kappa >= 0, not log-kappa.')
    return a


def log_surface(d):
    if int(d)!=d or d<2:raise ValueError('Ambient dimension must be an integer >=2')
    return float(math.log(2)+d/2*math.log(math.pi)-gammaln(d/2))


@_quiet_probability_underflow
def _series(a,k):
    """Convergent log-domain 0F1 series for Bessel underflow, not an asymptotic."""
    k=np.asarray(k,dtype=float); t=np.zeros_like(k); s=np.zeros_like(k)
    logz=2*np.log(k/2);active=np.ones(k.shape,bool)
    for j in range(1,100000):
        ix=np.flatnonzero(active)
        if not len(ix):return s
        t[ix]+=logz[ix]-np.log(j)-np.log(a+j-1)
        s[ix]=np.logaddexp(s[ix],t[ix])
        done=(logz[ix]<np.log(j+1)+np.log(a+j)) & (t[ix]-s[ix]<-37)
        active[ix[done]]=False
    raise FloatingPointError('vMF normalization series did not converge')


def centered_partition(k,d):
    """F_d(k)=log E_u[exp(k mu^T Z)]-k. Stable at k=0 and high dimension."""
    log_surface(d);k=check_kappa(k);x=np.atleast_1d(k).ravel();out=np.zeros_like(x)
    pos=x>0;v=x[pos]
    if len(v):
        nu=d/2-1;b=ive(nu,v);good=np.isfinite(b)&(b>1e-280)
        z=np.empty_like(v)
        z[good]=np.log(b[good])-nu*np.log(v[good]/2)+gammaln(d/2)
        z[~good]=_series(d/2,v[~good])-v[~good]
        out[pos]=z
    if not np.all(np.isfinite(out)):raise FloatingPointError('Nonfinite vMF normalizer')
    return out.reshape(k.shape)


def log_partition(k,d):return centered_partition(k,d)+np.asarray(k)


def log_nonspecificity(k,d):
    """For comparison only: log N = 2 A(k)-A(2k), retained even on exp underflow."""
    k=check_kappa(k);v=2*centered_partition(k,d)-centered_partition(2*k,d)
    if np.any(v>1e-7):raise FloatingPointError('Nonspecificity >1 beyond rounding error')
    return np.minimum(v,0.)


class PartitionTable:
    """Checked cubic interpolation of F_d(k) against log(1+k).

    Midpoint plus independent check-point errors are saved. They are a numerical
    audit, not an analytic uniform bound. Out-of-range inputs use exact evaluation.
    """
    def __init__(self,d,max_k=2e6,atol=2e-7,enabled=True):
        log_surface(d)
        if not np.isfinite(max_k) or max_k<=0 or not np.isfinite(atol) or atol<=0:raise ValueError('Invalid interpolation range or tolerance')
        self.d=int(d);self.max_k=float(max_k);self.atol=float(atol);self.sp=None
        self.nodes=0;self.checked_error=0.
        if not enabled:return
        for n in [512,1024,2048,4096,8192,16384,32768]:
            t=np.linspace(0,np.log1p(max_k),n)
            sp=CubicSpline(t,centered_partition(np.expm1(t),d))
            u=np.r_[(t[1:]+t[:-1])/2,np.random.default_rng(133).uniform(0,t[-1],2048)]
            err=float(np.max(np.abs(sp(u)-centered_partition(np.expm1(u),d))))
            if err<=atol:
                self.sp=sp;self.nodes=n;self.checked_error=err;break
        if self.sp is None:raise FloatingPointError(f'Cannot meet interpolation tolerance for d={d}')
    def __call__(self,k):
        k=check_kappa(k)
        if self.sp is None:return centered_partition(k,self.d)
        v=np.asarray(self.sp(np.log1p(np.minimum(k,self.max_k))))
        bad=k>self.max_k
        if np.any(bad):v[bad]=centered_partition(k[bad],self.d)
        return v
    def manifest(self):
        return dict(d=self.d,max_k=self.max_k,atol=self.atol,nodes=self.nodes,checked_error=self.checked_error)


def log_bayes_factors(cosines,kappa,gallery_kappa,d,table=None):
    c=np.asarray(cosines,dtype=np.float64)
    if c.ndim!=2 or np.any(~np.isfinite(c)) or np.any(np.abs(c)>1+1e-6):
        raise ValueError('Expected finite N x K cosine matrix')
    c=np.clip(c,-1,1);k=check_kappa(kappa).reshape(-1,1);g=check_kappa(gallery_kappa)
    g=np.full((1,c.shape[1]),float(g)) if g.ndim==0 else g.reshape(1,-1)
    if k.shape[0]!=c.shape[0] or g.shape[1]!=c.shape[1]:raise ValueError('Concentration shape mismatch')
    h=np.sqrt((k-g)**2+2*k*g*(1+c))
    den=h+k+g
    linear=np.divide(-2*k*g*(1-c),den,out=np.zeros_like(h),where=den>0)
    F=(lambda x:centered_partition(x,d)) if table is None else table
    ans=linear+F(h)-F(k)-F(g)
    ans[(k==0)|(g==0)]=0.
    if not np.all(np.isfinite(ans)):raise FloatingPointError('Nonfinite evidence')
    return ans


def class_log_weights(logB, beta, known_prior=None):
    """Unnormalized class log weights: unknown is column zero."""
    b = np.asarray(logB, dtype=np.float64)
    if b.ndim != 2 or b.shape[1] < 1 or not np.all(np.isfinite(b)):
        raise ValueError('Invalid log Bayes factors')
    if not np.isfinite(beta) or not 0 < beta < 1:
        raise ValueError('beta must be in (0,1), not an FPIR')
    K = b.shape[1]
    w = np.ones(K) / K if known_prior is None else np.asarray(known_prior, dtype=float)
    if w.shape != (K,) or np.any(~np.isfinite(w)) or np.any(w <= 0) or not np.isclose(w.sum(), 1):
        raise ValueError('Invalid conditional known prior')
    return np.column_stack((np.full(len(b), np.log(beta)),
                            np.log1p(-beta) + np.log(w)[None, :] + b))


@_quiet_probability_underflow
def log_posterior(logB, beta, known_prior=None):
    z = class_log_weights(logB, beta, known_prior)
    return z - logsumexp(z, axis=1, keepdims=True)


@_quiet_probability_underflow
def selected_log_odds(log_weights, actions):
    """Log(error mass / chosen-class mass), NEVER logit(1-p_selected).

    Accepts normalized or unnormalized log weights. All hypotheses in the evidence
    model have positive mass; finite log weights retain arbitrarily small alternatives.
    """
    z = np.asarray(log_weights, dtype=np.float64)
    a = np.asarray(actions)
    if z.ndim != 2 or z.shape[1] < 2 or np.any(~np.isfinite(z)):
        raise ValueError('Expected finite N x (K+1) class log weights')
    if a.shape != (len(z),) or not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a >= z.shape[1])):
        raise ValueError('actions must be N integer classes 0..K')
    selected = z[np.arange(len(z)), a]
    wrong = z.copy()
    wrong[np.arange(len(z)), a] = -np.inf
    return logsumexp(wrong, axis=1) - selected


@_quiet_probability_underflow
def risk_from_log_weights(log_weights, actions):
    """Stable risk, event log masses and fixed-action ranking score.

    Probabilities may unavoidably round to 0/1. The log odds are the ranking
    representation. Inactive events have log mass -inf, not an arbitrary epsilon.
    """
    z = np.asarray(log_weights, dtype=np.float64)
    a = np.asarray(actions)
    odds = selected_log_odds(z, a)
    n = len(z)
    lcorrect = -np.logaddexp(0., odds)
    lerror = -np.logaddexp(0., -odds)
    accepted = a > 0
    lfa = np.full(n, -np.inf)
    lid = np.full(n, -np.inf)
    lfr = np.where(accepted, -np.inf, lerror)
    ii = np.flatnonzero(accepted)
    if len(ii):
        alt_known = z[ii, 1:].copy()
        alt_known[np.arange(len(ii)), a[ii] - 1] = -np.inf
        other = logsumexp(alt_known, axis=1)
        unknown = z[ii, 0]
        # Split the error event in log space, avoiding normalizer cancellation.
        lfa[ii] = lerror[ii] - np.logaddexp(0., other - unknown)
        lid[ii] = lerror[ii] - np.logaddexp(0., unknown - other)
    events = np.column_stack((lcorrect, lfa, lid, lfr))
    odds0 = logsumexp(z[:, 1:], axis=1) - z[:, 0]
    logp0 = -np.logaddexp(0., odds0)
    lp = z - logsumexp(z, axis=1, keepdims=True)
    return dict(r_fa=np.exp(lfa), r_id=np.exp(lid), r_fr=np.exp(lfr),
                log_r_fa=lfa, log_r_id=lid, log_r_fr=lfr,
                log_correct_probability=lcorrect, log_error_probability=lerror,
                log_event_probabilities=events, risk=expit(odds), score_log_odds=odds,
                p0=expit(-odds0), log_p0=logp0, known_log_odds=odds0,
                selected_probability=expit(-odds), selected_log_probability=lcorrect,
                posterior_mass=np.exp(lp).sum(1), counterfactual_action=np.argmax(z, axis=1))


@_quiet_probability_underflow
def risk_components(lp, actions):
    """Backward-compatible entry point for an already normalized log posterior."""
    lp = np.asarray(lp, dtype=np.float64)
    if lp.ndim != 2 or np.any(~np.isfinite(lp)) or np.max(np.abs(logsumexp(lp, axis=1))) > 1e-7:
        raise ValueError('Posterior not normalized')
    return risk_from_log_weights(lp, actions)


@_quiet_probability_underflow
def cost_log_odds(log_events, costs):
    """Stable monotone score for a nonnegative weighted sum of error events.

    Columns: correct, FA, ID, FR (optionally further disjoint error events).
    Normalize loss by max(costs). Form both loss and remaining-to-max masses
    explicitly; do not subtract a near-one weighted probability from one.
    """
    e = np.asarray(log_events, dtype=float)
    w = np.asarray(costs, dtype=float)
    if e.ndim != 2 or w.shape != (e.shape[1]-1,) or np.any(np.isnan(e)) or np.any(e == np.inf):
        raise ValueError('Invalid event array/cost shape')
    if np.any(~np.isfinite(w)) or np.any(w < 0) or not np.any(w > 0):
        raise ValueError('Costs must be finite, nonnegative, and not all zero')
    w = np.r_[0., w / w.max()]
    lw = np.full_like(w, -np.inf)
    lc = np.full_like(w, -np.inf)
    np.log(w, out=lw, where=w>0)
    np.log1p(-w, out=lc, where=w<1)
    return logsumexp(e + lw, axis=1) - logsumexp(e + lc, axis=1)


@dataclass(frozen=True)
class Parameters:
    probe_scale:float=1.
    gallery_kappa:float=1000.
    beta:float=.5
    point:bool=False


@_quiet_probability_underflow
def evaluate(c,kappa,d,pars,actions,targets=None,table=None,batch=128,dense=False):
    c=np.asarray(c);kappa=check_kappa(kappa).reshape(-1);a=np.asarray(actions)
    if not np.issubdtype(a.dtype,np.integer) or a.ndim!=1:raise ValueError('actions must be a one-dimensional integer array')
    if not isinstance(batch,int) or batch<1:raise ValueError('batch must be a positive integer')
    if len(c)!=len(kappa) or len(c)!=len(a) or not len(c):raise ValueError('Mismatched or empty evaluation arrays')
    if pars.probe_scale<0 or pars.gallery_kappa<0:raise ValueError('Negative model parameter')
    out={}
    for start in range(0,len(c),batch):
        end=min(len(c),start+batch)
        cc=np.asarray(c[start:end],dtype=float)
        b=pars.gallery_kappa*(cc-1)-centered_partition(pars.gallery_kappa,d) if pars.point else log_bayes_factors(
            cc,kappa[start:end]*pars.probe_scale,pars.gallery_kappa,d,table)
        weights=class_log_weights(b,pars.beta)
        lp=weights-logsumexp(weights,axis=1,keepdims=True);p=np.exp(lp)
        v=risk_from_log_weights(weights,a[start:end])
        # Closed-set ablation: condition on known membership in log space.
        # A rejection is always wrong in that ablated problem.
        known_odds=np.full(end-start,np.inf)
        accepted=np.flatnonzero(a[start:end]>0)
        if len(accepted):
            known_odds[accepted]=(-np.inf if c.shape[1]==1 else
                selected_log_odds(weights[accepted,1:],a[start:end][accepted]-1))
        v['known_only_error_log_odds']=known_odds
        v['entropy']=-(p*lp).sum(1)
        v['log_known_bayes_factor']=logsumexp(b,axis=1)-np.log(c.shape[1])
        v['true_log_probability']=np.full(end-start,np.nan);v['class_brier']=np.full(end-start,np.nan)
        if targets is not None:
            y=np.asarray(targets,dtype=int)[start:end]
            if np.any((y<0)|(y>c.shape[1])):raise ValueError('Target index outside 0..K')
            truth_odds=selected_log_odds(weights,y)
            v['true_log_probability']=-np.logaddexp(0.,truth_odds)
            otherp=p.copy();otherp[np.arange(len(y)),y]=0.
            v['class_brier']=(otherp*otherp).sum(1)+expit(truth_odds)**2
        top=np.argsort(lp[:,1:],axis=1)[:,-min(c.shape[1],5):][:,::-1]+1
        v['top_classes']=top;v['top_probabilities']=np.take_along_axis(p,top,axis=1)
        if dense:v['log_posterior']=lp
        for name,x in v.items():out.setdefault(name,[]).append(x)
    return {name:np.concatenate(v) for name,v in out.items()}


def fit_parameters(c,kappa,d,y,fit_ix,select_ix,beta,table=None,point=False,maxiter=250,max_fit=3000,seed=777,report_path=None,
                   probe_scale_bounds=(1e-4,100.), gallery_kappa_bounds=(.1,1e5)):
    """Only supplied validation rows are used; beta is an explicitly fixed prior.

    Fits shared gallery concentration and probe concentration scale by multiclass
    NLL. Three starts are compared on disjoint selection rows. Audit/test absent.
    """
    fit_ix=np.asarray(fit_ix,dtype=int);select_ix=np.asarray(select_ix,dtype=int)
    if np.intersect1d(fit_ix,select_ix).size or min(len(fit_ix),len(select_ix))<4:
        raise ValueError('Need disjoint fit and selection observations')
    if max_fit and len(fit_ix)>max_fit:fit_ix=np.sort(np.random.default_rng(seed).choice(fit_ix,max_fit,replace=False))
    # Explicit predeclared ranges; a boundary result is reported, not repaired
    # by looking at a test metric. Defaults reproduce the historical ranges.
    for label, pair in [('probe scale',probe_scale_bounds),('gallery kappa',gallery_kappa_bounds)]:
        if len(pair)!=2 or not np.all(np.isfinite(pair)) or not 0<pair[0]<pair[1]:
            raise ValueError('Invalid '+label+' bounds')
    bounds=[tuple(np.log(gallery_kappa_bounds))] if point else [tuple(np.log(probe_scale_bounds)),tuple(np.log(gallery_kappa_bounds))]
    def decode(z):return Parameters(1. if point else float(np.exp(z[0])),float(np.exp(z[-1])),float(beta),point)
    def loss(z,ix):
        p=decode(z);s=0.
        for j in range(0,len(ix),128):
            ids=ix[j:j+128]
            b=p.gallery_kappa*(np.asarray(c[ids])-1)-centered_partition(p.gallery_kappa,d) if point else log_bayes_factors(
                c[ids],np.asarray(kappa)[ids]*p.probe_scale,p.gallery_kappa,d,table)
            weights=class_log_weights(b,beta)
            with np.errstate(under='ignore'):
                s+=np.logaddexp(0.,selected_log_odds(weights,np.asarray(y)[ids])).sum()
        return float(s/len(ix))
    history=[];best=None;best_index=None
    for scale,g in [(1.,d),(.1,2*d),(10.,d/2)]:
        print(f'[probability fit] point={point}, scale={scale:g}, kappa_g={g:g}',flush=True)
        x=np.log([g] if point else [scale,g]);x=np.clip(x,[b[0] for b in bounds],[b[1] for b in bounds])
        attempts=[]
        # Retry only an objectively nonconverged optimizer, never a weak test metric.
        for attempt,limit in enumerate([maxiter,4*maxiter]):
            res=minimize(lambda z:loss(z,fit_ix),x,method='L-BFGS-B',bounds=bounds,
                         options=dict(maxiter=limit,ftol=1e-10,maxls=50,eps=1e-5))
            attempts.append(dict(success=bool(res.success),message=str(res.message),iterations=int(res.nit),
                                 fit_nll=float(res.fun),maxiter=int(limit)))
            if res.success:break
            x=res.x
        score=loss(res.x,select_ix)
        eligible=bool(res.success and np.isfinite(score) and np.isfinite(res.fun))
        row=dict(parameters=asdict(decode(res.x)),fit_nll=float(res.fun),select_nll=score,
                 success=bool(res.success),eligible=eligible,message=str(res.message),iterations=int(res.nit),attempts=attempts,
                 at_boundary=bool(any(abs(v-lo)<.01 or abs(v-hi)<.01 for v,(lo,hi) in zip(res.x,bounds))))
        history.append(row)
        print(f'[probability fit] select NLL={score:.6g}; eligible={eligible}; params={row["parameters"]}',flush=True)
        if eligible and (best is None or score<best[0]):best=(score,decode(res.x));best_index=len(history)-1
    report=dict(candidates=history,fit_indices=fit_ix.tolist(),select_indices=select_ix.tolist(),
                beta_source='configured prior; never target FPIR',objective='validation multiclass NLL',
                selected_index=best_index,selected_converged=best is not None,
                optimization_bounds_log=bounds,bounds_automatically_expanded=False)
    if best is not None:
        z=np.log([best[1].gallery_kappa] if point else [best[1].probe_scale,best[1].gallery_kappa])
        # One-coordinate profiles at fixed other parameter, not profiled optima.
        profile=[]
        for j,parameter in enumerate(['gallery_kappa'] if point else ['probe_scale','gallery_kappa']):
            for factor in [.1,.3,1.,3.,10.]:
                zz=z.copy();zz[j]+=np.log(factor)
                profile.append(dict(parameter=parameter,factor=factor,parameters=asdict(decode(zz)),
                                    inside_fit_bounds=bool(all(lo<=v<=hi for v,(lo,hi) in zip(zz,bounds))),
                                    fit_nll=loss(zz,fit_ix),select_nll=loss(zz,select_ix),diagnostic_only=True))
        report['coordinate_slices']=profile
        report['selected_at_boundary']=history[best_index]['at_boundary']
        report['requires_boundary_review']=report['selected_at_boundary']
    if report_path is not None:
        from experiments.mprisk_evidence.artifacts import write_json
        write_json(report_path,report)
    if best is None:raise RuntimeError('No converged finite validation fit. See fit report; no unconverged candidate was selected.')
    return best[1],report
