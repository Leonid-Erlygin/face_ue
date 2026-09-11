"""OSR risk from reference-prior evidence (no temperature and no NS penalty).

q_x(z)=p_ref(z|x), reference prior u=uniform(S^(d-1)),
f_i=vMF(g_i,k_i), unknown f_0=u.  B_i=integral q_x f_i/u.
Class 0 denotes unknown, class i+1 denotes row i of the supplied gallery.
The action scored is supplied externally, never replaced by a posterior argmax.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np
from scipy.special import ive, gammaln, logsumexp
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

MODEL_VERSION = 'reference-prior-evidence-1.0'


def check_kappa(k):
    a=np.asarray(k,dtype=np.float64)
    if np.any(~np.isfinite(a)) or np.any(a<0):
        raise ValueError('Expected finite kappa >= 0, not log-kappa.')
    return a


def log_surface(d):
    if int(d)!=d or d<2:raise ValueError('Ambient dimension must be an integer >=2')
    return float(math.log(2)+d/2*math.log(math.pi)-gammaln(d/2))


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


def log_posterior(logB,beta,known_prior=None):
    b=np.asarray(logB,dtype=np.float64)
    if b.ndim!=2 or b.shape[1]<1 or not np.all(np.isfinite(b)):raise ValueError('Invalid log Bayes factors')
    if not np.isfinite(beta) or not 0<beta<1:raise ValueError('beta must be in (0,1), not an FPIR')
    K=b.shape[1];w=np.ones(K)/K if known_prior is None else np.asarray(known_prior,dtype=float)
    if w.shape!=(K,) or np.any(w<=0) or not np.isclose(w.sum(),1):raise ValueError('Invalid conditional known prior')
    z=np.column_stack((np.full(len(b),np.log(beta)),np.log1p(-beta)+np.log(w)[None,:]+b))
    return z-logsumexp(z,axis=1,keepdims=True)


def risk_components(lp,actions):
    lp=np.asarray(lp,dtype=np.float64);a=np.asarray(actions)
    n,J=lp.shape
    if a.shape!=(n,) or not np.issubdtype(a.dtype,np.integer) or np.any((a<0)|(a>=J)):
        raise ValueError('actions must be N integer classes 0..K')
    if np.any(~np.isfinite(lp)) or np.max(np.abs(logsumexp(lp,axis=1)))>1e-7:
        raise ValueError('Posterior not normalized')
    accepted=a>0;p=np.exp(lp);others=p[:,1:].copy();ix=np.flatnonzero(accepted)
    others[ix,a[ix]-1]=0
    fa=accepted*p[:,0];ident=accepted*others.sum(1);fr=(~accepted)*(-np.expm1(lp[:,0]))
    selected_lp=lp[np.arange(n),a];risk=-np.expm1(selected_lp)
    if not np.allclose(fa+ident+fr,risk,atol=2e-10,rtol=1e-7):raise AssertionError('Risk decomposition failed')
    return dict(r_fa=fa,r_id=ident,r_fr=fr,risk=risk,p0=p[:,0],selected_probability=np.exp(selected_lp),
                selected_log_probability=selected_lp,posterior_mass=p.sum(1),
                counterfactual_action=np.argmax(lp,axis=1))


@dataclass(frozen=True)
class Parameters:
    probe_scale:float=1.
    gallery_kappa:float=1000.
    beta:float=.5
    point:bool=False


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
        b=pars.gallery_kappa*cc-log_partition(pars.gallery_kappa,d) if pars.point else log_bayes_factors(
            cc,kappa[start:end]*pars.probe_scale,pars.gallery_kappa,d,table)
        lp=log_posterior(b,pars.beta);p=np.exp(lp);v=risk_components(lp,a[start:end])
        v['entropy']=-(p*lp).sum(1)
        v['log_known_bayes_factor']=logsumexp(b,axis=1)-np.log(c.shape[1])
        v['true_log_probability']=np.full(end-start,np.nan);v['class_brier']=np.full(end-start,np.nan)
        if targets is not None:
            y=np.asarray(targets,dtype=int)[start:end]
            if np.any((y<0)|(y>c.shape[1])):raise ValueError('Target index outside 0..K')
            v['true_log_probability']=lp[np.arange(len(y)),y]
            v['class_brier']=(p*p).sum(1)-2*np.exp(v['true_log_probability'])+1
        top=np.argsort(lp[:,1:],axis=1)[:,-min(c.shape[1],5):][:,::-1]+1
        v['top_classes']=top;v['top_probabilities']=np.take_along_axis(p,top,axis=1)
        if dense:v['log_posterior']=lp
        for name,x in v.items():out.setdefault(name,[]).append(x)
    return {name:np.concatenate(v) for name,v in out.items()}


def fit_parameters(c,kappa,d,y,fit_ix,select_ix,beta,table=None,point=False,maxiter=50,max_fit=3000,seed=777):
    """Only supplied validation rows are used; beta is an explicitly fixed prior.

    Fits shared gallery concentration and probe concentration scale by multiclass
    NLL. Three starts are compared on disjoint selection rows. Audit/test absent.
    """
    fit_ix=np.asarray(fit_ix,dtype=int);select_ix=np.asarray(select_ix,dtype=int)
    if np.intersect1d(fit_ix,select_ix).size or min(len(fit_ix),len(select_ix))<4:
        raise ValueError('Need disjoint fit and selection observations')
    if max_fit and len(fit_ix)>max_fit:fit_ix=np.sort(np.random.default_rng(seed).choice(fit_ix,max_fit,replace=False))
    bounds=[(-2.302585093,11.512925465)] if point else [(-9.210340372,4.605170186),(-2.302585093,11.512925465)]
    def decode(z):return Parameters(1. if point else float(np.exp(z[0])),float(np.exp(z[-1])),float(beta),point)
    def loss(z,ix):
        p=decode(z);s=0.
        for j in range(0,len(ix),128):
            ids=ix[j:j+128]
            b=p.gallery_kappa*np.asarray(c[ids])-log_partition(p.gallery_kappa,d) if point else log_bayes_factors(
                c[ids],np.asarray(kappa)[ids]*p.probe_scale,p.gallery_kappa,d,table)
            lp=log_posterior(b,beta);s-=lp[np.arange(len(ids)),np.asarray(y)[ids]].sum()
        return float(s/len(ix))
    history=[];best=None
    for scale,g in [(1.,d),(.1,2*d),(10.,d/2)]:
        print(f'[probability fit] point={point}, start scale={scale:g}, gallery_kappa={g:g}, n_fit={len(fit_ix)}, n_select={len(select_ix)}',flush=True)
        x=np.log([g] if point else [scale,g]);x=np.clip(x,[b[0] for b in bounds],[b[1] for b in bounds])
        res=minimize(lambda z:loss(z,fit_ix),x,method='L-BFGS-B',bounds=bounds,
                     options=dict(maxiter=maxiter,ftol=1e-9,maxls=30))
        score=loss(res.x,select_ix)
        row=dict(parameters=asdict(decode(res.x)),fit_nll=float(res.fun),select_nll=score,
                 success=bool(res.success),message=str(res.message),iterations=int(res.nit),
                 at_boundary=bool(any(abs(v-lo)<.01 or abs(v-hi)<.01 for v,(lo,hi) in zip(res.x,bounds))))
        history.append(row)
        print(f'[probability fit] select NLL={score:.6g}; converged={bool(res.success)}; params={row["parameters"]}',flush=True)
        if np.isfinite(score) and (best is None or score<best[0]):best=(score,decode(res.x))
    if best is None:raise FloatingPointError('No finite validation fit')
    return best[1],dict(candidates=history,fit_indices=fit_ix.tolist(),select_indices=select_ix.tolist(),
                         beta_source='configured prior; never target FPIR',objective='validation multiclass NLL')
