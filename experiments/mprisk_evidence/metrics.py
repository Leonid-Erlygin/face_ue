"""Fixed-action evaluation and validation-only score fitting."""
import numpy as np
from scipy.special import expit, logit
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression


def masks(a,y):
    a,y=np.asarray(a),np.asarray(y);known=y>0;accepted=a>0
    return dict(known=known,accepted=accepted,any_error=a!=y,tp=accepted&(a==y),fp=accepted&~known,
                fn=known&(a!=y),false_accept=accepted&~known,false_reject=~accepted&known,
                misidentification=accepted&known&(a!=y),true_reject=(a==0)&(y==0))


def auc(y,s):return float(roc_auc_score(y,s)) if len(np.unique(y))==2 else np.nan

def ap(y,s):return float(average_precision_score(y,s)) if len(np.unique(y))==2 else np.nan

def area(y,x):return float(np.trapezoid(y,x) if hasattr(np,'trapezoid') else np.trapz(y,x))

def f1(tp,fp,fn):
    den=2*tp+fp+fn
    return np.divide(2*tp,den,out=np.zeros_like(np.asarray(den),dtype=float),where=den>0)


class Metrics:
    """Repository F1 convention: incorrect identification contributes to FN.

    PRR uses its random/error-oracle normalization, not an asserted F1-optimal
    oracle. Constant-score ties use a shared label-independent permutation.
    AURC is the mean cumulative risk at coverages 1/N,...,1.
    """
    def __init__(self,a,y,seed=777,fractions=20):
        self.a=np.asarray(a,dtype=int);self.y=np.asarray(y,dtype=int);self.n=len(a)
        if not self.n or self.y.shape!=self.a.shape:raise ValueError('Invalid actions/targets')
        self.m=masks(self.a,self.y);rng=np.random.default_rng(seed);self.tie=rng.permutation(self.n)
        self.fracs=np.linspace(0,.5,int(fractions));self.keep=np.maximum(1,((1-self.fracs)*self.n).astype(int))
        self.random=rng.random(self.n);self.oracle=self.m['any_error'].astype(float)
        self.random_area=self.f1_area(self.random);self.oracle_area=self.f1_area(self.oracle)
    def order(self,s):
        s=np.asarray(s,dtype=float)
        if s.shape!=(self.n,) or not np.all(np.isfinite(s)):raise ValueError('Nonfinite/misaligned score')
        return np.lexsort((self.tie,s))
    def f1_curve(self,s):
        order=self.order(s);counts=[np.cumsum(self.m[k][order])[self.keep-1] for k in ['tp','fp','fn']]
        return f1(*counts)
    def f1_area(self,s):return area(self.f1_curve(s),self.fracs)
    def prr(self,s):
        den=self.oracle_area-self.random_area
        return (self.f1_area(s)-self.random_area)/den if abs(den)>1e-12 else np.nan
    def evaluate(self,s,probability=False):
        order=self.order(s);err=self.m['any_error'].astype(float);nk=self.m['known'].sum();nu=self.n-nk
        risk=np.cumsum(err[order])/np.arange(1,self.n+1)
        ans=dict(n=self.n,errors=int(err.sum()),error_rate=float(err.mean()),f1=float(f1(*[self.m[k].sum() for k in ['tp','fp','fn']])),
                 fnir=float(self.m['fn'].sum()/nk) if nk else np.nan,fpir=float(self.m['fp'].sum()/nu) if nu else np.nan,
                 prr_f1=float(self.prr(s)),error_auroc=auc(err,s),error_auprc=ap(err,s),aurc=float(risk.mean()),
                 excess_aurc=float(risk.mean()-np.mean(np.cumsum(np.sort(err))/np.arange(1,self.n+1))),
                 random_f1_area=self.random_area,error_oracle_f1_area=self.oracle_area,f1_area=self.f1_area(s),
                 score_min=float(np.min(s)),score_max=float(np.max(s)),is_probability=bool(probability))
        if probability:
            p=np.asarray(s)
            if np.any((p<-1e-8)|(p>1+1e-8)):raise ValueError('Score is not a probability')
            p=np.clip(p,0,1);v=np.clip(p,1e-15,1-1e-15)
            ans.update(error_brier=float(np.mean((p-err)**2)),error_nll=float(-np.mean(err*np.log(v)+(1-err)*np.log1p(-v))),
                       predicted_risk=float(p.mean()),calibration_gap=float(p.mean()-err.mean()))
        return ans
    def curves(self,s):
        order=self.order(s);sums={k:np.cumsum(self.m[k][order]) for k in ['tp','fp','fn','known','any_error']};rows=[]
        for f,n in zip(self.fracs,self.keep):
            nk=sums['known'][n-1];nu=n-nk
            rows.append(dict(filter_fraction=float(f),coverage=float(n/self.n),retained=int(n),
                  f1=float(f1(sums['tp'][n-1],sums['fp'][n-1],sums['fn'][n-1])),risk=float(sums['any_error'][n-1]/n),
                  fpir=float(sums['fp'][n-1]/nu) if nu else np.nan,fnir=float(sums['fn'][n-1]/nk) if nk else np.nan))
        rc=[dict(coverage=n/self.n,retained=int(n),risk=float(sums['any_error'][n-1]/n))
            for n in np.unique(np.maximum(1,np.rint(np.linspace(.01,1,100)*self.n).astype(int))) ]
        return rows,rc


def bins(p,y,number=15):
    p=np.asarray(p);y=np.asarray(y,dtype=float);g=np.minimum((p*number).astype(int),number-1);rows=[]
    for b in range(number):
        ix=g==b;rows.append(dict(bin=b,lo=b/number,hi=(b+1)/number,count=int(ix.sum()),
             predicted=float(p[ix].mean()) if ix.any() else np.nan,observed=float(y[ix].mean()) if ix.any() else np.nan))
    return rows


def fit_linear(X,a,y,budget=512,seed=777,positive=False,include=None):
    X=np.asarray(X,dtype=float);mean=X.mean(0);sd=X.std(0);sd=np.where(sd>1e-14,sd,1.);Z=(X-mean)/sd
    if not np.all(np.isfinite(Z)):raise ValueError('Invalid fusion features')
    d=X.shape[1];rng=np.random.default_rng(seed);m=Metrics(a,y,seed)
    candidates=[np.ones(d)/d,*np.eye(d)]
    if not positive:candidates+=list(-np.eye(d))
    if include is not None:candidates += [np.asarray(v)*sd for v in include]
    for _ in range(budget):
        w=np.exp(rng.uniform(-8,8,d))
        if not positive:w*=rng.choice([-1.,1.],d)
        candidates.append(w/(np.abs(w).sum()+1e-30))
    best=None
    for w in candidates:
        score=m.prr(Z@w)
        if np.isfinite(score) and (best is None or score>best[0]):best=(float(score),w)
    valid=best is not None
    if best is None:best=(np.nan,sd*np.ones(d)/d)
    return dict(mean=mean.tolist(),scale=sd.tolist(),weights=best[1].tolist(),raw_weights=(best[1]/sd).tolist(),
                validation_prr=best[0],identifiable=valid,positive=positive,budget=budget)


def apply_linear(X,p):return ((np.asarray(X)-p['mean'])/p['scale'])@np.asarray(p['weights'])


def fit_monotone(p,y):
    t=logit(np.clip(p,1e-12,1-1e-12));y=np.asarray(y,dtype=float)
    def loss(v):
        z=np.exp(v[0])*t+v[1];return float(np.mean(np.logaddexp(0,z)-y*z))
    r=minimize(loss,[0.,0.],method='L-BFGS-B',bounds=[(-8,8),(-30,30)])
    return dict(slope=float(np.exp(r.x[0])),intercept=float(r.x[1]),success=bool(r.success))


def apply_monotone(p,c):return expit(c['slope']*logit(np.clip(p,1e-12,1-1e-12))+c['intercept'])


def fit_logistic(X,y,seed=777):
    X=np.asarray(X);mean=X.mean(0);sd=X.std(0);sd=np.where(sd>1e-14,sd,1.)
    if len(np.unique(y))<2:return dict(mean=mean.tolist(),scale=sd.tolist(),coef=[0.]*X.shape[1],intercept=float(logit(np.clip(np.mean(y),1e-8,1-1e-8))))
    clf=LogisticRegression(C=1.,max_iter=2000,random_state=seed).fit((X-mean)/sd,y)
    return dict(mean=mean.tolist(),scale=sd.tolist(),coef=clf.coef_[0].tolist(),intercept=float(clf.intercept_[0]))


def apply_logistic(X,p):return expit(((np.asarray(X)-p['mean'])/p['scale'])@np.asarray(p['coef'])+p['intercept'])


def ranks(x,y,seed=777,pairs=50000):
    rng=np.random.default_rng(seed);a=rng.integers(len(x),size=pairs);b=rng.integers(len(x),size=pairs)
    dx=x[a]-x[b];dy=y[a]-y[b];valid=(dx!=0)&(dy!=0)
    return dict(correlation=float(spearmanr(x,y).statistic) if np.std(x)>0 and np.std(y)>0 else np.nan,
                inversions=float(np.mean(dx[valid]*dy[valid]<0)) if valid.any() else np.nan,non_tied_pairs=int(valid.sum()))


def bootstrap(a,y,scores,reference,groups,repeats=200,seed=777):
    g=np.asarray(groups).astype(str);unique,inv=np.unique(g,return_inverse=True);blocks=[np.flatnonzero(inv==i) for i in range(len(unique))]
    cluster=len(unique)>=20;unit='identity-clustered paired' if cluster else 'within-class probe bootstrap; conditional on these classes'
    rng=np.random.default_rng(seed);names=[n for n in scores if n!=reference];diff={n:[] for n in names}
    for _ in range(repeats):
        ix=np.concatenate([blocks[i] for i in rng.integers(len(blocks),size=len(blocks))]) if cluster else np.concatenate([rng.choice(b,len(b),replace=True) for b in blocks])
        m=Metrics(a[ix],y[ix],seed);base=m.evaluate(scores[reference][ix])
        for n in names:
            v=m.evaluate(scores[n][ix]);diff[n].append([v[k]-base[k] for k in ['prr_f1','error_auroc','aurc']])
    rows=[]
    for name,v in diff.items():
        ar=np.asarray(v)
        for j,k in enumerate(['prr_f1','error_auroc','aurc']):
            z=ar[:,j];z=z[np.isfinite(z)]
            rows.append(dict(method=name,reference=reference,metric=k,unit=unit,repeats=repeats,valid_repeats=len(z),
                   mean_difference=float(z.mean()) if len(z) else np.nan,
                   ci_low=float(np.quantile(z,.025)) if len(z) else np.nan,ci_high=float(np.quantile(z,.975)) if len(z) else np.nan))
    return rows


def fit_mlp(X,y,seed=777):
    """Small supervised baseline; all preprocessing/training uses selection only.

    Export plain weights instead of a pickle so predictions can be independently
    reconstructed and the shared results need not deserialize executable objects.
    """
    from sklearn.neural_network import MLPClassifier
    X=np.asarray(X,dtype=float);y=np.asarray(y,dtype=int)
    mean=X.mean(0);sd=X.std(0);sd=np.where(sd>1e-14,sd,1.)
    if len(np.unique(y))<2:
        return dict(mean=mean.tolist(),scale=sd.tolist(),constant=float(y.mean()),layers=[])
    early=len(y)>=30 and min(np.bincount(y))>=3
    clf=MLPClassifier(hidden_layer_sizes=(32,16),activation='relu',alpha=.01,
                      max_iter=400,random_state=seed,early_stopping=early,
                      validation_fraction=.15,n_iter_no_change=25)
    clf.fit((X-mean)/sd,y)
    return dict(mean=mean.tolist(),scale=sd.tolist(),constant=None,
                layers=[dict(weight=w.tolist(),bias=b.tolist()) for w,b in zip(clf.coefs_,clf.intercepts_)],
                iterations=int(clf.n_iter_),early_stopping=bool(early),seed=seed)


def apply_mlp(X,pars):
    if pars['constant'] is not None:return np.full(len(X),pars['constant'])
    z=(np.asarray(X)-pars['mean'])/pars['scale']
    for j,layer in enumerate(pars['layers']):
        z=z@np.asarray(layer['weight'])+np.asarray(layer['bias'])
        z=expit(z) if j==len(pars['layers'])-1 else np.maximum(z,0.)
    return z.reshape(-1)


def fit_positive_score_calibration(score,errors):
    """Monotone probability calibration for arbitrary (possibly negative) scores."""
    score=np.asarray(score,dtype=float);y=np.asarray(errors,dtype=float)
    mean=float(score.mean());sd=float(score.std());sd=sd if sd>1e-14 else 1.
    x=(score-mean)/sd
    def objective(v):
        z=np.exp(v[0])*x+v[1]
        return float(np.mean(np.logaddexp(0,z)-y*z))
    r=minimize(objective,[0.,0.],method='L-BFGS-B',bounds=[(-8,8),(-30,30)])
    return dict(mean=mean,scale=sd,slope=float(np.exp(r.x[0])),intercept=float(r.x[1]),
                success=bool(r.success),objective='unweighted binary NLL on validation selection')


def apply_positive_score_calibration(score,parameters):
    return expit(parameters['slope']*(np.asarray(score)-parameters['mean'])/parameters['scale']+parameters['intercept'])
