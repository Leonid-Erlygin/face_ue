"""Adapter to the supplied repository. No modern_ai modules are imported.

SCF files store log-kappa. Pooling follows PoolingDefault: mean within each
media, sum media representations, normalize; average concentration over media.
Pooling is reconstructed from source arrays to avoid shared stale cache keys.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
from scipy.special import logsumexp
from evaluation.open_set_methods.mprisk_evidence import log_partition,log_posterior,log_nonspecificity


def instantiate_config(config):
    """Instantiate the simple nested _target_ configurations used by this repo.

    No Hydra runtime / working-directory changes are needed. Complex Hydra
    directives are rejected explicitly instead of being interpreted differently.
    """
    import importlib
    from omegaconf import OmegaConf
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    if isinstance(config, list):
        return [instantiate_config(value) for value in config]
    if not isinstance(config, dict):
        return config
    if any(key in config for key in ('_partial_', '_recursive_', '_convert_', '_args_')):
        raise ValueError('Unsupported instantiation directive in source configuration')
    values = {key: instantiate_config(value) for key, value in config.items() if key != '_target_'}
    if '_target_' not in config:
        return values
    module, attribute = config['_target_'].rsplit('.', 1)
    factory = getattr(importlib.import_module(module), attribute)
    return factory(**values)


def digest_array(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256();h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes());return h.hexdigest()


def normalized(a):
    a=np.asarray(a,dtype=np.float64);n=np.linalg.norm(a,axis=1,keepdims=True)
    if a.ndim!=2 or np.any(~np.isfinite(a)) or np.any(n<=0):raise ValueError('Invalid representation direction')
    return a/n


@dataclass
class Data:
    mu:np.ndarray
    kappa:np.ndarray
    gallery:np.ndarray
    gallery_kappa:np.ndarray
    probe_ids:np.ndarray
    gallery_ids:np.ndarray
    template_ids:np.ndarray
    source:dict
    @property
    def d(self):return self.mu.shape[1]
    @property
    def n(self):return len(self.mu)
    @property
    def targets(self):
        index={str(v):i+1 for i,v in enumerate(self.gallery_ids)}
        return np.array([index.get(str(v),0) for v in self.probe_ids],dtype=int)
    def validate(self):
        self.mu=normalized(self.mu);self.gallery=normalized(self.gallery)
        self.kappa=np.asarray(self.kappa,dtype=float).reshape(-1);self.gallery_kappa=np.asarray(self.gallery_kappa,dtype=float).reshape(-1)
        if self.d!=self.gallery.shape[1] or not self.n or not len(self.gallery):raise ValueError('Empty data or dimension mismatch')
        if len(set(map(str,self.gallery_ids)))!=len(self.gallery):raise ValueError('Gallery identities must be unique; duplicate prototypes are not silently treated as new classes')
        for a in [self.kappa,self.probe_ids,self.template_ids]:
            if len(a)!=self.n:raise ValueError('Pooled probe metadata length mismatch; refusing to truncate')
        for a in [self.kappa,self.gallery_kappa]:
            if np.any(~np.isfinite(a)) or np.any(a<0):raise ValueError('Nonfinite/negative decoded concentration')
        if len(np.unique(self.template_ids))!=self.n:raise ValueError('Duplicate probe templates')
        return self
    def take(self,ix):
        ix=np.asarray(ix,dtype=int)
        return Data(self.mu[ix],self.kappa[ix],self.gallery,self.gallery_kappa,self.probe_ids[ix],self.gallery_ids,self.template_ids[ix],self.source).validate()
    def metadata(self):
        return dict(source=self.source,n=self.n,K=len(self.gallery),d=self.d,
                    unknown_fraction=float(np.mean(self.targets==0)),
                    mean_kappa=float(np.mean(self.kappa)),kappa_quantiles=np.quantile(self.kappa,[0,.1,.5,.9,1]).tolist(),
                    representations_sha256=digest_array(self.mu),concentrations_sha256=digest_array(self.kappa),
                    gallery_sha256=digest_array(self.gallery),templates_sha256=digest_array(self.template_ids))


def _pool(raw,logk,templates,medias,chosen,subjects):
    chosen=np.asarray(chosen).reshape(-1);subjects=np.asarray(subjects).reshape(-1)
    if len(chosen)!=len(subjects):raise ValueError('Protocol template/subject length mismatch')
    mapping={}
    for t,s in zip(chosen,subjects):
        if t in mapping and mapping[t]!=s:raise ValueError('Template assigned to multiple subjects')
        mapping[t]=s
    keys=np.array(sorted(mapping));order=np.argsort(templates,kind='stable');st=templates[order]
    mu=[];kap=[]
    for t in keys:
        lo, hi = np.searchsorted(st, t, 'left'), np.searchsorted(st, t, 'right')
        ix=order[lo:hi]
        if not len(ix):raise ValueError(f'Protocol template {t} is absent from raw representations')
        m=medias[ix];unique,inv=np.unique(m,return_inverse=True);counts=np.bincount(inv)
        vectors=np.zeros((len(unique),raw.shape[1]));np.add.at(vectors,inv,raw[ix]);vectors/=counts[:,None]
        concentrations=np.exp(logk[ix]);sums=np.zeros(len(unique));np.add.at(sums,inv,concentrations);sums/=counts
        mu.append(vectors.sum(0));kap.append(sums.mean())
    return normalized(np.array(mu)),np.array(kap),np.array([mapping[t] for t in keys]),keys


def load_native(dataset_cfg,gallery_name='g1'):
    ds=instantiate_config(dataset_cfg);path=Path(ds.dataset_path)/'embeddings'/f'scf_embs_{ds.dataset_name}.npz'
    if not path.is_file():raise FileNotFoundError(f'Required SCF file: {path}')
    with np.load(path,allow_pickle=False) as f:
        raw=np.asarray(f['embs']);lk=np.asarray(f['unc'])
    if lk.ndim==2 and lk.shape[1]==1:lk=lk[:,0]
    if lk.ndim!=1:raise ValueError(f'SCF must provide one log concentration per input: {path}')
    t=np.asarray(ds.templates);m=np.asarray(ds.medias)
    if len(raw)!=len(lk) or len(t)!=len(raw) or len(m)!=len(raw):raise ValueError('Raw representation/metadata alignment failure')
    probe=_pool(raw,lk,t,m,ds.probe_templates,ds.probe_ids)
    gal=_pool(raw,lk,t,m,getattr(ds,gallery_name+'_templates'),getattr(ds,gallery_name+'_ids'))
    overlap=np.intersect1d(probe[3],gal[3])
    if len(overlap):raise ValueError(f'{len(overlap)} exact templates overlap probe and gallery; resolve protocol leakage first')
    source=dict(dataset_path=str(Path(ds.dataset_path).resolve()),dataset_name=ds.dataset_name,embedding_file=str(path.resolve()),
                embedding_size=path.stat().st_size,embedding_mtime_ns=path.stat().st_mtime_ns,gallery_name=gallery_name,
                uncertainty_storage='SCF log-kappa; exponentiated once before media pooling',pooling='PoolingDefault equivalent, rebuilt; no shared cache')
    return Data(probe[0],probe[1],gal[0],gal[1],probe[2],gal[2],probe[3],source).validate()


def similarities(data,path,device='auto',batch=512):
    """Disk-backed N x K cosine cache, excluded from share ZIP (derived scratch)."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    c=np.lib.format.open_memmap(path,mode='w+',dtype='float64',shape=(data.n,len(data.gallery)))
    backend='numpy'
    if device!='cpu':
        import torch
        if torch.cuda.is_available():
            dev='cuda' if device=='auto' else device
            g=torch.as_tensor(data.gallery,dtype=torch.float64,device=dev);backend=dev
            for lo in range(0,data.n,batch):
                with torch.no_grad():
                    z=torch.as_tensor(data.mu[lo:lo+batch],dtype=torch.float64,device=dev)
                    c[lo:lo+batch]=(z@g.T).cpu().numpy()
            del g
        elif device not in ['auto','cpu']:raise RuntimeError(f'Requested device {device} unavailable')
    if backend=='numpy':
        for lo in range(0,data.n,batch):c[lo:lo+batch]=data.mu[lo:lo+batch]@data.gallery.T
    c.flush();return c,backend


def split_validation(data,seed=777):
    """60/20/20 fit/select/audit. Identity-disjoint when each stratum has >=6 IDs.

    Small topic galleries cannot support identity-disjoint partitions containing
    both known and unknown examples. For those, split templates within class and
    explicitly report that the audit is conditional on the represented classes.
    """
    rng=np.random.default_rng(seed);y=data.targets;ids=np.asarray(data.probe_ids).astype(str)
    use_groups=min(len(np.unique(ids[y==0])),len(np.unique(ids[y>0])))>=6
    parts=[[],[],[]]
    if use_groups:
        for known in [False,True]:
            keys=np.unique(ids[(y>0)==known]);rng.shuffle(keys);n=len(keys)
            n1=max(1,int(.6*n));n2=max(n1+1,int(.8*n));n2=min(n2,n-1)
            for p,ks in zip(parts,[keys[:n1],keys[n1:n2],keys[n2:]]):p.extend(np.flatnonzero(np.isin(ids,ks)))
        unit='identity-disjoint'
    else:
        for cls in np.unique(y):
            ix=np.flatnonzero(y==cls);rng.shuffle(ix)
            if len(ix)<5:raise ValueError('Need at least 5 validation probes per represented class for fit/select/audit')
            n1=max(1,int(.6*len(ix)));n2=min(len(ix)-1,max(n1+1,int(.8*len(ix))))
            for p,z in zip(parts,[ix[:n1],ix[n1:n2],ix[n2:]]):p.extend(z)
        unit='template-disjoint within class; conditional on these classes'
    out={n:np.array(sorted(p),dtype=int) for n,p in zip(['fit','select','audit'],parts)}
    if min(map(len,out.values()))<4:raise ValueError('Validation partitions too small')
    if len(np.unique(np.concatenate(list(out.values()))))!=data.n:raise AssertionError('Invalid validation split')
    return out,unit


def native_method(core,name):
    from omegaconf import OmegaConf
    for m in core.open_set_identification_methods:
        if str(m.pretty_name)==name:return OmegaConf.create(OmegaConf.to_container(m.recognition_method,resolve=True))
    raise KeyError(f'Method {name} absent in supplied core config')


def run_legacy(core,data,far,name='MPRisk raw',gallery_kappa=None):
    """Original code defines the reference decision, not the evidence posterior.

    This intentionally retains the source benchmark's target-FPIR operating-point
    construction using unknown protocol labels. It is NOT a prospective FPIR
    guarantee. Evidence parameters and score weights never see test labels.
    """
    from omegaconf import open_dict
    from evaluation.open_set_methods.mprisk_evidence import risk_components
    cfg=native_method(core,name)
    with open_dict(cfg):
        cfg.far=float(far);cfg.M=0;cfg.calibration_set=None;cfg.calibration_transform=None
        cfg.log_dir=None
        if 'tune_lambdas' in cfg:cfg.tune_lambdas=False
        if 'use_calibration' in cfg:cfg.use_calibration=False
        if gallery_kappa is not None:cfg.gallery_kappa=float(gallery_kappa)
    model=instantiate_config(cfg)
    model.setup(data.mu,data.kappa[:,None],data.gallery,data.gallery_kappa[:,None],
                g_unique_ids=data.gallery_ids,probe_unique_ids=data.probe_ids,dataset_name=data.source['dataset_name'])
    pred,reject=model.predict();actions=np.where(reject,0,np.asarray(pred,dtype=int)+1)
    known=np.asarray(model.mean_probs);p0=np.asarray(getattr(model,'oog_prob',1-known.sum(1))).reshape(-1)
    p=np.column_stack((p0,known));p=np.maximum(p,1e-300);p/=p.sum(1,keepdims=True)
    components=risk_components(np.log(p),actions)
    ns=np.asarray(getattr(model,'oog_nonspecificity',np.exp(log_nonspecificity(data.kappa,data.d)))).reshape(-1)
    out=dict(actions=actions,components=components,r_ns=(actions==0)*p[:,0]*ns,
             log_nonspecificity=log_nonspecificity(data.kappa,data.d),
             kl1=np.asarray(model.kl_1).reshape(-1),kl2=np.asarray(model.kl_2).reshape(-1),
             gallery_kappa=float(model.gallery_kappa),temperature=float(model.predict_T),gallery_prior=str(model.gallery_prior))
    del model
    try:
        import torch
        if torch.cuda.is_available():torch.cuda.empty_cache()
    except ImportError:pass
    return out


def synthetic_pair(seed=777,n=500,d=3):
    """Correctly specified observation channel; smoke data NEVER for dissertation tables."""
    from scipy.stats import vonmises_fisher
    rng=np.random.default_rng(seed);g=normalized(rng.normal(size=(4,d)))
    def make(offset):
        y=rng.choice(5,n,p=[.5,.125,.125,.125,.125]);k=np.exp(rng.uniform(np.log(.1),np.log(60),n));z=[]
        for yi,ki in zip(y,k):
            latent=normalized(rng.normal(size=(1,d)))[0] if yi==0 else np.asarray(vonmises_fisher(g[yi-1],20).rvs(random_state=rng)).reshape(-1)
            z.append(np.asarray(vonmises_fisher(latent,ki).rvs(random_state=rng)).reshape(-1))
        ids=np.where(y==0,-np.arange(1,n+1)-offset,y)
        return Data(np.array(z),k,g,np.full(4,20.),ids,np.arange(1,5),np.arange(n)+offset,
                    dict(dataset_name='synthetic',dataset_path=f'synthetic_{offset}',synthetic=True)).validate()
    return make(0),make(100000)


def synthetic_legacy(data,c,far,beta=.5,tau=None):
    from evaluation.open_set_methods.mprisk_evidence import risk_components
    scores=np.max(c,axis=1);unknown=scores[data.targets==0]
    if tau is None:
        nacc=int(far*len(unknown));tau=np.partition(unknown,len(unknown)-nacc)[len(unknown)-nacc] if nacc else np.nextafter(unknown.max(),np.inf)
    a=np.where(scores<tau,0,np.argmax(c,axis=1)+1)
    lp=log_posterior(20*np.asarray(c)-log_partition(20,data.d),beta)
    comp=risk_components(lp,a);ln=log_nonspecificity(data.kappa,data.d)
    return dict(actions=a,components=comp,r_ns=(a==0)*comp['p0']*np.exp(ln),log_nonspecificity=ln,
                kl1=-np.sum(np.exp(lp)*lp,axis=1),kl2=ln,gallery_kappa=20.,temperature=1.,gallery_prior='vMF',tau=tau)
