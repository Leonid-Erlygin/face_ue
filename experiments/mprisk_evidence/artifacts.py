"""Atomic reports and ZIP64 archives. Scratch cosine caches are not shared."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,zipfile,sys
import numpy as np
import pandas as pd


def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    if isinstance(v,np.bool_):return bool(v)
    if isinstance(v,Path):return str(v)
    return v


def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def archive(root):
    root=Path(root).resolve();target=Path(str(root)+'.zip');tmp=target.with_suffix('.zip.tmp')
    files=[p for p in sorted(root.rglob('*')) if p.is_file() and '_cache' not in p.relative_to(root).parts and not p.name.endswith('.tmp')]
    inventory={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in files if p.name!='file_inventory.json'}
    write_json(root/'file_inventory.json',inventory)
    files=[p for p in sorted(root.rglob('*')) if p.is_file() and '_cache' not in p.relative_to(root).parts and not p.name.endswith('.tmp')]
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
        for p in files:z.write(p,arcname=str(Path(root.name)/p.relative_to(root)))
    with zipfile.ZipFile(tmp) as z:
        bad=z.testzip()
        if bad:raise IOError('ZIP integrity error: '+bad)
    os.replace(tmp,target);target.with_suffix('.zip.sha256').write_text(sha256(target)+'  '+target.name+'\n')
    return target


class Tables:
    def __init__(self,root):self.root=Path(root);self.data={}
    def add(self,name,rows,**context):
        if isinstance(rows,dict):rows=[rows]
        self.data.setdefault(name,[]).extend([dict(context,**r) for r in rows])
    def save(self):
        d=self.root/'tables';d.mkdir(parents=True,exist_ok=True)
        for name,rows in self.data.items():
            path=d/(name+'.csv');tmp=path.with_suffix('.csv.tmp');pd.DataFrame(rows).to_csv(tmp,index=False);os.replace(tmp,path)
    def load(self):
        for p in (self.root/'tables').glob('*.csv'):
            try:self.data[p.stem]=pd.read_csv(p).to_dict('records')
            except pd.errors.EmptyDataError:self.data[p.stem]=[]


class Tee:
    def __init__(self,stream,file):self.stream=stream;self.file=file
    def write(self,s):self.stream.write(s);self.file.write(s);self.file.flush();return len(s)
    def flush(self):self.stream.flush();self.file.flush()
    def isatty(self):return False
