import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]


def invoke(arguments):
    env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg')
    return subprocess.run([sys.executable,str(ROOT/'experiments/mprisk_evidence_experiments.py'),*arguments],
                         cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)


def test_full_without_approval_is_blocked_and_packaged(tmp_path):
    root=tmp_path/'unapproved'
    p=invoke(['--stage','full','--domain','text','--run-dir',str(root)])
    assert p.returncode!=0
    m=json.loads((root/'manifest.json').read_text())
    assert m['status']=='failed'
    assert 'confirm-full' in m['failure']
    with zipfile.ZipFile(str(root)+'.zip') as z:assert z.testzip() is None


def test_synthetic_smoke_produces_aligned_outputs(tmp_path):
    root=tmp_path/'smoke'
    command=['--synthetic','--synthetic-n','100','--device','cpu','--fit-iterations','3','--search-budget','4','--run-dir',str(root)]
    p=invoke(command)
    assert p.returncode==0,p.stdout+p.stderr
    m=json.loads((root/'manifest.json').read_text())
    assert m['status']=='complete' and m['synthetic'] is True
    import numpy as np
    data=np.load(root/'datasets/synthetic/fpir_0.1/per_example/test.npz',allow_pickle=False)
    assert len(data['actions'])==data['scores'].shape[0]==100
    np.testing.assert_allclose(data['r_fa']+data['r_id']+data['r_fr'],data['risk'],atol=1e-9)
    assert 'MPRisk' in data['score_names']
    with zipfile.ZipFile(str(root)+'.zip') as z:
        assert z.testzip() is None
        assert any('/per_example/test.npz' in n for n in z.namelist())
        assert not any('/_cache/' in n for n in z.namelist())
    # Resume must not duplicate rows or rerun completed datasets.
    import pandas as pd
    before=pd.read_csv(root/'tables/main_mprisk_core_comparison.csv').shape
    resumed=invoke(command+['--resume'])
    assert resumed.returncode==0,resumed.stdout+resumed.stderr
    assert pd.read_csv(root/'tables/main_mprisk_core_comparison.csv').shape==before
