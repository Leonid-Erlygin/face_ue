#!/usr/bin/env python3
"""Repackage a stopped/incomplete run, preserving its actual status.

Use after a forced container kill prevented normal finalization. This never
marks incomplete results as successful. Do not run while the study is active.
"""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.mprisk_evidence.artifacts import archive,write_json
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('run_directory');p.add_argument('--stopped',action='store_true',help='Confirm no process is writing this directory')
a=p.parse_args();root=Path(a.run_directory).resolve()
if not a.stopped:raise SystemExit('Refusing concurrent packaging. Confirm the process stopped, then pass --stopped.')
if not (root/'manifest.json').is_file():raise SystemExit('No study manifest found')
s=json.loads((root/'manifest.json').read_text())
if s.get('status')=='running':
 s['status']='interrupted';s['failure']='Run stopped before normal finalization; manual recovery archive'
 write_json(root/'manifest.json',s)
(root/'.running.lock').unlink(missing_ok=True)
print(archive(root));print('Status:',s.get('status'))
