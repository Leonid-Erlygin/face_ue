#!/usr/bin/env python3
"""Run the PRR-selected EviRisk protocol with vMF GalUE on nine datasets / 41 points.

Apply the incremental patch from /app. Then:
  python experiments/evirisk_full_suite.py --check-inputs
  python experiments/evirisk_full_suite.py --out /app/outputs/evirisk_full_prr
No encoder training; uses the original SCF exports and supplied domain configs.
"""
from __future__ import annotations
import argparse, datetime, importlib, json, os, shutil, signal, subprocess, sys, traceback
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiments.evirisk_suite import PROTOCOL_VERSION
from experiments.evirisk_suite.protocol import (
    make_plan, plan_contract, code_files, assert_input_signatures, METHODS, DOMAINS, FPIRS)
from experiments.mprisk_evidence.artifacts import write_json, archive, sha256, Tables, Tee
from experiments.mprisk_evidence.metrics import Metrics


def parse():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out',default=str(ROOT/'outputs/evirisk_full_prr'))
    p.add_argument('--bio-config',default=str(ROOT/'configs/uncertainty_benchmark/mprisk_core_bio_complete.yaml'))
    p.add_argument('--text-config',default=str(ROOT/'configs/uncertainty_benchmark/mprisk_core_text_complete.yaml'))
    p.add_argument('--paths-json',help='Optional per-dataset test_root/validation_root overrides; no code editing needed.')
    p.add_argument('--datasets',nargs='+',help='Explicit subset only; default is all nine. Subsets are labelled partial scope.')
    p.add_argument('--seeds',nargs='+',type=int,default=[777],help='All listed seeds retained, never choose best test seed.')
    p.add_argument('--device',default='cpu',help='cpu, auto, cuda:0: cosine construction only; evidence is float64 CPU.')
    p.add_argument('--beta',type=float,default=.5)
    p.add_argument('--max-fit',type=int,default=3000,help='Cap probability-fitting rows only; zero removes cap. No test subsampling.')
    p.add_argument('--fit-iterations',type=int,default=250)
    p.add_argument('--search-budget',type=int,default=1024,help='Random weight vectors PER converged probability candidate.')
    p.add_argument('--bootstrap',type=int,default=200)
    p.add_argument('--validation-repeats',type=int,default=3)
    p.add_argument('--gallery-kappa-max',type=float,default=1e5)
    p.add_argument('--probe-scale-max',type=float,default=100.)
    p.add_argument('--supplementary',choices=['full','core'],default='full',help='full adds size/refit/sensitivity studies at predeclared FPIR=.1.')
    p.add_argument('--tie-policy',choices=['seeded-permutation','legacy'],default='seeded-permutation')
    p.add_argument('--resume',action='store_true',help='Skip verified completed dataset jobs; restart failed dataset jobs, not partial tables.')
    p.add_argument('--resume-audit-fix',action='store_true',
        help='With --resume only: verify and record the exact MS1M audit/warning patch; no other code/config changes allowed.')
    p.add_argument('--keep-cache',action='store_true',help='Keep recomputable disk-backed cosine matrices after successful jobs.')
    p.add_argument('--check-inputs',action='store_true',help='Resolve paths, dependency targets and immutable plan; no fitting.')
    p.add_argument('--smoke',action='store_true',help='Two small generated datasets; clearly synthetic, never dissertation evidence.')
    p.add_argument('--synthetic-n',type=int,default=500)
    p.add_argument('--_worker',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.resume_audit_fix and not a.resume: p.error('--resume-audit-fix requires --resume')
    if not 0<a.beta<1: p.error('beta must be in (0,1)')
    if min(a.fit_iterations,a.search_budget,a.validation_repeats,a.bootstrap)<1: p.error('All budgets must be positive')
    if a.max_fit<0 or a.gallery_kappa_max<=.1 or a.probe_scale_max<=1e-4: p.error('Invalid fitting limits')
    if len(set(a.seeds))!=len(a.seeds): p.error('Duplicate seeds')
    if not a.seeds or any(s<0 or s>2**32-1 for s in a.seeds): p.error('Seeds must be unsigned 32-bit integers')
    if a.synthetic_n<100: p.error('--synthetic-n must be >=100 for supported validation splits')
    if a.smoke and a.datasets: p.error('--smoke uses its own synthetic dataset names')
    if a.smoke:
        # Explicit debug-only budgets. Stored in the synthetic manifest.
        a.fit_iterations=min(a.fit_iterations,35);a.search_budget=min(a.search_budget,8)
        a.bootstrap=min(a.bootstrap,8);a.validation_repeats=min(a.validation_repeats,1)
        a.max_fit=min(a.max_fit or 120,120)
    return a


def dependency_check(jobs):
    failures=[]
    for module in ['numpy','scipy','pandas','sklearn','torch','omegaconf','mpmath']:
        try: importlib.import_module(module)
        except Exception as exc: failures.append(f'{module}: {exc}')
    if not all(j['synthetic'] for j in jobs):
        # Import the actual original targets before beginning the long suite.
        targets=set()
        def collect(obj):
            if isinstance(obj,dict):
                if '_target_' in obj: targets.add(obj['_target_'])
                for x in obj.values(): collect(x)
            elif isinstance(obj,list):
                for x in obj: collect(x)
        for j in jobs:
            for m in j['core']['open_set_identification_methods']:
                if m['pretty_name'] in ['MPRisk raw','HolUE']: collect(m['recognition_method'])
        for t in sorted(targets):
            try:
                mod,name=t.rsplit('.',1);getattr(importlib.import_module(mod),name)
            except Exception as exc: failures.append(f'{t}: {exc}')
    return failures


def job_key(j): return f"seed_{j['seed']}/{j['dataset']}"


def complete_inventory(root):
    return {str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha256(p))
            for p in sorted(root.rglob('*')) if p.is_file() and '_cache' not in p.parts
            and p.name not in ['completion_inventory.json'] and not p.name.endswith('.tmp')}


def verify_completed(root):
    status=json.loads((root/'manifest.json').read_text())
    if status.get('status')!='complete': return False
    inventory=json.loads((root/'completion_inventory.json').read_text())
    for name,row in inventory.items():
        p=root/name
        if not p.is_file() or p.stat().st_size!=row['bytes'] or sha256(p)!=row['sha256']:
            raise ValueError('Completed result changed: '+str(p))
    return True


def worker(specfile):
    spec=json.loads(Path(specfile).read_text());args=SimpleNamespace(**spec['worker_args']);job=spec['job']
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    state=dict(status='running',protocol=PROTOCOL_VERSION,synthetic=args.synthetic,
               arguments=vars(args),warnings=[],dataset=args.dataset,
               started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    print(f"[job start] dataset={args.dataset} seed={args.seed} report_dir={out}", flush=True)
    Metrics.default_tie_policy=args.tie_policy
    tables=Tables(out);oldout,olderr=sys.stdout,sys.stderr;log=(out/'run.log').open('w',buffering=1)
    sys.stdout=Tee(oldout,log);sys.stderr=Tee(olderr,log);code=0
    try:
        assert_input_signatures(job)
        write_json(out/'manifest.json',state)
        write_json(out/'method_registry.json',METHODS)
        from experiments.evirisk_suite.worker import run
        run(args,out,tables,state)
        assert_input_signatures(job)
    except BaseException as exc:
        print(f"[job failed] dataset={args.dataset} seed={args.seed} report_dir={out}: {exc}", file=sys.stderr, flush=True)
        code=1;state.update(status='failed',failure=str(exc))
        (out/'failure.txt').write_text(traceback.format_exc());traceback.print_exc()
    finally:
        state['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        tables.save();write_json(out/'manifest.json',state)
        sys.stdout.flush();sys.stderr.flush();sys.stdout,sys.stderr=oldout,olderr;log.close()
    if code==0:
        write_json(out/'completion_inventory.json',complete_inventory(out))
        if not args.keep_cache and (out/'_cache').exists(): shutil.rmtree(out/'_cache')
    return code


def package_and_summarize(root,jobs,completed):
    from experiments.evirisk_suite.collect import collect
    return collect(root,jobs,completed)


def main():
    args=parse()
    if args._worker: return worker(args._worker)
    root=Path(args.out).resolve()
    jobs=make_plan(args)
    missing=sorted({p for j in jobs for p in j.get('missing',[])})
    deps=dependency_check(jobs)
    print('Frozen protocol:',PROTOCOL_VERSION)
    for j in jobs:
        print(f"  {j['dataset']} | seed={j['seed']} | FPIR={j['fpirs']} | {len(j.get('missing',[]))} missing files")
    if missing or deps:
        print('\nINPUT/DEPENDENCY CHECK FAILED. No experiments started.',file=sys.stderr)
        for s in missing+deps: print('  '+s,file=sys.stderr)
        return 2
    Metrics.default_tie_policy=args.tie_policy
    contract=plan_contract(args,jobs)
    if args.check_inputs:
        print(json.dumps(dict(status='static input checks passed',jobs=len(jobs),
            operating_points=sum(len(j['fpirs']) for j in jobs),
            note='Existence/import check only; row alignment, numerical validation and provenance checks run in each worker.'),indent=2))
        return 0
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise ValueError('Output is not empty. Use a fresh --out or --resume: '+str(root))
    root.mkdir(parents=True,exist_ok=True)
    migration=None
    if args.resume:
        old=json.loads((root/'protocol.json').read_text())
        if old!=contract:
            if not args.resume_audit_fix:
                raise ValueError('Resume blocked: code/config/settings/input signatures/library versions differ. '
                                 'For the exact MS1M audit fix only, add --resume-audit-fix; otherwise use a fresh output directory.')
            from experiments.evirisk_suite.audit_fix_resume import validate_migration
            migration=validate_migration(root,old,contract,ROOT)

    lock=root/'.suite.lock'
    try:
        fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL);os.write(fd,str(os.getpid()).encode());os.close(fd)
    except FileExistsError:
        raise ValueError('Suite lock exists. Check for an active process before removing stale lock: '+str(lock))
    state=dict(status='running',protocol=PROTOCOL_VERSION,synthetic=args.smoke,
        args=vars(args),completed_jobs=[],failed_jobs=[],warnings=[],expected_jobs=len(jobs),
        expected_operating_points=sum(len(j['fpirs']) for j in jobs),
        scope='nine dissertation datasets' if not args.datasets and not args.smoke else 'explicit subset or synthetic smoke',
        source_code_sha256=contract['code_sha256'],started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    proc=None;exitcode=0
    def stop(signum,frame): raise KeyboardInterrupt('Received signal '+str(signum))
    oldterm=signal.signal(signal.SIGTERM,stop)
    try:
        if migration is not None:
            from experiments.evirisk_suite.audit_fix_resume import record_migration
            history=record_migration(root,old,contract,migration)
            state['audit_fix_migration']=str(history.relative_to(root))
            print('[resume audit fix] verified numerical protocol unchanged; history: '+str(history),flush=True)
        write_json(root/'protocol.json',contract);write_json(root/'manifest.json',state)
        write_json(root/'method_registry.json',METHODS)
        for p in code_files():
            dest=root/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        doc=ROOT/'experiments/evirisk_suite/README.md'
        if doc.is_file(): shutil.copy2(doc,root/'RUN_PROTOCOL.md')
        from experiments.mprisk_evidence.audits import run_audits
        math_audits=run_audits()
        write_json(root/'numerical_preflight.json',dict(checks=math_audits,all_passed=all(x['passed'] for x in math_audits)))
        for j in jobs:
            key=job_key(j);dest=root/'runs'/key
            if args.resume and dest.is_dir() and (dest/'completion_inventory.json').is_file() and verify_completed(dest):
                print('[resume] verified '+key,flush=True);state['completed_jobs'].append(key);continue
            # Never append duplicate rows from a failed partial dataset. Preserve
            # its outputs but exclude old attempts from final combined tables.
            if dest.exists():
                previous=root/'_cache'/'previous_attempts'/key/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
                previous.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(dest),str(previous))
            w=vars(args).copy();w.update(out=str(dest),dataset=j['dataset'],seed=j['seed'],
                fpirs=j['fpirs'],synthetic=j['synthetic'],resolved_core=j['core'],
                probability_selection='validation-prr',reuse_probability_fit=None,
                legacy_weight_tuning=False,include_holue=not j['synthetic'],include_mprisk=not j['synthetic'])
            spec=root/'jobs'/(key.replace('/','__')+'.json');write_json(spec,dict(job=j,worker_args=w))
            dest.parent.mkdir(parents=True,exist_ok=True)
            env=os.environ.copy();env['PYTHONHASHSEED']=str(j['seed']);env['MPLBACKEND']='Agg';env['PYTHONUNBUFFERED']='1'
            # Spawn fresh processes: native models cannot leak state between datasets.
            print(f'[job launch] {key} | output={dest}',flush=True)
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--_worker',str(spec)],cwd=ROOT,env=env)
            returncode=proc.wait();proc=None
            if returncode==0 and verify_completed(dest): state['completed_jobs'].append(key)
            else: state['failed_jobs'].append(dict(job=key,exit_code=returncode));exitcode=1
            write_json(root/'manifest.json',state)
        summary=package_and_summarize(root,jobs,state['completed_jobs'])
        state['summary']=summary
        if state['failed_jobs'] or not summary['coverage_complete']:
            state['status']='incomplete';exitcode=1
        elif summary.get('supplementary_failures'):
            state['status']='complete_with_diagnostic_failures'
        else: state['status']='complete'
    except BaseException as exc:
        if proc is not None:
            proc.terminate()
            try: proc.wait(timeout=20)
            except subprocess.TimeoutExpired: proc.kill();proc.wait()
        exitcode=1;state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',failure=str(exc))
        (root/'failure.txt').write_text(traceback.format_exc());traceback.print_exc()
    finally:
        state['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(root/'manifest.json',state)
        lock.unlink(missing_ok=True);signal.signal(signal.SIGTERM,oldterm)
        z=archive(root)
        print(f"\nSTATUS: {state['status']}\nRESULT ZIP: {z}\nReturn the whole ZIP, including manifests and per-example records.")
    return exitcode


if __name__=='__main__':
    try: raise SystemExit(main())
    except (ValueError,FileNotFoundError) as exc: raise SystemExit(str(exc))
