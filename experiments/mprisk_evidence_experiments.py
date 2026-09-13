#!/usr/bin/env python3
"""Two-stage evidence-based MPRisk study. Run --help for examples.

Every run has one results directory and one adjacent ZIP64 archive. Results from
synthetic smoke runs are marked and cannot authorize a real full run.
"""
from __future__ import annotations
import argparse,datetime,hashlib,importlib.metadata,json,os,platform,shutil,signal,subprocess,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from evaluation.open_set_methods.mprisk_evidence import MODEL_VERSION
from experiments.mprisk_evidence.artifacts import Tables,Tee,write_json,sha256,archive
from experiments.mprisk_evidence.audits import run_audits
from experiments.mprisk_evidence.study import run_dataset,cross_dataset_transfer


def code_hash():
    paths=[ROOT/'evaluation/open_set_methods/mprisk_evidence.py',Path(__file__),*sorted((ROOT/'experiments/mprisk_evidence').glob('*.py'))]
    paths += [ROOT/p for p in ['evaluation/open_set_methods/class_prob_models.py','evaluation/open_set_methods/kappa_utils.py','evaluation/open_set_methods/calibration_methods.py','evaluation/metrics.py','evaluation/test_datasets.py','evaluation/data_tools.py']]
    h=hashlib.sha256()
    for p in paths:h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
    return h.hexdigest(),paths


def parse():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage',choices=['sanity','full'],default='sanity')
    p.add_argument('--domain',choices=['text','bio'],default='text')
    p.add_argument('--run-dir',default=None)
    p.add_argument('--core-config',default=None,help='Defaults to the supplied mprisk_core_{domain}_complete.yaml')
    p.add_argument('--datasets',nargs='+',default=None)
    p.add_argument('--device',default='auto',help='auto, cpu, or cuda:0 for cosine matrix calculation')
    p.add_argument('--seed',type=int,default=777)
    p.add_argument('--beta',type=float,default=.5,help='Unknown class PRIOR, not target FPIR')
    p.add_argument('--sanity-val',type=int,default=6000)
    p.add_argument('--sanity-test',type=int,default=4000)
    p.add_argument('--max-fit',type=int,default=3000,help='Validation fit cap; 0 uses all fit rows')
    p.add_argument('--fit-iterations',type=int,default=250)
    p.add_argument('--search-budget',type=int,default=None)
    p.add_argument('--bootstrap',type=int,default=200)
    p.add_argument('--synthetic',action='store_true',help='Test pipeline only: generated data; never dissertation results')
    p.add_argument('--synthetic-n',type=int,default=500)
    p.add_argument('--approved-sanity',default=None,help='manifest.json of a completed, reviewed same-domain sanity run')
    p.add_argument('--confirm-full',action='store_true',help='Explicit human approval; no automatic performance-based promotion')
    p.add_argument('--previous-run',default=None,help='Previous completed same-domain sanity ZIP or run directory. Reuses exact subsets/splits; includes numerical-only replay.')
    p.add_argument('--replay-only',action='store_true',help='Do not refit or change reference decisions; report numerical-only replay, not approval for Stage B')
    p.add_argument('--reference-protocol',choices=['empirical','legacy'],default='empirical',help='Corrected sanity uses empirical benchmark thresholds; numerical replay always uses archived decisions.')
    p.add_argument('--review-resolution',default=None,help='Full only: JSON explicitly acknowledging the reviewed sanity warnings, with rationale and code hash')
    p.add_argument('--resume',action='store_true',help='Resume a run directory, retaining completed datasets only')
    args=p.parse_args()
    if args.search_budget is None:args.search_budget=128 if args.stage=='sanity' else 1024
    if not 0<args.beta<1:p.error('--beta must be in (0,1)')
    if min(args.fit_iterations,args.search_budget,args.bootstrap)<1:p.error('Iteration counts must be positive')
    if args.replay_only and not args.previous_run:p.error('--replay-only needs --previous-run')
    if args.previous_run and args.stage!='sanity':p.error('--previous-run belongs to sanity, not full')
    if args.reference_protocol=='legacy' and args.stage=='full':p.error('Full stage uses the reviewed empirical reference protocol')
    return args


def main():
    args=parse();stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run=Path(args.run_dir or ROOT/'outputs'/'mprisk_evidence'/f'{args.stage}_{args.domain}_{stamp}').resolve()
    if run.exists() and any(run.iterdir()) and not args.resume:raise SystemExit(f'{run} is not empty. Use a fresh directory or --resume.')
    run.mkdir(parents=True,exist_ok=True)
    lock=run/'.running.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    except FileExistsError:raise SystemExit(f'Run lock exists: {lock}. Check that no process is running before removing a stale lock.')
    old_stdout,old_stderr=sys.stdout,sys.stderr;log=open(run/'run.log','a',buffering=1)
    sys.stdout=Tee(old_stdout,log);sys.stderr=Tee(old_stderr,log)
    digest,paths=code_hash();tables=Tables(run);state=dict(status='running',stage=args.stage,domain=args.domain,synthetic=args.synthetic,
          model_version=MODEL_VERSION,code_sha256=digest,started_utc=stamp,arguments=vars(args),
          completed_datasets=[],decision_protocol='empirical cosine-threshold benchmark with explicit matching diagnostics' if args.reference_protocol=='empirical' else 'historical M=0 benchmark rule',
          fitting_protocol='validation fit/select/audit; no test labels used to fit evidence, probability calibration, or fusion',
          archive_excludes=['_cache: recomputable cosine matrices; raw image/audio/text inputs are not copied'])
    exit_code=0
    def stop(signum,frame):raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM,stop)
    try:
        if args.previous_run:
            from experiments.mprisk_evidence.recheck import PreviousRun
            origin=PreviousRun(args.previous_run,args.domain,args.seed,args.beta,args.synthetic)
            state['previous_run_fingerprint']=origin.fingerprint
            state['previous_run_manifest']=origin.manifest
            origin.close()
            if args.resume and json.loads((run/'manifest.json').read_text()).get('previous_run_fingerprint')!=state['previous_run_fingerprint']:
                raise ValueError('Previous-run input changed since interrupted run')
        if args.stage=='full' and not args.synthetic:
            if not args.confirm_full or not args.approved_sanity:raise ValueError('Full stage requires --confirm-full and --approved-sanity after human review.')
            approval=json.loads(Path(args.approved_sanity).read_text())
            if approval.get('status')!='complete' or approval.get('stage')!='sanity' or approval.get('synthetic') or approval.get('domain')!=args.domain:
                raise ValueError('Approval must refer to a successful REAL sanity run in the same domain')
            if approval.get('code_sha256')!=digest:raise ValueError('Code changed since the sanity run. Run sanity again or review the patch change explicitly.')
            if float(approval['arguments']['beta'])!=args.beta:raise ValueError('Prior beta differs from reviewed sanity. Review a sanity run with this prior first.')
            if approval.get('arguments',{}).get('replay_only'):raise ValueError('Numerical replay alone cannot authorize full experiments')
            if approval.get('arguments',{}).get('reference_protocol')!=args.reference_protocol:
                raise ValueError('Reference protocol differs from reviewed sanity')
            gate_path=Path(args.approved_sanity).parent/'review_gates.json'
            gates=json.loads(gate_path.read_text())
            if gates.get('blockers'):raise ValueError('Sanity has unresolved technical blockers: '+str(gates['blockers']))
            if gates.get('review_warnings'):
                if not args.review_resolution:raise ValueError('Sanity warnings require a reviewed --review-resolution; see review_gates.json')
                resolution=json.loads(Path(args.review_resolution).read_text())
                if resolution.get('sanity_code_sha256')!=digest or not str(resolution.get('rationale','')).strip():
                    raise ValueError('Review resolution needs the matching code hash and a scientific rationale')
                if set(resolution.get('acknowledged_warnings',[]))!=set(gates['review_warnings']):
                    raise ValueError('Review resolution must explicitly address exactly the listed warnings')
                state['review_resolution']=resolution
            state['reviewed_sanity']=str(Path(args.approved_sanity).resolve())
            state['reviewed_sanity_sha256']=sha256(args.approved_sanity)
        if args.resume:
            previous=json.loads((run/'manifest.json').read_text())
            if previous.get('code_sha256')!=digest or previous.get('stage')!=args.stage or previous.get('domain')!=args.domain:
                raise ValueError('Resume requires identical code, stage, and domain')
            relevant=['seed','beta','sanity_val','sanity_test','max_fit','fit_iterations','search_budget','bootstrap','synthetic','synthetic_n','datasets','previous_run','replay_only','reference_protocol']
            if any(previous['arguments'].get(k)!=vars(args).get(k) for k in relevant):raise ValueError('Resume settings changed')
            state['completed_datasets']=previous.get('completed_datasets',[]);tables.load()
        write_json(run/'manifest.json',state)
        snap=run/'reproducibility';snap.mkdir(exist_ok=True)
        write_json(snap/'arguments.json',vars(args))
        for document in (ROOT/'docs/mprisk_evidence').glob('*.md'):
            destination=snap/'protocol'/document.name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(document,destination)
        for path in paths:
            dest=snap/'source'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        versions={}
        for package in ['numpy','scipy','pandas','scikit-learn','torch','hydra-core','omegaconf','mpmath']:
            try:versions[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:versions[package]='missing'
        write_json(snap/'environment.json',dict(python=sys.version,platform=platform.platform(),packages=versions,
            argv=sys.argv,OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'),MKL_NUM_THREADS=os.environ.get('MKL_NUM_THREADS')))
        try:state['git_revision']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
        except subprocess.CalledProcessError:state['git_revision']=None
        np.random.seed(args.seed)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        try:
            import torch
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
            torch.use_deterministic_algorithms(True,warn_only=True)
            if hasattr(torch.backends,'cuda'):torch.backends.cuda.matmul.allow_tf32=False
        except ImportError:pass
        print('Running numerical checks before reading benchmark data.',flush=True)
        numeric=run_audits();tables.data['numerical_checks']=[];tables.add('numerical_checks',numeric);tables.save();state['numerical_checks_passed']=True
        if args.synthetic:core=None;names=['synthetic']
        else:
            from omegaconf import OmegaConf
            config=Path(args.core_config) if args.core_config else ROOT/'configs'/'uncertainty_benchmark'/f'mprisk_core_{args.domain}_complete.yaml'
            state['core_config_sha256']=sha256(config)
            if args.previous_run and state['previous_run_manifest'].get('core_config_sha256')!=state['core_config_sha256']:
                raise ValueError('Core configuration changed from previous run; resolve before claiming a same-protocol replay')
            if args.resume and previous.get('core_config_sha256')!=state['core_config_sha256']:
                raise ValueError('Source core configuration changed since the interrupted run')
            if args.stage=='full' and not args.synthetic and approval.get('core_config_sha256')!=state['core_config_sha256']:
                raise ValueError('Source core configuration differs from reviewed sanity')
            core=OmegaConf.load(config);core.exp_dir=str(run/'native_logs')
            if bool(core.get('use_two_galleries',False)):raise ValueError('This source configuration uses two galleries; supply the one-gallery dissertation configuration or extend adapter explicitly.')
            names=[str(d.dataset_name) for d in core.test_datasets]
            if args.datasets:
                if set(args.datasets)-set(names):raise ValueError('Requested dataset absent in source configuration')
                names=[n for n in names if n in args.datasets]
            OmegaConf.save(core,snap/'resolved_core_config.yaml',resolve=True)
            # Preserve the historical implementation used by benchmark baselines.
            for rel in ['evaluation/open_set_methods/class_prob_models.py','evaluation/open_set_methods/kappa_utils.py',
                        'evaluation/open_set_methods/calibration_methods.py','evaluation/metrics.py']:
                dest=snap/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
        if args.stage=='full' and not args.synthetic and not set(names)<=set(approval.get('completed_datasets',[])):
            raise ValueError('Full requests datasets absent from the reviewed sanity run')
        if args.previous_run and not set(names)<=set(state['previous_run_manifest'].get('completed_datasets',[])):
            raise ValueError('Requested dataset was not completed in previous sanity archive')
        state['datasets']=names;state['dataset_failures']={};write_json(run/'manifest.json',state)
        transfer={}
        for name in names:
            if name in state['completed_datasets']:
                print('Skipping completed dataset',name,flush=True);continue
            # Drop any partial table rows for this dataset before re-running it.
            for k in tables.data:tables.data[k]=[r for r in tables.data[k] if r.get('dataset')!=name]
            try:
                run_dataset(args,core,name,run,tables,transfer)
                state['completed_datasets'].append(name);write_json(run/'manifest.json',state)
            except Exception as exc:
                state['dataset_failures'][name]=str(exc)
                err=run/'errors';err.mkdir(exist_ok=True);(err/(name+'.txt')).write_text(traceback.format_exc())
                print(f'DATASET FAILED: {name}: {exc}',file=sys.stderr,flush=True)
                # Collect other domains/datasets rather than losing all diagnostic output.
        if args.stage=='full':
            # Reload lightweight transfer records for datasets skipped by --resume.
            from experiments.mprisk_evidence.study import reload_transfer
            transfer=reload_transfer(run,names)
            tables.data['cross_dataset_transfer']=[]
            cross_dataset_transfer(transfer,tables,args)
        if state['dataset_failures']:raise RuntimeError('One or more datasets failed; see errors and manifest, do not use this archive as complete results.')
        tables.save()
        from experiments.mprisk_evidence.plots import make_plots
        if not args.replay_only:make_plots(run)
        from experiments.mprisk_evidence.recheck import final_review_gates
        gates=final_review_gates(run,state,tables)
        if args.replay_only:
            gates['blockers'].append('replay_only:corrected_sanity_not_run');gates['status']='hold';write_json(run/'review_gates.json',gates)
        state['review_status']=gates['status']
        state['rank_score_semantics']='unsaturated log odds; error probabilities saved separately'
        state['input_corruption_or_new_validation_data_generated']=False
        state['status']='complete';print('ALL REQUESTED DATASETS COMPLETE',flush=True)
    except BaseException as exc:
        exit_code=1;state['status']='failed';state['failure']=str(exc);(run/'failure.txt').write_text(traceback.format_exc())
        print(traceback.format_exc(),file=sys.stderr,flush=True)
    finally:
        state['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();tables.save();write_json(run/'manifest.json',state)
        lock.unlink(missing_ok=True);sys.stdout.flush();sys.stderr.flush();sys.stdout,sys.stderr=old_stdout,old_stderr;log.close()
        z=archive(run);print(f'RESULT DIRECTORY: {run}\nRESULT ZIP: {z}\nSTATUS: {state["status"]}',flush=True)
    return exit_code


if __name__=='__main__':raise SystemExit(main())
