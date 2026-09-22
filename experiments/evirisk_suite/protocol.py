"""Configuration and input contracts; no model fitting or test-score selection."""
from __future__ import annotations
from pathlib import Path
import hashlib, importlib.metadata, json
from omegaconf import OmegaConf
from experiments.evirisk_suite import PROTOCOL_VERSION

ROOT = Path(__file__).resolve().parents[2]
DOMAINS = {
    'bio': ('IJBC', 'IJBB', 'whale', 'large_12k-perspk5'),
    'text': ('yahoo', 'dbpedia', 'agnews', 'clinc150', 'pan_test'),
}
PRETTY = dict(IJBC='IJB-C', IJBB='IJB-B', whale='Whale',
              **{'large_12k-perspk5': 'VoxBlink'}, yahoo='Yahoo Answers',
              dbpedia='DBPedia', agnews='AG News', clinc150='CLINC150', pan_test='PAN-20-AV')
FPIRS = {'bio': [.01, .05, .1, .2], 'text': [.1, .2, .3, .4, .5]}
METHODS = {
    'EviRisk': 'Finite-q likelihood-fit candidate AND 3 nonnegative weights selected by validation PRR; ordinal score, not probability.',
    'EviRisk-1': 'Unit-weight ablation using the SAME final probability model; conditional error-probability output.',
    'EviRisk NLL-selected control': 'Lowest validation multiclass-NLL candidate; 3 weights refitted on the common selection rows.',
    'GalUE': 'Point-representation GalUE with vMF class densities; gallery concentration is fitted by validation multiclass NLL; uncertainty is 1-max posterior probability over known and unknown classes.',
    'MPRisk reference (retuned 3)': 'Original M=0 FA/ID/FR probabilities; native density and temperature from supplied domain config; retuned on selection.',
    'MPRisk reference (retuned 4)': 'Same MPRisk posterior plus original NS; retuned raw sum including exact 3-component candidate.',
    'HolUE': 'Original HolUE features and calibrator from supplied domain config; trained on errors of the same fixed predictions.',
    'Lin-4': 'SCF, AccScr, MSP, Margin; standardization and weights on selection only.',
    'Lin-All': 'Explicit baseline registry in fusion_fit.json; nested Lin-4/KL models and exact standalone candidates.',
    'Lin-All+E': 'Baseline registry plus EviRisk unit-logit, weighted-loss logit, and 3 event probabilities; exact EviRisk candidate included.',
    'EviRisk-Cal': 'Positive-slope binary calibration of weighted-loss LOGIT, never batch-dependent ordinal ranks; original ranking retained.',
}


def versions():
    output = {}
    for name in ['numpy', 'scipy', 'scikit-learn', 'pandas', 'torch', 'omegaconf', 'mpmath']:
        try: output[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: output[name] = 'not installed'
    return output


def code_files(root=ROOT):
    # Include actual native implementations, not only the new driver.
    files = set()
    for rel in ['experiments/evirisk_suite', 'experiments/mprisk_evidence',
                'evaluation', 'utils']:
        files.update((root / rel).rglob('*.py'))
    for rel in ['experiments/evirisk_full_suite.py', 'experiments/evirisk_precision.py',
                'experiments/evirisk_whale_diagnostics.py', 'evaluation/metrics.py',
                'evaluation/test_datasets.py', 'evaluation/data_tools.py',
                'evaluation/template_pooling_strategies.py']:
        p = root / rel
        if p.is_file(): files.add(p)
    return sorted(files)


def code_digest(root=ROOT):
    h = hashlib.sha256()
    for p in code_files(root):
        h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def required_inputs(cfg):
    path = Path(cfg['dataset_path']); name = str(cfg['dataset_name']); stem = name.lower()
    return [path / 'embeddings' / f'scf_embs_{name}.npz',
            path / 'meta' / f'{stem}_face_tid_mid.txt',
            path / 'meta' / f'{stem}_1N_probe_mixed.csv',
            path / 'meta' / f'{stem}_1N_gallery_G1.csv']


def file_signature(path):
    p = Path(path).resolve(); s = p.stat()
    return dict(path=str(p), bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def make_plan(args):
    """Resolve all datasets now, so missing inputs cannot be silently skipped."""
    if args.smoke:
        names = ['synthetic-bio', 'synthetic-text']
        return [dict(dataset=n, pretty_name=n, domain=d, core=None,
                     fpirs=[.1], seed=seed, synthetic=True, inputs=[])
                for seed in args.seeds for n, d in zip(names, ['bio', 'text'])]
    requested = set(args.datasets or [n for names in DOMAINS.values() for n in names])
    unknown = requested - set(PRETTY)
    if unknown: raise ValueError('Unknown dataset identifiers: '+', '.join(sorted(unknown)))
    overrides = json.loads(Path(args.paths_json).read_text()) if args.paths_json else {}
    if set(overrides) - set(PRETTY): raise ValueError('Unknown keys in --paths-json')
    jobs = []
    for domain, names in DOMAINS.items():
        config_path = Path(getattr(args, domain+'_config')).resolve()
        cfg = OmegaConf.load(config_path)
        if bool(cfg.get('use_two_galleries', False)):
            raise ValueError('This frozen protocol uses gallery G1 only; use_two_galleries must be false: '+str(config_path))
        cfg.exp_dir = str(Path(args.out).resolve() / '_cache' / 'native_config')
        # Explicit protocol grid: four bio/audio and five text points, not a
        # dataset-dependent selection based on results. Source configuration kept.
        for name in names:
            if name not in requested: continue
            c = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            test = next((x for x in c.test_datasets if str(x.dataset_name) == name), None)
            if test is None: raise ValueError(f'{name} missing from {config_path}')
            val = c.dataset_name_to_calibration_set[name]
            override = overrides.get(name, {})
            if set(override) - {'test_root', 'validation_root'}: raise ValueError('Unknown path override for '+name)
            if override.get('test_root'): test.dataset_path = override['test_root']
            if override.get('validation_root'): val.dataset_path = override['validation_root']
            if Path(test.dataset_path).resolve() == Path(val.dataset_path).resolve():
                raise ValueError(name+': validation and test roots must differ')
            if float(c.dataset_name_to_T_scale[name]) <= 0: raise ValueError('Invalid native temperature')
            for method in c.open_set_identification_methods:
                if 'beta' in method.recognition_method: method.recognition_method.beta = args.beta
            native_names = {str(m.pretty_name) for m in c.open_set_identification_methods}
            if not {'HolUE', 'MPRisk raw'} <= native_names: raise ValueError('Missing mandatory native reference configs')
            inputs = required_inputs(val)+required_inputs(test)
            for ds in [val,test]:
                extra=Path(ds.dataset_path)/'meta'/f'{str(ds.dataset_name).lower()}_name_5pts_score.txt'
                if extra.is_file(): inputs.append(extra)
            inputs=sorted(set(inputs))
            missing = [str(p) for p in inputs if not p.is_file()]
            for seed in args.seeds:
                jobs.append(dict(dataset=name, pretty_name=PRETTY[name], domain=domain,
                    core=OmegaConf.to_container(c, resolve=True), fpirs=FPIRS[domain], seed=seed,
                    synthetic=False, config_source=str(config_path),
                    source_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    inputs=[file_signature(p) for p in inputs if p.is_file()], missing=missing))
    return jobs


def assert_input_signatures(job):
    for old in job['inputs']:
        if file_signature(old['path']) != old:
            raise ValueError('Input changed after plan was frozen: '+old['path'])


def plan_contract(args, jobs):
    fields = ['beta','max_fit','fit_iterations','search_budget','bootstrap','seeds',
              'datasets','smoke','synthetic_n','supplementary','validation_repeats',
              'gallery_kappa_max','probe_scale_max','device','tie_policy']
    return dict(protocol=PROTOCOL_VERSION, code_sha256=code_digest(), versions=versions(),
                settings={k:getattr(args,k) for k in fields}, jobs=jobs)
