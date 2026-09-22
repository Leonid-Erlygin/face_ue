"""Aggregate completed jobs only. Never choose a seed/model by test performance."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from experiments.mprisk_evidence.metrics import Metrics
from experiments.evirisk_precision import precise_cost_score
from experiments.mprisk_evidence.artifacts import Tables, write_json
from experiments.evirisk_suite.protocol import PRETTY


def collect(root,jobs,completed):
    root=Path(root);done=set(completed);alltables={};failures=[];coverage=[]
    for job in jobs:
        key=f"seed_{job['seed']}/{job['dataset']}";p=root/'runs'/key
        if key not in done: continue
        for src in (p/'tables').glob('*.csv'):
            try: frame=pd.read_csv(src)
            except pd.errors.EmptyDataError: continue
            frame['dataset']=job['dataset'];frame['seed']=job['seed'];frame['domain']=job['domain']
            frame['synthetic']=job['synthetic'];alltables.setdefault(src.stem,[]).append(frame)
        for f in job['fpirs']:
            case=p/f'fpir_{f:g}'
            status=json.loads((case/'status.json').read_text())
            for message in status.get('extras',{}).get('supplementary_failures',[]):
                failures.append(dict(job=key,fpir=f,message=message))
            coverage.append(dict(dataset=job['dataset'],seed=job['seed'],fpir=f,status=status['status']))
    target=root/'tables';target.mkdir(exist_ok=True)
    for name,frames in alltables.items():
        pd.concat(frames,ignore_index=True).to_csv(target/(name+'.csv'),index=False)
    expected=sum(len(j['fpirs']) for j in jobs)
    complete=len(coverage)==expected and all(r['status']=='complete' for r in coverage)
    if 'main_comparison' in alltables:
        main=pd.concat(alltables['main_comparison'],ignore_index=True)
        test=main[main.split=='test'].copy()
        dup=test.duplicated(['dataset','seed','target_fpir','method'])
        if dup.any(): raise ValueError('Duplicate final test rows; refusing ambiguous aggregation')
        primary=test[test.method=='EviRisk']
        complete=complete and len(primary)==expected and primary.prr_f1.notna().all()
        test.to_csv(target/'all_test_metrics.csv',index=False)
        test.pivot(index=['dataset','seed','target_fpir'],columns='method',values='prr_f1').to_csv(target/'test_prr_wide.csv')
        # FPIRs are correlated settings, not independent training replicates.
        avg=test.groupby(['dataset','seed','method'])['prr_f1'].agg(['mean','count']).reset_index()
        avg.rename(columns={'mean':'mean_over_operating_points','count':'operating_points'}).to_csv(target/'mean_prr_by_dataset_seed.csv',index=False)
        comparisons=[]
        for method in ['HolUE','MPRisk reference (retuned 3)','MPRisk reference (retuned 4)','Lin-All+E','EviRisk NLL-selected control']:
            pair=primary.merge(test[test.method==method],on=['dataset','seed','target_fpir'],suffixes=('_e','_other'))
            for _,r in pair.iterrows():
                delta=r.prr_f1_e-r.prr_f1_other
                comparisons.append(dict(dataset=r.dataset,seed=int(r.seed),fpir=r.target_fpir,comparison=method,
                    evirisk=r.prr_f1_e,baseline=r.prr_f1_other,difference=delta,
                    relative_prr_percent=100*delta/r.prr_f1_other if r.prr_f1_other>0 else np.nan,
                    note='Relative PRR change is not recognition-accuracy improvement'))
        pd.DataFrame(comparisons).to_csv(target/'paired_point_estimates.csv',index=False)
        generate_latex(test,root)
    transfer=Tables(root/'transfer')
    # Weights transfer only; destination posterior remains locally fitted.
    for source in jobs:
        skey=f"seed_{source['seed']}/{source['dataset']}"
        if skey not in done: continue
        for destination in jobs:
            dkey=f"seed_{destination['seed']}/{destination['dataset']}"
            if dkey not in done or source['dataset']==destination['dataset'] or source['seed']!=destination['seed']: continue
            for f in sorted(set(source['fpirs'])&set(destination['fpirs'])):
                s=root/'runs'/skey/f'fpir_{f:g}';d=root/'runs'/dkey/f'fpir_{f:g}'
                costs=json.loads((s/'weights.json').read_text())['EviRisk']['raw_weights']
                with np.load(d/'test.npz',allow_pickle=False) as a:
                    score=precise_cost_score(a['log_event_probabilities'],costs)
                    transfer.add('cross_dataset_weights',Metrics(a['actions'],a['targets'],source['seed']).evaluate(score),
                        source_dataset=source['dataset'],target_dataset=destination['dataset'],seed=source['seed'],fpir=f,
                        source_weights=costs,diagnostic_only=True,transferred='weights only; target posterior locally selected')
    transfer.save()
    rows=transfer.data.get('cross_dataset_weights',[])
    if rows: pd.DataFrame(rows).to_csv(target/'cross_dataset_weights.csv',index=False)
    report=dict(coverage_complete=bool(complete),expected_points=expected,completed_points=len(coverage),
        completed_jobs=len(completed),supplementary_failures=failures,coverage=coverage,
        estimator_split_and_fitting_seeds=sorted(set(j['seed'] for j in jobs)),
        encoder_retrained=False,
        fpir_points_are_independent_repeats=False,selected_best_test_seed=False,
        holdout_note='Protocol was revised after retrospective Whale debugging; no untouched-holdout claim for Whale.')
    write_json(root/'coverage.json',report)
    (root/'RESULTS_README.md').write_text('''# EviRisk full-suite results

Start with manifest.json and coverage.json. `complete` means every requested
point is present; explicit dataset subsets are not the complete nine-dataset scope.

Primary: EviRisk (finite probability candidate + three validation-trained weights).
EviRisk-1 is an ablation, not a replacement selected after inspecting test scores.

`tables/test_prr_wide.csv`: all test PRRs, including unrounded values.
`tables/paired_point_estimates.csv`: EviRisk minus comparator (bootstrap tables
retain the opposite, explicitly named comparator-minus-EviRisk orientation).
`tables/bootstrap.csv`: conditional paired bootstrap, not repeated training.
`tables/probability_calibration.csv`: calibration separate from ranking.
`tables/component_ablation.csv`: unit and selected-cost component ablations.
`tables/validation_size_weights.csv`: conditional cost-size study at FPIR .1.
`tables/probability_fit_size.csv`: regenerated likelihood candidates + PRR selection.
`tables/parameter_sensitivity.csv`: fixed-weights diagnostics, not model selection.
`tables/cross_dataset_weights.csv`: transfer of weights only.
`runs/seed_*/DATASET/fpir_*/`: exact selected models, costs, calibration/fusion
fits, indices, scores, labels and probability diagnostics for replay.

The benchmark uses unknown test labels to set target-FPIR thresholds, as in the
source protocol. These labels are not used for probability-model/cost fitting.
This is not a prospective deployment-threshold guarantee. SCF exports without
sample IDs retain the source pipeline's row-order assumption. Audit estimates
for small-class datasets are conditional on those represented classes.

No test subsampling, no best-test-seed selection, no stale table mixing.
Scratch cosine matrices and incomplete old attempts are excluded from the ZIP.
''')
    return report


def generate_latex(test,root):
    out=Path(root)/'latex';out.mkdir(exist_ok=True)
    methods=['EviRisk','EviRisk-1','HolUE','GalUE','MPRisk reference (retuned 3)','MPRisk reference (retuned 4)','Lin-4','Lin-All','Lin-All+E','EviRisk NLL-selected control']
    def esc(s): return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
    for seed,block in test.groupby('seed'):
        lines=[r'% Generated from this run only. One seed; FPIR rows are not replicates.',
            r'\begin{longtable}{llrr}',r'Набор & Метод & FPIR & PRR \\',r'\hline']
        for (dataset,fpir),part in block.groupby(['dataset','target_fpir'],sort=False):
            for method in methods:
                row=part[part.method==method]
                if row.empty: continue
                v=float(row.iloc[0].prr_f1)
                lines.append(f"{esc(PRETTY.get(dataset,dataset))} & {esc(method)} & {fpir:g} & {v:.6f} "+r'\\')
        lines.append(r'\end{longtable}')
        (out/f'prr_seed_{int(seed)}.tex').write_text('\n'.join(lines)+'\n')
