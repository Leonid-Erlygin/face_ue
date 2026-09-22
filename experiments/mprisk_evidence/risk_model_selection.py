"""Select a finite EviRisk fit AND its loss weights by validation PRR.

The NLL optimizer is left unchanged: it generates candidates. Its lowest-NLL
candidate must not silently become the final model of a PRR-optimized method.
Only validation selection rows enter this module's selection function. Audit
and test outcomes are not arguments. Point models and new score components are
not introduced. All eligible converged starts, including the incumbent, remain
in the search, and unit weights are evaluated for every candidate.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import numpy as np

from evaluation.open_set_methods.mprisk_evidence import Parameters, evaluate
from experiments.evirisk_precision import precise_cost_score, fit_precise_weights
from experiments.mprisk_evidence.metrics import Metrics, fit_risk_weights


def finite_candidates(fit_report: dict, beta: float):
    """Return every converged, eligible finite-q start, without cherry-picking."""
    output = []
    for i, row in enumerate(fit_report.get('candidates', [])):
        if not row.get('success', False) or not row.get('eligible', True):
            continue
        p = Parameters(**row['parameters'])
        if p.point:
            raise ValueError('Point fits are separate comparators, not EviRisk candidates')
        if not (np.isfinite(p.probe_scale) and p.probe_scale > 0 and
                np.isfinite(p.gallery_kappa) and p.gallery_kappa > 0 and
                np.isclose(p.beta, beta, rtol=0, atol=1e-14)):
            raise ValueError('Invalid or incompatible converged candidate')
        output.append((i, p))
    if not output:
        raise ValueError('No eligible converged finite-q candidate')
    incumbent = fit_report.get('selected_index')
    if incumbent is not None and incumbent not in [i for i, _ in output]:
        raise ValueError('The NLL incumbent is not in the converged candidate pool')
    return output


def select_evidence_model(cosines, kappa, d, targets, actions, select_indices,
                          fit_report, beta, table=None, budget=1024, seed=777,
                          legacy_weight_tuning=False):
    """Joint finite-model/three-cost search on ONE declared validation subset.

    Probability parameters were fitted on the disjoint `fit_indices`. Costs
    and model choice use select_indices; their PRR is an optimization objective,
    not an unbiased quality estimate. The separate audit split remains unused.
    """
    ix = np.asarray(select_indices, dtype=int)
    if ix.ndim != 1 or len(ix) < 4 or len(np.unique(ix)) != len(ix):
        raise ValueError('Need at least four unique validation selection rows')
    if np.any(ix < 0) or np.any(ix >= len(kappa)):
        raise ValueError('Selection row outside validation arrays')
    if np.intersect1d(ix, np.asarray(fit_report.get('fit_indices', []), int)).size:
        raise ValueError('Likelihood-fit and ranking-selection rows must be disjoint')
    y = np.asarray(targets, dtype=int)[ix]
    a = np.asarray(actions, dtype=int)[ix]
    metric = Metrics(a, y, seed)
    if metric.oracle_area - metric.random_area <= 1e-12:
        raise ValueError('Selection PRR has no positive reference denominator')
    fitter = fit_risk_weights if legacy_weight_tuning else fit_precise_weights
    rows, best = [], None
    for i, pars in finite_candidates(fit_report, beta):
        # Evaluate only allowed rows; audit predictions are not consulted here.
        result = evaluate(np.asarray(cosines)[ix], np.asarray(kappa)[ix], d,
                          pars, a, y, table)
        events = result['log_event_probabilities']
        fitted = fitter(events, a, y, budget, seed)
        score = precise_cost_score(events, fitted['raw_weights'])
        objective = float(metric.prr(score))
        unit = float(metric.prr(result['score_log_odds']))
        # Defensive inclusion of exact unit costs even in the legacy fast fitter.
        if unit > objective:
            fitted = dict(fitted, raw_weights=[1., 1., 1.],
                          validation_prr=unit, unit_fallback=True)
            objective = unit
        if not np.isfinite(objective):
            raise FloatingPointError('Non-finite candidate selection PRR')
        row = dict(candidate_index=i, parameters=asdict(pars),
                   class_select_nll=float(-result['true_log_probability'].mean()),
                   unit_validation_prr=unit, weighted_validation_prr=objective,
                   weight_fit=fitted, selected_by_nll=(i == fit_report.get('selected_index')),
                   at_fit_boundary=bool(fit_report['candidates'][i].get('at_boundary', False)))
        rows.append(row)
        print(f'[risk model selection] candidate={i}; alpha={pars.probe_scale:.7g}; '
              f'kappa_g={pars.gallery_kappa:.7g}; NLL={row["class_select_nll"]:.6g}; '
              f'unit PRR={unit:.6g}; weighted PRR={objective:.6g}', flush=True)
        # Deterministic tie breaking; NLL matters only when the PRR is identical.
        key = (objective, -row['class_select_nll'], -i)
        if best is None or key > best[0]:
            best = (key, pars, fitted, row)
    report = dict(schema='evirisk-finite-prr-selection-v13',
                  objective='validation selection PRR after three nonnegative cost fitting',
                  selected_index=best[3]['candidate_index'],
                  selected_parameters=asdict(best[1]), weight_fit=best[2],
                  nll_selected_index=fit_report.get('selected_index'),
                  candidates=rows, selection_indices=ix.tolist(), seed=int(seed),
                  weight_search_budget_per_candidate=int(budget),
                  candidate_count=len(rows), uses_audit_outcomes=False,
                  uses_test_outcomes=False, point_fallback=False,
                  prediction_formula_changed=False,
                  note='Selection PRR is in-sample for model/cost selection, not an unbiased estimate.')
    incumbent = [r for r in rows if r['selected_by_nll']]
    if incumbent and best[3]['weighted_validation_prr'] < incumbent[0]['weighted_validation_prr'] - 1e-12:
        raise AssertionError('Nested candidate search lost the NLL incumbent')
    return best[1], best[2], report


def reuse_probability_fit(previous_run, val, test, parts, args):
    """Reuse fitted candidates only when the archived input/split contract matches."""
    root = Path(previous_run).resolve()
    if not root.is_dir():
        raise ValueError('--reuse-probability-fit expects an extracted run directory')
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('status') != 'complete':
        raise ValueError('Only completed fit archives can be reused')
    if bool(manifest.get('synthetic', False)) != bool(args.synthetic):
        raise ValueError('Cannot mix synthetic and real run provenance')
    if int(manifest['arguments']['seed']) != args.seed:
        raise ValueError('Fit archive seed differs from this run')
    old_data = json.loads((root / 'data_manifest.json').read_text())
    fields = ['n', 'K', 'd', 'representations_sha256', 'concentrations_sha256',
              'gallery_sha256', 'templates_sha256']
    for label, data in [('validation', val), ('test', test)]:
        current = data.metadata()
        for key in fields:
            if old_data[label].get(key) != current[key]:
                raise ValueError(f'Fit reuse blocked: {label} {key} mismatch')
    old_split = json.loads((root / 'validation_split.json').read_text())
    for name, ids in parts.items():
        if not np.array_equal(np.asarray(old_split[name], int), ids):
            raise ValueError('Fit reuse blocked: validation '+name+' split mismatch')
    saved = json.loads((root / 'probability_model_fit.json').read_text())
    pars, point = Parameters(**saved['parameters']), Parameters(**saved['point_parameters'])
    for name, p in [('finite', pars), ('point', point)]:
        if not np.isclose(p.beta, args.beta, rtol=0, atol=1e-14):
            raise ValueError(name+' model has different beta')
    finite_candidates(saved['fit'], args.beta)
    if not saved['point_fit'].get('selected_converged', False) or not point.point:
        raise ValueError('Archived point comparator is not a converged point fit')
    return pars, saved['fit'], point, saved['point_fit'], str(root)


def fit_native_component_weights(components, actions, targets, budget=1024, seed=777,
                                 include=None):
    """Fresh nonnegative fusion of original MPRisk components, not evidence ones.

    No standardization. Includes exact zero-weight ablations. NS is a penalty,
    not a disjoint error probability, and is therefore NEVER passed as a fifth
    event to cost_log_odds. Its score is computed as the original raw sum.
    """
    x = np.asarray(components, dtype=float)
    if x.ndim != 2 or x.shape[1] not in (3, 4) or np.any(~np.isfinite(x)) or np.any(x < 0):
        raise ValueError('Expected N by 3/4 nonnegative native risk components')
    d = x.shape[1]; metric = Metrics(actions, targets, seed)
    candidates = [np.ones(d)]
    candidates += [np.array([(mask >> j) & 1 for j in range(d)], float)
                   for mask in range(1, 2**d)]
    candidates += [np.asarray(w, float) for w in (include or [])]
    candidates += list(np.exp(np.random.default_rng(seed).uniform(-10., 10., (budget, d))))
    best = None
    for w in candidates:
        w = w / w.max(); value = float(metric.prr(x @ w))
        if np.isfinite(value) and (best is None or value > best[0]):
            best = (value, w.copy())
    if best is None:
        raise ValueError('Native component PRR undefined on validation selection rows')
    return dict(raw_weights=best[1].tolist(), validation_prr=best[0],
                objective='validation selection PRR', budget=int(budget),
                component_count=d, normalization='max weight=1; no feature standardization',
                paper_comparator_retuned=True)
