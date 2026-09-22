#!/usr/bin/env python3
"""Diagnostic preview from archived top-class probabilities, NOT a benchmark run.

Recovers the retained gallery cosines by inverting the old finite-q evidence.
For a different archived converged fit it bounds the contribution of all omitted
classes. Score-interval overlap then gives conservative PRR ordering envelopes.
Only unit costs are previewed. The real runner always uses the full gallery.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluation.open_set_methods.mprisk_evidence import (
    Parameters, PartitionTable, log_bayes_factors, selected_log_odds)
from experiments.mprisk_evidence.metrics import Metrics, area


def interval_prr(metric, lower, upper):
    """Enclose F1 areas of every ordering allowed by the supplied score intervals.

    Disjoint interval groups have a fixed relative order. Inside an overlapping
    connected group arbitrary permutations are allowed, a conservative relaxation.
    This is an omitted-tail bound, not certified floating-point interval arithmetic.
    """
    lower = np.asarray(lower, float); upper = np.asarray(upper, float)
    if lower.shape != (metric.n,) or upper.shape != lower.shape:
        raise ValueError('Misaligned score intervals')
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or np.any(lower > upper):
        raise ValueError('Invalid score intervals')
    groups = []; current = []; end = -np.inf
    for i in np.argsort(lower):
        if current and lower[i] > end:
            groups.append(np.asarray(current, int)); current = []
        current.append(i); end = max(end, upper[i])
    if current:
        groups.append(np.asarray(current, int))
    counts = np.asarray([[len(g), metric.m['tp'][g].sum(), metric.m['any_error'][g].sum()]
                         for g in groups], dtype=int)
    ns = np.r_[0, np.cumsum(counts[:, 0])]
    tp = np.r_[0, np.cumsum(counts[:, 1])]
    err = np.r_[0, np.cumsum(counts[:, 2])]
    curves = [[], []]
    def f(t, e):
        return float(2*t/(2*t+e)) if 2*t+e else 0.
    for keep in metric.keep:
        j = np.searchsorted(ns, keep, side='right') - 1
        if j == len(groups):
            low = high = f(tp[-1], err[-1])
        else:
            q = keep - ns[j]; n, tg, eg = counts[j]; tn = n-tg-eg
            # Pack errors before correct knowns for the lower bound, vice versa
            # for the upper; correct unknowns occupy the remaining positions.
            low = f(tp[j]+max(0, q-eg-tn), err[j]+min(q, eg))
            high = f(tp[j]+min(q, tg), err[j]+max(0, q-tg-tn))
        curves[0].append(low); curves[1].append(high)
    areas = [area(curve, metric.fracs) for curve in curves]
    den = metric.oracle_area-metric.random_area
    if abs(den) <= 1e-12:
        raise ValueError('Undefined PRR reference denominator')
    endpoints = [(value-metric.random_area)/den for value in areas]
    return dict(prr_lower=float(min(endpoints)), prr_upper=float(max(endpoints)),
                ambiguous_groups=sum(len(g)>1 for g in groups),
                largest_ambiguous_group=max(map(len, groups)))


def recover_top_cosines(saved, params, d, table):
    probs = np.asarray(saved['top_probabilities'], float)
    kappa = np.asarray(saved['kappa'], float)
    K = len(saved['gallery_ids']); beta = params.beta
    if params.point or np.any(probs <= 0) or not np.all(np.isfinite(probs)):
        raise ValueError('Preview needs positive stored top probabilities and a finite-q original model')
    log_ratio = np.log((1-beta)/(beta*K))
    evidence = np.log(probs)-saved['log_p0'][:, None]-log_ratio
    lo = np.full(probs.shape, -1.); hi = np.full(probs.shape, 1.)
    effective = params.probe_scale*kappa
    left = log_bayes_factors(lo, effective, params.gallery_kappa, d, table)
    right = log_bayes_factors(hi, effective, params.gallery_kappa, d, table)
    if np.any(evidence < left-1e-7) or np.any(evidence > right+1e-7):
        raise ValueError('Stored probabilities incompatible with the archived model')
    for _ in range(48):
        mid = (lo+hi)/2
        value = log_bayes_factors(mid, effective, params.gallery_kappa, d, table)
        lo = np.where(value < evidence, mid, lo)
        hi = np.where(value >= evidence, mid, hi)
    c = (lo+hi)/2
    if np.any(np.diff(c, axis=1) > 1e-9):
        raise ValueError('Stored top-class probabilities are not sorted')
    residual = np.max(np.abs(log_bayes_factors(c, effective, params.gallery_kappa, d, table)-evidence))
    return c, float(residual)


def unit_score_bounds(c, kappa, K, d, params, accepted, table):
    count = c.shape[1]
    if count > K or count < 1:
        raise ValueError('Invalid retained class count')
    b = log_bayes_factors(c, kappa*params.probe_scale, params.gallery_kappa, d, table)
    z = np.column_stack((np.zeros(len(c)), b+np.log((1-params.beta)/(params.beta*K))))
    # For every accepted row the fixed action must be the first retained class.
    local_actions = np.where(accepted, 1, 0)
    lower = selected_log_odds(z, local_actions)
    if count == K:
        return lower, lower.copy()
    tail_upper = z[:, -1] + np.log(K-count)
    upper = selected_log_odds(np.column_stack((z, tail_upper)), local_actions)
    return lower, upper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--candidate-index', type=int, default=0)
    parser.add_argument('--numerical-padding', type=float, default=1e-4)
    args = parser.parse_args()
    if args.numerical_padding < 0 or not np.isfinite(args.numerical_padding):
        parser.error('Padding must be finite and nonnegative')
    root = args.results.resolve()
    fit = json.loads((root/'probability_model_fit.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    data_meta = json.loads((root/'data_manifest.json').read_text())
    seed = int(manifest['arguments']['seed'])
    if fit.get('probability_selection') == 'validation-prr':
        parser.error('This preview is for the old common-model archive, not a new per-FPIR selection run')
    candidate = fit['fit']['candidates'][args.candidate_index]
    if not candidate.get('success') or not candidate.get('eligible', True):
        raise ValueError('Candidate did not converge')
    old = Parameters(**fit['parameters']); new = Parameters(**candidate['parameters'])
    if new.point or old.beta != new.beta:
        raise ValueError('Preview requires finite-q models with the same prior')
    cases = sorted(root.glob('fpir_*'), key=lambda p: float(p.name.split('_')[1]))
    rows = []; checks = []; cached = {}
    for split in ('validation', 'test'):
        d = int(data_meta[split]['d']); table = PartitionTable(d)
        first = dict(np.load(cases[0]/f'{split}.npz', allow_pickle=False))
        c, residual = recover_top_cosines(first, old, d, table)
        cached[split] = c
        for case in cases:
            saved = dict(np.load(case/f'{split}.npz', allow_pickle=False))
            for name in ('top_probabilities', 'top_classes', 'log_p0', 'kappa', 'gallery_ids'):
                if not np.array_equal(first[name], saved[name]):
                    raise ValueError('The archive does not use one common probability model across FPIRs')
            actions = saved['actions']; targets = saved['targets']; K = len(saved['gallery_ids'])
            accept = actions > 0
            if np.any(actions[accept] != saved['top_classes'][accept, 0]):
                raise ValueError('Fixed accepted action differs from first retained known class')
            names = list(saved['score_names']); old_score = saved['scores'][:, names.index('EviRisk')]
            # Independent max-similarity check using the saved threshold-distance score.
            tau = json.loads((case/'reference_decisions.json').read_text())[split]['tau']
            accscr = saved['scores'][:, names.index('AccScr')]
            expected_max = tau + np.where(accept, -accscr, accscr)
            max_error = float(np.max(np.abs(expected_max-c[:, 0])))
            if max_error > 1e-7:
                raise ValueError('Recovered max similarity fails the independent AccScr check')
            checks.append(dict(split=split, fpir=float(case.name.split('_')[1]),
                               log_evidence_inversion_max_residual=residual,
                               recovered_max_cosine_discrepancy=max_error))
            low, high = unit_score_bounds(c, saved['kappa'], K, d, new, accept, table)
            parts = {'test': np.arange(len(actions))} if split == 'test' else {
                name: np.flatnonzero(saved['role'] == name) for name in ('fit', 'select', 'audit')}
            for part, ix in parts.items():
                metric = Metrics(actions[ix], targets[ix], seed)
                bound = interval_prr(metric, low[ix]-args.numerical_padding, high[ix]+args.numerical_padding)
                rows.append(dict(split=part, fpir=float(case.name.split('_')[1]), n=len(ix),
                                 old_weighted_prr=metric.prr(old_score[ix]),
                                 candidate_unit_topk_prr=metric.prr(low[ix]), **bound,
                                 candidate_index=args.candidate_index,
                                 maximum_log_odds_interval_width=float(np.max(high[ix]-low[ix]))))
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out/'candidate_preview.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report = dict(scope='Retrospective top-k diagnostic, not a refit or a new full-matrix benchmark',
                  selected_candidate_index=args.candidate_index, candidate_parameters=candidate['parameters'],
                  old_parameters=fit['parameters'], numerical_padding=args.numerical_padding,
                  interpretation='Bounds account for omitted gallery classes, conditional on recovered cosines. '
                    'Padding is a numerical sensitivity allowance, not formally certified roundoff. '
                    'Final weighted EviRisk must be evaluated by the full-data runner.',
                  weighted_costs_fitted=False, recognition_decisions_unchanged=True,
                  metric_convention='Original random/oracle draws restored by the v12 patch',
                  checks=checks)
    (args.out/'preview_manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Saved diagnostic preview to', args.out)


if __name__ == '__main__':
    main()
