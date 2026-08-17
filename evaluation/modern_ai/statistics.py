from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _metric(y: np.ndarray, score: np.ndarray, name: str) -> float:
    y = np.asarray(y, dtype=int).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    ok = np.isfinite(score)
    y, score = y[ok], score[ok]
    if len(np.unique(y)) < 2:
        return np.nan
    if name == "auroc":
        return float(roc_auc_score(y, score))
    if name == "auprc":
        return float(average_precision_score(y, score))
    if name == "aurc":
        order = np.argsort(score)
        yy = y[order]
        coverage = np.arange(1, len(yy) + 1, dtype=np.float64) / len(yy)
        risk = np.cumsum(yy) / np.arange(1, len(yy) + 1)
        return float(np.trapezoid(risk, coverage)) if len(yy) > 1 else float(risk[0])
    raise ValueError("metric must be auroc, auprc, or aurc")


def paired_group_bootstrap_difference(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    group_ids: Sequence[str],
    metric: str = "auroc",
    n_boot: int = 2000,
    seed: int = 777,
) -> Dict[str, float]:
    """Paired cluster bootstrap for comparing two uncertainty methods.

    Positive ``difference`` means A is larger than B.  For AURC, lower is
    better, so callers should interpret the sign accordingly.  Clustering is
    essential for RAGTruth because several model responses share a source.
    """

    y = np.asarray(y, dtype=int).reshape(-1)
    a = np.asarray(score_a, dtype=np.float64).reshape(-1)
    b = np.asarray(score_b, dtype=np.float64).reshape(-1)
    groups = np.asarray(list(map(str, group_ids)), dtype=object)
    if not (len(y) == len(a) == len(b) == len(groups)):
        raise ValueError("Inputs must have equal length.")
    unique = np.unique(groups)
    point_a = _metric(y, a, metric)
    point_b = _metric(y, b, metric)
    point = point_a - point_b
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(int(n_boot)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in sampled])
        ma = _metric(y[idx], a[idx], metric)
        mb = _metric(y[idx], b[idx], metric)
        if np.isfinite(ma) and np.isfinite(mb):
            diffs.append(ma - mb)
    if not diffs:
        return {
            "metric_a": point_a,
            "metric_b": point_b,
            "difference": point,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_two_sided": np.nan,
        }
    arr = np.asarray(diffs, dtype=np.float64)
    # Bootstrap sign probability is reported as a descriptive paired p-value.
    p = 2.0 * min(float(np.mean(arr <= 0.0)), float(np.mean(arr >= 0.0)))
    return {
        "metric_a": point_a,
        "metric_b": point_b,
        "difference": point,
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "p_two_sided": float(min(1.0, p)),
    }


def stratified_group_split(
    labels: np.ndarray,
    group_ids: Sequence[str],
    validation_fraction: float = 0.3,
    seed: int = 777,
    search_trials: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Leakage-safe group split chosen to preserve response-level label balance.

    Group-majority stratification can fail when every source contains a mix of
    correct and incorrect responses (common in RAG evaluation).  We therefore
    search whole-group partitions and choose the feasible split whose validation
    and test prevalence best match the global prevalence.
    """
    y = np.asarray(labels, dtype=int).reshape(-1)
    groups = np.asarray(list(map(str, group_ids)), dtype=object)
    if len(y) != len(groups):
        raise ValueError("labels and group_ids must have equal length")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("At least two groups are required")
    n_val_groups = int(round(len(unique) * float(validation_fraction)))
    n_val_groups = min(max(n_val_groups, 1), len(unique) - 1)
    rng = np.random.default_rng(seed)
    global_rate = float(np.mean(y))
    best = None
    best_obj = np.inf
    # Deterministic RNG with finite search; 512 is cheap even for thousands of groups.
    for _ in range(max(int(search_trials), 1)):
        val_groups = rng.choice(unique, size=n_val_groups, replace=False)
        val_mask = np.isin(groups, val_groups)
        test_mask = ~val_mask
        if not np.any(val_mask) or not np.any(test_mask):
            continue
        # Prefer splits on which discrimination metrics are identifiable.
        feasible = len(np.unique(y[val_mask])) >= 2 and len(np.unique(y[test_mask])) >= 2
        balance = abs(float(np.mean(y[val_mask])) - global_rate) + abs(float(np.mean(y[test_mask])) - global_rate)
        size_penalty = abs(float(np.mean(val_mask)) - float(validation_fraction))
        obj = balance + 0.25 * size_penalty + (0.0 if feasible else 100.0)
        if obj < best_obj:
            best_obj = obj
            best = (np.flatnonzero(val_mask), np.flatnonzero(test_mask))
    if best is None:
        raise ValueError("Could not construct a group-disjoint split")
    return best
