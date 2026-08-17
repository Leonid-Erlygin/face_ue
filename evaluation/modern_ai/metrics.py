from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .types import RetrievalProtocol, RetrievalScores


def retrieval_error_masks(protocol: RetrievalProtocol, scores: RetrievalScores) -> Dict[str, np.ndarray]:
    if tuple(protocol.query_ids) != tuple(scores.query_ids):
        raise ValueError("Protocol and score query order differ.")
    predicted = scores.predicted_doc_ids
    rejected = np.asarray(scores.was_rejected, dtype=bool)
    known = np.asarray(protocol.known_mask, dtype=bool)
    relevant = protocol.relevant_doc_ids
    correct_retrieval = np.asarray(
        [predicted[i] in set(relevant[i]) if known[i] else False for i in range(len(predicted))],
        dtype=bool,
    )
    false_accept = (~known) & (~rejected)
    false_reject = known & rejected
    misretrieval = known & (~rejected) & (~correct_retrieval)
    true_accept = known & (~rejected) & correct_retrieval
    true_reject = (~known) & rejected
    correct = true_accept | true_reject
    return {
        "known": known,
        "correct_retrieval": correct_retrieval,
        "false_accept": false_accept,
        "false_reject": false_reject,
        "misretrieval": misretrieval,
        "true_accept": true_accept,
        "true_reject": true_reject,
        "correct": correct,
        "any_error": ~correct,
    }


def open_set_retrieval_metrics(protocol: RetrievalProtocol, scores: RetrievalScores) -> Dict[str, float]:
    m = retrieval_error_masks(protocol, scores)
    known = m["known"]
    unknown = ~known
    tp = int(np.sum(m["true_accept"]))
    fp = int(np.sum(m["false_accept"]))
    fn = int(np.sum(m["false_reject"]) + np.sum(m["misretrieval"]))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "oser_accuracy": float(np.mean(m["correct"])),
        "oser_f1": float(f1),
        "fpir": float(np.mean(m["false_accept"][unknown])) if np.any(unknown) else np.nan,
        "fnir": float(1.0 - np.mean(m["true_accept"][known])) if np.any(known) else np.nan,
        "false_accept_rate_all": float(np.mean(m["false_accept"])),
        "false_reject_rate_all": float(np.mean(m["false_reject"])),
        "misretrieval_rate_all": float(np.mean(m["misretrieval"])),
        "coverage_accept": float(np.mean(~scores.was_rejected)),
        "num_queries": float(len(known)),
        "num_known": float(np.sum(known)),
        "num_unknown": float(np.sum(unknown)),
    }


def _safe_auc(y: np.ndarray, s: np.ndarray, kind: str) -> float:
    y = np.asarray(y, dtype=int).reshape(-1)
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if len(np.unique(y)) < 2:
        return np.nan
    if kind == "roc":
        return float(roc_auc_score(y, s))
    return float(average_precision_score(y, s))


def expected_calibration_error(y_error: np.ndarray, p_error: np.ndarray, bins: int = 15) -> float:
    y = np.asarray(y_error, dtype=np.float64).reshape(-1)
    p = np.clip(np.asarray(p_error, dtype=np.float64).reshape(-1), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    ece = 0.0
    for i in range(len(edges) - 1):
        if i == len(edges) - 2:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        if np.any(mask):
            ece += np.mean(mask) * abs(np.mean(p[mask]) - np.mean(y[mask]))
    return float(ece)


def risk_coverage_curve(y_error: np.ndarray, uncertainty: np.ndarray) -> Dict[str, np.ndarray | float]:
    y = np.asarray(y_error, dtype=np.float64).reshape(-1)
    u = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    order = np.argsort(u)  # keep least uncertain first
    y_sorted = y[order]
    n = len(y)
    coverages = np.arange(1, n + 1, dtype=np.float64) / max(n, 1)
    risks = np.cumsum(y_sorted) / np.arange(1, n + 1)
    aurc = float(np.trapezoid(risks, coverages)) if n > 1 else float(risks[0])
    return {"coverage": coverages, "risk": risks, "aurc": aurc}


def uncertainty_detection_metrics(
    masks: Mapping[str, np.ndarray],
    uncertainty: np.ndarray,
) -> Dict[str, float]:
    out = {}
    for label in ["any_error", "false_accept", "false_reject", "misretrieval"]:
        y = np.asarray(masks[label], dtype=int)
        out[f"{label}_auroc"] = _safe_auc(y, uncertainty, "roc")
        out[f"{label}_auprc"] = _safe_auc(y, uncertainty, "pr")
    rc = risk_coverage_curve(masks["any_error"], uncertainty)
    out["aurc"] = float(rc["aurc"])
    return out


def calibration_metrics(y_error: np.ndarray, p_error: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_error, dtype=int).reshape(-1)
    p = np.clip(np.asarray(p_error, dtype=np.float64).reshape(-1), 1e-8, 1.0 - 1e-8)
    return {
        "brier": float(brier_score_loss(y, p)),
        "nll": float(log_loss(y, p, labels=[0, 1])),
        "ece": expected_calibration_error(y, p),
    }


def ranking_metrics(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    protocol: RetrievalProtocol,
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    """Cosine retrieval metrics on known queries (full ranking, not OSR rejection)."""
    sims = np.asarray(query_embeddings) @ np.asarray(corpus_embeddings).T
    corpus_ids = np.asarray(list(map(str, protocol.corpus.keys())), dtype=object)
    known_idx = np.where(protocol.known_mask)[0]
    if not len(known_idx):
        return {f"recall@{k}": np.nan for k in ks}
    order = np.argsort(-sims[known_idx], axis=1)
    out = {}
    for k in ks:
        hits = []
        for local_i, qi in enumerate(known_idx):
            rel = set(protocol.relevant_doc_ids[qi])
            top = set(corpus_ids[order[local_i, : min(k, len(corpus_ids))]].tolist())
            hits.append(len(rel & top) > 0)
        out[f"recall@{k}"] = float(np.mean(hits))
    rr = []
    for local_i, qi in enumerate(known_idx):
        rel = set(protocol.relevant_doc_ids[qi])
        rank = 0.0
        for j, didx in enumerate(order[local_i], start=1):
            if corpus_ids[didx] in rel:
                rank = 1.0 / j
                break
        rr.append(rank)
    out["mrr"] = float(np.mean(rr))
    return out


def grouped_bootstrap_metric(
    y: np.ndarray,
    score: np.ndarray,
    group_ids: Sequence[str],
    metric: str = "auroc",
    n_boot: int = 1000,
    seed: int = 777,
) -> Dict[str, float]:
    """Cluster bootstrap, needed for repeated RAGTruth responses per source_id."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=np.float64)
    groups = np.asarray(list(map(str, group_ids)), dtype=object)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)

    def fn(yy, ss):
        return _safe_auc(yy, ss, "roc" if metric == "auroc" else "pr")

    point = fn(y, score)
    vals = []
    for _ in range(int(n_boot)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sampled])
        val = fn(y[idx], score[idx])
        if np.isfinite(val):
            vals.append(val)
    if not vals:
        return {"point": point, "ci_low": np.nan, "ci_high": np.nan}
    return {
        "point": point,
        "ci_low": float(np.quantile(vals, 0.025)),
        "ci_high": float(np.quantile(vals, 0.975)),
    }
