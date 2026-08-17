from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from .calibration import ErrorCalibratorBundle, binary_nll
from .embedders import BaseTextEmbedder
from .methods import ModernUncertaintyModel, PosteriorModelConfig
from .metrics import calibration_metrics, risk_coverage_curve
from .query_uncertainty import query_embeddings_and_kappa
from .types import ToolRoutingExample


def _stratified_indices(labels: np.ndarray, fraction: float, seed: int):
    y = np.asarray(labels, dtype=bool)
    rng = np.random.default_rng(seed)
    cal, test = [], []
    for cls in (False, True):
        idx = np.flatnonzero(y == cls)
        idx = rng.permutation(idx)
        if len(idx) <= 1:
            n = len(idx)
        else:
            n = min(max(int(round(len(idx) * fraction)), 1), len(idx) - 1)
        cal.extend(idx[:n]); test.extend(idx[n:])
    if not test:
        raise ValueError("Too few tool-routing examples for calibration/test split")
    return np.asarray(sorted(cal), dtype=int), np.asarray(sorted(test), dtype=int)


def _encode_examples(
    examples: Sequence[ToolRoutingExample],
    embedder: BaseTextEmbedder,
    rewrites: Optional[Mapping[str, Sequence[str]]],
    default_query_kappa: float,
    query_mean_mode: str,
):
    ids = [x.example_id for x in examples]
    queries = [x.query for x in examples]
    q, kappa, rbar = query_embeddings_and_kappa(
        ids, queries, embedder, rewrites=rewrites,
        default_kappa=default_query_kappa, mean_mode=query_mean_mode,
    )
    galleries = [embedder.encode_documents(x.tools) for x in examples]
    return q, kappa, rbar, galleries


def fit_variable_gallery_kappa(
    query_embeddings: np.ndarray,
    query_kappa: np.ndarray,
    galleries: Sequence[np.ndarray],
    known_mask: np.ndarray,
    model_config: PosteriorModelConfig,
    *,
    grid_size: int = 32,
) -> tuple[float, dict]:
    """Calibrate one concentration across variable-size function galleries.

    BFCL's candidate set changes per example, so the fixed-K boundary equation is
    inapplicable. We instead use a validation-only empirical operating-point fit:
    first minimize absolute FPIR error, then unknown-posterior NLL. This keeps the
    target FPIR semantics of the dissertation while respecting varying K.
    """
    if model_config.gallery_kappa is not None:
        return float(model_config.gallery_kappa), {"strategy": "fixed", "candidates": []}
    known = np.asarray(known_mask, dtype=bool)
    if not np.any(~known) or not np.any(known):
        raise ValueError("Variable-gallery kappa fitting needs known and irrelevant examples")
    candidates = np.exp(np.linspace(
        np.log(model_config.kappa_low), np.log(model_config.kappa_high), int(grid_size)
    ))
    records = []
    y_unknown = (~known).astype(int)
    for k in candidates:
        cfg = replace(model_config, gallery_kappa=float(k))
        model = ModernUncertaintyModel(cfg)
        s = model.score_variable_galleries(query_embeddings, query_kappa, galleries)
        rejected = s["rejected"].astype(bool)
        fpir = float(np.mean(~rejected[~known]))
        nll = binary_nll(y_unknown, s["unknown_probability"])
        records.append((float(k), fpir, nll))
    errors = np.asarray([abs(x[1] - model_config.target_fpir) for x in records])
    best = np.flatnonzero(errors <= np.min(errors) + 1e-12)
    idx = min(best.tolist(), key=lambda i: records[i][2])
    return records[idx][0], {
        "strategy": "variable_gallery_empirical_fpir_then_nll",
        "target_fpir": float(model_config.target_fpir),
        "chosen_fpir": records[idx][1],
        "chosen_nll": records[idx][2],
        "candidates": [
            {"kappa": k, "fpir": fpir, "unknown_nll": nll}
            for k, fpir, nll in records
        ],
    }


def _decision_labels(examples: Sequence[ToolRoutingExample], scores: Mapping[str, np.ndarray]):
    known = np.asarray([x.known for x in examples], dtype=bool)
    rejected = np.asarray(scores["rejected"], dtype=bool)
    pred = np.asarray(scores["predicted_index"], dtype=int)
    correct = ((~known) & rejected) | (known & ~rejected)
    selection_evaluable = np.asarray([
        bool(x.known and len(x.relevant_tool_indices)) for x in examples
    ], dtype=bool)
    for i in np.flatnonzero(selection_evaluable & (~rejected)):
        correct[i] = pred[i] in set(examples[i].relevant_tool_indices)
    return known, rejected, pred, correct, selection_evaluable


def _auc(y, s, kind="roc"):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y=np.asarray(y,dtype=int); s=np.asarray(s,dtype=float)
    ok=np.isfinite(s); y=y[ok]; s=s[ok]
    if len(np.unique(y))<2: return np.nan
    return float(roc_auc_score(y,s) if kind=="roc" else average_precision_score(y,s))


def run_tool_routing_experiment(
    examples: Sequence[ToolRoutingExample],
    embedder: BaseTextEmbedder,
    model_config: PosteriorModelConfig,
    *,
    rewrites: Optional[Mapping[str, Sequence[str]]] = None,
    default_query_kappa: float = 300.0,
    query_mean_mode: str = "original",
    calibration_fraction: float = 0.3,
    kappa_grid_size: int = 32,
    seed: int = 777,
    output_dir: Optional[str | Path] = None,
) -> dict:
    """Open-set function/tool routing experiment (BFCL relevance compatible)."""
    examples = list(examples)
    known = np.asarray([x.known for x in examples], dtype=bool)
    cal_idx, test_idx = _stratified_indices(known, calibration_fraction, seed)
    q, kappa, rbar, galleries = _encode_examples(
        examples, embedder, rewrites, default_query_kappa, query_mean_mode
    )
    fitted_kappa, kappa_meta = fit_variable_gallery_kappa(
        q[cal_idx], kappa[cal_idx], [galleries[i] for i in cal_idx], known[cal_idx],
        model_config, grid_size=kappa_grid_size,
    )
    cfg = replace(model_config, gallery_kappa=float(fitted_kappa))
    model = ModernUncertaintyModel(cfg)
    cal_s = model.score_variable_galleries(
        q[cal_idx], kappa[cal_idx], [galleries[i] for i in cal_idx], rbar[cal_idx]
    )
    test_s = model.score_variable_galleries(
        q[test_idx], kappa[test_idx], [galleries[i] for i in test_idx], rbar[test_idx]
    )
    cal_examples = [examples[i] for i in cal_idx]
    test_examples = [examples[i] for i in test_idx]
    _, _, _, cal_correct, _ = _decision_labels(cal_examples, cal_s)
    t_known, t_rej, t_pred, t_correct, selection_eval = _decision_labels(test_examples, test_s)
    y_cal = (~cal_correct).astype(int)
    y_test = (~t_correct).astype(int)

    scalar_cal = {
        k: v for k, v in cal_s.items()
        if k not in {"holue_kl_1", "holue_kl_2", "holue_kl_sum", "rejected", "predicted_index"}
    }
    bundle = ErrorCalibratorBundle.fit(
        scalar_cal, cal_s["holue_kl_1"], cal_s["holue_kl_2"], y_cal,
        exclude=("holue", "unknown_probability"),
    )
    calibrated = bundle.apply(
        {k:v for k,v in test_s.items() if k not in {"holue_kl_1","holue_kl_2","holue_kl_sum","rejected","predicted_index"}},
        test_s["holue_kl_1"], test_s["holue_kl_2"],
    )
    metrics = {
        "call_reject_accuracy": float(np.mean(((t_known & ~t_rej) | (~t_known & t_rej)))),
        "task_accuracy_with_tool_id_when_available": float(np.mean(t_correct)),
        "fpir_irrelevant_accepted": float(np.mean(~t_rej[~t_known])) if np.any(~t_known) else np.nan,
        "fnir_relevant_rejected": float(np.mean(t_rej[t_known])) if np.any(t_known) else np.nan,
        "num_test": int(len(test_idx)),
        "num_tool_id_evaluable": int(np.sum(selection_eval)),
        "irrelevance_auroc_p_unknown": _auc((~t_known).astype(int), test_s["unknown_probability"]),
        "irrelevance_auprc_p_unknown": _auc((~t_known).astype(int), test_s["unknown_probability"], "pr"),
    }
    if np.any(selection_eval):
        metrics["tool_id_accuracy_conditional_evaluable"] = float(np.mean(t_correct[selection_eval]))

    uncertainty = {}
    for name, raw in test_s.items():
        if name in {"rejected","predicted_index","holue_kl_1","holue_kl_2","holue_kl_sum"}: continue
        rc = risk_coverage_curve(y_test, raw)
        uncertainty[name] = {"error_auroc": _auc(y_test, raw), "error_auprc": _auc(y_test, raw,"pr"), "aurc": float(rc["aurc"])}
    if "holue" in calibrated:
        rc = risk_coverage_curve(y_test, calibrated["holue"])
        uncertainty["holue"] = {"error_auroc": _auc(y_test, calibrated["holue"]), "error_auprc": _auc(y_test, calibrated["holue"],"pr"), "aurc": float(rc["aurc"])}
    cal_metrics = {}
    for name,p in calibrated.items():
        ok=np.isfinite(p)
        if np.any(ok): cal_metrics[name]=calibration_metrics(y_test[ok],np.asarray(p)[ok])

    summary = {
        "experiment": "open_set_tool_routing",
        "embedder": getattr(embedder,"model_name",type(embedder).__name__),
        "fitted_gallery_kappa": float(fitted_kappa),
        "kappa_calibration": kappa_meta,
        "posterior_config": asdict(cfg),
        "split": {"seed":seed,"num_calibration":int(len(cal_idx)),"num_test":int(len(test_idx))},
        "decision_metrics": metrics,
        "uncertainty_detection": uncertainty,
        "calibration": cal_metrics,
        "bfcl_semantics": (
            "BFCL relevance/irrelevance files evaluate call-vs-reject only. "
            "Tool-ID accuracy is reported only when explicit relevant_tool_indices are supplied."
        ),
    }
    rows=[]
    for j,i in enumerate(test_idx):
        x=examples[i]
        row={
            "id":x.example_id,"known":bool(x.known),"num_tools":len(x.tools),
            "rejected":bool(t_rej[j]),"predicted_index":int(t_pred[j]),"correct":bool(t_correct[j]),
        }
        for k,v in test_s.items(): row[f"raw__{k}"]=float(v[j])
        for k,v in calibrated.items(): row[f"p_error__{k}"]=float(v[j])
        rows.append(row)
    if output_dir is not None:
        root=Path(output_dir); root.mkdir(parents=True,exist_ok=True)
        (root/"summary.json").write_text(json.dumps(summary,indent=2,default=float),encoding="utf-8")
        if rows:
            with (root/"per_example.csv").open("w",newline="",encoding="utf-8") as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return {"summary":summary,"rows":rows,"test_scores":test_s,"calibrated":calibrated}
