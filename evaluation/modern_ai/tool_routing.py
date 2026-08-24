from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from evaluation.open_set_methods.kappa_utils import vmf_log_normalizer_np

from .calibration import ErrorCalibratorBundle, binary_nll
from .embedders import BaseTextEmbedder, l2_normalize
from .methods import ModernUncertaintyModel, PosteriorModelConfig
from .metrics import calibration_metrics, risk_coverage_curve
from .query_uncertainty import query_embeddings_and_kappa
from .scf_diagnostics import vmf_mean_resultant
from .types import ToolRoutingExample




def _stratified_indices(labels: np.ndarray, fraction: float, seed: int):
    """Sample-level split for protocols whose gallery is intentionally fixed."""
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


def _gallery_group_id(example: ToolRoutingExample) -> str:
    """Stable identifier for the complete candidate-tool gallery of one example.

    BFCL can reuse the same candidate function set across several prompts.  Keeping
    a gallery wholly on one side of the calibration/test boundary prevents a
    downstream error calibrator from seeing the exact routing choice set later
    used for evaluation.
    """
    payload = "\x1e".join(str(x) for x in example.tools).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stratified_group_indices(
    examples: Sequence[ToolRoutingExample], fraction: float, seed: int
):
    """Split by known/unknown state while keeping candidate galleries disjoint."""
    rng = np.random.default_rng(seed)
    cal, test = [], []
    class_group_counts = {}
    for cls in (False, True):
        groups = {}
        for i, ex in enumerate(examples):
            if bool(ex.known) != cls:
                continue
            groups.setdefault(_gallery_group_id(ex), []).append(i)
        gids = list(groups)
        if len(gids) < 2:
            state = "known/relevant" if cls else "unknown/irrelevant"
            raise ValueError(
                "Internal tool-routing calibration requires at least two distinct "
                f"candidate galleries for the {state} class; found {len(gids)}. "
                "Provide an explicit calibration_examples split for small protocols."
            )
        gids = list(np.asarray(gids, dtype=object)[rng.permutation(len(gids))])
        n_cal_groups = min(
            max(int(round(len(gids) * float(fraction))), 1), len(gids) - 1
        )
        cal_gids = set(gids[:n_cal_groups])
        for gid, idx in groups.items():
            (cal if gid in cal_gids else test).extend(idx)
        class_group_counts[str(cls)] = {
            "total": int(len(gids)),
            "calibration": int(n_cal_groups),
            "test": int(len(gids) - n_cal_groups),
        }
    if not cal or not test:
        raise ValueError("Too few tool-routing examples for calibration/test split")
    return (
        np.asarray(sorted(cal), dtype=int),
        np.asarray(sorted(test), dtype=int),
        class_group_counts,
    )


def _fixed_tool_gallery(examples: Sequence[ToolRoutingExample]) -> Optional[tuple[str, ...]]:
    if not examples:
        return None
    first = tuple(examples[0].tools)
    return first if all(tuple(x.tools) == first for x in examples) else None


def _encode_examples(
    examples: Sequence[ToolRoutingExample],
    embedder: BaseTextEmbedder,
    rewrites: Optional[Mapping[str, Sequence[str]]],
    default_query_kappa: float,
    query_mean_mode: str,
    kappa_source: str,
):
    ids = [x.example_id for x in examples]
    queries = [x.query for x in examples]
    q, kappa, rbar = query_embeddings_and_kappa(
        ids, queries, embedder, rewrites=rewrites,
        default_kappa=default_query_kappa, mean_mode=query_mean_mode,
        kappa_source=kappa_source,
    )
    fixed = _fixed_tool_gallery(examples)
    if fixed is not None:
        g = embedder.encode_documents(fixed)
        galleries = [g] * len(examples)
    else:
        galleries = [embedder.encode_documents(x.tools) for x in examples]
    return q, kappa, rbar, galleries



def fit_query_kappa_scale(
    examples: Sequence[ToolRoutingExample],
    query_embeddings: np.ndarray,
    query_kappa: np.ndarray,
    galleries: Sequence[np.ndarray],
    *,
    min_scale: float = 0.05,
    max_scale: float = 20.0,
) -> tuple[float, dict]:
    """Fit one multiplicative SCF concentration scale on known calibration queries.

    SCF learns the *ordering* and approximate magnitude of ``kappa_x``.  Transfer
    to API identities unseen by the SCF stage can leave a global concentration
    bias even when ranking remains useful.  HolUE/MPRisk depend on the absolute
    concentration, so the primary ToolBench protocol is allowed one validation-
    only scalar calibration before gallery-kappa fitting.

    The scale minimizes the proper vMF negative log-likelihood of each known
    calibration query's true API-description direction.  Unknown calibration
    queries are deliberately excluded because they have no positive spherical
    target.  The fitted scalar is then frozen for the final test split.
    """
    if not (0.0 < float(min_scale) < float(max_scale)):
        raise ValueError("Need 0 < min_scale < max_scale for query-kappa calibration")
    q = l2_normalize(np.asarray(query_embeddings, dtype=np.float64))
    k = np.asarray(query_kappa, dtype=np.float64).reshape(-1)
    if len(q) != len(examples) or len(k) != len(examples) or len(galleries) != len(examples):
        raise ValueError("examples/query embeddings/query kappa/galleries lengths differ")
    if np.any(~np.isfinite(k)) or np.any(k <= 0):
        raise FloatingPointError("query-kappa calibration requires finite positive concentrations")

    positive_rows = []
    target_cosines = []
    for i, ex in enumerate(examples):
        if not bool(ex.known) or not ex.relevant_tool_indices:
            continue
        gallery = l2_normalize(np.asarray(galleries[i], dtype=np.float64))
        rel = [int(j) for j in ex.relevant_tool_indices if 0 <= int(j) < len(gallery)]
        if not rel:
            continue
        # ToolBench has one positive API.  Using max makes the calibration helper
        # well-defined for later multi-positive routing benchmarks too.
        cos = float(np.max(gallery[rel] @ q[i]))
        positive_rows.append(i)
        target_cosines.append(cos)
    if not positive_rows:
        raise ValueError("No known calibration queries with a relevant API for kappa scaling")

    idx = np.asarray(positive_rows, dtype=int)
    target = np.asarray(target_cosines, dtype=np.float64)
    base_k = k[idx]
    d = int(q.shape[1])

    def nll_for_log_scale(log_scale: float) -> float:
        scale = float(np.exp(log_scale))
        kk = np.maximum(base_k * scale, 1e-12)
        log_c = vmf_log_normalizer_np(kk, d=d)
        return float(np.mean(-(log_c + kk * target)))

    result = minimize_scalar(
        nll_for_log_scale,
        bounds=(float(np.log(min_scale)), float(np.log(max_scale))),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"SCF query-kappa scale calibration failed: {result}")
    scale = float(np.exp(result.x))

    raw_ad = vmf_mean_resultant(base_k, d)
    calibrated_ad = vmf_mean_resultant(base_k * scale, d)
    meta = {
        "strategy": "known_positive_vmf_nll_scale",
        "uses_unknown_queries": False,
        "num_known_positive_queries": int(len(idx)),
        "embedding_dim": d,
        "scale": scale,
        "bounds": [float(min_scale), float(max_scale)],
        "objective_nll_raw": nll_for_log_scale(0.0),
        "objective_nll_calibrated": float(result.fun),
        "target_cosine_mean": float(np.mean(target)),
        "raw_mean_A_d_kappa": float(np.mean(raw_ad)),
        "calibrated_mean_A_d_kappa": float(np.mean(calibrated_ad)),
        "raw_stationarity_mae": float(np.mean(np.abs(raw_ad - np.clip(target, 0.0, 1.0)))),
        "calibrated_stationarity_mae": float(np.mean(np.abs(calibrated_ad - np.clip(target, 0.0, 1.0)))),
        "optimizer_at_lower_bound": bool(scale <= min_scale * 1.001),
        "optimizer_at_upper_bound": bool(scale >= max_scale / 1.001),
    }
    return scale, meta

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
    inapplicable.  For dissertation-facing BFCL runs we transfer ``gallery_kappa``
    from the fixed-gallery ToolBench calibration.  If no value is supplied, this
    function provides an explicitly secondary validation-only empirical fallback:
    first minimize absolute FPIR error, then unknown-posterior NLL.
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


def _fixed_scores(model: ModernUncertaintyModel, examples, q, kappa, gallery, rbar):
    ids = [x.example_id for x in examples]
    resultant = None if np.all(~np.isfinite(rbar)) else rbar
    result = model.score(
        ids, [str(i) for i in range(len(gallery))], q, kappa, gallery,
        query_resultant_length=resultant,
    )
    scores = dict(result.scores)
    scores.update({
        "holue_kl_1": np.asarray(result.kl_1),
        "holue_kl_2": np.asarray(result.kl_2),
        "unknown_nonspecificity": np.asarray(result.unknown_nonspecificity),
        "rejected": np.asarray(result.was_rejected, dtype=float),
        "predicted_index": np.asarray(result.predicted_indices, dtype=float),
        "negative_query_kappa": -np.asarray(kappa, dtype=float).reshape(-1),
    })
    return scores


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


def _kappa_stats(values: np.ndarray) -> dict:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"count": 0, "mean": np.nan, "median": np.nan, "q05": np.nan, "q95": np.nan}
    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "q05": float(np.quantile(x, 0.05)),
        "q95": float(np.quantile(x, 0.95)),
    }


def run_tool_routing_experiment(
    examples: Sequence[ToolRoutingExample],
    embedder: BaseTextEmbedder,
    model_config: PosteriorModelConfig,
    *,
    calibration_examples: Optional[Sequence[ToolRoutingExample]] = None,
    rewrites: Optional[Mapping[str, Sequence[str]]] = None,
    calibration_rewrites: Optional[Mapping[str, Sequence[str]]] = None,
    default_query_kappa: float = 300.0,
    query_mean_mode: str = "original",
    kappa_source: str = "auto",
    calibration_fraction: float = 0.3,
    kappa_grid_size: int = 32,
    query_kappa_calibration: str = "none",
    query_kappa_scale_min: float = 0.05,
    query_kappa_scale_max: float = 20.0,
    seed: int = 777,
    output_dir: Optional[str | Path] = None,
) -> dict:
    """Open-set tool routing with either prescribed or internal calibration split.

    If ``calibration_examples`` is provided, no final-test example is used for
    gallery-kappa or error calibration.  Fixed-gallery protocols (ToolBench OSR)
    use the dissertation's fixed-K boundary-root calibration.  External BFCL runs
    should transfer that fitted concentration; an empirical BFCL fit exists only
    as a fallback when no transferred value is supplied.  Internal splits are
    grouped by candidate gallery to avoid exact-gallery calibration/test leakage.
    """
    examples = list(examples)
    if calibration_examples is None:
        internal_fixed_gallery = _fixed_tool_gallery(examples) is not None
        if internal_fixed_gallery:
            known_all = np.asarray([x.known for x in examples], dtype=bool)
            cal_idx, test_idx = _stratified_indices(
                known_all, calibration_fraction, seed
            )
            group_counts = None
        else:
            cal_idx, test_idx, group_counts = _stratified_group_indices(
                examples, calibration_fraction, seed
            )
        all_q, all_k, all_r, all_g = _encode_examples(
            examples, embedder, rewrites, default_query_kappa, query_mean_mode, kappa_source
        )
        cal_examples = [examples[i] for i in cal_idx]
        test_examples = [examples[i] for i in test_idx]
        q_cal, k_cal, r_cal = all_q[cal_idx], all_k[cal_idx], all_r[cal_idx]
        q_test, k_test, r_test = all_q[test_idx], all_k[test_idx], all_r[test_idx]
        g_cal = [all_g[i] for i in cal_idx]; g_test = [all_g[i] for i in test_idx]
        split_meta = {
            "strategy": (
                "internal_stratified_fixed_gallery"
                if internal_fixed_gallery else "internal_gallery_group_stratified"
            ),
            "seed": seed,
            "num_calibration": int(len(cal_idx)),
            "num_test": int(len(test_idx)),
        }
        if not internal_fixed_gallery:
            split_meta.update({
                "gallery_group_counts": group_counts,
                "calibration_gallery_hashes": sorted(
                    {_gallery_group_id(examples[i]) for i in cal_idx}
                ),
                "test_gallery_hashes": sorted(
                    {_gallery_group_id(examples[i]) for i in test_idx}
                ),
            })
    else:
        cal_examples = list(calibration_examples)
        test_examples = examples
        q_cal, k_cal, r_cal, g_cal = _encode_examples(
            cal_examples, embedder, calibration_rewrites, default_query_kappa,
            query_mean_mode, kappa_source,
        )
        q_test, k_test, r_test, g_test = _encode_examples(
            test_examples, embedder, rewrites, default_query_kappa,
            query_mean_mode, kappa_source,
        )
        split_meta = {"strategy": "prescribed_validation_test", "seed": seed,
                      "num_calibration": len(cal_examples), "num_test": len(test_examples)}

    raw_k_cal = np.asarray(k_cal, dtype=np.float64).copy()
    raw_k_test = np.asarray(k_test, dtype=np.float64).copy()
    kappa_scale_strategy = str(query_kappa_calibration).strip().lower()
    if kappa_scale_strategy in {"none", "off", "false"}:
        query_kappa_scale = 1.0
        query_kappa_calibration_meta = {
            "strategy": "none",
            "scale": 1.0,
            "uses_final_test_split": False,
        }
    elif kappa_scale_strategy in {"vmf_nll_scale", "known_positive_vmf_nll_scale"}:
        query_kappa_scale, query_kappa_calibration_meta = fit_query_kappa_scale(
            cal_examples, q_cal, raw_k_cal, g_cal,
            min_scale=query_kappa_scale_min, max_scale=query_kappa_scale_max,
        )
        query_kappa_calibration_meta["uses_final_test_split"] = False
    else:
        raise ValueError(
            "query_kappa_calibration must be 'none' or 'vmf_nll_scale'; "
            f"got {query_kappa_calibration!r}"
        )
    k_cal = raw_k_cal * float(query_kappa_scale)
    k_test = raw_k_test * float(query_kappa_scale)

    known_cal = np.asarray([x.known for x in cal_examples], dtype=bool)
    fixed_cal = _fixed_tool_gallery(cal_examples)
    fixed_test = _fixed_tool_gallery(test_examples)
    # A fixed-gallery protocol does not require the *identities* in calibration
    # and test galleries to be the same.  ToolBench v2 deliberately transfers a
    # concentration fitted on one K-class gallery to a disjoint K-class gallery,
    # mirroring identity-disjoint face-recognition evaluation.  The fixed-K
    # posterior is valid as long as each split has one fixed gallery and K agrees.
    use_fixed = (
        fixed_cal is not None and fixed_test is not None
        and len(fixed_cal) == len(fixed_test)
    )
    same_fixed_gallery = bool(use_fixed and fixed_cal == fixed_test)

    if use_fixed:
        calibration_gallery = g_cal[0]
        test_gallery = g_test[0]
        model = ModernUncertaintyModel(model_config)
        fitted_kappa = model.fit_gallery_kappa(q_cal, k_cal, calibration_gallery, known_cal)
        kappa_meta = {
            "strategy": str(model_config.gallery_kappa_strategy),
            "fixed_gallery": True,
            "same_gallery_identities_in_calibration_and_test": same_fixed_gallery,
            "num_gallery_tools": len(calibration_gallery),
            "fit_tau": float(getattr(model, "fit_tau_", np.nan)),
            "candidates": list(map(float, getattr(model, "kappa_candidates_", ()))),
            "candidate_nll": list(map(float, getattr(model, "kappa_candidate_nll_", ()))),
        }
        cfg = replace(model_config, gallery_kappa=float(fitted_kappa))
        # Fit only on calibration classes, then freeze kappa_g and score the
        # disjoint final gallery without refitting.
        cal_s = _fixed_scores(
            model, cal_examples, q_cal, k_cal, calibration_gallery, r_cal
        )
        test_s = _fixed_scores(
            model, test_examples, q_test, k_test, test_gallery, r_test
        )
    else:
        fitted_kappa, kappa_meta = fit_variable_gallery_kappa(
            q_cal, k_cal, g_cal, known_cal, model_config, grid_size=kappa_grid_size,
        )
        cfg = replace(model_config, gallery_kappa=float(fitted_kappa))
        model = ModernUncertaintyModel(cfg)
        cal_s = model.score_variable_galleries(
            q_cal, k_cal, g_cal, None if np.all(~np.isfinite(r_cal)) else r_cal
        )
        test_s = model.score_variable_galleries(
            q_test, k_test, g_test, None if np.all(~np.isfinite(r_test)) else r_test
        )
        cal_s["negative_query_kappa"] = -np.asarray(k_cal).reshape(-1)
        test_s["negative_query_kappa"] = -np.asarray(k_test).reshape(-1)

    _, _, _, cal_correct, _ = _decision_labels(cal_examples, cal_s)
    t_known, t_rej, t_pred, t_correct, selection_eval = _decision_labels(test_examples, test_s)
    y_cal = (~cal_correct).astype(int)
    y_test = (~t_correct).astype(int)

    excluded = {"holue_kl_1", "holue_kl_2", "holue_kl_sum", "rejected", "predicted_index"}
    scalar_cal = {k: v for k, v in cal_s.items() if k not in excluded}
    bundle = ErrorCalibratorBundle.fit(
        scalar_cal, cal_s["holue_kl_1"], cal_s["holue_kl_2"], y_cal,
        exclude=("holue", "unknown_probability"),
    )
    calibrated = bundle.apply(
        {k:v for k,v in test_s.items() if k not in excluded},
        test_s["holue_kl_1"], test_s["holue_kl_2"],
    )
    metrics = {
        "call_reject_accuracy": float(np.mean(((t_known & ~t_rej) | (~t_known & t_rej)))),
        "task_accuracy_with_tool_id_when_available": float(np.mean(t_correct)),
        "fpir_irrelevant_accepted": float(np.mean(~t_rej[~t_known])) if np.any(~t_known) else np.nan,
        "fnir_relevant_rejected": float(np.mean(t_rej[t_known])) if np.any(t_known) else np.nan,
        "num_test": int(len(test_examples)),
        "num_tool_id_evaluable": int(np.sum(selection_eval)),
        "irrelevance_auroc_p_unknown": _auc((~t_known).astype(int), test_s["unknown_probability"]),
        "irrelevance_auprc_p_unknown": _auc((~t_known).astype(int), test_s["unknown_probability"], "pr"),
    }
    if np.any(selection_eval):
        metrics["tool_id_accuracy_conditional_evaluable"] = float(np.mean(t_correct[selection_eval]))

    uncertainty = {}
    for name, raw in test_s.items():
        if name in excluded: continue
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
        "query_kappa_source": kappa_source,
        "query_kappa_calibration": query_kappa_calibration_meta,
        "query_kappa_stats": {
            "calibration_raw": _kappa_stats(raw_k_cal),
            "test_raw": _kappa_stats(raw_k_test),
            "calibration_effective": _kappa_stats(k_cal),
            "test_effective": _kappa_stats(k_test),
        },
        "fixed_gallery": bool(use_fixed),
        "same_gallery_identities_in_calibration_and_test": bool(same_fixed_gallery),
        "fitted_gallery_kappa": float(fitted_kappa),
        "kappa_calibration": kappa_meta,
        "posterior_config": asdict(cfg),
        "split": split_meta,
        "decision_metrics": metrics,
        "uncertainty_detection": uncertainty,
        "calibration": cal_metrics,
    }
    rows=[]
    for j,x in enumerate(test_examples):
        row={
            "id":x.example_id,"known":bool(x.known),"num_tools":len(x.tools),
            "rejected":bool(t_rej[j]),"predicted_index":int(t_pred[j]),"correct":bool(t_correct[j]),
            "relevant_tool_indices":";".join(map(str,x.relevant_tool_indices)),
            "group_id": str(x.metadata.get("tool_id") or x.metadata.get("source_file") or x.example_id),
            "query_kappa_raw": float(np.asarray(raw_k_test[j]).reshape(-1)[0]),
            "query_kappa_effective": float(np.asarray(k_test[j]).reshape(-1)[0]),
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
