#!/usr/bin/env python3

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import OmegaConf

from evaluation.reproducibility import seed_everything

import gc
from contextlib import nullcontext

try:
    import torch
except Exception:
    torch = None


def torch_inference_context():
    if torch is None:
        return nullcontext()
    return torch.inference_mode()


def cuda_cleanup(tag: str = ""):
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

        # Optional debug print:
        # allocated = torch.cuda.memory_allocated() / 2**30
        # reserved = torch.cuda.memory_reserved() / 2**30
        # print(f"[cuda-cleanup] {tag}: allocated={allocated:.2f} GiB reserved={reserved:.2f} GiB")


def as_numpy_copy(x):
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().numpy().copy()
    return np.asarray(x).copy()


def run_base_mprisk_safe(*args, **kwargs):
    cuda_cleanup("before run_base_mprisk")
    try:
        with torch_inference_context():
            return run_base_mprisk(*args, **kwargs)
    finally:
        cuda_cleanup("after run_base_mprisk")


def _first_dim(x) -> int:
    return int(np.asarray(x).shape[0])


def trim_feature_matrix(X: np.ndarray, n: int, name: str) -> np.ndarray:
    X = np.asarray(X)

    if X.shape[0] == n:
        return X

    print(f"[fair-tuning][align] trimming {name}: " f"{X.shape[0]} -> {n}")

    return X[:n]


def trim_vector(x: np.ndarray, n: int, name: str) -> np.ndarray:
    x = np.asarray(x)

    if x.shape[0] == n:
        return x

    print(f"[fair-tuning][align] trimming {name}: " f"{x.shape[0]} -> {n}")

    return x[:n]


def trim_components_to_n(
    components: Dict[str, np.ndarray],
    n: int,
    name: str,
) -> Dict[str, np.ndarray]:
    out = {}

    for key, value in components.items():
        value_np = np.asarray(value)

        if value_np.shape[0] == n:
            out[key] = value_np
        else:
            print(
                f"[fair-tuning][align] trimming {name}.{key}: "
                f"{value_np.shape[0]} -> {n}"
            )
            out[key] = value_np[:n]

    return out


def trim_result_to_n(result: Dict[str, Any], n: int, name: str) -> Dict[str, Any]:
    """
    Trim a result dictionary to the first n probes and recompute masks.
    g_unique_ids is not trimmed.
    """
    out = dict(result)

    for key in ["predicted_id", "was_rejected", "predicted_unc", "probe_unique_ids"]:
        if key in out:
            out[key] = trim_vector(out[key], n, f"{name}.{key}")

    out["masks"] = compute_osr_error_masks(
        predicted_id=out["predicted_id"],
        was_rejected=out["was_rejected"],
        g_unique_ids=out["g_unique_ids"],
        probe_unique_ids=out["probe_unique_ids"],
    )

    return out


def align_validation_blocks(
    X_simple_val: np.ndarray,
    X_post_val: np.ndarray,
    comps_val: Dict[str, np.ndarray],
    probe_ids_val: np.ndarray,
):
    lengths = [
        _first_dim(X_simple_val),
        _first_dim(X_post_val),
        _first_dim(comps_val["predicted_id"]),
        _first_dim(probe_ids_val),
    ]
    n = min(lengths)

    if len(set(lengths)) != 1:
        print(
            "[fair-tuning][align] validation length mismatch: "
            f"X_simple_val={lengths[0]}, X_post_val={lengths[1]}, "
            f"components={lengths[2]}, probe_ids={lengths[3]}; "
            f"using n={n}"
        )

    X_simple_val = trim_feature_matrix(X_simple_val, n, "X_simple_val")
    X_post_val = trim_feature_matrix(X_post_val, n, "X_post_val")
    comps_val = trim_components_to_n(comps_val, n, "calib_components")
    probe_ids_val = trim_vector(probe_ids_val, n, "probe_ids_val")

    return X_simple_val, X_post_val, comps_val, probe_ids_val


def align_test_blocks(
    X_simple_test: np.ndarray,
    X_post_test: np.ndarray,
    comps_test: Dict[str, np.ndarray],
    result_test: Dict[str, Any],
):
    lengths = [
        _first_dim(X_simple_test),
        _first_dim(X_post_test),
        _first_dim(comps_test["predicted_id"]),
        _first_dim(result_test["predicted_id"]),
        _first_dim(result_test["probe_unique_ids"]),
    ]
    n = min(lengths)

    if len(set(lengths)) != 1:
        print(
            "[fair-tuning][align] test length mismatch: "
            f"X_simple_test={lengths[0]}, X_post_test={lengths[1]}, "
            f"components={lengths[2]}, result_pred={lengths[3]}, "
            f"result_ids={lengths[4]}; using n={n}"
        )

    X_simple_test = trim_feature_matrix(X_simple_test, n, "X_simple_test")
    X_post_test = trim_feature_matrix(X_post_test, n, "X_post_test")
    comps_test = trim_components_to_n(comps_test, n, "test_components")
    result_test = trim_result_to_n(result_test, n, "result_test")

    return X_simple_test, X_post_test, comps_test, result_test


def maybe_set_predict_T_from_dataset(
    core_cfg,
    recognition_method,
    dataset_name_for_T: str,
):
    """
    Normal evaluation.evaluate.py sets GalUE predict_T from cfg.dataset_name_to_T_scale.
    Fair-tuning runs methods manually, so reproduce the same behavior here.

    Any method with predict_T=None gets a dataset-specific value.
    """
    if not hasattr(recognition_method, "predict_T"):
        return

    if getattr(recognition_method, "predict_T") is not None:
        return

    try:
        predict_T = getattr(core_cfg.dataset_name_to_T_scale, dataset_name_for_T)
        recognition_method.predict_T = predict_T
        print(
            f"[fair-tuning] set predict_T={predict_T} "
            f"for dataset={dataset_name_for_T}"
        )
    except Exception:
        recognition_method.predict_T = 1.0
        print(
            f"[fair-tuning] warning: no dataset-specific predict_T for "
            f"{dataset_name_for_T}; using predict_T=1.0"
        )


def ensure_calibration_dataset_for_mprisk(
    core_cfg,
    recognition_method,
    dataset_name: str,
):
    """
    External fair-tuning needs validation components for MPRisk.
    Attach the instantiated calibration set if needed.
    """
    if (
        not hasattr(recognition_method, "calibration_set")
        or recognition_method.calibration_set is None
        or recognition_method.calibration_set is True
    ):
        calib_set_cfg = getattr(core_cfg.dataset_name_to_calibration_set, dataset_name)
        recognition_method.calibration_set = instantiate(calib_set_cfg)

    if (
        not hasattr(recognition_method, "calibration_embs_name")
        or recognition_method.calibration_embs_name is None
    ):
        recognition_method.calibration_embs_name = "scf"


from experiments.mprisk_core_experiments import (
    build_tester,
    compute_osr_error_masks,
    run_method_raw,
    safe_auc,
    safe_auprc,
    safe_spearman,
    self_normalized_prr,
    slugify,
)

from experiments.mprisk_tuning_experiments import (
    build_lambda_candidates,
    component_score,
    compute_mprisk_calibration_components,
    run_base_mprisk,
    tune_lambdas_for_components,
)


# ---------------------------------------------------------------------
# Config utilities
# ---------------------------------------------------------------------


def load_core_cfg(path: str, exp_dir: str):
    core_cfg = OmegaConf.load(path)
    core_cfg.exp_dir = exp_dir
    OmegaConf.resolve(core_cfg)
    return core_cfg


def cfg_to_container(x):
    return OmegaConf.to_container(x, resolve=True)


def copy_cfg_node(x):
    return OmegaConf.create(cfg_to_container(x))


def get_dataset_cfg(core_cfg, dataset_name: str):
    for ds in core_cfg.test_datasets:
        if str(ds.dataset_name) == str(dataset_name):
            return ds
    raise KeyError(f"Dataset {dataset_name!r} not found in core config.")


def get_method_cfg(core_cfg, pretty_name: str):
    for m in core_cfg.open_set_identification_methods:
        if str(m.pretty_name) == str(pretty_name):
            return m
    raise KeyError(f"Method {pretty_name!r} not found in core config.")


def set_method_override(method_cfg, key: str, value: Any):
    OmegaConf.update(
        method_cfg,
        f"recognition_method.{key}",
        value,
        force_add=True,
    )


def apply_overrides(method_cfg, overrides: Optional[Dict[str, Any]]):
    method_cfg = copy_cfg_node(method_cfg)
    if overrides is None:
        return method_cfg

    for key, value in overrides.items():
        set_method_override(method_cfg, key, value)

    return method_cfg


def disable_internal_supervision(method_cfg):
    """
    For feature extraction we want raw uncertainty scores, not internally
    calibrated or internally tuned variants.
    """
    method_cfg = copy_cfg_node(method_cfg)

    if "calibration_set" in method_cfg.recognition_method:
        method_cfg.recognition_method.calibration_set = None

    if "calibration_transform" in method_cfg.recognition_method:
        method_cfg.recognition_method.calibration_transform = None

    if "tune_lambdas" in method_cfg.recognition_method:
        method_cfg.recognition_method.tune_lambdas = False

    if "use_calibration" in method_cfg.recognition_method:
        method_cfg.recognition_method.use_calibration = False

    return method_cfg


# ---------------------------------------------------------------------
# Generic scoring utilities
# ---------------------------------------------------------------------


def make_fraction_grid(spec) -> np.ndarray:
    spec = np.asarray(spec, dtype=float).reshape(-1)
    if spec.shape[0] == 3 and spec[2] > 1 and float(spec[2]).is_integer():
        return np.linspace(float(spec[0]), float(spec[1]), int(spec[2]))
    return spec


def category_arrays_from_osr_masks(
    masks: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Convert OSR masks to TP/FP/FN boolean arrays for F1 computation.
    """
    tp = np.asarray(masks["true_accept_true_ident"], dtype=bool)
    fp = np.asarray(masks["false_accept"], dtype=bool)
    fn = np.logical_or(
        np.asarray(masks["false_reject"], dtype=bool),
        np.asarray(masks["misidentification"], dtype=bool),
    )
    err = np.asarray(masks["any_error"], dtype=bool)

    return {"tp": tp, "fp": fp, "fn": fn, "err": err}


def category_arrays_from_prediction(
    predicted_id: np.ndarray,
    was_rejected: np.ndarray,
    g_unique_ids: np.ndarray,
    probe_unique_ids: np.ndarray,
) -> Dict[str, np.ndarray]:
    masks = compute_osr_error_masks(
        predicted_id=predicted_id,
        was_rejected=was_rejected,
        g_unique_ids=g_unique_ids,
        probe_unique_ids=probe_unique_ids,
    )
    return category_arrays_from_osr_masks(masks)


def f1_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    denom = 2.0 * tp + fp + fn
    out = np.zeros_like(denom, dtype=np.float64)
    ok = denom > 0
    out[ok] = 2.0 * tp[ok] / denom[ok]
    return out


def f1_auc_fast(
    uncertainty: np.ndarray,
    cats: Dict[str, np.ndarray],
    fractions: np.ndarray,
) -> float:
    """
    Fast F1 rejection-curve AUC.

    Repository convention:
      lower uncertainty is kept first,
      higher uncertainty is filtered first.
    """
    uncertainty = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    n = len(uncertainty)

    order = np.argsort(uncertainty)

    tp_sorted = cats["tp"][order].astype(np.int64)
    fp_sorted = cats["fp"][order].astype(np.int64)
    fn_sorted = cats["fn"][order].astype(np.int64)

    tp_cum = np.concatenate([[0], np.cumsum(tp_sorted)])
    fp_cum = np.concatenate([[0], np.cumsum(fp_sorted)])
    fn_cum = np.concatenate([[0], np.cumsum(fn_sorted)])

    keep_counts = np.asarray(
        [int((1.0 - float(frac)) * n) for frac in fractions],
        dtype=np.int64,
    )
    keep_counts = np.clip(keep_counts, 0, n)

    tp = tp_cum[keep_counts]
    fp = fp_cum[keep_counts]
    fn = fn_cum[keep_counts]

    f1_values = f1_from_counts(tp, fp, fn)

    return float(np.trapz(f1_values, fractions))


def prr_fast(
    uncertainty: np.ndarray,
    cats: Dict[str, np.ndarray],
    fractions: np.ndarray,
    random_auc: float,
    oracle_auc: float,
) -> float:
    auc_value = f1_auc_fast(
        uncertainty=uncertainty,
        cats=cats,
        fractions=fractions,
    )

    denom = oracle_auc - random_auc
    if abs(denom) < 1e-12:
        return auc_value

    return float((auc_value - random_auc) / denom)


def reference_auc_fast(
    cats: Dict[str, np.ndarray],
    fractions: np.ndarray,
    seed: int,
) -> Tuple[float, float]:
    n = len(cats["tp"])
    rng = np.random.default_rng(seed)

    random_unc = rng.random(n)

    # Oracle should filter errors first, so errors get larger uncertainty.
    oracle_unc = cats["err"].astype(np.float64) + 1e-9 * rng.random(n)

    random_auc = f1_auc_fast(random_unc, cats, fractions)
    oracle_auc = f1_auc_fast(oracle_unc, cats, fractions)

    return random_auc, oracle_auc


def eval_uncertainty_score(
    score: np.ndarray,
    result: Dict[str, Any],
    fractions: np.ndarray,
    seed: int,
) -> Dict[str, float]:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    masks = result["masks"]

    prr, curve, _, _ = self_normalized_prr(
        uncertainty=score,
        predicted_id=result["predicted_id"],
        was_rejected=result["was_rejected"],
        g_unique_ids=result["g_unique_ids"],
        probe_unique_ids=result["probe_unique_ids"],
        fractions=fractions,
        metric_name="f1_class",
        seed=seed,
    )

    return {
        "test_prr_f1": prr,
        "test_error_auroc": safe_auc(masks["any_error"], score),
        "test_error_auprc": safe_auprc(masks["any_error"], score),
        "test_fa_auroc": safe_auc(masks["false_accept"], score),
        "test_fr_auroc": safe_auc(masks["false_reject"], score),
        "test_id_auroc": safe_auc(masks["misidentification"], score),
        "f1_at_0_filter": float(curve["f1_class"].iloc[0]),
        "f1_at_max_filter": float(curve["f1_class"].iloc[-1]),
    }


def normalize_features(
    X_val: np.ndarray,
    X_test: np.ndarray,
    eps: float = 1e-8,
):
    X_val = np.asarray(X_val, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)

    mean = np.nanmean(X_val, axis=0)
    std = np.nanstd(X_val, axis=0)
    std = np.maximum(std, eps)

    X_val_norm = (X_val - mean) / std
    X_test_norm = (X_test - mean) / std

    X_val_norm = np.nan_to_num(X_val_norm, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_norm = np.nan_to_num(X_test_norm, nan=0.0, posinf=0.0, neginf=0.0)

    return X_val_norm, X_test_norm, mean, std


def signed_feature_expansion(
    X_val: np.ndarray,
    X_test: np.ndarray,
    names: List[str],
):
    X_val = np.asarray(X_val, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)

    X_val_exp = np.concatenate([X_val, -X_val], axis=1)
    X_test_exp = np.concatenate([X_test, -X_test], axis=1)
    names_exp = names + [f"neg_{n}" for n in names]

    return X_val_exp, X_test_exp, names_exp


def linear_candidates(
    dim: int, num_random: int, log_low: float, log_high: float, seed: int
):
    rng = np.random.default_rng(seed)

    candidates = []

    candidates.append(np.ones(dim, dtype=np.float64))
    for j in range(dim):
        e = np.zeros(dim, dtype=np.float64)
        e[j] = 1.0
        candidates.append(e)

    for _ in range(int(num_random)):
        w = np.exp(rng.uniform(log_low, log_high, size=dim))
        w = w / max(np.max(w), 1e-12)
        candidates.append(w.astype(np.float64))

    return candidates


def slice_components_local(
    components: Dict[str, np.ndarray],
    idx: np.ndarray,
) -> Dict[str, np.ndarray]:
    idx = np.asarray(idx, dtype=int)
    return {k: np.asarray(v)[idx] for k, v in components.items()}


def tune_lambdas_for_components_fast(
    calib_components: Dict[str, np.ndarray],
    g_unique_ids_calib: np.ndarray,
    probe_unique_ids_calib: np.ndarray,
    fractions: np.ndarray,
    num_random: int,
    log_low: float,
    log_high: float,
    seed: int,
    subset_idx: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    if subset_idx is not None:
        calib_components_used = slice_components_local(calib_components, subset_idx)
        probe_ids_used = np.asarray(probe_unique_ids_calib)[subset_idx]
    else:
        calib_components_used = calib_components
        probe_ids_used = np.asarray(probe_unique_ids_calib)

    cats = category_arrays_from_prediction(
        predicted_id=calib_components_used["predicted_id"],
        was_rejected=calib_components_used["was_rejected"],
        g_unique_ids=g_unique_ids_calib,
        probe_unique_ids=probe_ids_used,
    )

    random_auc, oracle_auc = reference_auc_fast(
        cats=cats,
        fractions=fractions,
        seed=seed,
    )

    candidates = build_lambda_candidates(
        num_random=num_random,
        log_low=log_low,
        log_high=log_high,
        seed=seed,
    )

    best_lambdas = np.ones(4, dtype=np.float64)
    best_prr = -np.inf
    best_score = None

    for lambdas in candidates:
        score = component_score(calib_components_used, lambdas)
        prr = prr_fast(
            uncertainty=score,
            cats=cats,
            fractions=fractions,
            random_auc=random_auc,
            oracle_auc=oracle_auc,
        )

        if prr > best_prr:
            best_prr = float(prr)
            best_lambdas = np.asarray(lambdas, dtype=np.float64).copy()
            best_score = score

    return {
        "lambdas": best_lambdas,
        "val_prr": float(best_prr),
        "val_auc": (
            safe_auc(cats["err"], best_score) if best_score is not None else np.nan
        ),
        "val_auprc": (
            safe_auprc(cats["err"], best_score) if best_score is not None else np.nan
        ),
        "subset_size": len(probe_ids_used),
    }


def tune_linear_score(
    X_val: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    calib_components: Dict[str, np.ndarray],
    g_unique_ids_calib: np.ndarray,
    probe_unique_ids_calib: np.ndarray,
    result_test: Dict[str, Any],
    fractions: np.ndarray,
    num_random: int,
    log_low: float,
    log_high: float,
    seed: int,
    expand_signed: bool = True,
):
    if expand_signed:
        X_val, X_test, feature_names = signed_feature_expansion(
            X_val,
            X_test,
            feature_names,
        )

    X_val_norm, X_test_norm, mean, std = normalize_features(X_val, X_test)

    cats_val = category_arrays_from_prediction(
        predicted_id=calib_components["predicted_id"],
        was_rejected=calib_components["was_rejected"],
        g_unique_ids=g_unique_ids_calib,
        probe_unique_ids=probe_unique_ids_calib,
    )

    random_auc, oracle_auc = reference_auc_fast(
        cats=cats_val,
        fractions=fractions,
        seed=seed,
    )

    best_w = None
    best_prr = -np.inf
    best_score_val = None
    best_score_test = None

    candidates = linear_candidates(
        dim=X_val_norm.shape[1],
        num_random=num_random,
        log_low=log_low,
        log_high=log_high,
        seed=seed,
    )

    for w in candidates:
        score_val = X_val_norm @ w

        prr = prr_fast(
            uncertainty=score_val,
            cats=cats_val,
            fractions=fractions,
            random_auc=random_auc,
            oracle_auc=oracle_auc,
        )

        if prr > best_prr:
            best_prr = float(prr)
            best_w = np.asarray(w, dtype=np.float64).copy()
            best_score_val = score_val.copy()
            best_score_test = X_test_norm @ w

    test_eval = eval_uncertainty_score(
        score=best_score_test,
        result=result_test,
        fractions=fractions,
        seed=seed,
    )

    weight_dict = {name: float(weight) for name, weight in zip(feature_names, best_w)}

    return {
        "score_val": best_score_val,
        "score_test": best_score_test,
        "val_prr": float(best_prr),
        "test_eval": test_eval,
        "weights": weight_dict,
        "feature_names": feature_names,
        "feature_mean": mean,
        "feature_std": std,
    }


def fit_logistic_or_mlp(
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_error_val: np.ndarray,
    kind: str,
    seed: int,
):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_val_s = scaler.fit_transform(
        np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    )
    X_test_s = scaler.transform(np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0))

    if len(np.unique(y_error_val.astype(int))) < 2:
        p_val = np.full(len(y_error_val), float(np.mean(y_error_val)))
        p_test = np.full(X_test.shape[0], float(np.mean(y_error_val)))
        return p_val, p_test

    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
            solver="lbfgs",
        )
    elif kind == "mlp":
        from sklearn.neural_network import MLPClassifier

        clf = MLPClassifier(
            hidden_layer_sizes=(16,),
            activation="relu",
            alpha=1e-3,
            max_iter=2000,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.15,
        )
    else:
        raise ValueError(f"Unknown supervised baseline kind={kind}")

    clf.fit(X_val_s, y_error_val.astype(int))

    if hasattr(clf, "predict_proba"):
        p_val = clf.predict_proba(X_val_s)[:, 1]
        p_test = clf.predict_proba(X_test_s)[:, 1]
    else:
        p_val = clf.decision_function(X_val_s)
        p_test = clf.decision_function(X_test_s)

    return p_val, p_test


# ---------------------------------------------------------------------
# Running methods for features
# ---------------------------------------------------------------------


def run_plain_method(
    core_cfg,
    method_cfg,
    dataset_cfg,
    far: float,
    beta: float,
    suffix: str,
    temperature_dataset_name: Optional[str] = None,
):
    method_cfg = disable_internal_supervision(method_cfg)

    dataset = instantiate(dataset_cfg)
    rm = instantiate(method_cfg.recognition_method)
    rm.far = far
    rm.beta = beta

    if temperature_dataset_name is None:
        temperature_dataset_name = str(dataset.dataset_name)

    maybe_set_predict_T_from_dataset(
        core_cfg=core_cfg,
        recognition_method=rm,
        dataset_name_for_T=temperature_dataset_name,
    )

    method_name = (
        f"fair_{slugify(str(method_cfg.pretty_name))}"
        f"_{slugify(dataset.dataset_name)}"
        f"_far_{far}_beta_{beta}_{suffix}"
    )

    tt = build_tester(
        cfg=core_cfg,
        method_cfg=method_cfg,
        test_dataset=dataset,
        recognition_method=rm,
        method_name=method_name,
        pretty_name=str(method_cfg.pretty_name),
    )

    try:
        with torch_inference_context():
            result = run_method_raw(tt, gallery_name="g1")
        return result
    finally:
        del tt
        del rm
        del dataset
        cuda_cleanup(f"after plain method {method_name}")


def build_simple_feature_matrix(
    core_cfg,
    simple_method_names: List[str],
    test_dataset_cfg,
    calib_dataset_cfg,
    far: float,
    beta: float,
    seed: int,
):
    X_test_cols = []
    X_val_cols = []
    names = []

    simple_results_test = {}
    simple_results_val = {}

    main_dataset_name = str(test_dataset_cfg.dataset_name)

    for method_name in simple_method_names:
        method_cfg = get_method_cfg(core_cfg, method_name)

        print(f"[features] running simple method={method_name} on test")
        res_test = run_plain_method(
            core_cfg=core_cfg,
            method_cfg=method_cfg,
            dataset_cfg=test_dataset_cfg,
            far=far,
            beta=beta,
            suffix="test",
            temperature_dataset_name=main_dataset_name,
        )

        print(f"[features] running simple method={method_name} on validation")
        res_val = run_plain_method(
            core_cfg=core_cfg,
            method_cfg=method_cfg,
            dataset_cfg=calib_dataset_cfg,
            far=far,
            beta=beta,
            suffix="val",
            temperature_dataset_name=main_dataset_name,
        )

        simple_results_test[method_name] = res_test
        simple_results_val[method_name] = res_val

        X_test_cols.append(
            np.asarray(res_test["predicted_unc"], dtype=np.float64).reshape(-1)
        )
        X_val_cols.append(
            np.asarray(res_val["predicted_unc"], dtype=np.float64).reshape(-1)
        )
        names.append(method_name)

    # Robustly align columns. This handles rare one-template differences
    # between Recognition_test and prepare_calibration_dataset.
    n_test = min(len(x) for x in X_test_cols)
    n_val = min(len(x) for x in X_val_cols)

    if len(set(len(x) for x in X_test_cols)) != 1:
        print(
            "[fair-tuning][align] simple test feature length mismatch: "
            + ", ".join(f"{n}:{len(x)}" for n, x in zip(names, X_test_cols))
            + f"; using n={n_test}"
        )

    if len(set(len(x) for x in X_val_cols)) != 1:
        print(
            "[fair-tuning][align] simple val feature length mismatch: "
            + ", ".join(f"{n}:{len(x)}" for n, x in zip(names, X_val_cols))
            + f"; using n={n_val}"
        )

    X_test = np.stack([x[:n_test] for x in X_test_cols], axis=1)
    X_val = np.stack([x[:n_val] for x in X_val_cols], axis=1)

    return X_val, X_test, names, simple_results_val, simple_results_test


def posterior_feature_matrix_from_mprisk(entry, calib):
    rm = entry["recognition_method"]

    # Test features
    comps_test = entry["test_components"]
    kl1_test = np.asarray(rm.kl_1, dtype=np.float64).reshape(-1)
    kl2_test = np.asarray(rm.kl_2, dtype=np.float64).reshape(-1)
    oog_test = np.asarray(getattr(rm, "oog_prob"), dtype=np.float64).reshape(-1)
    nonspec_test = np.asarray(
        getattr(rm, "oog_nonspecificity"), dtype=np.float64
    ).reshape(-1)

    mprisk_raw_test = component_score(comps_test, np.ones(4))

    # Validation features
    comps_val = calib["calib_components"]
    kl1_val = np.asarray(calib["kl_1_calib"], dtype=np.float64).reshape(-1)
    kl2_val = np.asarray(calib["kl_2_calib"], dtype=np.float64).reshape(-1)
    oog_val = np.asarray(calib["oog_prob_calib"], dtype=np.float64).reshape(-1)
    nonspec_val = np.asarray(
        calib["oog_nonspecificity_calib"], dtype=np.float64
    ).reshape(-1)

    mprisk_raw_val = component_score(comps_val, np.ones(4))

    names = [
        "neg_kl1",
        "neg_kl2",
        "neg_kl_sum",
        "oog_prob",
        "unknown_nonspecificity",
        "r_fa",
        "r_id",
        "r_fr",
        "r_ns",
        "mprisk_raw",
    ]

    X_test = np.stack(
        [
            -kl1_test,
            -kl2_test,
            -(kl1_test + kl2_test),
            oog_test,
            nonspec_test,
            comps_test["r_fa"],
            comps_test["r_id"],
            comps_test["r_fr"],
            comps_test["r_ns"],
            mprisk_raw_test,
        ],
        axis=1,
    )

    X_val = np.stack(
        [
            -kl1_val,
            -kl2_val,
            -(kl1_val + kl2_val),
            oog_val,
            nonspec_val,
            comps_val["r_fa"],
            comps_val["r_id"],
            comps_val["r_fr"],
            comps_val["r_ns"],
            mprisk_raw_val,
        ],
        axis=1,
    )

    return X_val, X_test, names


# ---------------------------------------------------------------------
# Main experiments
# ---------------------------------------------------------------------


def run_one_dataset_far(
    cfg,
    core_cfg,
    dataset_name: str,
    far: float,
    beta: float,
):
    fractions = make_fraction_grid(cfg.rejection_fractions)

    dataset_cfg = get_dataset_cfg(core_cfg, dataset_name)
    calib_dataset_cfg = getattr(core_cfg.dataset_name_to_calibration_set, dataset_name)

    # Base raw MPRisk run on test + calibration components.
    mprisk_method_cfg = get_method_cfg(core_cfg, cfg.mprisk_method_name)

    print(f"[MPRisk base] dataset={dataset_name}, far={far}, beta={beta}")
    entry = run_base_mprisk_safe(
        cfg=core_cfg,
        method_cfg=mprisk_method_cfg,
        test_dataset=instantiate(dataset_cfg),
        far=far,
        beta=beta,
        method_name_suffix="fair_base",
    )
    ensure_calibration_dataset_for_mprisk(
        core_cfg=core_cfg,
        recognition_method=entry["recognition_method"],
        dataset_name=dataset_name,
    )

    calib = compute_mprisk_calibration_components(entry["recognition_method"])
    result_test = entry["result"]
    comps_test = entry["test_components"]
    comps_val = calib["calib_components"]

    g_ids_val = calib["g_unique_ids_calib"]
    probe_ids_val = calib["probe_unique_ids_calib"]

    # Extract posterior/KL features while recognition_method is still alive.
    X_post_val, X_post_test, post_names = posterior_feature_matrix_from_mprisk(
        entry,
        calib,
    )

    # Now the GPU-heavy recognition_method is no longer needed.
    entry["recognition_method"] = None

    # These calibration arrays are already copied into X_post_val, so remove them
    # if present.
    for k in [
        "kl_1_calib",
        "kl_2_calib",
        "oog_prob_calib",
        "oog_nonspecificity_calib",
    ]:
        calib.pop(k, None)

    cuda_cleanup("after extracting MPRisk posterior features")

    masks_val = compute_osr_error_masks(
        predicted_id=comps_val["predicted_id"],
        was_rejected=comps_val["was_rejected"],
        g_unique_ids=g_ids_val,
        probe_unique_ids=probe_ids_val,
    )
    y_error_val = masks_val["any_error"].astype(bool)

    rows = []
    weight_rows = []

    # --------------------------------------------------------------
    # MPRisk raw and tuned
    # --------------------------------------------------------------

    score_raw_test = component_score(comps_test, np.ones(4))
    raw_eval = eval_uncertainty_score(
        score_raw_test,
        result_test,
        fractions,
        seed=int(cfg.seed),
    )
    rows.append(
        {
            "dataset": dataset_name,
            "far": far,
            "beta": beta,
            "method": "MPRisk raw external",
            "val_prr": np.nan,
            **raw_eval,
        }
    )

    tune_full = tune_lambdas_for_components_fast(
        calib_components=comps_val,
        g_unique_ids_calib=g_ids_val,
        probe_unique_ids_calib=probe_ids_val,
        fractions=fractions,
        num_random=int(cfg.lambda_search.num_random),
        log_low=float(cfg.lambda_search.log_low),
        log_high=float(cfg.lambda_search.log_high),
        seed=int(cfg.seed) + 11,
    )
    score_full_test = component_score(comps_test, tune_full["lambdas"])
    full_eval = eval_uncertainty_score(
        score_full_test,
        result_test,
        fractions,
        seed=int(cfg.seed),
    )
    rows.append(
        {
            "dataset": dataset_name,
            "far": far,
            "beta": beta,
            "method": "MPRisk tuned external",
            "val_prr": tune_full["val_prr"],
            **full_eval,
        }
    )
    weight_rows.append(
        {
            "dataset": dataset_name,
            "far": far,
            "beta": beta,
            "method": "MPRisk tuned external",
            "lambda_fa": tune_full["lambdas"][0],
            "lambda_id": tune_full["lambdas"][1],
            "lambda_fr": tune_full["lambdas"][2],
            "lambda_ns": tune_full["lambdas"][3],
        }
    )

    # Tune no-NS variant by setting r_ns = 0.
    comps_val_no_ns = copy.deepcopy(comps_val)
    comps_test_no_ns = copy.deepcopy(comps_test)
    comps_val_no_ns["r_ns"] = np.zeros_like(comps_val_no_ns["r_ns"])
    comps_test_no_ns["r_ns"] = np.zeros_like(comps_test_no_ns["r_ns"])

    tune_no_ns = tune_lambdas_for_components_fast(
        calib_components=comps_val_no_ns,
        g_unique_ids_calib=g_ids_val,
        probe_unique_ids_calib=probe_ids_val,
        fractions=fractions,
        num_random=int(cfg.lambda_search.num_random),
        log_low=float(cfg.lambda_search.log_low),
        log_high=float(cfg.lambda_search.log_high),
        seed=int(cfg.seed) + 12,
    )
    score_no_ns_test = component_score(comps_test_no_ns, tune_no_ns["lambdas"])
    no_ns_eval = eval_uncertainty_score(
        score_no_ns_test,
        result_test,
        fractions,
        seed=int(cfg.seed),
    )
    rows.append(
        {
            "dataset": dataset_name,
            "far": far,
            "beta": beta,
            "method": "MPRisk tuned no NS external",
            "val_prr": tune_no_ns["val_prr"],
            **no_ns_eval,
        }
    )
    weight_rows.append(
        {
            "dataset": dataset_name,
            "far": far,
            "beta": beta,
            "method": "MPRisk tuned no NS external",
            "lambda_fa": tune_no_ns["lambdas"][0],
            "lambda_id": tune_no_ns["lambdas"][1],
            "lambda_fr": tune_no_ns["lambdas"][2],
            "lambda_ns": tune_no_ns["lambdas"][3],
        }
    )

    # --------------------------------------------------------------
    # Simple baseline features
    # --------------------------------------------------------------

    (
        X_simple_val,
        X_simple_test,
        simple_names,
        simple_val_results,
        simple_test_results,
    ) = build_simple_feature_matrix(
        core_cfg=core_cfg,
        simple_method_names=list(cfg.simple_methods),
        test_dataset_cfg=dataset_cfg,
        calib_dataset_cfg=calib_dataset_cfg,
        far=far,
        beta=beta,
        seed=int(cfg.seed),
    )

    # --------------------------------------------------------------
    # Posterior / KL / MPRisk features
    # --------------------------------------------------------------

    # ------------------------------------------------------------------
    # Align all validation/test feature blocks and MPRisk components.
    # This prevents one-template mismatches from causing broadcast errors.
    # ------------------------------------------------------------------
    X_simple_val, X_post_val, comps_val, probe_ids_val = align_validation_blocks(
        X_simple_val=X_simple_val,
        X_post_val=X_post_val,
        comps_val=comps_val,
        probe_ids_val=probe_ids_val,
    )

    X_simple_test, X_post_test, comps_test, result_test = align_test_blocks(
        X_simple_test=X_simple_test,
        X_post_test=X_post_test,
        comps_test=comps_test,
        result_test=result_test,
    )

    # Recompute validation masks after alignment.
    masks_val = compute_osr_error_masks(
        predicted_id=comps_val["predicted_id"],
        was_rejected=comps_val["was_rejected"],
        g_unique_ids=g_ids_val,
        probe_unique_ids=probe_ids_val,
    )
    y_error_val = masks_val["any_error"].astype(bool)

    # Recompute aligned full MPRisk score.
    score_full_test = component_score(comps_test, tune_full["lambdas"])
    score_full_val = component_score(comps_val, tune_full["lambdas"])
    # --------------------------------------------------------------
    # Tuned HolUE/KL linear
    # --------------------------------------------------------------

    holue_features = ["neg_kl1", "neg_kl2", "neg_kl_sum"]
    holue_idx = [post_names.index(n) for n in holue_features]

    tuned_kl = tune_linear_score(
        X_val=X_post_val[:, holue_idx],
        X_test=X_post_test[:, holue_idx],
        feature_names=holue_features,
        calib_components=comps_val,
        g_unique_ids_calib=g_ids_val,
        probe_unique_ids_calib=probe_ids_val,
        result_test=result_test,
        fractions=fractions,
        num_random=int(cfg.linear_search.num_random),
        log_low=float(cfg.linear_search.log_low),
        log_high=float(cfg.linear_search.log_high),
        seed=int(cfg.seed) + 21,
        expand_signed=True,
    )
    rows.append(
        {
            "dataset": dataset_name,
            "far": far,
            "beta": beta,
            "method": "Tuned HolUE KL-linear",
            "val_prr": tuned_kl["val_prr"],
            **tuned_kl["test_eval"],
        }
    )
    for fname, w in tuned_kl["weights"].items():
        weight_rows.append(
            {
                "dataset": dataset_name,
                "far": far,
                "beta": beta,
                "method": "Tuned HolUE KL-linear",
                "feature": fname,
                "weight": w,
            }
        )

    # --------------------------------------------------------------
    # Tuned linear combination of simple baselines
    # --------------------------------------------------------------

    tuned_simple = tune_linear_score(
        X_val=X_simple_val,
        X_test=X_simple_test,
        feature_names=simple_names,
        calib_components=comps_val,
        g_unique_ids_calib=g_ids_val,
        probe_unique_ids_calib=probe_ids_val,
        result_test=result_test,
        fractions=fractions,
        num_random=int(cfg.linear_search.num_random),
        log_low=float(cfg.linear_search.log_low),
        log_high=float(cfg.linear_search.log_high),
        seed=int(cfg.seed) + 22,
        expand_signed=True,
    )
    rows.append(
        {
            "dataset": dataset_name,
            "far": far,
            "beta": beta,
            "method": "Tuned simple linear",
            "val_prr": tuned_simple["val_prr"],
            **tuned_simple["test_eval"],
        }
    )
    for fname, w in tuned_simple["weights"].items():
        weight_rows.append(
            {
                "dataset": dataset_name,
                "far": far,
                "beta": beta,
                "method": "Tuned simple linear",
                "feature": fname,
                "weight": w,
            }
        )

    # --------------------------------------------------------------
    # Decision-conditioned rejected-SCF baseline
    # --------------------------------------------------------------

    if "SCF" in simple_names:
        scf_idx = simple_names.index("SCF")
        scf_val = X_simple_val[:, scf_idx]
        scf_test = X_simple_test[:, scf_idx]

        low_val = np.nanmin(scf_val) - np.nanstd(scf_val)
        low_test = np.nanmin(scf_test) - np.nanstd(scf_test)

        rej_scf_val = np.where(comps_val["was_rejected"], scf_val, low_val)
        rej_scf_test = np.where(comps_test["was_rejected"], scf_test, low_test)

        rej_scf_eval = eval_uncertainty_score(
            rej_scf_test,
            result_test,
            fractions,
            seed=int(cfg.seed),
        )
        val_rej_scf_eval = eval_uncertainty_score(
            rej_scf_val,
            {
                "predicted_id": comps_val["predicted_id"],
                "was_rejected": comps_val["was_rejected"],
                "g_unique_ids": g_ids_val,
                "probe_unique_ids": probe_ids_val,
                "masks": masks_val,
            },
            fractions,
            seed=int(cfg.seed),
        )
        rows.append(
            {
                "dataset": dataset_name,
                "far": far,
                "beta": beta,
                "method": "Rejected-SCF",
                "val_prr": val_rej_scf_eval["test_prr_f1"],
                **rej_scf_eval,
            }
        )

    # --------------------------------------------------------------
    # Supervised logistic and MLP predictors
    # --------------------------------------------------------------

    X_all_val = np.concatenate([X_simple_val, X_post_val], axis=1)
    X_all_test = np.concatenate([X_simple_test, X_post_test], axis=1)
    all_names = simple_names + post_names

    for kind in ["logistic", "mlp"]:
        p_val, p_test = fit_logistic_or_mlp(
            X_val=X_all_val,
            X_test=X_all_test,
            y_error_val=y_error_val,
            kind=kind,
            seed=int(cfg.seed) + (31 if kind == "logistic" else 32),
        )

        val_eval = eval_uncertainty_score(
            p_val,
            {
                "predicted_id": comps_val["predicted_id"],
                "was_rejected": comps_val["was_rejected"],
                "g_unique_ids": g_ids_val,
                "probe_unique_ids": probe_ids_val,
                "masks": masks_val,
            },
            fractions,
            seed=int(cfg.seed),
        )
        test_eval = eval_uncertainty_score(
            p_test,
            result_test,
            fractions,
            seed=int(cfg.seed),
        )

        rows.append(
            {
                "dataset": dataset_name,
                "far": far,
                "beta": beta,
                "method": f"Supervised {kind}",
                "val_prr": val_eval["test_prr_f1"],
                **test_eval,
            }
        )

    # --------------------------------------------------------------
    # Hybrid KL + MPRisk
    # --------------------------------------------------------------

    X_hybrid_val = np.stack(
        [
            tuned_kl["score_val"],
            score_full_val,
        ],
        axis=1,
    )
    X_hybrid_test = np.stack(
        [
            tuned_kl["score_test"],
            score_full_test,
        ],
        axis=1,
    )

    tuned_hybrid = tune_linear_score(
        X_val=X_hybrid_val,
        X_test=X_hybrid_test,
        feature_names=["tuned_kl", "tuned_mprisk"],
        calib_components=comps_val,
        g_unique_ids_calib=g_ids_val,
        probe_unique_ids_calib=probe_ids_val,
        result_test=result_test,
        fractions=fractions,
        num_random=int(cfg.linear_search.num_random),
        log_low=float(cfg.linear_search.log_low),
        log_high=float(cfg.linear_search.log_high),
        seed=int(cfg.seed) + 41,
        expand_signed=False,
    )

    rows.append(
        {
            "dataset": dataset_name,
            "far": far,
            "beta": beta,
            "method": "Hybrid KL+MPRisk",
            "val_prr": tuned_hybrid["val_prr"],
            **tuned_hybrid["test_eval"],
        }
    )
    for fname, w in tuned_hybrid["weights"].items():
        weight_rows.append(
            {
                "dataset": dataset_name,
                "far": far,
                "beta": beta,
                "method": "Hybrid KL+MPRisk",
                "feature": fname,
                "weight": w,
            }
        )

    # --------------------------------------------------------------
    # Full component ablation
    # --------------------------------------------------------------

    component_rows = build_full_component_ablation(
        dataset_name=dataset_name,
        far=far,
        beta=beta,
        comps_val=comps_val,
        comps_test=comps_test,
        g_ids_val=g_ids_val,
        probe_ids_val=probe_ids_val,
        result_test=result_test,
        fractions=fractions,
        seed=int(cfg.seed),
        tune_full=tune_full,
        tune_no_ns=tune_no_ns,
        rej_scf_val=rej_scf_val if "SCF" in simple_names else None,
        rej_scf_test=rej_scf_test if "SCF" in simple_names else None,
    )

    base_score_for_approx = as_numpy_copy(component_score(comps_test, np.ones(4)))
    base_rns_for_approx = as_numpy_copy(comps_test["r_ns"])

    calib_for_stability = {
        "calib_components": {k: as_numpy_copy(v) for k, v in comps_val.items()},
        "g_unique_ids_calib": as_numpy_copy(g_ids_val),
        "probe_unique_ids_calib": as_numpy_copy(probe_ids_val),
    }

    # Drop large references before returning.
    entry = None
    calib = None
    simple_test_results = None
    simple_val_results = None

    cuda_cleanup("leaving run_one_dataset_far")

    return (
        rows,
        weight_rows,
        component_rows,
        {
            "calib": calib_for_stability,
            "base_score": base_score_for_approx,
            "base_rns": base_rns_for_approx,
        },
    )


def build_full_component_ablation(
    dataset_name,
    far,
    beta,
    comps_val,
    comps_test,
    g_ids_val,
    probe_ids_val,
    result_test,
    fractions,
    seed,
    tune_full,
    tune_no_ns,
    rej_scf_val=None,
    rej_scf_test=None,
):
    masks_val = compute_osr_error_masks(
        comps_val["predicted_id"],
        comps_val["was_rejected"],
        g_ids_val,
        probe_ids_val,
    )

    val_result = {
        "predicted_id": comps_val["predicted_id"],
        "was_rejected": comps_val["was_rejected"],
        "g_unique_ids": g_ids_val,
        "probe_unique_ids": probe_ids_val,
        "masks": masks_val,
    }

    variants = {
        "r_FA only": comps_test["r_fa"],
        "r_ID only": comps_test["r_id"],
        "r_FR only": comps_test["r_fr"],
        "r_NS only": comps_test["r_ns"],
        "r_FA+r_ID": comps_test["r_fa"] + comps_test["r_id"],
        "r_FR+r_NS": comps_test["r_fr"] + comps_test["r_ns"],
        "ordinary no NS": comps_test["r_fa"] + comps_test["r_id"] + comps_test["r_fr"],
        "full raw": component_score(comps_test, np.ones(4)),
        "tuned no NS": component_score(comps_test, tune_no_ns["lambdas"]),
        "full tuned": component_score(comps_test, tune_full["lambdas"]),
    }

    if rej_scf_test is not None:
        variants["Rejected-SCF"] = rej_scf_test

    rows = []
    for variant, score_test in variants.items():
        eval_test = eval_uncertainty_score(
            score_test,
            result_test,
            fractions,
            seed=seed,
        )
        rows.append(
            {
                "dataset": dataset_name,
                "far": far,
                "beta": beta,
                "variant": variant,
                **eval_test,
            }
        )

    return rows


def run_lambda_stability(
    cfg,
    dataset_name,
    far,
    beta,
    calib,
    fractions,
):
    if not bool(cfg.lambda_stability.enabled):
        return []
    allowed_fars = cfg.lambda_stability.get("fars", None)
    if allowed_fars is not None:
        if not any(np.isclose(float(far), float(x)) for x in allowed_fars):
            return []
    comps_val = calib["calib_components"]
    g_ids_val = calib["g_unique_ids_calib"]
    probe_ids_val = calib["probe_unique_ids_calib"]
    n = len(probe_ids_val)

    rows = []

    for frac in cfg.lambda_stability.validation_fractions:
        for rep in range(int(cfg.lambda_stability.repeats)):
            rng = np.random.default_rng(int(cfg.seed) + rep + int(float(frac) * 10000))

            subset_size = max(
                int(cfg.lambda_stability.min_subset_size),
                int(round(float(frac) * n)),
            )
            subset_size = min(subset_size, n)
            subset_idx = rng.choice(n, size=subset_size, replace=False)

            tune = tune_lambdas_for_components_fast(
                calib_components=comps_val,
                g_unique_ids_calib=g_ids_val,
                probe_unique_ids_calib=probe_ids_val,
                fractions=fractions,
                num_random=int(cfg.lambda_search.num_random),
                log_low=float(cfg.lambda_search.log_low),
                log_high=float(cfg.lambda_search.log_high),
                seed=int(cfg.seed) + 1000 * rep,
                subset_idx=subset_idx,
            )

            rows.append(
                {
                    "dataset": dataset_name,
                    "far": far,
                    "beta": beta,
                    "validation_fraction": float(frac),
                    "repeat": rep,
                    "subset_size": subset_size,
                    "val_prr": tune["val_prr"],
                    "lambda_fa": tune["lambdas"][0],
                    "lambda_id": tune["lambdas"][1],
                    "lambda_fr": tune["lambdas"][2],
                    "lambda_ns": tune["lambdas"][3],
                }
            )

    return rows


def run_approximation_variants(
    cfg,
    core_cfg,
    dataset_name,
    far,
    beta,
    base_score,
    base_rns,
    fractions,
):
    if not bool(cfg.approximation_check.enabled):
        return []
    allowed_fars = cfg.approximation_check.get("fars", None)
    if allowed_fars is not None:
        if not any(np.isclose(float(far), float(x)) for x in allowed_fars):
            return []
    dataset_cfg = get_dataset_cfg(core_cfg, dataset_name)
    base_method_cfg = get_method_cfg(core_cfg, cfg.mprisk_method_name)

    rows = []

    for variant in cfg.approximation_check.variants:
        overrides = cfg_to_container(variant.get("recognition_method", {}))
        method_cfg = apply_overrides(base_method_cfg, overrides)
        # Approximation variants with MC samples are memory-heavy.
        # Force conservative posterior batching unless explicitly specified.
        if "prob_batch_size" not in overrides:
            set_method_override(
                method_cfg,
                "prob_batch_size",
                int(cfg.approximation_check.get("prob_batch_size", 32)),
            )

        if "max_prob_elements" not in overrides:
            set_method_override(
                method_cfg,
                "max_prob_elements",
                int(cfg.approximation_check.get("max_prob_elements", 2_000_000)),
            )
        max_m = cfg.approximation_check.get("max_M", None)
        if max_m is not None:
            m_val = overrides.get("M", 0)
            if int(m_val) > int(max_m):
                print(
                    f"[approximation] skip {variant.name}: "
                    f"M={m_val} > max_M={max_m}"
                )
                continue
        entry = None
        test_dataset = None

        try:
            cuda_cleanup(f"before approximation variant {variant.name}")

            test_dataset = instantiate(dataset_cfg)

            entry = run_base_mprisk_safe(
                cfg=core_cfg,
                method_cfg=method_cfg,
                test_dataset=test_dataset,
                far=far,
                beta=beta,
                method_name_suffix=f"approx_{variant.name}",
            )

            comps = entry["test_components"]
            score = as_numpy_copy(component_score(comps, np.ones(4)))
            rns = as_numpy_copy(comps["r_ns"])

            eval_test = eval_uncertainty_score(
                score,
                entry["result"],
                fractions,
                seed=int(cfg.seed),
            )

            rows.append(
                {
                    "dataset": dataset_name,
                    "far": far,
                    "beta": beta,
                    "variant": str(variant.name),
                    "score_spearman_vs_base": safe_spearman(base_score, score),
                    "rns_spearman_vs_base": safe_spearman(base_rns, rns),
                    **eval_test,
                }
            )

        finally:
            entry = None
            test_dataset = None
            cuda_cleanup(f"after approximation variant {variant.name}")
    return rows


def _row_key_tuple(row: Dict[str, Any]) -> Tuple[str, float, float, str]:
    return (
        str(row["dataset"]),
        float(row["far"]),
        float(row["beta"]),
        str(row.get("method", row.get("variant", ""))),
    )


def load_existing_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except Exception:
        return []


def save_all_tables(
    tables_dir: Path,
    all_rows: List[Dict[str, Any]],
    all_weights: List[Dict[str, Any]],
    all_components: List[Dict[str, Any]],
    all_lambda_stability: List[Dict[str, Any]],
    all_approx: List[Dict[str, Any]],
) -> None:
    pd.DataFrame(all_rows).to_csv(
        tables_dir / "fair_tuning_comparison.csv",
        index=False,
    )
    pd.DataFrame(all_weights).to_csv(
        tables_dir / "fair_tuning_weights.csv",
        index=False,
    )
    pd.DataFrame(all_components).to_csv(
        tables_dir / "full_component_ablation.csv",
        index=False,
    )
    pd.DataFrame(all_lambda_stability).to_csv(
        tables_dir / "lambda_stability.csv",
        index=False,
    )
    pd.DataFrame(all_approx).to_csv(
        tables_dir / "approximation_check.csv",
        index=False,
    )


def done_marker_path(tables_dir: Path, dataset: str, far: float, beta: float) -> Path:
    done_dir = tables_dir / "_done"
    done_dir.mkdir(parents=True, exist_ok=True)
    return done_dir / f"{slugify(dataset)}_far_{far}_beta_{beta}.done"


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


@hydra.main(
    config_path=str(
        Path(__file__).resolve().parents[1] / "configs/uncertainty_benchmark"
    ),
    config_name="mprisk_fair_tuning_bio",
    version_base="1.2",
)
def main(cfg):
    if torch is not None:
        torch.set_grad_enabled(False)

    seed_everything(
        seed=int(cfg.seed),
        deterministic=bool(cfg.get("deterministic", True)),
    )

    exp_dir = Path(cfg.exp_dir)
    tables_dir = exp_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    all_rows = load_existing_rows(tables_dir / "fair_tuning_comparison.csv")
    all_weights = load_existing_rows(tables_dir / "fair_tuning_weights.csv")
    all_components = load_existing_rows(tables_dir / "full_component_ablation.csv")
    all_lambda_stability = load_existing_rows(tables_dir / "lambda_stability.csv")
    all_approx = load_existing_rows(tables_dir / "approximation_check.csv")
    core_cfg = load_core_cfg(
        path=str(cfg.core_config_path),
        exp_dir=str(exp_dir),
    )
    if "recompute_template_pooling" in cfg:
        core_cfg.recompute_template_pooling = bool(cfg.recompute_template_pooling)
    fractions = make_fraction_grid(cfg.rejection_fractions)

    # all_rows = []
    # all_weights = []
    # all_components = []
    # all_lambda_stability = []
    # all_approx = []

    for dataset_name in cfg.datasets:
        for far in cfg.far_list:
            for beta in cfg.beta_list:
                marker = done_marker_path(
                    tables_dir=tables_dir,
                    dataset=str(dataset_name),
                    far=float(far),
                    beta=float(beta),
                )

                if marker.is_file() and not bool(cfg.get("force_recompute", False)):
                    print(
                        f"[fair-tuning] skip completed "
                        f"dataset={dataset_name}, far={far}, beta={beta}"
                    )
                    continue
                print("=" * 100)
                print(f"[Fair tuning] dataset={dataset_name} far={far} beta={beta}")
                print("=" * 100)

                rows, weights, components, cache = run_one_dataset_far(
                    cfg=cfg,
                    core_cfg=core_cfg,
                    dataset_name=str(dataset_name),
                    far=float(far),
                    beta=float(beta),
                )

                all_rows.extend(rows)
                all_weights.extend(weights)
                all_components.extend(components)

                stability_rows = run_lambda_stability(
                    cfg=cfg,
                    dataset_name=str(dataset_name),
                    far=float(far),
                    beta=float(beta),
                    calib=cache["calib"],
                    fractions=fractions,
                )
                all_lambda_stability.extend(stability_rows)

                # Approximation variants do not need calibration.
                cache.pop("calib", None)
                cuda_cleanup("before approximation variants")

                base_score = cache["base_score"]
                base_rns = cache["base_rns"]

                approx_rows = run_approximation_variants(
                    cfg=cfg,
                    core_cfg=core_cfg,
                    dataset_name=str(dataset_name),
                    far=float(far),
                    beta=float(beta),
                    base_score=base_score,
                    base_rns=base_rns,
                    fractions=fractions,
                )
                all_approx.extend(approx_rows)
                save_all_tables(
                    tables_dir=tables_dir,
                    all_rows=all_rows,
                    all_weights=all_weights,
                    all_components=all_components,
                    all_lambda_stability=all_lambda_stability,
                    all_approx=all_approx,
                )

                marker.write_text("done\n")
                print(f"[fair-tuning] wrote done marker: {marker}")

    save_all_tables(
        tables_dir=tables_dir,
        all_rows=all_rows,
        all_weights=all_weights,
        all_components=all_components,
        all_lambda_stability=all_lambda_stability,
        all_approx=all_approx,
    )

    print("\nSaved:")
    print(tables_dir / "fair_tuning_comparison.csv")
    print(tables_dir / "fair_tuning_weights.csv")
    print(tables_dir / "full_component_ablation.csv")
    print(tables_dir / "lambda_stability.csv")
    print(tables_dir / "approximation_check.csv")


if __name__ == "__main__":
    main()
