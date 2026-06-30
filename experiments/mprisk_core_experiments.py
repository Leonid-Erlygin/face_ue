#!/usr/bin/env python3

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import OmegaConf

from evaluation.recognition_test import Recognition_test
from evaluation.reproducibility import seed_everything


try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception:
    roc_auc_score = None
    average_precision_score = None

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------


def slugify(x: str) -> str:
    x = str(x)
    x = x.replace("+", "_plus_")
    x = x.replace("@", "_at_")
    x = re.sub(r"[^a-zA-Z0-9_.-]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def as_np_1d(x: Any) -> np.ndarray:
    return np.asarray(x).reshape(-1)


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(bool).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)

    if roc_auc_score is None:
        return np.nan
    if len(np.unique(y_true)) < 2:
        return np.nan

    try:
        return float(roc_auc_score(y_true.astype(int), score))
    except Exception:
        return np.nan


def safe_auprc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(bool).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)

    if average_precision_score is None:
        return np.nan
    if len(np.unique(y_true)) < 2:
        return np.nan

    try:
        return float(average_precision_score(y_true.astype(int), score))
    except Exception:
        return np.nan


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if spearmanr is None:
        return np.nan
    if len(x) < 3:
        return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan

    try:
        return float(spearmanr(x, y).correlation)
    except Exception:
        return np.nan


def sampled_rank_inversion_rate(
    score_a: np.ndarray,
    score_b: np.ndarray,
    num_pairs: int = 200_000,
    seed: int = 777,
) -> float:
    """
    Pairwise rank inversion rate between two scores.

    Both score_a and score_b are assumed to be higher = more risky/uncertain.
    """
    score_a = np.asarray(score_a, dtype=np.float64).reshape(-1)
    score_b = np.asarray(score_b, dtype=np.float64).reshape(-1)

    n = len(score_a)
    if n < 2:
        return np.nan

    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=num_pairs)
    j = rng.integers(0, n, size=num_pairs)

    da = score_a[i] - score_a[j]
    db = score_b[i] - score_b[j]

    non_tie = np.logical_and(np.abs(da) > 1e-12, np.abs(db) > 1e-12)
    if not np.any(non_tie):
        return np.nan

    inv = (da[non_tie] * db[non_tie]) < 0
    return float(np.mean(inv))


def np_trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(
        np.trapezoid(np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64))
    )


# ---------------------------------------------------------------------
# OSR masks and metrics
# ---------------------------------------------------------------------


def compute_osr_error_masks(
    predicted_id: np.ndarray,
    was_rejected: np.ndarray,
    g_unique_ids: np.ndarray,
    probe_unique_ids: np.ndarray,
) -> Dict[str, np.ndarray]:
    predicted_id = np.asarray(predicted_id, dtype=int).reshape(-1)
    was_rejected = np.asarray(was_rejected, dtype=bool).reshape(-1)
    probe_unique_ids = np.asarray(probe_unique_ids).reshape(-1)
    g_unique_ids = np.asarray(g_unique_ids).reshape(-1)

    n = len(probe_unique_ids)
    is_seen = np.isin(probe_unique_ids, g_unique_ids)

    predicted_subject = np.full(n, -(10**18), dtype=np.int64)
    if len(g_unique_ids) > 0:
        predicted_subject = g_unique_ids[predicted_id]

    true_accept = np.logical_and(is_seen, ~was_rejected)
    true_reject = np.logical_and(~is_seen, was_rejected)

    true_accept_true_ident = np.logical_and(
        true_accept,
        predicted_subject == probe_unique_ids,
    )

    false_accept = np.logical_and(~is_seen, ~was_rejected)
    false_reject = np.logical_and(is_seen, was_rejected)
    misidentification = np.logical_and(
        true_accept,
        predicted_subject != probe_unique_ids,
    )

    correct = np.logical_or(true_accept_true_ident, true_reject)
    any_error = ~correct

    return {
        "is_seen": is_seen,
        "correct": correct,
        "any_error": any_error,
        "false_accept": false_accept,
        "false_reject": false_reject,
        "misidentification": misidentification,
        "true_reject": true_reject,
        "true_accept_true_ident": true_accept_true_ident,
    }


def f1_classic(
    predicted_id: np.ndarray,
    was_rejected: np.ndarray,
    g_unique_ids: np.ndarray,
    probe_unique_ids: np.ndarray,
) -> float:
    masks = compute_osr_error_masks(
        predicted_id,
        was_rejected,
        g_unique_ids,
        probe_unique_ids,
    )

    tp = int(np.sum(masks["true_accept_true_ident"]))
    fp = int(np.sum(masks["false_accept"]))
    fn = int(np.sum(masks["false_reject"]) + np.sum(masks["misidentification"]))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return float(2 * precision * recall / (precision + recall))


def fnir_fpir(
    predicted_id: np.ndarray,
    was_rejected: np.ndarray,
    g_unique_ids: np.ndarray,
    probe_unique_ids: np.ndarray,
) -> Tuple[float, float]:
    masks = compute_osr_error_masks(
        predicted_id,
        was_rejected,
        g_unique_ids,
        probe_unique_ids,
    )

    n_seen = int(np.sum(masks["is_seen"]))
    n_unknown = int(np.sum(~masks["is_seen"]))

    tp = int(np.sum(masks["true_accept_true_ident"]))
    fp = int(np.sum(masks["false_accept"]))

    fnir = 1.0 - tp / n_seen if n_seen > 0 else np.nan
    fpir = fp / n_unknown if n_unknown > 0 else np.nan

    return float(fnir), float(fpir)


def rejection_curve_for_score(
    uncertainty: np.ndarray,
    predicted_id: np.ndarray,
    was_rejected: np.ndarray,
    g_unique_ids: np.ndarray,
    probe_unique_ids: np.ndarray,
    fractions: np.ndarray,
) -> pd.DataFrame:
    """
    Repository convention:
      predicted_unc is sorted ascending.
      Small = confident, kept first.
      Large = uncertain, filtered first.
    """
    uncertainty = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    n = len(uncertainty)

    order = np.argsort(uncertainty)

    rows = []
    for frac in fractions:
        keep_count = int((1.0 - float(frac)) * n)
        keep_count = max(0, min(n, keep_count))
        keep_idx = order[:keep_count]

        pred_i = np.asarray(predicted_id)[keep_idx]
        rej_i = np.asarray(was_rejected)[keep_idx]
        probe_i = np.asarray(probe_unique_ids)[keep_idx]

        f1 = f1_classic(pred_i, rej_i, g_unique_ids, probe_i)
        fnir, fpir = fnir_fpir(pred_i, rej_i, g_unique_ids, probe_i)

        masks_i = compute_osr_error_masks(pred_i, rej_i, g_unique_ids, probe_i)

        rows.append(
            {
                "fraction": float(frac),
                "f1_class": f1,
                "fnir": fnir,
                "fpir": fpir,
                "error_rate": (
                    float(np.mean(masks_i["any_error"]))
                    if len(keep_idx) > 0
                    else np.nan
                ),
                "false_accept_count": int(np.sum(masks_i["false_accept"])),
                "false_reject_count": int(np.sum(masks_i["false_reject"])),
                "misidentification_count": int(np.sum(masks_i["misidentification"])),
                "kept_count": int(keep_count),
            }
        )

    return pd.DataFrame(rows)


def self_normalized_prr(
    uncertainty: np.ndarray,
    predicted_id: np.ndarray,
    was_rejected: np.ndarray,
    g_unique_ids: np.ndarray,
    probe_unique_ids: np.ndarray,
    fractions: np.ndarray,
    metric_name: str = "f1_class",
    seed: int = 777,
) -> Tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    masks = compute_osr_error_masks(
        predicted_id,
        was_rejected,
        g_unique_ids,
        probe_unique_ids,
    )

    rng = np.random.default_rng(seed)
    n = len(probe_unique_ids)

    random_score = rng.random(n)

    oracle_score = masks["any_error"].astype(np.float64)
    oracle_score = oracle_score + 1e-9 * rng.random(n)

    curve = rejection_curve_for_score(
        uncertainty,
        predicted_id,
        was_rejected,
        g_unique_ids,
        probe_unique_ids,
        fractions,
    )
    random_curve = rejection_curve_for_score(
        random_score,
        predicted_id,
        was_rejected,
        g_unique_ids,
        probe_unique_ids,
        fractions,
    )
    oracle_curve = rejection_curve_for_score(
        oracle_score,
        predicted_id,
        was_rejected,
        g_unique_ids,
        probe_unique_ids,
        fractions,
    )

    area = np_trapz(curve[metric_name].values, curve["fraction"].values)
    random_area = np_trapz(
        random_curve[metric_name].values, random_curve["fraction"].values
    )
    oracle_area = np_trapz(
        oracle_curve[metric_name].values, oracle_curve["fraction"].values
    )

    denom = oracle_area - random_area
    if abs(denom) < 1e-12:
        prr = np.nan
    else:
        prr = (area - random_area) / denom

    return float(prr), curve, random_curve, oracle_curve


# ---------------------------------------------------------------------
# Running existing repository methods and extracting internals
# ---------------------------------------------------------------------


def instantiate_list(query_list):
    return [instantiate(value) for value in query_list]


def get_used_galleries(tt: Recognition_test) -> List[str]:
    used = ["g1"]
    if (
        tt.use_two_galleries
        and tt.test_dataset is not None
        and tt.test_dataset.g2_templates.shape != ()
    ):
        used += ["g2"]
    return used


def maybe_attach_calibration_set(cfg, recognition_method, dataset_name: str):
    if (
        hasattr(recognition_method, "calibration_set")
        and recognition_method.calibration_set is True
    ):
        calib_set_cfg = getattr(cfg.dataset_name_to_calibration_set, dataset_name)
        recognition_method.calibration_set = instantiate(calib_set_cfg)


def maybe_attach_dataset_temperature(
    cfg, recognition_method, pretty_name: str, dataset_name: str
):
    if pretty_name == "GalUE" and hasattr(cfg, "dataset_name_to_T_scale"):
        recognition_method.predict_T = getattr(
            cfg.dataset_name_to_T_scale, dataset_name
        )


def build_tester(
    cfg,
    method_cfg,
    test_dataset,
    recognition_method,
    method_name: str,
    pretty_name: str,
) -> Recognition_test:
    gallery_template_pooling_strategy = instantiate(
        method_cfg.gallery_template_pooling_strategy
    )
    probe_template_pooling_strategy = instantiate(
        method_cfg.probe_template_pooling_strategy
    )

    dataset_name = test_dataset.dataset_name
    embeddings_path = (
        Path(test_dataset.dataset_path)
        / f"embeddings/{method_cfg.embeddings}_embs_{dataset_name}.npz"
    )

    tt = Recognition_test(
        task_type="open_set_identification",
        method_name=method_name,
        pretty_name=pretty_name,
        recognition_method=recognition_method,
        test_dataset=test_dataset,
        embedding_type=method_cfg.embeddings,
        embeddings_path=embeddings_path,
        gallery_template_pooling_strategy=gallery_template_pooling_strategy,
        probe_template_pooling_strategy=probe_template_pooling_strategy,
        use_two_galleries=cfg.use_two_galleries,
        recompute_template_pooling=cfg.recompute_template_pooling,
        recognition_metrics={"open_set_identification": []},
        uncertainty_metrics={"open_set_identification": []},
        use_detector_score=bool(method_cfg.get("use_detector_score", False)),
    )
    return tt


def run_method_raw(
    tt: Recognition_test,
    gallery_name: str = "g1",
) -> Dict[str, Any]:
    rm = tt.recognition_method

    probe_feats = tt.probe_pooled_templates[gallery_name]["template_pooled_features"]
    probe_unc = tt.probe_pooled_templates[gallery_name]["template_pooled_data_unc"]
    gallery_feats = tt.gallery_pooled_templates[gallery_name][
        "template_pooled_features"
    ]
    gallery_unc = tt.gallery_pooled_templates[gallery_name]["template_pooled_data_unc"]
    g_unique_ids = tt.gallery_pooled_templates[gallery_name][
        "template_subject_ids_sorted"
    ]
    probe_unique_ids = tt.probe_pooled_templates[gallery_name][
        "template_subject_ids_sorted"
    ]

    rm.setup(
        probe_feats,
        probe_unc,
        gallery_feats,
        gallery_unc,
        g_unique_ids=g_unique_ids,
        probe_unique_ids=probe_unique_ids,
        dataset_name=tt.test_dataset.dataset_name,
    )

    predicted_id, was_rejected = rm.predict()
    predicted_unc = rm.predict_uncertainty()

    predicted_id = np.asarray(predicted_id, dtype=int).reshape(-1)
    was_rejected = np.asarray(was_rejected, dtype=bool).reshape(-1)
    predicted_unc = np.asarray(predicted_unc, dtype=np.float64).reshape(-1)

    masks = compute_osr_error_masks(
        predicted_id,
        was_rejected,
        g_unique_ids,
        probe_unique_ids,
    )

    return {
        "recognition_method": rm,
        "predicted_id": predicted_id,
        "was_rejected": was_rejected,
        "predicted_unc": predicted_unc,
        "g_unique_ids": np.asarray(g_unique_ids),
        "probe_unique_ids": np.asarray(probe_unique_ids),
        "masks": masks,
        "gallery_name": gallery_name,
    }


def extract_method_arrays(recognition_method) -> Dict[str, np.ndarray]:
    """
    Extract useful internal arrays from MPRisk/HolUE/GalUE-like methods.
    Missing attributes are skipped.
    """
    names = [
        # MPRisk components
        "r_fa",
        "r_id",
        "r_fr",
        "r_ns",
        "risk_main",
        "risk_ns",
        "mprisk",
        "oog_prob",
        "oog_nonspecificity",
        # HolUE KL terms
        "kl_1",
        "kl_2",
        # posterior probabilities
        "mean_probs",
        "all_classes_log_prob",
        "similarity_matrix",
        "probe_score",
    ]

    out = {}
    for name in names:
        if hasattr(recognition_method, name):
            value = getattr(recognition_method, name)
            try:
                arr = np.asarray(value)
                if arr.size > 0:
                    out[name] = arr
            except Exception:
                pass

    return out


def extract_lambdas(recognition_method) -> Dict[str, float]:
    keys = ["lambda_fa", "lambda_id", "lambda_fr", "lambda_ns"]
    out = {}
    for k in keys:
        out[k] = float(getattr(recognition_method, k, np.nan))
    return out


# ---------------------------------------------------------------------
# Analysis blocks
# ---------------------------------------------------------------------


def error_type_detection_rows(
    dataset_name: str,
    pretty_name: str,
    far: float,
    beta: float,
    score: np.ndarray,
    masks: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    rows = []
    labels = {
        "any_error": masks["any_error"],
        "false_accept": masks["false_accept"],
        "false_reject": masks["false_reject"],
        "misidentification": masks["misidentification"],
    }

    for label_name, y in labels.items():
        rows.append(
            {
                "dataset": dataset_name,
                "method": pretty_name,
                "far": far,
                "beta": beta,
                "target": label_name,
                "positive_count": int(np.sum(y)),
                "total_count": int(len(y)),
                "auroc": safe_auc(y, score),
                "auprc": safe_auprc(y, score),
            }
        )

    return rows


def component_ablation_rows(
    dataset_name: str,
    pretty_name: str,
    far: float,
    beta: float,
    result: Dict[str, Any],
    method_arrays: Dict[str, np.ndarray],
    fractions: np.ndarray,
    seed: int,
) -> List[Dict[str, Any]]:
    needed = ["r_fa", "r_id", "r_fr", "r_ns"]
    if not all(k in method_arrays for k in needed):
        return []

    r_fa = as_np_1d(method_arrays["r_fa"])
    r_id = as_np_1d(method_arrays["r_id"])
    r_fr = as_np_1d(method_arrays["r_fr"])
    r_ns = as_np_1d(method_arrays["r_ns"])

    variants = {
        "FA only": r_fa,
        "ID only": r_id,
        "FR only": r_fr,
        "NS only": r_ns,
        "ordinary equal": r_fa + r_id + r_fr,
        "full equal": r_fa + r_id + r_fr + r_ns,
    }

    if "risk_main" in method_arrays:
        variants["ordinary current"] = as_np_1d(method_arrays["risk_main"])
    if "mprisk" in method_arrays:
        variants["MPRisk current"] = as_np_1d(method_arrays["mprisk"])
    if "risk_ns" in method_arrays:
        variants["NS current"] = as_np_1d(method_arrays["risk_ns"])

    rows = []
    masks = result["masks"]

    for variant_name, score in variants.items():
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

        rows.append(
            {
                "dataset": dataset_name,
                "source_method": pretty_name,
                "variant": variant_name,
                "far": far,
                "beta": beta,
                "prr_f1": prr,
                "error_auroc": safe_auc(masks["any_error"], score),
                "error_auprc": safe_auprc(masks["any_error"], score),
                "f1_at_0_filter": float(curve["f1_class"].iloc[0]),
                "f1_at_max_filter": float(curve["f1_class"].iloc[-1]),
            }
        )

    return rows


def kl_rank_inversion_rows(
    dataset_name: str,
    pretty_name: str,
    far: float,
    beta: float,
    result: Dict[str, Any],
    method_arrays: Dict[str, np.ndarray],
    seed: int,
    num_pairs: int,
) -> List[Dict[str, Any]]:
    if "kl_1" not in method_arrays or "kl_2" not in method_arrays:
        return []

    kl_1 = as_np_1d(method_arrays["kl_1"])
    kl_2 = as_np_1d(method_arrays["kl_2"])

    # KL is information gain / confidence. Convert to uncertainty:
    # larger value should mean more risky.
    kl_unc = -(kl_1 + kl_2)

    rows = []

    refs = {
        "empirical_error": result["masks"]["any_error"].astype(np.float64),
        "method_uncertainty": as_np_1d(result["predicted_unc"]),
    }

    if "mprisk" in method_arrays:
        refs["mprisk"] = as_np_1d(method_arrays["mprisk"])
    if "risk_main" in method_arrays:
        refs["ordinary_risk"] = as_np_1d(method_arrays["risk_main"])

    for ref_name, ref_score in refs.items():
        rows.append(
            {
                "dataset": dataset_name,
                "method": pretty_name,
                "far": far,
                "beta": beta,
                "kl_reference": ref_name,
                "spearman": safe_spearman(kl_unc, ref_score),
                "rank_inversion_rate": sampled_rank_inversion_rate(
                    kl_unc,
                    ref_score,
                    num_pairs=num_pairs,
                    seed=seed,
                ),
                "kl_error_auroc": safe_auc(result["masks"]["any_error"], kl_unc),
                "kl_error_auprc": safe_auprc(result["masks"]["any_error"], kl_unc),
            }
        )

    return rows


def save_kl_scatter(
    out_dir: Path,
    dataset_name: str,
    pretty_name: str,
    far: float,
    method_arrays: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    max_points: int = 5000,
    seed: int = 777,
):
    if "kl_1" not in method_arrays or "kl_2" not in method_arrays:
        return
    if "mprisk" not in method_arrays:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    kl_unc = -(as_np_1d(method_arrays["kl_1"]) + as_np_1d(method_arrays["kl_2"]))
    risk = as_np_1d(method_arrays["mprisk"])

    n = len(kl_unc)
    rng = np.random.default_rng(seed)

    if n > max_points:
        idx = rng.choice(np.arange(n), size=max_points, replace=False)
    else:
        idx = np.arange(n)

    error_kind = np.full(n, "correct", dtype=object)
    error_kind[masks["false_accept"]] = "false_accept"
    error_kind[masks["false_reject"]] = "false_reject"
    error_kind[masks["misidentification"]] = "misidentification"

    colors = {
        "correct": "tab:green",
        "false_accept": "tab:red",
        "false_reject": "tab:orange",
        "misidentification": "tab:blue",
    }

    plt.figure(figsize=(7, 5))

    for kind, color in colors.items():
        mask = error_kind[idx] == kind
        if np.any(mask):
            plt.scatter(
                kl_unc[idx][mask],
                risk[idx][mask],
                s=8,
                alpha=0.55,
                label=kind,
                c=color,
                linewidths=0,
            )

    plt.xlabel("Raw KL uncertainty: -(KL1 + KL2)")
    plt.ylabel("MPRisk")
    plt.title(f"{pretty_name}, {dataset_name}, FPIR={far}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(markerscale=2, fontsize=8)
    plt.tight_layout()

    file_name = f"{slugify(dataset_name)}_{slugify(pretty_name)}_far_{far}_kl_vs_mprisk"
    plt.savefig(out_dir / f"{file_name}.png", dpi=300)
    plt.savefig(out_dir / f"{file_name}.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def save_per_example_npz(
    out_path: Path,
    result: Dict[str, Any],
    method_arrays: Dict[str, np.ndarray],
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "predicted_id": result["predicted_id"],
        "was_rejected": result["was_rejected"],
        "predicted_unc": result["predicted_unc"],
        "g_unique_ids": result["g_unique_ids"],
        "probe_unique_ids": result["probe_unique_ids"],
    }

    for k, v in result["masks"].items():
        save_dict[f"mask_{k}"] = np.asarray(v)

    for k, v in method_arrays.items():
        save_dict[k] = np.asarray(v)

    np.savez_compressed(out_path, **save_dict)


def save_main_rejection_curve_plots(
    all_curves: Dict[Tuple[str, float, float, str], pd.DataFrame],
    all_prr: Dict[Tuple[str, float, float, str], float],
    reference_curves: Dict[Tuple[str, float, float], Dict[str, pd.DataFrame]],
    cfg,
    out_dir: Path,
):
    """
    Save original-paper-style rejection curves.

    Includes:
      - method curves;
      - Random curve;
      - Oracle curve;
      - PRR value in legend.

    `all_curves` contains method curves:
        (dataset, far, beta, method) -> curve

    `all_prr` contains method PRR values:
        (dataset, far, beta, method) -> PRR

    `reference_curves` contains Random/Oracle curves:
        (dataset, far, beta) -> {"Random": df, "Oracle": df}
    """
    try:
        enabled = bool(cfg.plots.rejection_curves.enabled)
    except Exception:
        enabled = True

    if not enabled:
        return

    try:
        metrics = list(cfg.plots.rejection_curves.metrics)
    except Exception:
        metrics = ["f1_class", "fpir", "fnir"]

    try:
        method_order = list(cfg.plots.rejection_curves.method_order)
    except Exception:
        method_order = []

    try:
        display_random_curve = bool(cfg.plots.rejection_curves.display_random_curve)
    except Exception:
        display_random_curve = True

    try:
        display_oracle_curve = bool(cfg.plots.rejection_curves.display_oracle_curve)
    except Exception:
        display_oracle_curve = True

    try:
        prr_in_legend = bool(cfg.plots.rejection_curves.prr_in_legend)
    except Exception:
        prr_in_legend = True

    try:
        legend_fontsize = float(cfg.plots.rejection_curves.legend_fontsize)
    except Exception:
        legend_fontsize = 8.0

    try:
        figsize = tuple(cfg.plots.rejection_curves.figsize)
    except Exception:
        figsize = (7.0, 4.8)

    plot_dir = out_dir / "rejection_curves"
    plot_dir.mkdir(parents=True, exist_ok=True)

    groups = sorted(
        set((dataset, far, beta) for dataset, far, beta, _ in all_curves.keys())
    )

    metric_labels = {
        "f1_class": "$F_1$",
        "fpir": "FPIR",
        "fnir": "FNIR",
        "error_rate": "Error rate",
        "false_accept_count": "False accept count",
        "false_reject_count": "False reject count",
        "misidentification_count": "Misidentification count",
    }

    for dataset, far, beta in groups:
        group_key = (dataset, far, beta)

        group_methods = [
            method
            for d, f, b, method in all_curves.keys()
            if d == dataset and f == far and b == beta
        ]

        ordered_methods = [m for m in method_order if m in group_methods]
        ordered_methods += [m for m in group_methods if m not in ordered_methods]

        group_dir = plot_dir / slugify(dataset) / f"far_{far}_beta_{beta}"
        group_dir.mkdir(parents=True, exist_ok=True)

        refs = reference_curves.get(group_key, {})
        random_curve = refs.get("Random", None)
        oracle_curve = refs.get("Oracle", None)

        # --------------------------------------------------------------
        # Save all curves into one CSV, including Random and Oracle.
        # --------------------------------------------------------------
        curve_rows = []

        if display_random_curve and random_curve is not None:
            tmp = random_curve.copy()
            tmp.insert(0, "method", "Random")
            tmp.insert(0, "prr_f1", 0.0)
            tmp.insert(0, "beta", beta)
            tmp.insert(0, "far", far)
            tmp.insert(0, "dataset", dataset)
            curve_rows.append(tmp)

        if display_oracle_curve and oracle_curve is not None:
            tmp = oracle_curve.copy()
            tmp.insert(0, "method", "Oracle")
            tmp.insert(0, "prr_f1", 1.0)
            tmp.insert(0, "beta", beta)
            tmp.insert(0, "far", far)
            tmp.insert(0, "dataset", dataset)
            curve_rows.append(tmp)

        for method in ordered_methods:
            curve = all_curves[(dataset, far, beta, method)]
            prr_value = all_prr.get((dataset, far, beta, method), np.nan)

            tmp = curve.copy()
            tmp.insert(0, "method", method)
            tmp.insert(0, "prr_f1", prr_value)
            tmp.insert(0, "beta", beta)
            tmp.insert(0, "far", far)
            tmp.insert(0, "dataset", dataset)
            curve_rows.append(tmp)

        if len(curve_rows) > 0:
            pd.concat(curve_rows, ignore_index=True).to_csv(
                group_dir / "all_rejection_curves.csv",
                index=False,
            )

        # Also save a compact PRR table for the plotted methods.
        prr_rows = []
        if display_random_curve and random_curve is not None:
            prr_rows.append({"method": "Random", "prr_f1": 0.0})
        if display_oracle_curve and oracle_curve is not None:
            prr_rows.append({"method": "Oracle", "prr_f1": 1.0})
        for method in ordered_methods:
            prr_rows.append(
                {
                    "method": method,
                    "prr_f1": all_prr.get((dataset, far, beta, method), np.nan),
                }
            )

        pd.DataFrame(prr_rows).to_csv(group_dir / "prr_values.csv", index=False)

        # --------------------------------------------------------------
        # Draw one plot per metric.
        # --------------------------------------------------------------
        for metric in metrics:
            plt.figure(figsize=figsize)

            # Plot Random first.
            if (
                display_random_curve
                and random_curve is not None
                and metric in random_curve.columns
            ):
                label = "Random"
                if prr_in_legend:
                    label += ", PRR=0.00"

                plt.plot(
                    random_curve["fraction"].values,
                    random_curve[metric].values,
                    label=label,
                    linewidth=2.0,
                    alpha=0.8,
                    color="gray",
                    linestyle="--",
                )

            # Plot Oracle second.
            if (
                display_oracle_curve
                and oracle_curve is not None
                and metric in oracle_curve.columns
            ):
                label = "Oracle"
                if prr_in_legend:
                    label += ", PRR=1.00"

                plt.plot(
                    oracle_curve["fraction"].values,
                    oracle_curve[metric].values,
                    label=label,
                    linewidth=2.0,
                    alpha=0.8,
                    color="black",
                    linestyle="--",
                )

            # Plot actual methods.
            for method in ordered_methods:
                curve = all_curves[(dataset, far, beta, method)]

                if metric not in curve.columns:
                    continue

                prr_value = all_prr.get((dataset, far, beta, method), np.nan)

                if prr_in_legend and np.isfinite(prr_value):
                    label = f"{method}, PRR={prr_value:.2f}"
                else:
                    label = method

                linewidth = 2.8 if ("MPRisk" in method or "HolUE" in method) else 1.8
                alpha = 0.95

                plt.plot(
                    curve["fraction"].values,
                    curve[metric].values,
                    label=label,
                    linewidth=linewidth,
                    alpha=alpha,
                )

            plt.xlabel("Filtered-out probe fraction")
            plt.ylabel(metric_labels.get(metric, metric))
            # plt.title(f"{dataset}, FPIR={far}, beta={beta}")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.legend(fontsize=legend_fontsize)
            plt.tight_layout()

            out_stem = group_dir / f"{metric}_rejection_curve"
            plt.savefig(out_stem.with_suffix(".png"), dpi=300)
            plt.savefig(out_stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
            plt.close()

            print(f"[saved] {out_stem.with_suffix('.png')}")
            print(f"[saved] {out_stem.with_suffix('.pdf')}")


# ---------------------------------------------------------------------
# Main Hydra entry
# ---------------------------------------------------------------------


@hydra.main(
    config_path=str(
        Path(__file__).resolve().parents[1] / "configs/uncertainty_benchmark"
    ),
    config_name="mprisk_core_experiments",
    version_base="1.2",
)
def main(cfg):
    seed_everything(
        seed=int(cfg.get("seed", 777)),
        deterministic=bool(cfg.get("deterministic", True)),
    )

    exp_dir = Path(cfg.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    tables_dir = exp_dir / "tables"
    curves_dir = exp_dir / "curves"
    arrays_dir = exp_dir / "per_example_npz"
    plots_dir = exp_dir / "plots"

    for d in [tables_dir, curves_dir, arrays_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    fractions = np.linspace(
        float(cfg.rejection_fractions[0]),
        float(cfg.rejection_fractions[1]),
        int(cfg.rejection_fractions[2]),
    )

    test_datasets = instantiate_list(cfg.test_datasets)

    main_rows = []
    error_type_rows = []
    component_rows = []
    inversion_rows = []

    # Curves for paper-style plots.
    all_curves = {}
    all_prr = {}
    reference_curves = {}
    for test_dataset in test_datasets:
        dataset_name = test_dataset.dataset_name

        for method_cfg in cfg.open_set_identification_methods:
            for far in cfg.far_list:
                for beta in cfg.beta_list:
                    pretty_name = str(method_cfg.pretty_name)
                    if len(cfg.beta_list) > 1:
                        pretty_name = f"{pretty_name}_beta-{beta}"

                    method_name = (
                        f"{slugify(pretty_name)}"
                        f"_dataset_{slugify(dataset_name)}"
                        f"_far_{far}"
                        f"_beta_{beta}"
                    )

                    print("=" * 100)
                    print(
                        f"[MPRiskCore] dataset={dataset_name} method={pretty_name} far={far} beta={beta}"
                    )
                    print("=" * 100)

                    recognition_method = instantiate(method_cfg.recognition_method)

                    maybe_attach_calibration_set(cfg, recognition_method, dataset_name)
                    maybe_attach_dataset_temperature(
                        cfg,
                        recognition_method,
                        str(method_cfg.pretty_name),
                        dataset_name,
                    )

                    recognition_method.far = far
                    recognition_method.beta = beta

                    tt = build_tester(
                        cfg=cfg,
                        method_cfg=method_cfg,
                        test_dataset=test_dataset,
                        recognition_method=recognition_method,
                        method_name=method_name,
                        pretty_name=pretty_name,
                    )

                    used_galleries = get_used_galleries(tt)
                    if len(used_galleries) != 1:
                        raise NotImplementedError(
                            "This core script currently expects use_two_galleries=False."
                        )

                    result = run_method_raw(tt, gallery_name=used_galleries[0])
                    rm = result["recognition_method"]
                    method_arrays = extract_method_arrays(rm)
                    lambdas = extract_lambdas(rm)

                    save_per_example_npz(
                        arrays_dir / f"{method_name}.npz",
                        result=result,
                        method_arrays=method_arrays,
                    )

                    prr, curve, random_curve, oracle_curve = self_normalized_prr(
                        uncertainty=result["predicted_unc"],
                        predicted_id=result["predicted_id"],
                        was_rejected=result["was_rejected"],
                        g_unique_ids=result["g_unique_ids"],
                        probe_unique_ids=result["probe_unique_ids"],
                        fractions=fractions,
                        metric_name="f1_class",
                        seed=int(cfg.get("seed", 777)),
                    )

                    curve_out_dir = (
                        curves_dir
                        / slugify(dataset_name)
                        / slugify(pretty_name)
                        / f"far_{far}"
                    )
                    curve_out_dir.mkdir(parents=True, exist_ok=True)
                    curve.to_csv(curve_out_dir / "curve.csv", index=False)
                    random_curve.to_csv(curve_out_dir / "random_curve.csv", index=False)
                    oracle_curve.to_csv(curve_out_dir / "oracle_curve.csv", index=False)
                    group_key = (dataset_name, float(far), float(beta))
                    method_curve_key = (
                        dataset_name,
                        float(far),
                        float(beta),
                        pretty_name,
                    )

                    all_curves[method_curve_key] = curve
                    all_prr[method_curve_key] = float(prr)

                    # Store one Random/Oracle reference pair per dataset/far/beta.
                    # If a reference_method is specified, prefer its references.
                    try:
                        reference_method = str(
                            cfg.plots.rejection_curves.reference_method
                        )
                    except Exception:
                        reference_method = None

                    if group_key not in reference_curves or (
                        reference_method is not None and pretty_name == reference_method
                    ):
                        reference_curves[group_key] = {
                            "Random": random_curve,
                            "Oracle": oracle_curve,
                        }

                    masks = result["masks"]
                    fnir, fpir = fnir_fpir(
                        result["predicted_id"],
                        result["was_rejected"],
                        result["g_unique_ids"],
                        result["probe_unique_ids"],
                    )
                    f1 = f1_classic(
                        result["predicted_id"],
                        result["was_rejected"],
                        result["g_unique_ids"],
                        result["probe_unique_ids"],
                    )

                    main_rows.append(
                        {
                            "dataset": dataset_name,
                            "method": pretty_name,
                            "far": far,
                            "beta": beta,
                            "prr_f1": prr,
                            "base_f1_class": f1,
                            "base_fnir": fnir,
                            "base_fpir": fpir,
                            "error_rate": float(np.mean(masks["any_error"])),
                            "false_accept_count": int(np.sum(masks["false_accept"])),
                            "false_reject_count": int(np.sum(masks["false_reject"])),
                            "misidentification_count": int(
                                np.sum(masks["misidentification"])
                            ),
                            "error_auroc": safe_auc(
                                masks["any_error"], result["predicted_unc"]
                            ),
                            "error_auprc": safe_auprc(
                                masks["any_error"], result["predicted_unc"]
                            ),
                            **lambdas,
                        }
                    )

                    error_type_rows.extend(
                        error_type_detection_rows(
                            dataset_name=dataset_name,
                            pretty_name=pretty_name,
                            far=far,
                            beta=beta,
                            score=result["predicted_unc"],
                            masks=masks,
                        )
                    )

                    component_rows.extend(
                        component_ablation_rows(
                            dataset_name=dataset_name,
                            pretty_name=pretty_name,
                            far=far,
                            beta=beta,
                            result=result,
                            method_arrays=method_arrays,
                            fractions=fractions,
                            seed=int(cfg.get("seed", 777)),
                        )
                    )

                    inversion_rows.extend(
                        kl_rank_inversion_rows(
                            dataset_name=dataset_name,
                            pretty_name=pretty_name,
                            far=far,
                            beta=beta,
                            result=result,
                            method_arrays=method_arrays,
                            seed=int(cfg.get("seed", 777)),
                            num_pairs=int(cfg.get("rank_inversion_num_pairs", 200000)),
                        )
                    )

                    if bool(cfg.plots.kl_scatter.enabled):
                        save_kl_scatter(
                            out_dir=plots_dir / "kl_vs_mprisk",
                            dataset_name=dataset_name,
                            pretty_name=pretty_name,
                            far=far,
                            method_arrays=method_arrays,
                            masks=masks,
                            max_points=int(cfg.plots.kl_scatter.max_points),
                            seed=int(cfg.get("seed", 777)),
                        )

    main_df = pd.DataFrame(main_rows)
    error_type_df = pd.DataFrame(error_type_rows)
    component_df = pd.DataFrame(component_rows)
    inversion_df = pd.DataFrame(inversion_rows)

    main_df.to_csv(tables_dir / "main_mprisk_core_comparison.csv", index=False)
    error_type_df.to_csv(tables_dir / "error_type_detection.csv", index=False)
    component_df.to_csv(tables_dir / "mprisk_component_ablation.csv", index=False)
    inversion_df.to_csv(tables_dir / "kl_rank_inversion.csv", index=False)
    save_main_rejection_curve_plots(
        all_curves=all_curves,
        all_prr=all_prr,
        reference_curves=reference_curves,
        cfg=cfg,
        out_dir=plots_dir,
    )
    print("\nSaved:")
    print(tables_dir / "main_mprisk_core_comparison.csv")
    print(tables_dir / "error_type_detection.csv")
    print(tables_dir / "mprisk_component_ablation.csv")
    print(tables_dir / "kl_rank_inversion.csv")
    print(f"\nExperiment directory: {exp_dir}")


if __name__ == "__main__":
    main()
