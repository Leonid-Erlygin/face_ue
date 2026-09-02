#!/usr/bin/env python3
"""Geometric SCF calibration and HolUE fusion ablations on dissertation OSR data.

This runner intentionally uses only the repository's existing biometric/text OSR
protocols.  It does not depend on the modern-AI/tool-routing datasets.

Main questions:
  1. Does SCF concentration predict held-out true-class angular geometry?
  2. Is there a transferable global scale s such that A_d(s*kappa) matches that
     geometry better on unseen test identities/classes?
  3. If kappa is geometrically calibrated without OSR error labels, how much of
     HolUE's supervised MLP fusion is still needed?
  4. How does the same scale change the reject non-specificity diagnostic r_NS?
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import OmegaConf

from evaluation.open_set_methods.kappa_utils import (
    fit_vmf_kappa_scale,
    vmf_kappa_scale_nll,
    vmf_mean_resultant_np,
    vmf_nonspecificity_np,
)
from evaluation.reproducibility import seed_everything
from experiments.mprisk_core_experiments import (
    build_tester,
    extract_method_arrays,
    f1_classic,
    fnir_fpir,
    maybe_attach_calibration_set,
    run_method_raw,
    safe_auc,
    safe_auprc,
    self_normalized_prr,
)


def slugify(value: str) -> str:
    value = str(value).replace("+", "_plus_").replace("@", "_at_")
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def copy_cfg(node):
    return OmegaConf.create(OmegaConf.to_container(node, resolve=True))


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2-D feature matrix, got {x.shape}")
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(~np.isfinite(norm)) or np.any(norm <= 0):
        raise FloatingPointError("Non-finite or zero-norm embedding encountered")
    return x / norm


def _id_key(x: Any):
    """Convert NumPy scalar IDs to stable Python dictionary keys."""
    return x.item() if isinstance(x, np.generic) else x


def _template_subject_map(
    protocol_templates: np.ndarray,
    protocol_subject_ids: np.ndarray,
) -> Dict[Any, Any]:
    templates = np.asarray(protocol_templates).reshape(-1)
    subjects = np.asarray(protocol_subject_ids).reshape(-1)
    if len(templates) != len(subjects):
        raise ValueError("Protocol template/subject arrays have different lengths")

    mapping: Dict[Any, Any] = {}
    for template, subject in zip(templates, subjects):
        tk, sk = _id_key(template), _id_key(subject)
        if tk in mapping and mapping[tk] != sk:
            raise ValueError(f"Template {tk!r} maps to multiple subject IDs")
        mapping[tk] = sk
    return mapping


def _gallery_proxy_map(
    gallery_features: np.ndarray,
    gallery_subject_ids: np.ndarray,
) -> Dict[Any, np.ndarray]:
    features = _normalize_rows(gallery_features)
    subject_ids = np.asarray(gallery_subject_ids).reshape(-1)
    if len(features) != len(subject_ids):
        raise ValueError("Gallery features/subject IDs have different lengths")

    grouped: Dict[Any, List[np.ndarray]] = {}
    for feature, subject in zip(features, subject_ids):
        grouped.setdefault(_id_key(subject), []).append(feature)

    proxies: Dict[Any, np.ndarray] = {}
    for subject, rows in grouped.items():
        mean = np.mean(np.stack(rows, axis=0), axis=0)
        norm = np.linalg.norm(mean)
        if not np.isfinite(norm) or norm <= 0:
            raise FloatingPointError(f"Invalid gallery proxy for subject {subject!r}")
        proxies[subject] = mean / norm
    return proxies


def _decoded_scf_kappa(tt) -> np.ndarray:
    kappa = np.asarray(tt._decode_uncertainty(tt.unc), dtype=np.float64)
    if kappa.ndim == 1:
        kappa = kappa[:, None]
    if kappa.ndim != 2 or kappa.shape[1] != 1:
        raise ValueError(
            "Geometric SCF experiment requires one scalar concentration per sample; "
            f"got uncertainty shape {kappa.shape}."
        )
    kappa = kappa[:, 0]
    if np.any(~np.isfinite(kappa)) or np.any(kappa <= 0):
        raise FloatingPointError("SCF kappa must be finite and strictly positive")
    return kappa


def extract_sample_geometry(tt, gallery_name: str = "g1") -> pd.DataFrame:
    """Pair raw probe samples with an independent enrolled true-class proxy.

    The SCF stationarity relation is a *sample-level* statement.  Therefore we
    deliberately use unpooled sample embeddings/kappas here, while the true-class
    direction comes from the independent gallery enrollment template.
    """
    dataset = tt.test_dataset
    gallery = tt.gallery_pooled_templates[gallery_name]
    gallery_features = np.asarray(gallery["template_pooled_features"])
    gallery_ids = np.asarray(gallery["template_subject_ids_sorted"]).reshape(-1)
    proxies = _gallery_proxy_map(gallery_features, gallery_ids)

    probe_map = _template_subject_map(dataset.probe_templates, dataset.probe_ids)
    sample_templates = np.asarray(dataset.templates).reshape(-1)
    sample_features = _normalize_rows(tt.image_input_feats)
    sample_kappa = _decoded_scf_kappa(tt)

    rows = []
    gallery_matrix = _normalize_rows(gallery_features)
    for idx, (template, feature, kappa) in enumerate(
        zip(sample_templates, sample_features, sample_kappa)
    ):
        subject = probe_map.get(_id_key(template))
        if subject is None or subject not in proxies:
            continue
        proxy = proxies[subject]
        cosine = float(np.clip(np.dot(feature, proxy), -1.0, 1.0))
        nearest = int(np.argmax(gallery_matrix @ feature))
        nearest_subject = _id_key(gallery_ids[nearest])
        rows.append(
            {
                "row_index": int(idx),
                "template_id": _id_key(template),
                "subject_id": subject,
                "kappa": float(kappa),
                "true_proxy_cosine": cosine,
                "positive_cosine_target": float(np.clip(cosine, 0.0, 1.0)),
                "nearest_gallery_subject_id": nearest_subject,
                "nearest_gallery_correct": bool(nearest_subject == subject),
            }
        )

    if not rows:
        raise ValueError(
            f"No seen raw probe samples could be matched to gallery proxies for "
            f"{dataset.dataset_name}."
        )
    return pd.DataFrame(rows)


def extract_template_geometry(tt, gallery_name: str = "g1") -> pd.DataFrame:
    """Deployment-level diagnostic using the pooled quantities consumed by HolUE."""
    gallery = tt.gallery_pooled_templates[gallery_name]
    probe = tt.probe_pooled_templates[gallery_name]

    gallery_features = np.asarray(gallery["template_pooled_features"])
    gallery_ids = np.asarray(gallery["template_subject_ids_sorted"]).reshape(-1)
    probe_features = _normalize_rows(probe["template_pooled_features"])
    probe_ids = np.asarray(probe["template_subject_ids_sorted"]).reshape(-1)
    probe_kappa = np.asarray(probe["template_pooled_data_unc"], dtype=np.float64)
    if probe_kappa.ndim == 1:
        probe_kappa = probe_kappa[:, None]
    if probe_kappa.shape[1] != 1:
        raise ValueError("Template-level geometric calibration requires scalar SCF kappa")
    probe_kappa = probe_kappa[:, 0]

    proxies = _gallery_proxy_map(gallery_features, gallery_ids)
    gallery_matrix = _normalize_rows(gallery_features)

    rows = []
    for idx, (feature, kappa, subject) in enumerate(
        zip(probe_features, probe_kappa, probe_ids)
    ):
        sk = _id_key(subject)
        if sk not in proxies:
            continue
        cosine = float(np.clip(np.dot(feature, proxies[sk]), -1.0, 1.0))
        nearest = int(np.argmax(gallery_matrix @ feature))
        nearest_subject = _id_key(gallery_ids[nearest])
        rows.append(
            {
                "row_index": int(idx),
                "subject_id": sk,
                "kappa": float(kappa),
                "true_proxy_cosine": cosine,
                "positive_cosine_target": float(np.clip(cosine, 0.0, 1.0)),
                "nearest_gallery_subject_id": nearest_subject,
                "nearest_gallery_correct": bool(nearest_subject == sk),
            }
        )

    if not rows:
        raise ValueError(
            f"No seen pooled probes could be matched to gallery proxies for "
            f"{tt.test_dataset.dataset_name}."
        )
    return pd.DataFrame(rows)


def reliability_bins(
    kappa: np.ndarray,
    cosine: np.ndarray,
    *,
    scale: float,
    embedding_dim: int,
    num_bins: int,
) -> pd.DataFrame:
    kappa = np.asarray(kappa, dtype=np.float64).reshape(-1)
    cosine = np.asarray(cosine, dtype=np.float64).reshape(-1)
    rho = vmf_mean_resultant_np(scale * kappa, d=embedding_dim)
    target = np.clip(cosine, 0.0, 1.0)

    # Equal-count bins are much more informative than fixed-width bins because
    # high-dimensional SCF outputs often occupy a narrow high-confidence range.
    order = np.argsort(rho)
    chunks = np.array_split(order, min(int(num_bins), len(order)))
    rows = []
    for bin_index, idx in enumerate(chunks, start=1):
        if len(idx) == 0:
            continue
        rows.append(
            {
                "bin": int(bin_index),
                "count": int(len(idx)),
                "rho_mean": float(np.mean(rho[idx])),
                "rho_min": float(np.min(rho[idx])),
                "rho_max": float(np.max(rho[idx])),
                "cosine_mean": float(np.mean(cosine[idx])),
                "positive_cosine_mean": float(np.mean(target[idx])),
                "absolute_gap": float(abs(np.mean(rho[idx]) - np.mean(target[idx]))),
            }
        )
    return pd.DataFrame(rows)


def geometry_metrics(
    points: pd.DataFrame,
    *,
    scale: float,
    embedding_dim: int,
    num_bins: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    kappa = points["kappa"].to_numpy(dtype=np.float64)
    cosine = points["true_proxy_cosine"].to_numpy(dtype=np.float64)
    target = np.clip(cosine, 0.0, 1.0)
    rho = vmf_mean_resultant_np(scale * kappa, d=embedding_dim)

    bins = reliability_bins(
        kappa,
        cosine,
        scale=scale,
        embedding_dim=embedding_dim,
        num_bins=num_bins,
    )
    total = max(int(bins["count"].sum()), 1)
    gce = float(np.sum(bins["count"] * bins["absolute_gap"]) / total)

    if len(kappa) > 1 and np.std(np.log(kappa)) > 0 and np.std(cosine) > 0:
        from scipy.stats import spearmanr

        spearman = float(spearmanr(np.log(kappa), cosine).statistic)
    else:
        spearman = float("nan")

    # Linear diagnostic only; it is not used to calibrate or tune the model.
    if len(rho) >= 2 and np.std(rho) > 1e-12:
        slope, intercept = np.polyfit(rho, target, deg=1)
    else:
        slope, intercept = np.nan, np.nan

    metrics = {
        "count": int(len(kappa)),
        "scale": float(scale),
        "mean_kappa": float(np.mean(kappa)),
        "median_kappa": float(np.median(kappa)),
        "mean_rho": float(np.mean(rho)),
        "mean_true_cosine": float(np.mean(cosine)),
        "mean_positive_cosine_target": float(np.mean(target)),
        "negative_cosine_fraction": float(np.mean(cosine < 0.0)),
        "mae_stationarity": float(np.mean(np.abs(rho - target))),
        "rmse_stationarity": float(np.sqrt(np.mean((rho - target) ** 2))),
        "mean_signed_stationarity": float(np.mean(rho - target)),
        "geometric_calibration_error": gce,
        "spearman_log_kappa_vs_true_cosine": spearman,
        "nearest_gallery_accuracy": float(
            np.mean(points["nearest_gallery_correct"].to_numpy(dtype=bool))
        ),
        "reliability_slope": float(slope),
        "reliability_intercept": float(intercept),
        "vmf_nll": vmf_kappa_scale_nll(scale, kappa, cosine, d=embedding_dim),
    }
    return metrics, bins


def save_reliability_plot(
    raw_bins: pd.DataFrame,
    calibrated_bins: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="ideal")
    ax.plot(
        raw_bins["rho_mean"],
        raw_bins["positive_cosine_mean"],
        marker="o",
        label="raw $\\kappa$",
    )
    ax.plot(
        calibrated_bins["rho_mean"],
        calibrated_bins["positive_cosine_mean"],
        marker="o",
        label="geometrically calibrated $\\kappa$",
    )
    ax.set_xlabel(r"Predicted mean resultant length $A_d(\kappa)$")
    ax.set_ylabel("Observed true-proxy cosine (positive target)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=250)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def dataset_fars(cfg, dataset_name: str) -> List[float]:
    if "dataset_far_lists" in cfg and dataset_name in cfg.dataset_far_lists:
        return [float(x) for x in cfg.dataset_far_lists[dataset_name]]
    return [float(x) for x in cfg.far_list]


def make_geometry_tester(cfg, dataset, base_method_cfg, tag: str):
    method_cfg = copy_cfg(base_method_cfg)
    rm_cfg = method_cfg.recognition_method
    rm_cfg.calibration_set = None
    rm_cfg.calibration_transform = None
    recognition_method = instantiate(rm_cfg)
    recognition_method.far = float(cfg.far_list[0])
    recognition_method.beta = float(cfg.beta_list[0])
    return build_tester(
        cfg=cfg,
        method_cfg=method_cfg,
        test_dataset=dataset,
        recognition_method=recognition_method,
        method_name=f"geometry_loader_{slugify(tag)}",
        pretty_name="geometry_loader",
    )


def holue_variant_cfg(base_method_cfg, *, scale: float, use_mlp: bool):
    method_cfg = copy_cfg(base_method_cfg)
    OmegaConf.update(
        method_cfg,
        "recognition_method.kappa_input_scale",
        float(scale),
        force_add=True,
    )
    if use_mlp:
        OmegaConf.update(
            method_cfg,
            "recognition_method.calibration_set",
            True,
            force_add=True,
        )
    else:
        OmegaConf.update(
            method_cfg,
            "recognition_method.calibration_set",
            None,
            force_add=True,
        )
        OmegaConf.update(
            method_cfg,
            "recognition_method.calibration_transform",
            None,
            force_add=True,
        )
    return method_cfg


def run_holue_variant(
    cfg,
    *,
    dataset,
    method_cfg,
    variant_name: str,
    far: float,
    beta: float,
    scale: float,
    fractions: np.ndarray,
    out_dir: Path,
    uses_supervised_fusion: bool,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, Any]]:
    recognition_method = instantiate(method_cfg.recognition_method)
    maybe_attach_calibration_set(cfg, recognition_method, dataset.dataset_name)
    recognition_method.far = float(far)
    recognition_method.beta = float(beta)

    method_name = (
        f"{slugify(variant_name)}_dataset_{slugify(dataset.dataset_name)}"
        f"_far_{far}_beta_{beta}"
    )

    # The original HolUE implementation writes calibration diagnostics through
    # recognition_method.log_dir and the nested calibration transform.  Give each
    # run a private directory so datasets/FPIRs/variants cannot overwrite one
    # another while these ablations are running.
    run_log_dir = out_dir / "calibrator_logs" / method_name
    run_log_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(recognition_method, "log_dir"):
        recognition_method.log_dir = str(run_log_dir)
    calibration_transform = getattr(recognition_method, "calibration_transform", None)
    if calibration_transform is not None and hasattr(calibration_transform, "log_dir"):
        calibration_transform.log_dir = str(run_log_dir)
    tt = build_tester(
        cfg=cfg,
        method_cfg=method_cfg,
        test_dataset=dataset,
        recognition_method=recognition_method,
        method_name=method_name,
        pretty_name=variant_name,
    )
    result = run_method_raw(tt, gallery_name="g1")
    arrays = extract_method_arrays(result["recognition_method"])

    prr, curve, random_curve, oracle_curve = self_normalized_prr(
        uncertainty=result["predicted_unc"],
        predicted_id=result["predicted_id"],
        was_rejected=result["was_rejected"],
        g_unique_ids=result["g_unique_ids"],
        probe_unique_ids=result["probe_unique_ids"],
        fractions=fractions,
        metric_name="f1_class",
        seed=int(cfg.seed),
    )
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

    curve_dir = (
        out_dir
        / "curves"
        / slugify(dataset.dataset_name)
        / f"far_{far}"
        / slugify(variant_name)
    )
    curve_dir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(curve_dir / "curve.csv", index=False)
    random_curve.to_csv(curve_dir / "random_curve.csv", index=False)
    oracle_curve.to_csv(curve_dir / "oracle_curve.csv", index=False)

    rm = result["recognition_method"]
    probe_kappa = np.asarray(
        tt.probe_pooled_templates["g1"]["template_pooled_data_unc"],
        dtype=np.float64,
    ).reshape(-1)
    npz_path = out_dir / "per_example_npz" / f"{method_name}.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        predicted_id=result["predicted_id"],
        was_rejected=result["was_rejected"],
        predicted_unc=result["predicted_unc"],
        probe_unique_ids=result["probe_unique_ids"],
        g_unique_ids=result["g_unique_ids"],
        probe_kappa_raw=probe_kappa,
        probe_kappa_scaled=probe_kappa * float(scale),
        kl_1=np.asarray(arrays.get("kl_1", [])),
        kl_2=np.asarray(arrays.get("kl_2", [])),
        mean_probs=np.asarray(arrays.get("mean_probs", [])),
    )

    row = {
        "dataset": dataset.dataset_name,
        "variant": variant_name,
        "far": float(far),
        "beta": float(beta),
        "kappa_scale": float(scale),
        "uses_error_supervised_fusion": bool(uses_supervised_fusion),
        "prr_f1": float(prr),
        "base_f1_class": float(f1),
        "base_fnir": float(fnir),
        "base_fpir": float(fpir),
        "error_rate": float(np.mean(result["masks"]["any_error"])),
        "error_auroc": safe_auc(result["masks"]["any_error"], result["predicted_unc"]),
        "error_auprc": safe_auprc(result["masks"]["any_error"], result["predicted_unc"]),
        "false_accept_count": int(np.sum(result["masks"]["false_accept"])),
        "false_reject_count": int(np.sum(result["masks"]["false_reject"])),
        "misidentification_count": int(np.sum(result["masks"]["misidentification"])),
        "gallery_kappa": float(getattr(rm, "gallery_kappa", np.nan)),
        "predict_T": float(
            getattr(rm, "predict_T", np.nan).detach().cpu()
            if hasattr(getattr(rm, "predict_T", None), "detach")
            else getattr(rm, "predict_T", np.nan)
        ),
    }
    return result, arrays, row


def r_ns_rows(
    *,
    dataset_name: str,
    far: float,
    beta: float,
    scale: float,
    result: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    probe_kappa_raw: np.ndarray,
    embedding_dim: int,
) -> List[Dict[str, Any]]:
    """Diagnose what geometric kappa calibration adds for rejected probes.

    Besides r_NS itself, report the exact collapsed false-rejection term
    ``r_FR = 1-p0`` and non-specificity alone.  High-p0 strata are important:
    they isolate the regime where the collapsed posterior strongly supports
    "unknown" and r_NS is intended to add information about how specific that
    unknown explanation is.
    """
    if "mean_probs" not in arrays:
        return []
    mean_probs = np.asarray(arrays["mean_probs"], dtype=np.float64)
    if mean_probs.ndim != 2:
        return []

    p0 = np.clip(1.0 - np.sum(mean_probs, axis=1), 0.0, 1.0)
    rejected = np.asarray(result["was_rejected"], dtype=bool)
    k = np.asarray(probe_kappa_raw, dtype=np.float64).reshape(-1)
    if len(k) != len(p0):
        return []

    n_raw = vmf_nonspecificity_np(k, d=embedding_dim)
    n_gc = vmf_nonspecificity_np(scale * k, d=embedding_dim)
    scores = {
        "r_FR_collapsed": rejected.astype(np.float64) * (1.0 - p0),
        "N_raw_kappa": rejected.astype(np.float64) * n_raw,
        "N_geometric_kappa": rejected.astype(np.float64) * n_gc,
        "r_NS_raw_kappa": rejected.astype(np.float64) * p0 * n_raw,
        "r_NS_geometric_kappa": rejected.astype(np.float64) * p0 * n_gc,
    }

    masks = result["masks"]
    rejected_known_or_unknown = masks["false_reject"] | masks["true_reject"]
    binary_fr = masks["false_reject"][rejected_known_or_unknown]
    p0_rejected = p0[rejected_known_or_unknown]

    if np.any(rejected_known_or_unknown):
        q50 = float(np.quantile(p0_rejected, 0.50))
        q75 = float(np.quantile(p0_rejected, 0.75))
    else:
        q50 = q75 = np.nan

    rows = []
    for name, score in scores.items():
        local_score = score[rejected_known_or_unknown]
        base = {
            "dataset": dataset_name,
            "far": float(far),
            "beta": float(beta),
            "variant": name,
            "kappa_scale": (
                float(scale)
                if name in {"N_geometric_kappa", "r_NS_geometric_kappa"}
                else 1.0
            ),
            "rejected_count": int(np.sum(rejected_known_or_unknown)),
            "false_reject_count": int(np.sum(masks["false_reject"])),
            "true_reject_count": int(np.sum(masks["true_reject"])),
            "false_reject_vs_true_reject_auroc": safe_auc(binary_fr, local_score),
            "false_reject_mean": float(np.mean(score[masks["false_reject"]]))
            if np.any(masks["false_reject"])
            else np.nan,
            "true_reject_mean": float(np.mean(score[masks["true_reject"]]))
            if np.any(masks["true_reject"])
            else np.nan,
            "all_error_auroc": safe_auc(masks["any_error"], score),
            "p0_rejected_median": q50,
            "p0_rejected_q75": q75,
        }
        for label, threshold in [("high_p0_q50", q50), ("high_p0_q75", q75)]:
            if not np.isfinite(threshold):
                base[f"false_reject_vs_true_reject_auroc_{label}"] = np.nan
                base[f"count_{label}"] = 0
                continue
            stratum = rejected_known_or_unknown & (p0 >= threshold)
            y = masks["false_reject"][stratum]
            sc = score[stratum]
            base[f"false_reject_vs_true_reject_auroc_{label}"] = safe_auc(y, sc)
            base[f"count_{label}"] = int(np.sum(stratum))
        rows.append(base)
    return rows


def save_r_ns_per_example(
    out_path: Path,
    *,
    result: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    probe_kappa_raw: np.ndarray,
    scale: float,
    embedding_dim: int,
) -> None:
    if "mean_probs" not in arrays:
        return
    mean_probs = np.asarray(arrays["mean_probs"], dtype=np.float64)
    p0 = np.clip(1.0 - np.sum(mean_probs, axis=1), 0.0, 1.0)
    k = np.asarray(probe_kappa_raw, dtype=np.float64).reshape(-1)
    if len(k) != len(p0):
        return

    masks = result["masks"]
    rejected = np.asarray(result["was_rejected"], dtype=bool)
    n_raw = vmf_nonspecificity_np(k, d=embedding_dim)
    n_gc = vmf_nonspecificity_np(scale * k, d=embedding_dim)
    df = pd.DataFrame(
        {
            "probe_subject_id": result["probe_unique_ids"],
            "was_rejected": rejected,
            "false_reject": masks["false_reject"],
            "true_reject": masks["true_reject"],
            "false_accept": masks["false_accept"],
            "misidentification": masks["misidentification"],
            "p0": p0,
            "kappa_raw": k,
            "kappa_geometric": scale * k,
            "N_raw": n_raw,
            "N_geometric": n_gc,
            "r_FR": rejected.astype(np.float64) * (1.0 - p0),
            "r_NS_raw": rejected.astype(np.float64) * p0 * n_raw,
            "r_NS_geometric": rejected.astype(np.float64) * p0 * n_gc,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def delta_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    keys = ["dataset", "far", "beta"]
    metrics = ["prr_f1", "error_auroc", "error_auprc"]
    pivots = {
        metric: df.pivot_table(index=keys, columns="variant", values=metric, aggfunc="first")
        for metric in metrics
    }
    comparisons = [
        ("GC direct - raw direct", "HolUE-GC direct", "HolUE direct"),
        ("raw supervised - raw direct", "HolUE raw supervised", "HolUE direct"),
        ("GC supervised - GC direct", "HolUE-GC supervised", "HolUE-GC direct"),
        ("GC direct - raw supervised", "HolUE-GC direct", "HolUE raw supervised"),
    ]
    rows = []
    for idx in pivots[metrics[0]].index:
        base = dict(zip(keys, idx if isinstance(idx, tuple) else (idx,)))
        for label, a, b in comparisons:
            row = {**base, "comparison": label}
            for metric in metrics:
                p = pivots[metric]
                row[f"delta_{metric}"] = (
                    float(p.loc[idx, a] - p.loc[idx, b])
                    if a in p.columns and b in p.columns
                    else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


@hydra.main(
    config_path=str(Path(__file__).resolve().parents[1] / "configs/uncertainty_benchmark"),
    config_name="scf_geometry_holue_text",
    version_base="1.2",
)
def main(cfg):
    seed_everything(
        seed=int(cfg.get("seed", 777)),
        deterministic=bool(cfg.get("deterministic", True)),
    )

    out_dir = Path(cfg.exp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "geometry").mkdir(parents=True, exist_ok=True)

    fractions = np.linspace(
        float(cfg.rejection_fractions[0]),
        float(cfg.rejection_fractions[1]),
        int(cfg.rejection_fractions[2]),
    )
    base_method_cfg = cfg.holue_method
    test_datasets = [instantiate(x) for x in cfg.test_datasets]

    geometry_rows: List[Dict[str, Any]] = []
    variant_rows: List[Dict[str, Any]] = []
    rns_all_rows: List[Dict[str, Any]] = []
    scale_manifest: Dict[str, Any] = {}

    for test_dataset in test_datasets:
        dataset_name = test_dataset.dataset_name
        calib_cfg = getattr(cfg.dataset_name_to_calibration_set, dataset_name)
        calibration_dataset = instantiate(calib_cfg)

        print("=" * 100)
        print(f"[SCF geometry] dataset={dataset_name}")
        print("=" * 100)

        calib_tt = make_geometry_tester(
            cfg,
            calibration_dataset,
            base_method_cfg,
            tag=f"{dataset_name}_calibration",
        )
        test_tt = make_geometry_tester(
            cfg,
            test_dataset,
            base_method_cfg,
            tag=f"{dataset_name}_test",
        )

        calib_sample = extract_sample_geometry(calib_tt)
        calib_template = extract_template_geometry(calib_tt)
        test_sample = extract_sample_geometry(test_tt)
        test_template = extract_template_geometry(test_tt)
        embedding_dim = int(calib_tt.image_input_feats.shape[1])

        fit_level = str(cfg.geometric_calibration.get("fit_level", "sample")).lower()
        if fit_level == "sample":
            fit_points = calib_sample
        elif fit_level == "template":
            fit_points = calib_template
        else:
            raise ValueError("geometric_calibration.fit_level must be sample or template")

        fit = fit_vmf_kappa_scale(
            fit_points["kappa"].to_numpy(),
            fit_points["true_proxy_cosine"].to_numpy(),
            d=embedding_dim,
            max_scale=float(cfg.geometric_calibration.get("max_scale", 1_000_000.0)),
        )
        scale = float(fit["scale"])

        # Oracle test scale is diagnostic only and is never used for HolUE.
        oracle_test_fit = fit_vmf_kappa_scale(
            test_sample["kappa"].to_numpy(),
            test_sample["true_proxy_cosine"].to_numpy(),
            d=embedding_dim,
            max_scale=float(cfg.geometric_calibration.get("max_scale", 1_000_000.0)),
        )

        scale_manifest[dataset_name] = {
            "fit_level": fit_level,
            "validation_scale": scale,
            "validation_fit": fit,
            "oracle_test_scale_diagnostic_only": float(oracle_test_fit["scale"]),
            "oracle_test_to_validation_scale_ratio": float(oracle_test_fit["scale"] / scale),
            "embedding_dim": embedding_dim,
        }

        geom_dir = out_dir / "geometry" / slugify(dataset_name)
        geom_dir.mkdir(parents=True, exist_ok=True)
        calib_sample.to_csv(geom_dir / "calibration_sample_points.csv", index=False)
        calib_template.to_csv(geom_dir / "calibration_template_points.csv", index=False)
        test_sample.to_csv(geom_dir / "test_sample_points.csv", index=False)
        test_template.to_csv(geom_dir / "test_template_points.csv", index=False)

        for split_name, level_name, points in [
            ("validation", "sample", calib_sample),
            ("validation", "template", calib_template),
            ("test", "sample", test_sample),
            ("test", "template", test_template),
        ]:
            raw_metrics, raw_bins = geometry_metrics(
                points,
                scale=1.0,
                embedding_dim=embedding_dim,
                num_bins=int(cfg.geometric_calibration.get("num_bins", 10)),
            )
            cal_metrics, cal_bins = geometry_metrics(
                points,
                scale=scale,
                embedding_dim=embedding_dim,
                num_bins=int(cfg.geometric_calibration.get("num_bins", 10)),
            )
            for variant, metrics in [("raw", raw_metrics), ("geometric", cal_metrics)]:
                geometry_rows.append(
                    {
                        "dataset": dataset_name,
                        "split": split_name,
                        "level": level_name,
                        "variant": variant,
                        "fitted_on": f"validation_{fit_level}",
                        "validation_scale": scale,
                        "oracle_test_scale_diagnostic_only": float(oracle_test_fit["scale"]),
                        **metrics,
                    }
                )
            raw_bins.to_csv(
                geom_dir / f"{split_name}_{level_name}_reliability_raw.csv", index=False
            )
            cal_bins.to_csv(
                geom_dir / f"{split_name}_{level_name}_reliability_geometric.csv",
                index=False,
            )
            save_reliability_plot(
                raw_bins,
                cal_bins,
                geom_dir / f"{split_name}_{level_name}_reliability",
                title=f"{dataset_name}: {split_name}, {level_name}",
            )

        for far in dataset_fars(cfg, dataset_name):
            for beta in cfg.beta_list:
                variants = [
                    ("HolUE raw supervised", 1.0, True),
                    ("HolUE direct", 1.0, False),
                    ("HolUE-GC direct", scale, False),
                    ("HolUE-GC supervised", scale, True),
                ]
                run_cache = {}
                for variant_name, variant_scale, use_mlp in variants:
                    print(
                        f"[HolUE geometry] dataset={dataset_name} far={far} "
                        f"variant={variant_name} scale={variant_scale:.6g}"
                    )
                    method_cfg = holue_variant_cfg(
                        base_method_cfg,
                        scale=variant_scale,
                        use_mlp=use_mlp,
                    )
                    result, arrays, row = run_holue_variant(
                        cfg,
                        dataset=test_dataset,
                        method_cfg=method_cfg,
                        variant_name=variant_name,
                        far=float(far),
                        beta=float(beta),
                        scale=float(variant_scale),
                        fractions=fractions,
                        out_dir=out_dir,
                        uses_supervised_fusion=use_mlp,
                    )
                    variant_rows.append(row)
                    run_cache[variant_name] = (result, arrays)

                # Use the direct raw posterior as a fixed p0/decision reference for
                # isolating what geometric kappa scaling does to r_NS.
                result, arrays = run_cache["HolUE direct"]
                probe_kappa_raw = np.asarray(
                    test_tt.probe_pooled_templates["g1"]["template_pooled_data_unc"],
                    dtype=np.float64,
                ).reshape(-1)
                rns_all_rows.extend(
                    r_ns_rows(
                        dataset_name=dataset_name,
                        far=float(far),
                        beta=float(beta),
                        scale=scale,
                        result=result,
                        arrays=arrays,
                        probe_kappa_raw=probe_kappa_raw,
                        embedding_dim=embedding_dim,
                    )
                )
                save_r_ns_per_example(
                    out_dir
                    / "r_ns_per_example"
                    / slugify(dataset_name)
                    / f"far_{far}_beta_{beta}.csv",
                    result=result,
                    arrays=arrays,
                    probe_kappa_raw=probe_kappa_raw,
                    scale=scale,
                    embedding_dim=embedding_dim,
                )

    geometry_df = pd.DataFrame(geometry_rows)
    variants_df = pd.DataFrame(variant_rows)
    rns_df = pd.DataFrame(rns_all_rows)
    deltas_df = delta_table(variants_df)

    geometry_df.to_csv(out_dir / "tables" / "scf_geometric_calibration.csv", index=False)
    variants_df.to_csv(out_dir / "tables" / "holue_geometry_variants.csv", index=False)
    rns_df.to_csv(out_dir / "tables" / "r_ns_geometry_diagnostics.csv", index=False)
    deltas_df.to_csv(out_dir / "tables" / "holue_geometry_deltas.csv", index=False)
    with open(out_dir / "tables" / "geometric_scale_manifest.json", "w", encoding="utf-8") as f:
        json.dump(scale_manifest, f, indent=2, ensure_ascii=False)

    print("\nSaved:")
    print(out_dir / "tables" / "scf_geometric_calibration.csv")
    print(out_dir / "tables" / "holue_geometry_variants.csv")
    print(out_dir / "tables" / "r_ns_geometry_diagnostics.csv")
    print(out_dir / "tables" / "holue_geometry_deltas.csv")
    print(out_dir / "tables" / "geometric_scale_manifest.json")


if __name__ == "__main__":
    main()
