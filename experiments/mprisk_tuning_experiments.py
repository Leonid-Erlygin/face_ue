#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import OmegaConf

from evaluation.reproducibility import seed_everything
from evaluation.open_set_methods.calibration_utils import prepare_calibration_dataset
from evaluation.open_set_methods.class_prob_models import FarLossCalc
from utils.golden_section import golden_selection_search

from experiments.mprisk_core_experiments import (
    build_tester,
    component_ablation_rows,
    compute_osr_error_masks,
    f1_classic,
    fnir_fpir,
    maybe_attach_calibration_set,
    maybe_attach_dataset_temperature,
    rejection_curve_for_score,
    safe_auc,
    safe_auprc,
    self_normalized_prr,
    slugify,
    run_method_raw,
)
#---------------------------------------------------------------------
# Generic component utilities
# ---------------------------------------------------------------------


MPRISK_COMPONENT_KEYS = ["r_fa", "r_id", "r_fr", "r_ns"]


def copy_method_cfg(method_cfg):
    return OmegaConf.create(OmegaConf.to_container(method_cfg, resolve=True))


def apply_recognition_overrides(method_cfg, overrides: Optional[Dict[str, Any]]):
    if overrides is None:
        return method_cfg

    method_cfg = copy_method_cfg(method_cfg)

    for key, value in overrides.items():
        OmegaConf.update(
            method_cfg,
            f"recognition_method.{key}",
            value,
            force_add=True,
        )

    return method_cfg


def extract_mprisk_components_from_result(result: Dict[str, Any]) -> Dict[str, np.ndarray]:
    rm = result["recognition_method"]

    missing = []
    for key in MPRISK_COMPONENT_KEYS:
        if not hasattr(rm, key):
            missing.append(key)

    if missing:
        raise AttributeError(
            f"Recognition method does not expose MPRisk components: {missing}. "
            "Make sure MPRiskPredictiveProb stores r_fa, r_id, r_fr, r_ns."
        )

    components = {
        "predicted_id": np.asarray(result["predicted_id"], dtype=int),
        "was_rejected": np.asarray(result["was_rejected"], dtype=bool),
        "r_fa": np.asarray(rm.r_fa, dtype=np.float64).reshape(-1),
        "r_id": np.asarray(rm.r_id, dtype=np.float64).reshape(-1),
        "r_fr": np.asarray(rm.r_fr, dtype=np.float64).reshape(-1),
        "r_ns": np.asarray(rm.r_ns, dtype=np.float64).reshape(-1),
    }

    return components


def component_score(
    components: Dict[str, np.ndarray],
    lambdas: np.ndarray,
) -> np.ndarray:
    lambdas = np.asarray(lambdas, dtype=np.float64).reshape(4)

    return (
        lambdas[0] * np.asarray(components["r_fa"], dtype=np.float64)
        + lambdas[1] * np.asarray(components["r_id"], dtype=np.float64)
        + lambdas[2] * np.asarray(components["r_fr"], dtype=np.float64)
        + lambdas[3] * np.asarray(components["r_ns"], dtype=np.float64)
    )


def slice_components(
    components: Dict[str, np.ndarray],
    idx: np.ndarray,
) -> Dict[str, np.ndarray]:
    idx = np.asarray(idx, dtype=int)

    return {
        "predicted_id": np.asarray(components["predicted_id"])[idx],
        "was_rejected": np.asarray(components["was_rejected"])[idx],
        "r_fa": np.asarray(components["r_fa"])[idx],
        "r_id": np.asarray(components["r_id"])[idx],
        "r_fr": np.asarray(components["r_fr"])[idx],
        "r_ns": np.asarray(components["r_ns"])[idx],
    }


def build_lambda_candidates(
    num_random: int,
    log_low: float,
    log_high: float,
    seed: int,
    include_structured: bool = True,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)

    candidates: List[np.ndarray] = []

    if include_structured:
        candidates.extend(
            [
                np.array([1.0, 1.0, 1.0, 1.0]),
                np.array([1.0, 1.0, 1.0, 0.0]),
                np.array([1.0, 1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 1.0, 1.0]),
                np.array([1.0, 0.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 0.0, 1.0]),
                np.array([1.0, 1.0, 1.0, 2.0]),
                np.array([1.0, 1.0, 1.0, 5.0]),
                np.array([1.0, 1.0, 1.0, 10.0]),
                np.array([1.0, 1.0, 1.0, 50.0]),
                np.array([1.0, 1.0, 1.0, 100.0]),
                np.array([1.0, 1.0, 0.1, 10.0]),
                np.array([1.0, 1.0, 0.01, 10.0]),
                np.array([1.0, 0.5, 0.1, 10.0]),
                np.array([2.0, 1.0, 0.1, 10.0]),
                np.array([10.0, 1.0, 0.1, 10.0]),
            ]
        )

    for _ in range(int(num_random)):
        log_l = rng.uniform(log_low, log_high, size=4)
        l = np.exp(log_l)

        # Scale-invariant normalization. Only ranking matters.
        l = l / max(float(np.max(l)), 1e-12)
        candidates.append(l.astype(np.float64))

    cleaned = []
    for l in candidates:
        l = np.asarray(l, dtype=np.float64).reshape(4)
        l = np.maximum(l, 0.0)

        if np.max(l) > 0:
            l = l / np.max(l)

        cleaned.append(l)

    return cleaned


def tune_lambdas_for_components(
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
        calib_components_sub = slice_components(calib_components, subset_idx)
        probe_ids_sub = np.asarray(probe_unique_ids_calib)[subset_idx]
    else:
        calib_components_sub = calib_components
        probe_ids_sub = np.asarray(probe_unique_ids_calib)

    masks = compute_osr_error_masks(
        predicted_id=calib_components_sub["predicted_id"],
        was_rejected=calib_components_sub["was_rejected"],
        g_unique_ids=g_unique_ids_calib,
        probe_unique_ids=probe_ids_sub,
    )

    if len(probe_ids_sub) < 2:
        return {
            "lambdas": np.ones(4, dtype=np.float64),
            "val_prr": np.nan,
            "val_auc": np.nan,
            "val_auprc": np.nan,
            "subset_size": len(probe_ids_sub),
        }

    candidates = build_lambda_candidates(
        num_random=num_random,
        log_low=log_low,
        log_high=log_high,
        seed=seed,
    )

    # Build oracle/random once for self-normalized PRR.
    rng = np.random.default_rng(seed)
    n = len(probe_ids_sub)

    random_score = rng.random(n)
    oracle_score = masks["any_error"].astype(np.float64)
    oracle_score = oracle_score + 1e-9 * rng.random(n)

    random_curve = rejection_curve_for_score(
        uncertainty=random_score,
        predicted_id=calib_components_sub["predicted_id"],
        was_rejected=calib_components_sub["was_rejected"],
        g_unique_ids=g_unique_ids_calib,
        probe_unique_ids=probe_ids_sub,
        fractions=fractions,
    )

    oracle_curve = rejection_curve_for_score(
        uncertainty=oracle_score,
        predicted_id=calib_components_sub["predicted_id"],
        was_rejected=calib_components_sub["was_rejected"],
        g_unique_ids=g_unique_ids_calib,
        probe_unique_ids=probe_ids_sub,
        fractions=fractions,
    )

    random_area = float(np.trapezoid(random_curve["f1_class"].values, random_curve["fraction"].values))
    oracle_area = float(np.trapezoid(oracle_curve["f1_class"].values, oracle_curve["fraction"].values))
    denom = oracle_area - random_area

    best_l = np.ones(4, dtype=np.float64)
    best_prr = -np.inf
    best_score = None

    for l in candidates:
        score = component_score(calib_components_sub, l)

        curve = rejection_curve_for_score(
            uncertainty=score,
            predicted_id=calib_components_sub["predicted_id"],
            was_rejected=calib_components_sub["was_rejected"],
            g_unique_ids=g_unique_ids_calib,
            probe_unique_ids=probe_ids_sub,
            fractions=fractions,
        )

        area = float(np.trapezoid(curve["f1_class"].values, curve["fraction"].values))
        if abs(denom) < 1e-12:
            prr = area
        else:
            prr = (area - random_area) / denom

        if prr > best_prr:
            best_prr = float(prr)
            best_l = l.copy()
            best_score = score.copy()

    return {
        "lambdas": best_l,
        "val_prr": float(best_prr),
        "val_auc": safe_auc(masks["any_error"], best_score) if best_score is not None else np.nan,
        "val_auprc": safe_auprc(masks["any_error"], best_score) if best_score is not None else np.nan,
        "subset_size": len(probe_ids_sub),
    }


# ---------------------------------------------------------------------
# Running MPRisk and computing calibration components
# ---------------------------------------------------------------------


def run_base_mprisk(
    cfg,
    method_cfg,
    test_dataset,
    far: float,
    beta: float,
    method_name_suffix: str,
):
    dataset_name = test_dataset.dataset_name

    recognition_method = instantiate(method_cfg.recognition_method)

    # For these experiments we want raw components. Lambda tuning is done
    # outside the method for controlled ablations.
    if hasattr(recognition_method, "tune_lambdas"):
        recognition_method.tune_lambdas = False
    if hasattr(recognition_method, "use_calibration"):
        recognition_method.use_calibration = False

    maybe_attach_calibration_set(cfg, recognition_method, dataset_name)
    maybe_attach_dataset_temperature(
        cfg,
        recognition_method,
        str(method_cfg.pretty_name),
        dataset_name,
    )

    recognition_method.far = far
    recognition_method.beta = beta

    pretty_name = f"{method_cfg.pretty_name}_{method_name_suffix}"
    method_name = (
        f"{slugify(pretty_name)}"
        f"_dataset_{slugify(dataset_name)}"
        f"_far_{far}"
        f"_beta_{beta}"
    )

    tt = build_tester(
        cfg=cfg,
        method_cfg=method_cfg,
        test_dataset=test_dataset,
        recognition_method=recognition_method,
        method_name=method_name,
        pretty_name=pretty_name,
    )

    result = run_method_raw(tt, gallery_name="g1")
    test_components = extract_mprisk_components_from_result(result)

    return {
        "tt": tt,
        "result": result,
        "recognition_method": result["recognition_method"],
        "test_components": test_components,
    }


def compute_mprisk_calibration_components(
    recognition_method,
) -> Dict[str, Any]:
    rm = recognition_method

    if rm.calibration_set is None or rm.calibration_set is True:
        raise ValueError(
            "recognition_method.calibration_set must be an instantiated dataset."
        )

    gallery_pooled_templates_calib, probe_pooled_templates_calib = (
        prepare_calibration_dataset(
            rm.calibration_set,
            rm.calibration_embs_name,
        )
    )

    g_unique_ids_calib = gallery_pooled_templates_calib["g1"][
        "template_subject_ids_sorted"
    ]
    probe_unique_ids_calib = probe_pooled_templates_calib["g1"][
        "template_subject_ids_sorted"
    ]

    probe_feats_calib = probe_pooled_templates_calib["g1"][
        "template_pooled_features"
    ]
    probe_unc_calib = probe_pooled_templates_calib["g1"][
        "template_pooled_data_unc"
    ]

    gallery_feats_calib = gallery_pooled_templates_calib["g1"][
        "template_pooled_features"
    ]
    gallery_unc_calib = gallery_pooled_templates_calib["g1"][
        "template_pooled_data_unc"
    ]

    is_seen_calib = np.isin(probe_unique_ids_calib, g_unique_ids_calib)

    probe_unc_calib_scaled = probe_unc_calib * rm.kappa_input_scale

    far_loss_func_calib = FarLossCalc(
        probe_feats_calib,
        probe_unc_calib_scaled,
        gallery_feats_calib,
        gallery_unc_calib,
        rm.predict_T,
        rm.far,
        is_seen_calib,
        rm,
        verbose=False,
    )

    calibration_kappa = golden_selection_search(
        rm.kappa_high,
        rm.kappa_low,
        rm.eps,
        rm.max_iter,
        far_loss_func_calib,
        verbose=False,
    )

    print(
        f"[MPRiskTuning] calibration kappa={np.round(calibration_kappa, 4)} "
        f"for far={rm.far}"
    )

    (
        mean_probs_calib,
        kl_1_calib,
        kl_2_calib,
        oog_prob_calib,
        oog_nonspecificity_calib,
    ) = rm._compute_probs_aux(
        probe_feats=probe_feats_calib,
        probe_unc=probe_unc_calib,
        gallery_feats=gallery_feats_calib,
        gallery_unc=gallery_unc_calib,
        gallery_kappa=calibration_kappa,
    )

    calib_components = rm._risk_components(
        mean_probs_calib,
        oog_prob_calib,
        oog_nonspecificity_calib,
    )

    return {
        "calib_components": calib_components,
        "g_unique_ids_calib": np.asarray(g_unique_ids_calib),
        "probe_unique_ids_calib": np.asarray(probe_unique_ids_calib),
        "calibration_kappa": float(calibration_kappa),
        "kl_1_calib": kl_1_calib,
        "kl_2_calib": kl_2_calib,
        "oog_prob_calib": oog_prob_calib,
        "oog_nonspecificity_calib": oog_nonspecificity_calib,
    }


def evaluate_lambdas_on_test(
    lambdas: np.ndarray,
    test_components: Dict[str, np.ndarray],
    result: Dict[str, Any],
    fractions: np.ndarray,
    seed: int,
) -> Dict[str, Any]:
    score = component_score(test_components, lambdas)

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

    return {
        "test_prr_f1": prr,
        "test_error_auroc": safe_auc(masks["any_error"], score),
        "test_error_auprc": safe_auprc(masks["any_error"], score),
        "base_f1_class": f1,
        "base_fnir": fnir,
        "base_fpir": fpir,
        "f1_at_max_filter": float(curve["f1_class"].iloc[-1]),
        "fnir_at_max_filter": float(curve["fnir"].iloc[-1]),
        "fpir_at_max_filter": float(curve["fpir"].iloc[-1]),
    }


# ---------------------------------------------------------------------
# Experiment 1: validation-size ablation
# ---------------------------------------------------------------------


def run_validation_size_ablation(cfg, cache: Dict, out_dir: Path) -> pd.DataFrame:
    if not bool(cfg.validation_size_ablation.enabled):
        return pd.DataFrame()

    print("\n" + "=" * 100)
    print("[Experiment] Validation-size ablation")
    print("=" * 100)

    rows = []
    fractions = np.linspace(
        float(cfg.rejection_fractions[0]),
        float(cfg.rejection_fractions[1]),
        int(cfg.rejection_fractions[2]),
    )

    val_fracs = list(cfg.validation_size_ablation.validation_fractions)
    repeats = int(cfg.validation_size_ablation.repeats)

    for cache_key, entry in cache.items():
        dataset_name, variant_name, far, beta = cache_key

        calib = entry["calib"]
        calib_components = calib["calib_components"]
        probe_ids_calib = calib["probe_unique_ids_calib"]
        g_ids_calib = calib["g_unique_ids_calib"]

        n_calib = len(probe_ids_calib)

        for val_frac in val_fracs:
            for rep in range(repeats):
                seed = int(cfg.seed) + 1000 * rep + int(float(val_frac) * 10000)

                if float(val_frac) >= 1.0:
                    subset_idx = np.arange(n_calib)
                else:
                    rng = np.random.default_rng(seed)
                    subset_size = max(
                        int(cfg.validation_size_ablation.min_subset_size),
                        int(round(float(val_frac) * n_calib)),
                    )
                    subset_size = min(subset_size, n_calib)
                    subset_idx = rng.choice(n_calib, size=subset_size, replace=False)

                tune = tune_lambdas_for_components(
                    calib_components=calib_components,
                    g_unique_ids_calib=g_ids_calib,
                    probe_unique_ids_calib=probe_ids_calib,
                    fractions=fractions,
                    num_random=int(cfg.lambda_search.num_random),
                    log_low=float(cfg.lambda_search.log_low),
                    log_high=float(cfg.lambda_search.log_high),
                    seed=seed,
                    subset_idx=subset_idx,
                )

                test_eval = evaluate_lambdas_on_test(
                    lambdas=tune["lambdas"],
                    test_components=entry["test_components"],
                    result=entry["result"],
                    fractions=fractions,
                    seed=seed,
                )

                rows.append(
                    {
                        "dataset": dataset_name,
                        "variant": variant_name,
                        "far": far,
                        "beta": beta,
                        "validation_fraction": float(val_frac),
                        "repeat": rep,
                        "subset_size": tune["subset_size"],
                        "val_prr": tune["val_prr"],
                        "val_auc": tune["val_auc"],
                        "val_auprc": tune["val_auprc"],
                        "lambda_fa": tune["lambdas"][0],
                        "lambda_id": tune["lambdas"][1],
                        "lambda_fr": tune["lambdas"][2],
                        "lambda_ns": tune["lambdas"][3],
                        **test_eval,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "validation_size_ablation.csv", index=False)

    if len(df) > 0:
        summary = (
            df.groupby(["dataset", "variant", "far", "beta", "validation_fraction"])
            .agg(
                test_prr_f1_mean=("test_prr_f1", "mean"),
                test_prr_f1_std=("test_prr_f1", "std"),
                val_prr_mean=("val_prr", "mean"),
                lambda_fa_mean=("lambda_fa", "mean"),
                lambda_id_mean=("lambda_id", "mean"),
                lambda_fr_mean=("lambda_fr", "mean"),
                lambda_ns_mean=("lambda_ns", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(out_dir / "validation_size_ablation_summary.csv", index=False)

    return df


# ---------------------------------------------------------------------
# Experiment 2: operating-point transfer
# ---------------------------------------------------------------------


def run_operating_point_transfer(cfg, cache: Dict, out_dir: Path) -> pd.DataFrame:
    if not bool(cfg.operating_point_transfer.enabled):
        return pd.DataFrame()

    print("\n" + "=" * 100)
    print("[Experiment] Operating-point transfer")
    print("=" * 100)

    rows = []
    fractions = np.linspace(
        float(cfg.rejection_fractions[0]),
        float(cfg.rejection_fractions[1]),
        int(cfg.rejection_fractions[2]),
    )

    train_fars = list(cfg.operating_point_transfer.train_fars)
    eval_fars = list(cfg.operating_point_transfer.eval_fars)

    for dataset_name in sorted({k[0] for k in cache.keys()}):
        for variant_name in sorted({k[1] for k in cache.keys() if k[0] == dataset_name}):
            for beta in sorted({k[3] for k in cache.keys() if k[0] == dataset_name and k[1] == variant_name}):
                for train_far in train_fars:
                    train_key = (dataset_name, variant_name, float(train_far), float(beta))
                    if train_key not in cache:
                        continue

                    train_entry = cache[train_key]
                    calib = train_entry["calib"]

                    tune = tune_lambdas_for_components(
                        calib_components=calib["calib_components"],
                        g_unique_ids_calib=calib["g_unique_ids_calib"],
                        probe_unique_ids_calib=calib["probe_unique_ids_calib"],
                        fractions=fractions,
                        num_random=int(cfg.lambda_search.num_random),
                        log_low=float(cfg.lambda_search.log_low),
                        log_high=float(cfg.lambda_search.log_high),
                        seed=int(cfg.seed) + int(float(train_far) * 100000),
                    )

                    for eval_far in eval_fars:
                        eval_key = (dataset_name, variant_name, float(eval_far), float(beta))
                        if eval_key not in cache:
                            continue

                        eval_entry = cache[eval_key]

                        test_eval = evaluate_lambdas_on_test(
                            lambdas=tune["lambdas"],
                            test_components=eval_entry["test_components"],
                            result=eval_entry["result"],
                            fractions=fractions,
                            seed=int(cfg.seed),
                        )

                        rows.append(
                            {
                                "dataset": dataset_name,
                                "variant": variant_name,
                                "beta": beta,
                                "train_far": float(train_far),
                                "eval_far": float(eval_far),
                                "val_prr_at_train_far": tune["val_prr"],
                                "lambda_fa": tune["lambdas"][0],
                                "lambda_id": tune["lambdas"][1],
                                "lambda_fr": tune["lambdas"][2],
                                "lambda_ns": tune["lambdas"][3],
                                **test_eval,
                            }
                        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "operating_point_transfer.csv", index=False)
    return df


# ---------------------------------------------------------------------
# Experiment 3: cross-dataset transfer
# ---------------------------------------------------------------------


def run_cross_dataset_transfer(cfg, cache: Dict, out_dir: Path) -> pd.DataFrame:
    if not bool(cfg.cross_dataset_transfer.enabled):
        return pd.DataFrame()

    print("\n" + "=" * 100)
    print("[Experiment] Cross-dataset lambda transfer")
    print("=" * 100)

    rows = []
    fractions = np.linspace(
        float(cfg.rejection_fractions[0]),
        float(cfg.rejection_fractions[1]),
        int(cfg.rejection_fractions[2]),
    )

    datasets = sorted({k[0] for k in cache.keys()})
    variants = sorted({k[1] for k in cache.keys()})

    for variant_name in variants:
        for far in cfg.cross_dataset_transfer.fars:
            for beta in cfg.beta_list:
                for source_dataset in datasets:
                    source_key = (
                        source_dataset,
                        variant_name,
                        float(far),
                        float(beta),
                    )
                    if source_key not in cache:
                        continue

                    source_entry = cache[source_key]
                    calib = source_entry["calib"]

                    tune = tune_lambdas_for_components(
                        calib_components=calib["calib_components"],
                        g_unique_ids_calib=calib["g_unique_ids_calib"],
                        probe_unique_ids_calib=calib["probe_unique_ids_calib"],
                        fractions=fractions,
                        num_random=int(cfg.lambda_search.num_random),
                        log_low=float(cfg.lambda_search.log_low),
                        log_high=float(cfg.lambda_search.log_high),
                        seed=int(cfg.seed) + abs(hash(source_dataset)) % 100000,
                    )

                    for target_dataset in datasets:
                        target_key = (
                            target_dataset,
                            variant_name,
                            float(far),
                            float(beta),
                        )
                        if target_key not in cache:
                            continue

                        target_entry = cache[target_key]

                        test_eval = evaluate_lambdas_on_test(
                            lambdas=tune["lambdas"],
                            test_components=target_entry["test_components"],
                            result=target_entry["result"],
                            fractions=fractions,
                            seed=int(cfg.seed),
                        )

                        rows.append(
                            {
                                "variant": variant_name,
                                "far": float(far),
                                "beta": float(beta),
                                "source_dataset": source_dataset,
                                "target_dataset": target_dataset,
                                "source_val_prr": tune["val_prr"],
                                "lambda_fa": tune["lambdas"][0],
                                "lambda_id": tune["lambdas"][1],
                                "lambda_fr": tune["lambdas"][2],
                                "lambda_ns": tune["lambdas"][3],
                                **test_eval,
                            }
                        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "cross_dataset_transfer.csv", index=False)
    return df


# ---------------------------------------------------------------------
# Experiment 4: hyperparameter sensitivity
# ---------------------------------------------------------------------


def run_hyperparameter_sensitivity(
    cfg,
    test_datasets,
    base_method_cfg,
    out_dir: Path,
) -> pd.DataFrame:
    if not bool(cfg.hyperparameter_sensitivity.enabled):
        return pd.DataFrame()

    print("\n" + "=" * 100)
    print("[Experiment] Hyperparameter sensitivity")
    print("=" * 100)

    rows = []
    fractions = np.linspace(
        float(cfg.rejection_fractions[0]),
        float(cfg.rejection_fractions[1]),
        int(cfg.rejection_fractions[2]),
    )

    for variant in cfg.hyperparameter_sensitivity.variants:
        variant_name = str(variant.name)
        overrides = OmegaConf.to_container(
            variant.get("recognition_method", {}),
            resolve=True,
        )

        method_cfg = apply_recognition_overrides(base_method_cfg, overrides)

        for test_dataset in test_datasets:
            dataset_name = test_dataset.dataset_name

            for far in cfg.hyperparameter_sensitivity.fars:
                for beta in cfg.beta_list:
                    print(
                        f"[Hyperparam] variant={variant_name} "
                        f"dataset={dataset_name} far={far} beta={beta}"
                    )

                    entry = run_base_mprisk(
                        cfg=cfg,
                        method_cfg=method_cfg,
                        test_dataset=test_dataset,
                        far=float(far),
                        beta=float(beta),
                        method_name_suffix=variant_name,
                    )

                    calib = compute_mprisk_calibration_components(
                        entry["recognition_method"]
                    )

                    tune = tune_lambdas_for_components(
                        calib_components=calib["calib_components"],
                        g_unique_ids_calib=calib["g_unique_ids_calib"],
                        probe_unique_ids_calib=calib["probe_unique_ids_calib"],
                        fractions=fractions,
                        num_random=int(cfg.lambda_search.num_random),
                        log_low=float(cfg.lambda_search.log_low),
                        log_high=float(cfg.lambda_search.log_high),
                        seed=int(cfg.seed) + abs(hash(variant_name)) % 100000,
                    )

                    test_eval = evaluate_lambdas_on_test(
                        lambdas=tune["lambdas"],
                        test_components=entry["test_components"],
                        result=entry["result"],
                        fractions=fractions,
                        seed=int(cfg.seed),
                    )

                    rows.append(
                        {
                            "variant": variant_name,
                            "dataset": dataset_name,
                            "far": float(far),
                            "beta": float(beta),
                            "val_prr": tune["val_prr"],
                            "lambda_fa": tune["lambdas"][0],
                            "lambda_id": tune["lambdas"][1],
                            "lambda_fr": tune["lambdas"][2],
                            "lambda_ns": tune["lambdas"][3],
                            **{
                                f"override_{k}": v
                                for k, v in overrides.items()
                            },
                            **test_eval,
                        }
                    )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "hyperparameter_sensitivity.csv", index=False)
    return df


# ---------------------------------------------------------------------
# Cache construction
# ---------------------------------------------------------------------


def build_cache_for_main_tuning_experiments(
    cfg,
    test_datasets,
    base_method_cfg,
) -> Dict[Tuple[str, str, float, float], Dict[str, Any]]:
    cache: Dict[Tuple[str, str, float, float], Dict[str, Any]] = {}

    far_values = set()
    far_values.update([float(x) for x in cfg.far_list])

    if bool(cfg.operating_point_transfer.enabled):
        far_values.update([float(x) for x in cfg.operating_point_transfer.train_fars])
        far_values.update([float(x) for x in cfg.operating_point_transfer.eval_fars])

    if bool(cfg.cross_dataset_transfer.enabled):
        far_values.update([float(x) for x in cfg.cross_dataset_transfer.fars])

    variant_name = str(base_method_cfg.pretty_name)

    for test_dataset in test_datasets:
        dataset_name = test_dataset.dataset_name

        for far in sorted(far_values):
            for beta in cfg.beta_list:
                key = (dataset_name, variant_name, float(far), float(beta))

                print(
                    "\n" + "-" * 100 + "\n"
                    f"[Cache] dataset={dataset_name} variant={variant_name} "
                    f"far={far} beta={beta}\n"
                    + "-" * 100
                )

                entry = run_base_mprisk(
                    cfg=cfg,
                    method_cfg=base_method_cfg,
                    test_dataset=test_dataset,
                    far=float(far),
                    beta=float(beta),
                    method_name_suffix="cache",
                )

                calib = compute_mprisk_calibration_components(
                    entry["recognition_method"]
                )

                entry["calib"] = calib
                cache[key] = entry

    return cache

# ---------------------------------------------------------------------
# Main Hydra entry
# ---------------------------------------------------------------------


@hydra.main(
    config_path=str(Path(__file__).resolve().parents[1] / "configs/uncertainty_benchmark"),
    config_name="mprisk_tuning_experiments",
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
    tables_dir.mkdir(parents=True, exist_ok=True)

    test_datasets = [instantiate(x) for x in cfg.test_datasets]

    base_method_cfg = cfg.base_mprisk_method

    cache = build_cache_for_main_tuning_experiments(
        cfg=cfg,
        test_datasets=test_datasets,
        base_method_cfg=base_method_cfg,
    )

    # Save basic cache summary.
    cache_rows = []
    for key, entry in cache.items():
        dataset_name, variant_name, far, beta = key
        rm = entry["recognition_method"]
        result = entry["result"]
        masks = result["masks"]

        cache_rows.append(
            {
                "dataset": dataset_name,
                "variant": variant_name,
                "far": far,
                "beta": beta,
                "gallery_kappa": float(getattr(rm, "gallery_kappa", np.nan)),
                "calibration_kappa": entry["calib"]["calibration_kappa"],
                "base_error_rate": float(np.mean(masks["any_error"])),
                "base_false_accept": int(np.sum(masks["false_accept"])),
                "base_false_reject": int(np.sum(masks["false_reject"])),
                "base_misidentification": int(np.sum(masks["misidentification"])),
            }
        )

    pd.DataFrame(cache_rows).to_csv(tables_dir / "cache_summary.csv", index=False)

    validation_size_df = run_validation_size_ablation(cfg, cache, tables_dir)
    operating_point_df = run_operating_point_transfer(cfg, cache, tables_dir)
    cross_dataset_df = run_cross_dataset_transfer(cfg, cache, tables_dir)
    sensitivity_df = run_hyperparameter_sensitivity(
        cfg=cfg,
        test_datasets=test_datasets,
        base_method_cfg=base_method_cfg,
        out_dir=tables_dir,
    )

    print("\nSaved tuning experiment outputs:")
    print(tables_dir / "cache_summary.csv")

    if len(validation_size_df) > 0:
        print(tables_dir / "validation_size_ablation.csv")
        print(tables_dir / "validation_size_ablation_summary.csv")

    if len(operating_point_df) > 0:
        print(tables_dir / "operating_point_transfer.csv")

    if len(cross_dataset_df) > 0:
        print(tables_dir / "cross_dataset_transfer.csv")

    if len(sensitivity_df) > 0:
        print(tables_dir / "hyperparameter_sensitivity.csv")

    print(f"\nExperiment directory: {exp_dir}")


if __name__ == "__main__":
    main()