#!/usr/bin/env python3

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import OmegaConf

from evaluation.reproducibility import seed_everything

from experiments.mprisk_core_experiments import (
    build_tester,
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
    extract_method_arrays,
)


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------


def copy_cfg(cfg_node):
    return OmegaConf.create(OmegaConf.to_container(cfg_node, resolve=True))


def apply_recognition_overrides(method_cfg, overrides: Optional[Dict[str, Any]]):
    method_cfg = copy_cfg(method_cfg)
    if overrides is None:
        return method_cfg

    for key, value in overrides.items():
        OmegaConf.update(
            method_cfg,
            f"recognition_method.{key}",
            value,
            force_add=True,
        )

    return method_cfg


def np_trapz(y, x) -> float:
    return float(
        np.trapezoid(np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64))
    )


def score_quantiles(score: np.ndarray) -> Dict[str, float]:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    qs = np.quantile(score[np.isfinite(score)], [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    return {
        "q000": float(qs[0]),
        "q001": float(qs[1]),
        "q005": float(qs[2]),
        "q050": float(qs[3]),
        "q095": float(qs[4]),
        "q099": float(qs[5]),
        "q100": float(qs[6]),
    }


def method_key(
    dataset_name: str, pretty_name: str, far: float, beta: float
) -> Tuple[str, str, float, float]:
    return dataset_name, pretty_name, float(far), float(beta)


def valid_probability_from_predicted_unc(
    predicted_unc: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Convert repository uncertainty convention into P(error), if possible.

    For calibrated HolUE/ScalarRiskCalibration:
      predicted_unc = -P(correct)
      therefore P(error) = 1 + predicted_unc.

    If input is already in [0,1], we treat it as P(error).
    Otherwise return None.
    """
    u = np.asarray(predicted_unc, dtype=np.float64).reshape(-1)

    if np.nanmin(u) >= -1.001 and np.nanmax(u) <= 0.001:
        p_error = 1.0 + u
        return np.clip(p_error, 0.0, 1.0)

    if np.nanmin(u) >= -0.001 and np.nanmax(u) <= 1.001:
        return np.clip(u, 0.0, 1.0)

    return None


# ---------------------------------------------------------------------
# Running methods
# ---------------------------------------------------------------------


def run_configured_methods(cfg, test_datasets, out_dir: Path):
    """
    Run all methods from config and collect outputs.
    """
    results = {}
    runtime_rows = []

    for test_dataset in test_datasets:
        dataset_name = test_dataset.dataset_name

        for method_cfg in cfg.open_set_identification_methods:
            base_pretty_name = str(method_cfg.pretty_name)

            for far in cfg.far_list:
                for beta in cfg.beta_list:
                    pretty_name = base_pretty_name
                    if len(cfg.beta_list) > 1:
                        pretty_name += f"_beta-{beta}"

                    print("=" * 100)
                    print(
                        f"[Diagnostics] dataset={dataset_name} method={pretty_name} far={far} beta={beta}"
                    )
                    print("=" * 100)

                    recognition_method = instantiate(method_cfg.recognition_method)

                    maybe_attach_calibration_set(cfg, recognition_method, dataset_name)
                    maybe_attach_dataset_temperature(
                        cfg,
                        recognition_method,
                        base_pretty_name,
                        dataset_name,
                    )

                    recognition_method.far = float(far)
                    recognition_method.beta = float(beta)

                    unique_method_name = (
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
                        method_name=unique_method_name,
                        pretty_name=pretty_name,
                    )

                    t0 = time.perf_counter()
                    result = run_method_raw(tt, gallery_name="g1")
                    elapsed = time.perf_counter() - t0

                    rm = result["recognition_method"]
                    arrays = extract_method_arrays(rm)

                    key = method_key(dataset_name, pretty_name, float(far), float(beta))
                    results[key] = {
                        "result": result,
                        "arrays": arrays,
                        "recognition_method": rm,
                        "test_dataset": test_dataset,
                        "method_cfg": method_cfg,
                        "pretty_name": pretty_name,
                        "far": float(far),
                        "beta": float(beta),
                    }

                    n = len(result["probe_unique_ids"])
                    runtime_rows.append(
                        {
                            "dataset": dataset_name,
                            "method": pretty_name,
                            "far": float(far),
                            "beta": float(beta),
                            "num_probes": int(n),
                            "elapsed_sec_total": float(elapsed),
                            "elapsed_ms_per_probe": float(1000.0 * elapsed / max(n, 1)),
                        }
                    )

    pd.DataFrame(runtime_rows).to_csv(out_dir / "runtime_overhead.csv", index=False)
    return results


# ---------------------------------------------------------------------
# Experiment 1: mixed-prior necessity
# ---------------------------------------------------------------------


def collapsed_unknown_risk(
    result: Dict[str, Any], arrays: Dict[str, np.ndarray]
) -> Optional[np.ndarray]:
    """
    Collapsed unknown baseline risk:

      R_collapsed = 1 - probability of the selected OSR action.

    If accepted as gallery class i:
      action probability = pi_i.

    If rejected:
      action probability = pi_0.

    This intentionally collapses the continuous unknown identity space to one
    reject action and therefore removes the non-specificity correction.
    """
    if "mean_probs" not in arrays:
        return None

    mean_probs = np.asarray(arrays["mean_probs"], dtype=np.float64)
    if mean_probs.ndim != 2:
        return None

    n, K = mean_probs.shape

    if "oog_prob" in arrays:
        oog_prob = np.asarray(arrays["oog_prob"], dtype=np.float64).reshape(-1)
    else:
        oog_prob = 1.0 - np.sum(mean_probs, axis=1)

    predicted_id = np.asarray(result["predicted_id"], dtype=int).reshape(-1)
    was_rejected = np.asarray(result["was_rejected"], dtype=bool).reshape(-1)

    pi_hat = mean_probs[np.arange(n), predicted_id]
    pi_action = np.where(was_rejected, oog_prob, pi_hat)

    return 1.0 - np.clip(pi_action, 0.0, 1.0)


def mixed_prior_variant_scores(
    result: Dict[str, Any], arrays: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    scores = {}

    if all(k in arrays for k in ["r_fa", "r_id", "r_fr", "r_ns"]):
        r_fa = np.asarray(arrays["r_fa"], dtype=np.float64).reshape(-1)
        r_id = np.asarray(arrays["r_id"], dtype=np.float64).reshape(-1)
        r_fr = np.asarray(arrays["r_fr"], dtype=np.float64).reshape(-1)
        r_ns = np.asarray(arrays["r_ns"], dtype=np.float64).reshape(-1)

        scores["r_FA"] = r_fa
        scores["r_ID"] = r_id
        scores["r_FR"] = r_fr
        scores["r_NS"] = r_ns

        scores["ordinary_risk_no_NS"] = r_fa + r_id + r_fr
        scores["equal_full_risk"] = r_fa + r_id + r_fr + r_ns

    if "mprisk" in arrays:
        scores["MPRisk_current"] = np.asarray(
            arrays["mprisk"], dtype=np.float64
        ).reshape(-1)

    collapsed = collapsed_unknown_risk(result, arrays)
    if collapsed is not None:
        scores["collapsed_unknown_risk"] = collapsed

    if "oog_nonspecificity" in arrays:
        scores["unknown_nonspecificity"] = np.asarray(
            arrays["oog_nonspecificity"],
            dtype=np.float64,
        ).reshape(-1)

    return scores


def run_mixed_prior_necessity(cfg, results: Dict, out_dir: Path):
    if not bool(cfg.mixed_prior_necessity.enabled):
        return pd.DataFrame()

    rows = []
    fractions = np.linspace(
        float(cfg.rejection_fractions[0]),
        float(cfg.rejection_fractions[1]),
        int(cfg.rejection_fractions[2]),
    )

    for key, entry in results.items():
        dataset_name, pretty_name, far, beta = key

        # Usually this experiment is only meaningful for MPRisk methods.
        if cfg.mixed_prior_necessity.method_filter not in pretty_name:
            continue

        result = entry["result"]
        arrays = entry["arrays"]
        masks = result["masks"]

        scores = mixed_prior_variant_scores(result, arrays)

        for variant_name, score in scores.items():
            prr, curve, _, _ = self_normalized_prr(
                uncertainty=score,
                predicted_id=result["predicted_id"],
                was_rejected=result["was_rejected"],
                g_unique_ids=result["g_unique_ids"],
                probe_unique_ids=result["probe_unique_ids"],
                fractions=fractions,
                metric_name="f1_class",
                seed=int(cfg.seed),
            )

            row = {
                "dataset": dataset_name,
                "source_method": pretty_name,
                "variant": variant_name,
                "far": far,
                "beta": beta,
                "prr_f1": prr,
                "any_error_auroc": safe_auc(masks["any_error"], score),
                "any_error_auprc": safe_auprc(masks["any_error"], score),
                "false_accept_auroc": safe_auc(masks["false_accept"], score),
                "false_reject_auroc": safe_auc(masks["false_reject"], score),
                "misidentification_auroc": safe_auc(masks["misidentification"], score),
                "f1_at_0_filter": float(curve["f1_class"].iloc[0]),
                "f1_at_max_filter": float(curve["f1_class"].iloc[-1]),
            }

            row.update({f"score_{k}": v for k, v in score_quantiles(score).items()})
            rows.append(row)

        # Dedicated NS separation table: false rejects vs true rejects.
        if "r_ns" in arrays:
            r_ns = np.asarray(arrays["r_ns"], dtype=np.float64).reshape(-1)

            for group_name, group_mask in {
                "false_reject": masks["false_reject"],
                "true_reject": masks["true_reject"],
                "false_accept": masks["false_accept"],
                "misidentification": masks["misidentification"],
                "correct": masks["correct"],
            }.items():
                vals = r_ns[group_mask]
                if len(vals) == 0:
                    continue
                rows.append(
                    {
                        "dataset": dataset_name,
                        "source_method": pretty_name,
                        "variant": "r_NS_group_summary",
                        "group": group_name,
                        "far": far,
                        "beta": beta,
                        "count": int(len(vals)),
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "median": float(np.median(vals)),
                        "q90": float(np.quantile(vals, 0.9)),
                        "q99": float(np.quantile(vals, 0.99)),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "mixed_prior_necessity.csv", index=False)
    return df


# ---------------------------------------------------------------------
# Experiment 2: reliability / calibration quality
# ---------------------------------------------------------------------


def calibration_metrics_from_prob(
    p_error: np.ndarray,
    y_error: np.ndarray,
    num_bins: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    p_error = np.asarray(p_error, dtype=np.float64).reshape(-1)
    y_error = np.asarray(y_error, dtype=bool).reshape(-1)

    p_error = np.clip(p_error, 1e-8, 1.0 - 1e-8)

    brier = float(np.mean((p_error - y_error.astype(np.float64)) ** 2))
    nll = float(
        -np.mean(
            y_error.astype(np.float64) * np.log(p_error)
            + (1.0 - y_error.astype(np.float64)) * np.log(1.0 - p_error)
        )
    )

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_id = np.digitize(p_error, bins, right=True)
    bin_id = np.clip(bin_id, 1, num_bins)

    rows = []
    ece = 0.0
    mce = 0.0

    for b in range(1, num_bins + 1):
        mask = bin_id == b
        count = int(np.sum(mask))
        if count == 0:
            rows.append(
                {
                    "bin": b,
                    "left": bins[b - 1],
                    "right": bins[b],
                    "count": 0,
                    "mean_pred_error": np.nan,
                    "empirical_error": np.nan,
                    "gap": np.nan,
                }
            )
            continue

        mean_pred = float(np.mean(p_error[mask]))
        empirical = float(np.mean(y_error[mask]))
        gap = abs(mean_pred - empirical)

        ece += count / len(p_error) * gap
        mce = max(mce, gap)

        rows.append(
            {
                "bin": b,
                "left": bins[b - 1],
                "right": bins[b],
                "count": count,
                "mean_pred_error": mean_pred,
                "empirical_error": empirical,
                "gap": gap,
            }
        )

    metrics = {
        "brier": brier,
        "nll": nll,
        "ece": float(ece),
        "mce": float(mce),
        "error_auroc": safe_auc(y_error, p_error),
        "error_auprc": safe_auprc(y_error, p_error),
    }

    return metrics, pd.DataFrame(rows)


def plot_reliability_diagram(bin_df: pd.DataFrame, title: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    valid = bin_df["count"].values > 0
    df = bin_df[valid].copy()

    fig, ax = plt.subplots(figsize=(5.2, 5.2))

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.5)

    widths = df["right"].values - df["left"].values
    centers = 0.5 * (df["left"].values + df["right"].values)

    ax.bar(
        centers,
        df["empirical_error"].values,
        width=0.9 * widths,
        alpha=0.65,
        edgecolor="black",
        label="empirical error",
    )

    ax.scatter(
        df["mean_pred_error"].values,
        df["empirical_error"].values,
        c="red",
        s=35,
        zorder=5,
        label="bins",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted error probability")
    ax.set_ylabel("Empirical error frequency")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()

    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_reliability_experiment(cfg, results: Dict, out_dir: Path):
    if not bool(cfg.reliability.enabled):
        return pd.DataFrame()

    rows = []
    bins_out_dir = out_dir / "reliability_bins"
    plots_out_dir = out_dir / "reliability_plots"
    bins_out_dir.mkdir(parents=True, exist_ok=True)
    plots_out_dir.mkdir(parents=True, exist_ok=True)

    method_allowlist = set(str(x) for x in cfg.reliability.methods)

    for key, entry in results.items():
        dataset_name, pretty_name, far, beta = key

        if method_allowlist and pretty_name not in method_allowlist:
            continue

        result = entry["result"]
        masks = result["masks"]

        p_error = valid_probability_from_predicted_unc(result["predicted_unc"])
        if p_error is None:
            print(f"[Reliability] skip {pretty_name}: score is not probability-like")
            continue

        metrics, bin_df = calibration_metrics_from_prob(
            p_error=p_error,
            y_error=masks["any_error"],
            num_bins=int(cfg.reliability.num_bins),
        )

        rows.append(
            {
                "dataset": dataset_name,
                "method": pretty_name,
                "far": far,
                "beta": beta,
                **metrics,
            }
        )

        stem = f"{slugify(dataset_name)}_{slugify(pretty_name)}_far_{far}_beta_{beta}"
        bin_df.to_csv(bins_out_dir / f"{stem}.csv", index=False)

        title = f"{pretty_name}, {dataset_name}, FPIR={far}"
        plot_reliability_diagram(
            bin_df,
            title=title,
            out_path=plots_out_dir / stem,
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "reliability_metrics.csv", index=False)
    return df


# ---------------------------------------------------------------------
# Experiment 3: paired bootstrap significance
# ---------------------------------------------------------------------


def prr_for_bootstrap_sample(
    score: np.ndarray,
    result: Dict[str, Any],
    idx: np.ndarray,
    fractions: np.ndarray,
    seed: int,
) -> float:
    return self_normalized_prr(
        uncertainty=np.asarray(score)[idx],
        predicted_id=np.asarray(result["predicted_id"])[idx],
        was_rejected=np.asarray(result["was_rejected"])[idx],
        g_unique_ids=result["g_unique_ids"],
        probe_unique_ids=np.asarray(result["probe_unique_ids"])[idx],
        fractions=fractions,
        metric_name="f1_class",
        seed=seed,
    )[0]


def run_bootstrap_significance(cfg, results: Dict, out_dir: Path):
    if not bool(cfg.bootstrap.enabled):
        return pd.DataFrame()

    rows = []
    fractions = np.linspace(
        float(cfg.rejection_fractions[0]),
        float(cfg.rejection_fractions[1]),
        int(cfg.rejection_fractions[2]),
    )

    num_bootstrap = int(cfg.bootstrap.num_bootstrap)
    rng = np.random.default_rng(int(cfg.seed))

    for pair_cfg in cfg.bootstrap.pairs:
        dataset_name = str(pair_cfg.dataset)
        far = float(pair_cfg.far)
        beta = float(pair_cfg.beta)
        method_a = str(pair_cfg.method_a)
        method_b = str(pair_cfg.method_b)

        key_a = method_key(dataset_name, method_a, far, beta)
        key_b = method_key(dataset_name, method_b, far, beta)

        if key_a not in results:
            print(f"[Bootstrap] missing {key_a}")
            continue
        if key_b not in results:
            print(f"[Bootstrap] missing {key_b}")
            continue

        res_a = results[key_a]["result"]
        res_b = results[key_b]["result"]

        if not np.array_equal(res_a["probe_unique_ids"], res_b["probe_unique_ids"]):
            print(f"[Bootstrap] skip {method_a} vs {method_b}: probe order differs")
            continue

        score_a = res_a["predicted_unc"]
        score_b = res_b["predicted_unc"]

        n = len(score_a)

        prr_a_full = prr_for_bootstrap_sample(
            score_a,
            res_a,
            np.arange(n),
            fractions,
            seed=int(cfg.seed),
        )
        prr_b_full = prr_for_bootstrap_sample(
            score_b,
            res_b,
            np.arange(n),
            fractions,
            seed=int(cfg.seed),
        )

        diffs = []
        for b in range(num_bootstrap):
            idx = rng.integers(0, n, size=n)

            prr_a = prr_for_bootstrap_sample(
                score_a,
                res_a,
                idx,
                fractions,
                seed=int(cfg.seed) + b,
            )
            prr_b = prr_for_bootstrap_sample(
                score_b,
                res_b,
                idx,
                fractions,
                seed=int(cfg.seed) + 100000 + b,
            )

            diffs.append(prr_a - prr_b)

        diffs = np.asarray(diffs, dtype=np.float64)

        rows.append(
            {
                "dataset": dataset_name,
                "far": far,
                "beta": beta,
                "method_a": method_a,
                "method_b": method_b,
                "prr_a_full": prr_a_full,
                "prr_b_full": prr_b_full,
                "delta_full": prr_a_full - prr_b_full,
                "delta_boot_mean": float(np.mean(diffs)),
                "delta_boot_std": float(np.std(diffs)),
                "ci95_low": float(np.quantile(diffs, 0.025)),
                "ci95_high": float(np.quantile(diffs, 0.975)),
                "p_delta_le_0": float(np.mean(diffs <= 0.0)),
                "num_bootstrap": num_bootstrap,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "bootstrap_prr_differences.csv", index=False)
    return df


# ---------------------------------------------------------------------
# Experiment 4: qualitative examples
# ---------------------------------------------------------------------


def error_kind_for_index(masks: Dict[str, np.ndarray], i: int) -> str:
    if masks["false_accept"][i]:
        return "false_accept"
    if masks["false_reject"][i]:
        return "false_reject"
    if masks["misidentification"][i]:
        return "misidentification"
    if masks["true_reject"][i]:
        return "true_reject"
    if masks["true_accept_true_ident"][i]:
        return "true_accept_true_ident"
    return "unknown"


def top_indices_by_score(score: np.ndarray, mask: np.ndarray, k: int) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return idx
    order = np.argsort(score[idx])[::-1]
    return idx[order[:k]]


def infer_template_ids(test_dataset, n_probes: int) -> np.ndarray:
    """
    Recognition_test pools probe templates in sorted unique-template order for
    PoolingDefault. This matches np.unique(probe_templates) in current pipeline.
    """
    try:
        template_ids = np.unique(test_dataset.probe_templates)
        if len(template_ids) == n_probes:
            return template_ids
    except Exception:
        pass

    return np.arange(n_probes)


def run_qualitative_examples(cfg, results: Dict, out_dir: Path):
    if not bool(cfg.qualitative.enabled):
        return pd.DataFrame()

    rows = []

    for item in cfg.qualitative.methods:
        dataset_name = str(item.dataset)
        method = str(item.method)
        far = float(item.far)
        beta = float(item.beta)

        key = method_key(dataset_name, method, far, beta)
        if key not in results:
            print(f"[Qualitative] missing {key}")
            continue

        entry = results[key]
        result = entry["result"]
        arrays = entry["arrays"]
        test_dataset = entry["test_dataset"]

        n = len(result["probe_unique_ids"])
        probe_template_ids = infer_template_ids(test_dataset, n)

        masks = result["masks"]

        if "mean_probs" in arrays:
            mean_probs = np.asarray(arrays["mean_probs"], dtype=np.float64)
            top_gallery = np.argsort(mean_probs, axis=1)[:, ::-1]
        else:
            mean_probs = None
            top_gallery = None

        score_sources = {
            "predicted_unc": np.asarray(result["predicted_unc"], dtype=np.float64),
        }

        for name in ["r_fa", "r_id", "r_fr", "r_ns", "mprisk", "risk_main", "risk_ns"]:
            if name in arrays:
                score_sources[name] = np.asarray(
                    arrays[name], dtype=np.float64
                ).reshape(-1)

        for score_name, score in score_sources.items():
            for group_name, group_mask in {
                "any_error": masks["any_error"],
                "false_accept": masks["false_accept"],
                "false_reject": masks["false_reject"],
                "misidentification": masks["misidentification"],
                "correct": masks["correct"],
            }.items():
                selected = top_indices_by_score(
                    score,
                    group_mask,
                    int(cfg.qualitative.top_k_per_group),
                )

                for rank, i in enumerate(selected):
                    predicted_gallery_index = int(result["predicted_id"][i])
                    predicted_subject = int(
                        result["g_unique_ids"][predicted_gallery_index]
                    )

                    row = {
                        "dataset": dataset_name,
                        "method": method,
                        "far": far,
                        "beta": beta,
                        "score_name": score_name,
                        "group": group_name,
                        "rank": rank,
                        "probe_index": int(i),
                        "probe_template_id": int(probe_template_ids[i]),
                        "true_subject_id": int(result["probe_unique_ids"][i]),
                        "predicted_gallery_index": predicted_gallery_index,
                        "predicted_subject_id": predicted_subject,
                        "was_rejected": bool(result["was_rejected"][i]),
                        "error_kind": error_kind_for_index(masks, i),
                        "score_value": float(score[i]),
                    }

                    for name in [
                        "r_fa",
                        "r_id",
                        "r_fr",
                        "r_ns",
                        "mprisk",
                        "risk_main",
                        "risk_ns",
                        "oog_prob",
                        "oog_nonspecificity",
                    ]:
                        if name in arrays:
                            row[name] = float(np.asarray(arrays[name]).reshape(-1)[i])

                    if mean_probs is not None:
                        row["max_gallery_prob"] = float(np.max(mean_probs[i]))
                        row["sum_gallery_prob"] = float(np.sum(mean_probs[i]))

                    if top_gallery is not None:
                        for j in range(
                            min(
                                int(cfg.qualitative.num_nearest_gallery),
                                top_gallery.shape[1],
                            )
                        ):
                            gid_index = int(top_gallery[i, j])
                            row[f"top{j+1}_gallery_index"] = gid_index
                            row[f"top{j+1}_subject_id"] = int(
                                result["g_unique_ids"][gid_index]
                            )
                            row[f"top{j+1}_posterior_prob"] = float(
                                mean_probs[i, gid_index]
                            )

                    rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "qualitative_examples.csv", index=False)
    return df


# ---------------------------------------------------------------------
# Experiment 5: embedding-quality stress test
# ---------------------------------------------------------------------


def run_quality_stress(cfg, test_datasets, base_method_cfg, out_dir: Path):
    """
    Rerun MPRisk while changing kappa_input_scale.

    This is not an image-level corruption experiment. It is an embedding-space
    quality stress test: lowering kappa_input_scale makes p(z|x) more diffuse.
    """
    if not bool(cfg.quality_stress.enabled):
        return pd.DataFrame()

    rows = []
    fractions = np.linspace(
        float(cfg.rejection_fractions[0]),
        float(cfg.rejection_fractions[1]),
        int(cfg.rejection_fractions[2]),
    )

    for scale in cfg.quality_stress.kappa_input_scales:
        method_cfg = apply_recognition_overrides(
            base_method_cfg,
            {
                "kappa_input_scale": float(scale),
                "tune_lambdas": False,
                "use_calibration": False,
            },
        )

        for test_dataset in test_datasets:
            dataset_name = test_dataset.dataset_name

            for far in cfg.quality_stress.fars:
                for beta in cfg.beta_list:
                    print(
                        f"[QualityStress] dataset={dataset_name} scale={scale} far={far} beta={beta}"
                    )

                    recognition_method = instantiate(method_cfg.recognition_method)
                    maybe_attach_calibration_set(cfg, recognition_method, dataset_name)

                    recognition_method.far = float(far)
                    recognition_method.beta = float(beta)

                    unique_method_name = (
                        f"quality_stress_scale_{scale}"
                        f"_dataset_{slugify(dataset_name)}"
                        f"_far_{far}_beta_{beta}"
                    )

                    tt = build_tester(
                        cfg=cfg,
                        method_cfg=method_cfg,
                        test_dataset=test_dataset,
                        recognition_method=recognition_method,
                        method_name=unique_method_name,
                        pretty_name=f"quality_stress_scale_{scale}",
                    )

                    result = run_method_raw(tt, gallery_name="g1")
                    arrays = extract_method_arrays(result["recognition_method"])
                    masks = result["masks"]

                    scores = mixed_prior_variant_scores(result, arrays)

                    for score_name, score in scores.items():
                        prr, curve, _, _ = self_normalized_prr(
                            uncertainty=score,
                            predicted_id=result["predicted_id"],
                            was_rejected=result["was_rejected"],
                            g_unique_ids=result["g_unique_ids"],
                            probe_unique_ids=result["probe_unique_ids"],
                            fractions=fractions,
                            metric_name="f1_class",
                            seed=int(cfg.seed),
                        )

                        rows.append(
                            {
                                "dataset": dataset_name,
                                "far": float(far),
                                "beta": float(beta),
                                "kappa_input_scale": float(scale),
                                "score": score_name,
                                "prr_f1": prr,
                                "false_reject_auroc": safe_auc(
                                    masks["false_reject"], score
                                ),
                                "true_reject_mean_score": (
                                    float(np.mean(score[masks["true_reject"]]))
                                    if np.any(masks["true_reject"])
                                    else np.nan
                                ),
                                "false_reject_mean_score": (
                                    float(np.mean(score[masks["false_reject"]]))
                                    if np.any(masks["false_reject"])
                                    else np.nan
                                ),
                                "f1_at_max_filter": float(curve["f1_class"].iloc[-1]),
                            }
                        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "quality_stress.csv", index=False)
    return df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


@hydra.main(
    config_path=str(
        Path(__file__).resolve().parents[1] / "configs/uncertainty_benchmark"
    ),
    config_name="mprisk_diagnostics_experiments",
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

    results = run_configured_methods(
        cfg=cfg,
        test_datasets=test_datasets,
        out_dir=tables_dir,
    )

    mixed_df = run_mixed_prior_necessity(cfg, results, tables_dir)
    reliability_df = run_reliability_experiment(cfg, results, tables_dir)
    bootstrap_df = run_bootstrap_significance(cfg, results, tables_dir)
    qualitative_df = run_qualitative_examples(cfg, results, tables_dir)

    if "quality_stress_base_method" in cfg:
        quality_stress_df = run_quality_stress(
            cfg=cfg,
            test_datasets=test_datasets,
            base_method_cfg=cfg.quality_stress_base_method,
            out_dir=tables_dir,
        )
    else:
        quality_stress_df = pd.DataFrame()

    print("\nSaved diagnostics outputs:")
    print(tables_dir / "runtime_overhead.csv")

    if len(mixed_df) > 0:
        print(tables_dir / "mixed_prior_necessity.csv")
    if len(reliability_df) > 0:
        print(tables_dir / "reliability_metrics.csv")
    if len(bootstrap_df) > 0:
        print(tables_dir / "bootstrap_prr_differences.csv")
    if len(qualitative_df) > 0:
        print(tables_dir / "qualitative_examples.csv")
    if len(quality_stress_df) > 0:
        print(tables_dir / "quality_stress.csv")

    print(f"\nExperiment directory: {exp_dir}")


if __name__ == "__main__":
    main()
