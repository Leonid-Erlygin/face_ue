from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from .calibration import ErrorCalibratorBundle
from .data import nested_corpus_protocols, stratified_protocol_split
from .embedders import BaseTextEmbedder
from .methods import ModernUncertaintyModel, PosteriorModelConfig
from .metrics import (
    calibration_metrics,
    open_set_retrieval_metrics,
    ranking_metrics,
    retrieval_error_masks,
    uncertainty_detection_metrics,
)
from .query_uncertainty import query_embeddings_and_kappa
from .types import RetrievalProtocol, RetrievalScores


def _jsonable(x):
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _encode_protocol(
    protocol: RetrievalProtocol,
    embedder: BaseTextEmbedder,
    rewrites: Optional[Mapping[str, Sequence[str]]],
    default_query_kappa: float,
    query_mean_mode: str,
):
    corpus_ids = tuple(map(str, protocol.corpus.keys()))
    gallery = embedder.encode_documents([protocol.corpus[d] for d in corpus_ids])
    q, kappa, rbar = query_embeddings_and_kappa(
        protocol.query_ids,
        protocol.queries,
        embedder,
        rewrites=rewrites,
        default_kappa=default_query_kappa,
        mean_mode=query_mean_mode,
    )
    return corpus_ids, np.asarray(gallery), q, kappa, rbar


def _take_rows(protocol: RetrievalProtocol, full_protocol: RetrievalProtocol, arr: np.ndarray) -> np.ndarray:
    pos = {qid: i for i, qid in enumerate(full_protocol.query_ids)}
    return np.asarray(arr)[[pos[q] for q in protocol.query_ids]]


def _multiple_relevance_diagnostic(
    protocol: RetrievalProtocol,
    scores: RetrievalScores,
) -> Dict[str, float]:
    known = np.asarray(protocol.known_mask, dtype=bool)
    counts = np.asarray([len(x) for x in protocol.relevant_doc_ids], dtype=int)
    out = {
        "mean_relevant_docs_known": float(np.mean(counts[known])) if np.any(known) else np.nan,
        "fraction_multirelevant_known": float(np.mean(counts[known] > 1)) if np.any(known) else np.nan,
    }
    # The dissertation identity-risk term assumes a single correct gallery class.
    # With qrels, probability on a second relevant passage is not a retrieval error.
    # Quantify exactly how much r_ID over-counts that mass when a dense posterior is
    # available. This is a diagnostic, not a qrel-informed inference method.
    probs = np.asarray(scores.mean_known_probs)
    if probs.ndim == 2 and probs.shape[1] == len(scores.corpus_ids):
        did_to_idx = {d: i for i, d in enumerate(scores.corpus_ids)}
        over = []
        set_aware = []
        original = []
        for i in np.flatnonzero(known & ~np.asarray(scores.was_rejected, dtype=bool)):
            rel_idx = [did_to_idx[d] for d in protocol.relevant_doc_ids[i] if d in did_to_idx]
            if not rel_idx:
                continue
            pred = int(scores.predicted_indices[i])
            p0 = float(scores.unknown_prob[i])
            p_pred = float(probs[i, pred])
            original_rid = max(0.0, 1.0 - p0 - p_pred)
            acceptable_mass = float(np.sum(probs[i, rel_idx]))
            set_rid = max(0.0, 1.0 - p0 - acceptable_mass)
            original.append(original_rid)
            set_aware.append(set_rid)
            over.append(original_rid - set_rid)
        out.update({
            "mean_original_identity_risk_accepted": float(np.mean(original)) if original else np.nan,
            "mean_set_aware_identity_risk_accepted": float(np.mean(set_aware)) if set_aware else np.nan,
            "mean_multirelevance_risk_overcount": float(np.mean(over)) if over else np.nan,
        })
    else:
        out.update({
            "mean_original_identity_risk_accepted": np.nan,
            "mean_set_aware_identity_risk_accepted": np.nan,
            "mean_multirelevance_risk_overcount": np.nan,
        })
    return out


def _score_table(
    protocol: RetrievalProtocol,
    scores: RetrievalScores,
    calibrated: Mapping[str, np.ndarray],
) -> list[dict]:
    masks = retrieval_error_masks(protocol, scores)
    predicted = scores.predicted_doc_ids
    rows = []
    for i, qid in enumerate(protocol.query_ids):
        row = {
            "query_id": qid,
            "known": bool(protocol.known_mask[i]),
            "designated_unknown": bool(protocol.designated_unknown_mask[i]),
            "num_relevant": len(protocol.relevant_doc_ids[i]),
            "predicted_doc_id": predicted[i],
            "rejected": bool(scores.was_rejected[i]),
            "correct_retrieval": bool(masks["correct_retrieval"][i]),
            "false_accept": bool(masks["false_accept"][i]),
            "false_reject": bool(masks["false_reject"][i]),
            "misretrieval": bool(masks["misretrieval"][i]),
            "any_error": bool(masks["any_error"][i]),
            "p_unknown": float(scores.unknown_prob[i]),
            "unknown_nonspecificity": float(scores.unknown_nonspecificity[i]),
            "kl_1": float(scores.kl_1[i]),
            "kl_2": float(scores.kl_2[i]),
        }
        for name, vals in scores.scores.items():
            row[f"raw__{name}"] = float(np.asarray(vals)[i])
        for name, vals in calibrated.items():
            row[f"p_error__{name}"] = float(np.asarray(vals)[i])
        rows.append(row)
    return rows


def _save_result(output_dir: Path, summary: dict, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True), encoding="utf-8"
    )
    if rows:
        with (output_dir / "per_query.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def run_retrieval_experiment(
    protocol: RetrievalProtocol,
    embedder: BaseTextEmbedder,
    model_config: PosteriorModelConfig,
    *,
    rewrites: Optional[Mapping[str, Sequence[str]]] = None,
    default_query_kappa: float = 300.0,
    query_mean_mode: str = "original",
    calibration_fraction: float = 0.3,
    seed: int = 777,
    output_dir: Optional[str | Path] = None,
) -> dict:
    """Run the primary Open-Set Evidence Retrieval (OSER) experiment.

    The gallery concentration and all error-probability calibrators are fit on a
    query-disjoint validation split.  The active corpus is shared across splits,
    preserving the retrieval problem.  Multiple qrel passages are treated as a
    *set* of acceptable outputs for evaluation and audited separately because the
    original MPRisk identity term assumes one correct gallery identity.
    """

    cal_protocol, test_protocol = stratified_protocol_split(
        protocol, calibration_fraction=calibration_fraction, seed=seed
    )
    corpus_ids, gallery, q_all, k_all, rbar_all = _encode_protocol(
        protocol, embedder, rewrites, default_query_kappa, query_mean_mode
    )
    q_cal = _take_rows(cal_protocol, protocol, q_all)
    k_cal = _take_rows(cal_protocol, protocol, k_all)
    r_cal = _take_rows(cal_protocol, protocol, rbar_all)
    q_test = _take_rows(test_protocol, protocol, q_all)
    k_test = _take_rows(test_protocol, protocol, k_all)
    r_test = _take_rows(test_protocol, protocol, rbar_all)

    model = ModernUncertaintyModel(model_config)
    fitted_kappa = model.fit_gallery_kappa(q_cal, k_cal, gallery, cal_protocol.known_mask)
    cal_scores = model.score(
        cal_protocol.query_ids, corpus_ids, q_cal, k_cal, gallery, r_cal
    )
    test_scores = model.score(
        test_protocol.query_ids, corpus_ids, q_test, k_test, gallery, r_test
    )

    cal_masks = retrieval_error_masks(cal_protocol, cal_scores)
    test_masks = retrieval_error_masks(test_protocol, test_scores)
    y_cal = cal_masks["any_error"].astype(int)

    # HolUE's native object is two-dimensional (KL1, KL2). Do not give a scalar
    # sum an unfair privileged status; learn a simple validation-only logistic map.
    scalar_cal_scores = {
        k: v for k, v in cal_scores.scores.items()
        if k != "holue_kl_sum"
    }
    bundle = ErrorCalibratorBundle.fit(
        scalar_cal_scores, cal_scores.kl_1, cal_scores.kl_2, y_cal,
        exclude=("holue", "unknown_probability"),
    )
    calibrated_test = bundle.apply(
        {k: v for k, v in test_scores.scores.items() if k != "holue_kl_sum"},
        test_scores.kl_1,
        test_scores.kl_2,
    )

    raw_detection = {}
    for name, u in test_scores.scores.items():
        if name == "holue_kl_sum" or not np.any(np.isfinite(u)):
            continue
        raw_detection[name] = uncertainty_detection_metrics(test_masks, u)
    # HolUE needs validation-fitted two-feature direction, so its calibrated
    # probability is also its held-out ranking score.
    if "holue" in calibrated_test:
        raw_detection["holue"] = uncertainty_detection_metrics(
            test_masks, calibrated_test["holue"]
        )

    calibration = {}
    for name, p in calibrated_test.items():
        ok = np.isfinite(p)
        if np.any(ok):
            calibration[name] = calibration_metrics(
                test_masks["any_error"][ok].astype(int), np.asarray(p)[ok]
            )

    rank = {}
    # Standard full-ranking metrics are useful, but the dense NxK matrix is not
    # necessary for the scientific OSER result and is skipped at very large scale.
    if len(test_protocol.query_ids) * len(corpus_ids) <= model_config.streaming_threshold_elements:
        rank = ranking_metrics(q_test, gallery, test_protocol)
    else:
        rank["recall@1_from_streamed_top1"] = float(np.mean(
            [
                test_scores.predicted_doc_ids[i] in set(test_protocol.relevant_doc_ids[i])
                for i in np.flatnonzero(test_protocol.known_mask)
            ]
        )) if np.any(test_protocol.known_mask) else np.nan

    from sklearn.metrics import average_precision_score, roc_auc_score
    unknown_y = (~np.asarray(test_protocol.known_mask, dtype=bool)).astype(int)
    unknown_state_detection = {}
    for name in ["unknown_probability", "max_similarity", "margin"]:
        vals = np.asarray(test_scores.scores[name], dtype=np.float64)
        ok = np.isfinite(vals)
        if len(np.unique(unknown_y[ok])) >= 2:
            unknown_state_detection[name] = {
                "auroc": float(roc_auc_score(unknown_y[ok], vals[ok])),
                "auprc": float(average_precision_score(unknown_y[ok], vals[ok])),
            }
        else:
            unknown_state_detection[name] = {"auroc": np.nan, "auprc": np.nan}

    summary = {
        "experiment": "open_set_evidence_retrieval",
        "protocol": protocol.name,
        "protocol_metadata": dict(protocol.metadata),
        "embedder": getattr(embedder, "model_name", type(embedder).__name__),
        "query_mean_mode": query_mean_mode,
        "default_query_kappa": float(default_query_kappa),
        "fitted_gallery_kappa": float(fitted_kappa),
        "posterior_config": asdict(model_config),
        "posterior_metadata": dict(test_scores.metadata),
        "split": {
            "seed": int(seed),
            "calibration_fraction": float(calibration_fraction),
            "num_calibration": len(cal_protocol.query_ids),
            "num_test": len(test_protocol.query_ids),
            "num_calibration_known": int(np.sum(cal_protocol.known_mask)),
            "num_calibration_unknown": int(np.sum(~cal_protocol.known_mask)),
            "num_test_known": int(np.sum(test_protocol.known_mask)),
            "num_test_unknown": int(np.sum(~test_protocol.known_mask)),
        },
        "open_set": open_set_retrieval_metrics(test_protocol, test_scores),
        "ranking": rank,
        "uncertainty_detection": raw_detection,
        "unknown_state_detection": unknown_state_detection,
        "calibration": calibration,
        "multiple_relevance_audit": _multiple_relevance_diagnostic(test_protocol, test_scores),
    }
    rows = _score_table(test_protocol, test_scores, calibrated_test)
    if output_dir is not None:
        _save_result(Path(output_dir), summary, rows)
    return {
        "summary": summary,
        "rows": rows,
        "calibration_protocol": cal_protocol,
        "test_protocol": test_protocol,
        "test_scores": test_scores,
        "calibrated_test": calibrated_test,
        "model": model,
    }


def run_corpus_scaling_experiment(
    protocol: RetrievalProtocol,
    embedder: BaseTextEmbedder,
    model_config: PosteriorModelConfig,
    target_sizes: Sequence[int],
    *,
    rewrites: Optional[Mapping[str, Sequence[str]]] = None,
    default_query_kappa: float = 300.0,
    query_mean_mode: str = "original",
    calibration_fraction: float = 0.3,
    seed: int = 777,
    calibration_mode: str = "refit",
    output_dir: Optional[str | Path] = None,
) -> dict:
    """Stress-test corpus growth using nested negative document sets.

    ``refit`` re-estimates gallery kappa for each K, testing achievable calibrated
    operation. ``fixed_reference`` fits kappa on the smallest corpus and freezes it,
    directly testing whether the model remains valid when a deployed corpus grows.
    """
    if calibration_mode not in {"refit", "fixed_reference"}:
        raise ValueError("calibration_mode must be refit or fixed_reference")
    protocols = nested_corpus_protocols(protocol, target_sizes, seed=seed)
    results = []
    reference_kappa = model_config.gallery_kappa
    for i, p in enumerate(protocols):
        cfg = model_config
        if calibration_mode == "fixed_reference" and i > 0:
            cfg = replace(model_config, gallery_kappa=float(reference_kappa))
        subdir = Path(output_dir) / f"K_{len(p.corpus)}" if output_dir else None
        res = run_retrieval_experiment(
            p, embedder, cfg, rewrites=rewrites,
            default_query_kappa=default_query_kappa,
            query_mean_mode=query_mean_mode,
            calibration_fraction=calibration_fraction,
            seed=seed,
            output_dir=subdir,
        )
        if calibration_mode == "fixed_reference" and i == 0:
            reference_kappa = res["summary"]["fitted_gallery_kappa"]
        row = {
            "corpus_size": len(p.corpus),
            "gallery_kappa": res["summary"]["fitted_gallery_kappa"],
            **res["summary"]["open_set"],
        }
        for method in ("mprisk", "holue", "galue_entropy", "max_similarity"):
            mm = res["summary"]["uncertainty_detection"].get(method, {})
            row[f"{method}__error_auroc"] = mm.get("any_error_auroc", np.nan)
            row[f"{method}__aurc"] = mm.get("aurc", np.nan)
        results.append(row)
    summary = {
        "experiment": "corpus_scaling",
        "protocol": protocol.name,
        "calibration_mode": calibration_mode,
        "target_sizes": list(map(int, target_sizes)),
        "results": results,
    }
    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "scaling_summary.json").write_text(
            json.dumps(_jsonable(summary), indent=2, sort_keys=True), encoding="utf-8"
        )
        if results:
            with (root / "scaling.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                w.writeheader(); w.writerows(results)
    return summary


def run_rewrite_ablation(
    protocol: RetrievalProtocol,
    embedder: BaseTextEmbedder,
    model_config: PosteriorModelConfig,
    rewrites: Mapping[str, Sequence[str]],
    *,
    default_query_kappa: float = 300.0,
    calibration_fraction: float = 0.3,
    seed: int = 777,
    output_dir: Optional[str | Path] = None,
) -> dict:
    """Separate HolUE uncertainty gains from query-representation rewriting gains."""
    conditions = {
        "deterministic_original": (None, "original"),
        "rewrite_kappa_original_mean": (rewrites, "original"),
        "rewrite_kappa_rewrite_mean": (rewrites, "rewrite_mean"),
    }
    summaries = {}
    for name, (rw, mean_mode) in conditions.items():
        sub = Path(output_dir) / name if output_dir else None
        res = run_retrieval_experiment(
            protocol, embedder, model_config,
            rewrites=rw, default_query_kappa=default_query_kappa,
            query_mean_mode=mean_mode, calibration_fraction=calibration_fraction,
            seed=seed, output_dir=sub,
        )
        summaries[name] = res["summary"]
    out = {
        "experiment": "query_rewrite_ablation",
        "scientific_contrast": {
            "rewrite_kappa_original_mean": "changes uncertainty only",
            "rewrite_kappa_rewrite_mean": "changes uncertainty and representation",
        },
        "conditions": summaries,
    }
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "rewrite_ablation.json").write_text(
            json.dumps(_jsonable(out), indent=2, sort_keys=True), encoding="utf-8"
        )
    return out


def run_kappa_root_sensitivity(
    protocol: RetrievalProtocol,
    embedder: BaseTextEmbedder,
    model_config: PosteriorModelConfig,
    *,
    rewrites: Optional[Mapping[str, Sequence[str]]] = None,
    default_query_kappa: float = 300.0,
    query_mean_mode: str = "original",
    calibration_fraction: float = 0.3,
    seed: int = 777,
    output_dir: Optional[str | Path] = None,
) -> dict:
    """Expose sensitivity to every gallery-kappa root matching the same FPIR boundary.

    The power-spherical boundary can admit multiple concentrations.  The main
    method selects a root on validation NLL; this experiment reports what would
    have happened under *each* mathematically admissible root so reviewers can
    see whether conclusions depend on the root-selection convention.
    """
    discovery_cfg = replace(
        model_config,
        gallery_kappa=None,
        gallery_kappa_strategy="boundary_roots_calibrated",
        mc_samples=0,
    )
    discovery = run_retrieval_experiment(
        protocol, embedder, discovery_cfg, rewrites=rewrites,
        default_query_kappa=default_query_kappa, query_mean_mode=query_mean_mode,
        calibration_fraction=calibration_fraction, seed=seed,
        output_dir=(Path(output_dir) / "selected_root" if output_dir else None),
    )
    candidates = list(discovery["model"].kappa_candidates_)
    if not candidates:
        candidates = [float(discovery["summary"]["fitted_gallery_kappa"])]
    rows = []
    for i, kappa in enumerate(candidates):
        cfg = replace(model_config, gallery_kappa=float(kappa), mc_samples=0)
        res = run_retrieval_experiment(
            protocol, embedder, cfg, rewrites=rewrites,
            default_query_kappa=default_query_kappa, query_mean_mode=query_mean_mode,
            calibration_fraction=calibration_fraction, seed=seed,
            output_dir=(Path(output_dir) / f"root_{i}_{kappa:.6g}" if output_dir else None),
        )
        m = res["summary"]
        row = {
            "root_index": i,
            "gallery_kappa": float(kappa),
            "selected_by_validation": bool(np.isclose(kappa, discovery["summary"]["fitted_gallery_kappa"])),
            **m["open_set"],
        }
        for method in ["mprisk", "holue", "galue_entropy", "max_similarity"]:
            mm = m["uncertainty_detection"].get(method, {})
            row[f"{method}__error_auroc"] = mm.get("any_error_auroc", np.nan)
            row[f"{method}__aurc"] = mm.get("aurc", np.nan)
        rows.append(row)
    out = {
        "experiment": "gallery_kappa_root_sensitivity",
        "selected_gallery_kappa": discovery["summary"]["fitted_gallery_kappa"],
        "fit_tau": discovery["model"].fit_tau_,
        "candidate_validation_nll": list(discovery["model"].kappa_candidate_nll_),
        "results": rows,
    }
    if output_dir is not None:
        root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
        (root / "kappa_root_sensitivity.json").write_text(
            json.dumps(_jsonable(out), indent=2, sort_keys=True), encoding="utf-8"
        )
        if rows:
            with (root / "kappa_root_sensitivity.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return out


def run_mc_sensitivity_experiment(
    protocol: RetrievalProtocol,
    embedder: BaseTextEmbedder,
    model_config: PosteriorModelConfig,
    mc_samples: Sequence[int] = (0, 8, 32, 128),
    *,
    kappa_mode: str = "fixed_m0",
    repeats: int = 1,
    rewrites: Optional[Mapping[str, Sequence[str]]] = None,
    default_query_kappa: float = 300.0,
    query_mean_mode: str = "original",
    calibration_fraction: float = 0.3,
    seed: int = 777,
    output_dir: Optional[str | Path] = None,
) -> dict:
    """Test the dissertation's deterministic mean-embedding approximation.

    `fixed_m0` first calibrates gallery kappa with M=0 and freezes it for every
    Monte-Carlo condition, isolating posterior integration from re-calibration.
    `refit` lets each M re-fit its operating point (using empirical-grid fitting
    for M>0), answering a different operational question.
    """
    from evaluation.reproducibility import seed_everything
    from scipy.stats import spearmanr

    if kappa_mode not in {"fixed_m0", "refit"}:
        raise ValueError("kappa_mode must be fixed_m0 or refit")
    values = sorted(set(int(x) for x in mc_samples))
    if 0 not in values:
        values = [0] + values

    base_cfg = replace(model_config, mc_samples=0)
    base = run_retrieval_experiment(
        protocol, embedder, base_cfg, rewrites=rewrites,
        default_query_kappa=default_query_kappa, query_mean_mode=query_mean_mode,
        calibration_fraction=calibration_fraction, seed=seed,
        output_dir=(Path(output_dir) / "M_0" if output_dir else None),
    )
    fixed_kappa = float(base["summary"]["fitted_gallery_kappa"])
    base_rows = {r["query_id"]: r for r in base["rows"]}
    results = []

    for M in values:
        for rep in range(int(repeats)):
            if M == 0 and rep == 0:
                res = base
            elif M == 0:
                # Deterministic M=0 has no sampling variance; do not duplicate work.
                continue
            else:
                seed_everything(seed + 1000 * M + rep)
                if kappa_mode == "fixed_m0":
                    cfg = replace(
                        model_config, mc_samples=M, gallery_kappa=fixed_kappa,
                        mc_num_workers=1,
                    )
                else:
                    cfg = replace(
                        model_config, mc_samples=M, gallery_kappa=None,
                        gallery_kappa_strategy="empirical_grid", mc_num_workers=1,
                    )
                sub = Path(output_dir) / f"M_{M}" / f"rep_{rep}" if output_dir else None
                res = run_retrieval_experiment(
                    protocol, embedder, cfg, rewrites=rewrites,
                    default_query_kappa=default_query_kappa, query_mean_mode=query_mean_mode,
                    calibration_fraction=calibration_fraction, seed=seed,
                    output_dir=sub,
                )
            rows = {r["query_id"]: r for r in res["rows"]}
            common = sorted(set(base_rows) & set(rows))
            reject_disagree = np.mean([
                bool(base_rows[q]["rejected"]) != bool(rows[q]["rejected"]) for q in common
            ]) if common else np.nan
            pred_disagree = np.mean([
                str(base_rows[q]["predicted_doc_id"]) != str(rows[q]["predicted_doc_id"]) for q in common
            ]) if common else np.nan
            bscore = np.asarray([base_rows[q]["raw__mprisk"] for q in common], dtype=float)
            mscore = np.asarray([rows[q]["raw__mprisk"] for q in common], dtype=float)
            rho = float(spearmanr(bscore, mscore).correlation) if len(common) >= 3 and np.std(bscore) > 0 and np.std(mscore) > 0 else np.nan
            sm = res["summary"]
            results.append({
                "M": int(M), "repeat": int(rep),
                "gallery_kappa": sm["fitted_gallery_kappa"],
                "oser_accuracy": sm["open_set"]["oser_accuracy"],
                "fpir": sm["open_set"]["fpir"], "fnir": sm["open_set"]["fnir"],
                "reject_disagreement_vs_M0": float(reject_disagree),
                "top1_disagreement_vs_M0": float(pred_disagree),
                "mprisk_spearman_vs_M0": rho,
                "mprisk_error_auroc": sm["uncertainty_detection"].get("mprisk", {}).get("any_error_auroc", np.nan),
                "holue_error_auroc": sm["uncertainty_detection"].get("holue", {}).get("any_error_auroc", np.nan),
            })

    out = {
        "experiment": "mc_posterior_sensitivity",
        "kappa_mode": kappa_mode,
        "m0_gallery_kappa": fixed_kappa,
        "repeats": int(repeats),
        "results": results,
        "interpretation": (
            "fixed_m0 isolates posterior-integration effects; refit combines integration and operating-point recalibration"
        ),
    }
    if output_dir is not None:
        root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
        (root / "mc_sensitivity.json").write_text(json.dumps(_jsonable(out), indent=2), encoding="utf-8")
        if results:
            with (root / "mc_sensitivity.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)
    return out


def run_posterior_hyperparameter_sensitivity(
    protocol: RetrievalProtocol,
    embedder: BaseTextEmbedder,
    model_config: PosteriorModelConfig,
    sweeps: Mapping[str, Sequence[float]],
    *,
    rewrites: Optional[Mapping[str, Sequence[str]]] = None,
    default_query_kappa: float = 300.0,
    query_mean_mode: str = "original",
    calibration_fraction: float = 0.3,
    seed: int = 777,
    output_dir: Optional[str | Path] = None,
) -> dict:
    """One-factor-at-a-time sensitivity for prior and decision-cost assumptions.

    Every variant uses the same query split. Gallery kappa is re-identified for
    parameters that alter the posterior/operating point (``beta``,
    ``target_fpir``, ``gallery_prior``, ``predict_T``); pure MPRisk cost changes
    reuse the base posterior concentration when one was explicitly supplied.

    The function is intentionally one-factor-at-a-time: a full Cartesian grid can
    obscure which assumption drives a result and becomes prohibitively expensive
    for large retrieval corpora. Interaction sweeps can still be launched through
    CLI overrides after the main-effects audit.
    """

    allowed = set(asdict(model_config))
    unknown = sorted(set(sweeps) - allowed)
    if unknown:
        raise ValueError(f"Unknown PosteriorModelConfig sensitivity fields: {unknown}")

    posterior_fields = {"beta", "target_fpir", "gallery_prior", "predict_T"}
    rows = []

    def run_variant(label: str, cfg: PosteriorModelConfig, subdir: Optional[Path]):
        res = run_retrieval_experiment(
            protocol, embedder, cfg, rewrites=rewrites,
            default_query_kappa=default_query_kappa,
            query_mean_mode=query_mean_mode,
            calibration_fraction=calibration_fraction, seed=seed,
            output_dir=subdir,
        )
        sm = res["summary"]
        row = {
            "variant": label,
            "fitted_gallery_kappa": sm["fitted_gallery_kappa"],
            "oser_accuracy": sm["open_set"].get("oser_accuracy", np.nan),
            "fpir": sm["open_set"].get("fpir", np.nan),
            "fnir": sm["open_set"].get("fnir", np.nan),
        }
        for method in ("mprisk", "mprisk_no_ns", "holue", "galue_entropy", "max_similarity"):
            mm = sm["uncertainty_detection"].get(method, {})
            row[f"{method}__error_auroc"] = mm.get("any_error_auroc", np.nan)
            row[f"{method}__aurc"] = mm.get("aurc", np.nan)
        rows.append(row)
        return res

    base_dir = Path(output_dir) / "baseline" if output_dir else None
    base = run_variant("baseline", model_config, base_dir)
    base_kappa = float(base["summary"]["fitted_gallery_kappa"])

    for field, values in sweeps.items():
        reference = getattr(model_config, field)
        for raw_value in values:
            # YAML can represent all these main scientific sweeps numerically.
            value = raw_value
            if isinstance(reference, int) and not isinstance(reference, bool):
                value = int(raw_value)
            elif isinstance(reference, float):
                value = float(raw_value)
            kwargs = {field: value}
            if field in posterior_fields:
                kwargs["gallery_kappa"] = None
            elif model_config.gallery_kappa is None:
                # Freeze the baseline concentration for pure decision-cost sweeps
                # so they do not accidentally test a second calibration change.
                kwargs["gallery_kappa"] = base_kappa
            cfg = replace(model_config, **kwargs)
            safe_value = str(value).replace("/", "_").replace(" ", "_")
            sub = Path(output_dir) / f"{field}_{safe_value}" if output_dir else None
            run_variant(f"{field}={value}", cfg, sub)
            rows[-1]["parameter"] = field
            rows[-1]["value"] = value

    rows[0]["parameter"] = "baseline"
    rows[0]["value"] = np.nan
    summary = {
        "experiment": "posterior_hyperparameter_sensitivity",
        "protocol": protocol.name,
        "seed": int(seed),
        "sweeps": {k: list(v) for k, v in sweeps.items()},
        "baseline_gallery_kappa": base_kappa,
        "results": rows,
        "interpretation": (
            "One-factor-at-a-time audit. Posterior/operating-point parameters are re-calibrated; "
            "pure MPRisk loss-weight sweeps freeze the baseline posterior concentration."
        ),
    }
    if output_dir is not None:
        root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
        (root / "posterior_sensitivity.json").write_text(
            json.dumps(_jsonable(summary), indent=2, sort_keys=True), encoding="utf-8"
        )
        if rows:
            fieldnames = []
            for r in rows:
                for k in r:
                    if k not in fieldnames: fieldnames.append(k)
            with (root / "posterior_sensitivity.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
    return summary
