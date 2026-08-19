from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METHOD_ORDER = [
    "max_similarity",
    "margin",
    "softmax_msp",
    "softmax_entropy",
    "galue_msp",
    "galue_entropy",
    "holue",
    "mprisk_no_ns",
    "mprisk",
    "retrieval_combined",
    "generator_mean_nll",
    "generator_mean_entropy",
    "generator_semantic_entropy_discrete",
    "generator_semantic_entropy_normalized",
    "generator_self_consistency",
    "generator_only",
    "hybrid",
]

PRETTY_METHOD = {
    "max_similarity": "MaxSim",
    "margin": "Margin",
    "softmax_msp": "MSP",
    "softmax_entropy": "Softmax entropy",
    "galue_msp": "GalUE MSP",
    "galue_entropy": "GalUE",
    "holue": "HolUE",
    "mprisk_no_ns": "MPRisk no NS",
    "mprisk": "MPRisk",
    "unknown_probability": "$p(\\mathrm{unknown})$",
    "retrieval_combined": "Retrieval combined",
    "generator_mean_nll": "Generator NLL",
    "generator_mean_entropy": "Token entropy",
    "generator_worst_token_nll": "Worst-token NLL",
    "generator_semantic_entropy_discrete": "Semantic entropy",
    "generator_semantic_entropy_normalized": "Semantic entropy (norm.)",
    "generator_self_consistency": "Self-consistency",
    "generator_only": "Generator combined",
    "hybrid": "Hybrid",
}

METRIC_LABELS = {
    "oser_f1": "$F_1$",
    "task_accuracy": "Task accuracy",
    "call_reject_accuracy": "Call/reject accuracy",
    "accuracy": "Accuracy",
    "error_rate": "Error rate",
    "fpir": "FPIR",
    "fnir": "FNIR",
}


# -----------------------------------------------------------------------------
# Configuration / generic helpers
# -----------------------------------------------------------------------------


def _plain(x: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(x):
            return OmegaConf.to_container(x, resolve=True)
    except Exception:
        pass
    return x


def _report_cfg(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    cfg = _plain(cfg)
    if isinstance(cfg, Mapping):
        value = cfg.get("reporting", {})
        return dict(_plain(value) or {})
    return {}


def reporting_enabled(cfg: Any) -> bool:
    return bool(_report_cfg(cfg).get("enabled", True))


def make_fraction_grid(spec: Optional[Sequence[float]] = None) -> np.ndarray:
    """Create the same [start, stop, count] rejection grid used by old code."""
    if spec is None:
        spec = (0.0, 0.5, 20)
    values = list(spec)
    if len(values) == 3:
        start, stop, count = float(values[0]), float(values[1]), int(values[2])
        if count >= 2:
            grid = np.linspace(start, stop, count)
        else:
            grid = np.asarray([start], dtype=np.float64)
    else:
        grid = np.asarray(values, dtype=np.float64)
    if grid.size == 0:
        raise ValueError("rejection_fractions must not be empty")
    if np.any(~np.isfinite(grid)) or np.any(grid < 0) or np.any(grid >= 1):
        raise ValueError("Rejection fractions must be finite and satisfy 0 <= f < 1")
    return np.unique(grid)


def _slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    return text or "report"


def _as_bool(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    lowered = series.astype(str).str.lower()
    return lowered.isin(["true", "1", "yes"]).to_numpy(dtype=bool)


def _safe_float(x: Any) -> float:
    try:
        value = float(x)
    except Exception:
        return np.nan
    return value if np.isfinite(value) else np.nan


def _curve_area(curve: pd.DataFrame, metric: str) -> float:
    if metric not in curve.columns or len(curve) < 2:
        return np.nan
    x = pd.to_numeric(curve["fraction"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(curve[metric], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if np.sum(ok) < 2:
        return np.nan
    return float(np.trapezoid(y[ok], x[ok]))


def _prr(curve: pd.DataFrame, random_curve: pd.DataFrame, oracle_curve: pd.DataFrame, metric: str) -> float:
    area = _curve_area(curve, metric)
    random_area = _curve_area(random_curve, metric)
    oracle_area = _curve_area(oracle_curve, metric)
    denom = oracle_area - random_area
    if not np.isfinite(area + random_area + oracle_area) or abs(denom) < 1e-12:
        return np.nan
    return float((area - random_area) / denom)


def _ordered_methods(methods: Iterable[str], cfg: Any) -> list[str]:
    methods = list(dict.fromkeys(map(str, methods)))
    rcfg = _report_cfg(cfg)
    included = rcfg.get("methods")
    if included:
        include_order = list(map(str, included))
        methods = [m for m in include_order if m in methods]
    configured = list(map(str, rcfg.get("method_order", [])))
    order = configured or DEFAULT_METHOD_ORDER
    out = [m for m in order if m in methods]
    out.extend(m for m in methods if m not in out)
    return out


def _pretty_method(method: str, cfg: Any) -> str:
    rcfg = _report_cfg(cfg)
    custom = dict(rcfg.get("pretty_method", {}) or {})
    return str(custom.get(method, PRETTY_METHOD.get(method, method.replace("_", " "))))


# -----------------------------------------------------------------------------
# Rejection curves
# -----------------------------------------------------------------------------


def _subset_by_fraction(df: pd.DataFrame, score: np.ndarray, fraction: float) -> pd.DataFrame:
    n = len(df)
    keep_count = int((1.0 - float(fraction)) * n)
    keep_count = max(1, min(n, keep_count))
    order = np.argsort(np.asarray(score, dtype=np.float64), kind="stable")
    return df.iloc[order[:keep_count]]


def retrieval_rejection_curve(df: pd.DataFrame, score: np.ndarray, fractions: np.ndarray) -> pd.DataFrame:
    rows = []
    for fraction in fractions:
        kept = _subset_by_fraction(df, score, float(fraction))
        known = _as_bool(kept["known"])
        rejected = _as_bool(kept["rejected"])
        correct_retrieval = _as_bool(kept["correct_retrieval"])
        true_accept = known & (~rejected) & correct_retrieval
        false_accept = (~known) & (~rejected)
        any_error = _as_bool(kept["any_error"])
        tp = int(np.sum(true_accept))
        fp = int(np.sum(false_accept))
        fn = int(np.sum(known) - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "fraction": float(fraction),
                "oser_f1": float(f1),
                "fpir": float(np.mean(false_accept[~known])) if np.any(~known) else np.nan,
                "fnir": float(1.0 - np.mean(true_accept[known])) if np.any(known) else np.nan,
                "error_rate": float(np.mean(any_error)),
                "kept_count": int(len(kept)),
            }
        )
    return pd.DataFrame(rows)


def tool_rejection_curve(df: pd.DataFrame, score: np.ndarray, fractions: np.ndarray) -> pd.DataFrame:
    rows = []
    for fraction in fractions:
        kept = _subset_by_fraction(df, score, float(fraction))
        known = _as_bool(kept["known"])
        rejected = _as_bool(kept["rejected"])
        correct = _as_bool(kept["correct"])
        call_reject_correct = (known & (~rejected)) | ((~known) & rejected)
        rows.append(
            {
                "fraction": float(fraction),
                "task_accuracy": float(np.mean(correct)),
                "call_reject_accuracy": float(np.mean(call_reject_correct)),
                "fpir": float(np.mean(~rejected[~known])) if np.any(~known) else np.nan,
                "fnir": float(np.mean(rejected[known])) if np.any(known) else np.nan,
                "error_rate": float(np.mean(~correct)),
                "kept_count": int(len(kept)),
            }
        )
    return pd.DataFrame(rows)


def binary_rejection_curve(df: pd.DataFrame, score: np.ndarray, fractions: np.ndarray) -> pd.DataFrame:
    rows = []
    for fraction in fractions:
        kept = _subset_by_fraction(df, score, float(fraction))
        y = pd.to_numeric(kept["target"], errors="coerce").to_numpy(dtype=float)
        risk = float(np.mean(y))
        rows.append(
            {
                "fraction": float(fraction),
                "accuracy": float(1.0 - risk),
                "error_rate": risk,
                "kept_count": int(len(kept)),
            }
        )
    return pd.DataFrame(rows)


def _score_column_for_method(df: pd.DataFrame, method: str, experiment: str) -> Optional[str]:
    if experiment in {"ragtruth_hallucination_risk", "end_to_end_rag_incremental_risk"}:
        candidates = [f"model__{method}", f"p_error__{method}", method]
    else:
        if method == "holue":
            candidates = ["p_error__holue", "raw__holue", "holue"]
        else:
            candidates = [f"raw__{method}", f"p_error__{method}", method]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _candidate_methods(summary: Mapping[str, Any], experiment: str) -> list[str]:
    if experiment == "open_set_evidence_retrieval":
        return list((summary.get("uncertainty_detection") or {}).keys())
    if experiment == "open_set_tool_routing":
        return list((summary.get("uncertainty_detection") or {}).keys())
    return list((summary.get("metrics") or {}).keys())


def _curve_spec(experiment: str):
    if experiment == "open_set_evidence_retrieval":
        return retrieval_rejection_curve, ["oser_f1", "fpir", "fnir", "error_rate"], "oser_f1", "any_error"
    if experiment == "open_set_tool_routing":
        return tool_rejection_curve, ["task_accuracy", "call_reject_accuracy", "fpir", "fnir", "error_rate"], "task_accuracy", "correct"
    return binary_rejection_curve, ["accuracy", "error_rate"], "accuracy", "target"


def generate_rejection_curves(
    output_dir: str | Path,
    summary: Mapping[str, Any],
    df: pd.DataFrame,
    cfg: Any = None,
) -> pd.DataFrame:
    experiment = str(summary.get("experiment", ""))
    curve_fn, default_metrics, prr_metric, oracle_field = _curve_spec(experiment)
    rcfg = _report_cfg(cfg)
    fractions = make_fraction_grid(rcfg.get("rejection_fractions", [0.0, 0.5, 20]))
    plot_metrics = list(map(str, rcfg.get("rejection_metrics", default_metrics)))

    methods = _ordered_methods(_candidate_methods(summary, experiment), cfg)
    score_cols: Dict[str, str] = {}
    for method in methods:
        col = _score_column_for_method(df, method, experiment)
        if col is not None:
            score_cols[method] = col

    # To make Random/Oracle and all plotted methods directly comparable, use one
    # common finite test population.  This also avoids silently changing N between
    # methods when an optional generator feature is missing.
    if score_cols:
        common = np.ones(len(df), dtype=bool)
        for col in score_cols.values():
            common &= np.isfinite(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float))
        if np.sum(common) >= 2:
            df = df.loc[common].reset_index(drop=True)
        else:
            score_cols = {}

    if not score_cols or len(df) < 2:
        return pd.DataFrame()

    seed = int(rcfg.get("random_seed", 777))
    rng = np.random.default_rng(seed)
    random_score = rng.random(len(df))
    if oracle_field == "correct":
        y_error = (~_as_bool(df["correct"])).astype(float)
    elif oracle_field == "any_error":
        y_error = _as_bool(df["any_error"]).astype(float)
    else:
        y_error = pd.to_numeric(df[oracle_field], errors="coerce").to_numpy(dtype=float)
    oracle_score = y_error + 1e-9 * rng.random(len(df))
    random_curve = curve_fn(df, random_score, fractions)
    oracle_curve = curve_fn(df, oracle_score, fractions)

    curves: Dict[str, pd.DataFrame] = {}
    prr_values: Dict[str, float] = {}
    for method, col in score_cols.items():
        score = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        curve = curve_fn(df, score, fractions)
        curves[method] = curve
        prr_values[method] = _prr(curve, random_curve, oracle_curve, prr_metric)

    root = Path(output_dir) / "rejection_curves"
    root.mkdir(parents=True, exist_ok=True)

    combined = []
    for name, curve, prr_value in [
        ("Random", random_curve, 0.0),
        ("Oracle", oracle_curve, 1.0),
    ]:
        tmp = curve.copy()
        tmp.insert(0, "prr", prr_value)
        tmp.insert(0, "method", name)
        combined.append(tmp)
    for method in methods:
        if method not in curves:
            continue
        tmp = curves[method].copy()
        tmp.insert(0, "prr", prr_values[method])
        tmp.insert(0, "method", method)
        combined.append(tmp)
    all_curves = pd.concat(combined, ignore_index=True)
    all_curves.to_csv(root / "all_rejection_curves.csv", index=False)
    pd.DataFrame(
        [{"method": m, "prr": prr_values[m]} for m in methods if m in prr_values]
    ).to_csv(root / "prr_values.csv", index=False)

    figsize = tuple(rcfg.get("figsize", [6.4, 4.8]))
    legend_fontsize = float(rcfg.get("legend_fontsize", 8))
    prr_in_legend = bool(rcfg.get("prr_in_legend", True))
    display_random = bool(rcfg.get("display_random_curve", True))
    display_oracle = bool(rcfg.get("display_oracle_curve", True))

    for metric in plot_metrics:
        if metric not in random_curve.columns:
            continue
        plt.figure(figsize=figsize)
        if display_random:
            label = "Random" + (", PRR=0.00" if prr_in_legend else "")
            plt.plot(random_curve["fraction"], random_curve[metric], "--", color="gray", linewidth=2.0, alpha=0.8, label=label)
        if display_oracle:
            label = "Oracle" + (", PRR=1.00" if prr_in_legend else "")
            plt.plot(oracle_curve["fraction"], oracle_curve[metric], "--", color="black", linewidth=2.0, alpha=0.8, label=label)
        for method in methods:
            curve = curves.get(method)
            if curve is None or metric not in curve.columns:
                continue
            prr_value = prr_values.get(method, np.nan)
            label = _pretty_method(method, cfg)
            if prr_in_legend and np.isfinite(prr_value):
                label += f", PRR={prr_value:.2f}"
            linewidth = 2.8 if method in {"holue", "mprisk", "hybrid"} else 1.8
            plt.plot(curve["fraction"], curve[metric], linewidth=linewidth, alpha=0.95, label=label)
        plt.xlabel("Filtered-out sample fraction")
        plt.ylabel(METRIC_LABELS.get(metric, metric))
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(fontsize=legend_fontsize)
        plt.tight_layout()
        stem = root / f"{_slug(metric)}_rejection_curve"
        plt.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.savefig(stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
        plt.close()

    return pd.DataFrame([{"method": m, "prr": prr_values[m]} for m in methods if m in prr_values])


# -----------------------------------------------------------------------------
# LaTeX tables
# -----------------------------------------------------------------------------


def _latex_escape(value: Any) -> str:
    s = str(value)
    if "\\" in s or "$" in s:
        return s
    for old, new in [
        ("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"),
        ("{", r"\{"), ("}", r"\}"),
    ]:
        s = s.replace(old, new)
    return s


def _format_value(value: Any, digits: int = 3) -> str:
    value = _safe_float(value)
    if not np.isfinite(value):
        return "--"
    if abs(value) >= 10000 or (abs(value) > 0 and abs(value) < 1e-4):
        return f"{value:.2e}"
    out = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return "0" if out == "-0" else out


def _best_second(values: pd.Series, direction: str, digits: int) -> tuple[Optional[float], Optional[float]]:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return None, None
    rounded = np.unique(np.round(x.to_numpy(dtype=float), digits))
    rounded = np.sort(rounded)
    if direction == "high":
        rounded = rounded[::-1]
    best = float(rounded[0]) if len(rounded) else None
    second = float(rounded[1]) if len(rounded) > 1 else None
    return best, second


def _render_latex_table(
    numeric: pd.DataFrame,
    *,
    caption: str,
    label: str,
    directions: Mapping[str, str],
    pretty_columns: Mapping[str, str],
    cfg: Any,
) -> str:
    rcfg = _report_cfg(cfg)
    digits = int(rcfg.get("round_num", 3))
    highlight = bool(rcfg.get("highlight_best", True))
    first_col = numeric.columns[0]
    best_second = {
        col: _best_second(numeric[col], directions.get(col, "high"), digits)
        for col in numeric.columns[1:]
    }
    display_rows = []
    for _, row in numeric.iterrows():
        vals = [_latex_escape(row[first_col])]
        for col in numeric.columns[1:]:
            value = _safe_float(row[col])
            cell = _format_value(value, digits)
            if highlight and np.isfinite(value):
                rounded = float(np.round(value, digits))
                best, second = best_second[col]
                if best is not None and rounded == best:
                    cell = r"\textbf{" + cell + "}"
                elif second is not None and rounded == second:
                    cell = r"\underline{" + cell + "}"
            vals.append(cell)
        display_rows.append(vals)

    headers = [pretty_columns.get(c, c) for c in numeric.columns]
    col_fmt = "l" + "c" * (len(headers) - 1)
    out = "\\begin{table}[t]\n\\centering\n\\footnotesize\n"
    out += "\\setlength\\tabcolsep{4pt}\n"
    out += f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
    if len(headers) > 6:
        out += "\\resizebox{\\linewidth}{!}{%\n"
    out += f"\\begin{{tabular}}{{{col_fmt}}}\n\\toprule\n"
    out += " & ".join(headers) + " \\\\\n\\midrule\n"
    for vals in display_rows:
        out += " & ".join(vals) + " \\\\\n"
    out += "\\bottomrule\n\\end{tabular}\n"
    if len(headers) > 6:
        out += "}%\n"
    out += "\\end{table}\n"
    return out


def _write_table(
    root: Path,
    name: str,
    numeric: pd.DataFrame,
    *,
    caption: str,
    label: str,
    directions: Mapping[str, str],
    pretty_columns: Mapping[str, str],
    cfg: Any,
) -> None:
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    numeric.to_csv(tables / f"{name}.csv", index=False)
    latex = _render_latex_table(
        numeric,
        caption=caption,
        label=label,
        directions=directions,
        pretty_columns=pretty_columns,
        cfg=cfg,
    )
    (tables / f"{name}.tex").write_text(latex, encoding="utf-8")


def _prr_lookup(prr_df: pd.DataFrame) -> Dict[str, float]:
    if prr_df.empty:
        return {}
    return {
        str(row["method"]): _safe_float(row["prr"])
        for _, row in prr_df.iterrows()
        if str(row["method"]) not in {"Random", "Oracle"}
    }


def _retrieval_tables(root: Path, summary: Mapping[str, Any], prr_df: pd.DataFrame, cfg: Any) -> None:
    prr = _prr_lookup(prr_df)
    detection = summary.get("uncertainty_detection") or {}
    calibration = summary.get("calibration") or {}
    methods = _ordered_methods(detection.keys(), cfg)
    rows = []
    for method in methods:
        d = detection.get(method, {}) or {}
        c = calibration.get(method, {}) or {}
        rows.append({
            "method": _pretty_method(method, cfg),
            "prr": prr.get(method, np.nan),
            "auroc": d.get("any_error_auroc", np.nan),
            "auprc": d.get("any_error_auprc", np.nan),
            "aurc": d.get("aurc", np.nan),
            "brier": c.get("brier", np.nan),
            "ece": c.get("ece", np.nan),
            "nll": c.get("nll", np.nan),
        })
    if rows:
        _write_table(
            root, "uncertainty_metrics", pd.DataFrame(rows),
            caption="Uncertainty quality for open-set evidence retrieval on the held-out test split.",
            label=f"tab:{_slug(root.name)}_oser_uncertainty",
            directions={"prr":"high","auroc":"high","auprc":"high","aurc":"low","brier":"low","ece":"low","nll":"low"},
            pretty_columns={"method":"Method","prr":"PRR $\\uparrow$","auroc":"AUROC $\\uparrow$","auprc":"AUPRC $\\uparrow$","aurc":"AURC $\\downarrow$","brier":"Brier $\\downarrow$","ece":"ECE $\\downarrow$","nll":"NLL $\\downarrow$"},
            cfg=cfg,
        )
    osr = summary.get("open_set") or {}
    ranking = summary.get("ranking") or {}
    op = pd.DataFrame([{
        "setting": str(summary.get("protocol", root.name)),
        "oser_accuracy": osr.get("oser_accuracy", np.nan),
        "oser_f1": osr.get("oser_f1", np.nan),
        "fpir": osr.get("fpir", np.nan),
        "fnir": osr.get("fnir", np.nan),
        "recall1": ranking.get("recall@1", ranking.get("recall@1_from_streamed_top1", np.nan)),
        "mrr": ranking.get("mrr", np.nan),
        "gallery_kappa": summary.get("fitted_gallery_kappa", np.nan),
    }])
    _write_table(
        root, "operating_point", op,
        caption="Open-set evidence-retrieval operating point and ranking quality.",
        label=f"tab:{_slug(root.name)}_oser_operating",
        directions={c:"high" for c in op.columns[1:]},
        pretty_columns={"setting":"Setting","oser_accuracy":"OSER Acc.","oser_f1":"OSER $F_1$","fpir":"FPIR","fnir":"FNIR","recall1":"Recall@1","mrr":"MRR","gallery_kappa":"$\\kappa_g$"},
        cfg=cfg,
    )


def _tool_tables(root: Path, summary: Mapping[str, Any], prr_df: pd.DataFrame, cfg: Any) -> None:
    prr = _prr_lookup(prr_df)
    detection = summary.get("uncertainty_detection") or {}
    calibration = summary.get("calibration") or {}
    methods = _ordered_methods(detection.keys(), cfg)
    rows = []
    for method in methods:
        d = detection.get(method, {}) or {}
        c = calibration.get(method, {}) or {}
        rows.append({
            "method": _pretty_method(method, cfg),
            "prr": prr.get(method, np.nan),
            "auroc": d.get("error_auroc", np.nan),
            "auprc": d.get("error_auprc", np.nan),
            "aurc": d.get("aurc", np.nan),
            "brier": c.get("brier", np.nan),
            "ece": c.get("ece", np.nan),
        })
    if rows:
        _write_table(
            root, "uncertainty_metrics", pd.DataFrame(rows),
            caption="Uncertainty quality for open-set tool routing on the held-out test split.",
            label=f"tab:{_slug(root.name)}_tool_uncertainty",
            directions={"prr":"high","auroc":"high","auprc":"high","aurc":"low","brier":"low","ece":"low"},
            pretty_columns={"method":"Method","prr":"PRR $\\uparrow$","auroc":"AUROC $\\uparrow$","auprc":"AUPRC $\\uparrow$","aurc":"AURC $\\downarrow$","brier":"Brier $\\downarrow$","ece":"ECE $\\downarrow$"},
            cfg=cfg,
        )
    d = summary.get("decision_metrics") or {}
    op = pd.DataFrame([{
        "setting": root.name,
        "call_reject_accuracy": d.get("call_reject_accuracy", np.nan),
        "task_accuracy": d.get("task_accuracy_with_tool_id_when_available", np.nan),
        "fpir": d.get("fpir_irrelevant_accepted", np.nan),
        "fnir": d.get("fnir_relevant_rejected", np.nan),
        "gallery_kappa": summary.get("fitted_gallery_kappa", np.nan),
    }])
    _write_table(
        root, "operating_point", op,
        caption="Open-set tool-routing operating point.",
        label=f"tab:{_slug(root.name)}_tool_operating",
        directions={c:"high" for c in op.columns[1:]},
        pretty_columns={"setting":"Setting","call_reject_accuracy":"Call/reject Acc.","task_accuracy":"Task Acc.","fpir":"FPIR","fnir":"FNIR","gallery_kappa":"$\\kappa_g$"},
        cfg=cfg,
    )


def _rag_tables(root: Path, summary: Mapping[str, Any], prr_df: pd.DataFrame, cfg: Any) -> None:
    prr = _prr_lookup(prr_df)
    metrics = summary.get("metrics") or {}
    methods = _ordered_methods(metrics.keys(), cfg)
    rows = []
    for method in methods:
        m = metrics.get(method, {}) or {}
        rows.append({
            "method": _pretty_method(method, cfg),
            "prr": prr.get(method, np.nan),
            "auroc": m.get("auroc", np.nan),
            "auprc": m.get("auprc", np.nan),
            "aurc": m.get("aurc", np.nan),
            "risk50": m.get("risk_at_50pct_coverage", np.nan),
            "risk80": m.get("risk_at_80pct_coverage", np.nan),
            "risk90": m.get("risk_at_90pct_coverage", np.nan),
            "brier": m.get("calibration_brier", np.nan),
            "ece": m.get("calibration_ece", np.nan),
        })
    if rows:
        _write_table(
            root, "uncertainty_metrics", pd.DataFrame(rows),
            caption="Held-out error/hallucination prediction and selective-generation performance.",
            label=f"tab:{_slug(root.name)}_rag_uncertainty",
            directions={"prr":"high","auroc":"high","auprc":"high","aurc":"low","risk50":"low","risk80":"low","risk90":"low","brier":"low","ece":"low"},
            pretty_columns={"method":"Method","prr":"PRR $\\uparrow$","auroc":"AUROC $\\uparrow$","auprc":"AUPRC $\\uparrow$","aurc":"AURC $\\downarrow$","risk50":"Risk@50 $\\downarrow$","risk80":"Risk@80 $\\downarrow$","risk90":"Risk@90 $\\downarrow$","brier":"Brier $\\downarrow$","ece":"ECE $\\downarrow$"},
            cfg=cfg,
        )

    comps = summary.get("paired_comparisons") or {}
    comp_rows = []
    for name, values in comps.items():
        if not isinstance(values, Mapping):
            continue
        comp_rows.append({
            "comparison": name.replace("_", " "),
            "delta": values.get("delta", values.get("point", np.nan)),
            "ci_low": values.get("ci_low", np.nan),
            "ci_high": values.get("ci_high", np.nan),
        })
    if comp_rows:
        _write_table(
            root, "paired_comparisons", pd.DataFrame(comp_rows),
            caption="Paired group-bootstrap comparison of modern RAG uncertainty models.",
            label=f"tab:{_slug(root.name)}_rag_comparison",
            directions={"delta":"high","ci_low":"high","ci_high":"high"},
            pretty_columns={"comparison":"Comparison","delta":"$\\Delta$ AUROC","ci_low":"95\\% CI low","ci_high":"95\\% CI high"},
            cfg=cfg,
        )


def generate_latex_tables(
    output_dir: str | Path,
    summary: Mapping[str, Any],
    prr_df: pd.DataFrame,
    cfg: Any = None,
) -> None:
    root = Path(output_dir)
    experiment = str(summary.get("experiment", ""))
    if experiment == "open_set_evidence_retrieval":
        _retrieval_tables(root, summary, prr_df, cfg)
    elif experiment == "open_set_tool_routing":
        _tool_tables(root, summary, prr_df, cfg)
    elif experiment in {"ragtruth_hallucination_risk", "end_to_end_rag_incremental_risk"}:
        _rag_tables(root, summary, prr_df, cfg)


def _aggregate_csv_table(path: Path, cfg: Any) -> Optional[Path]:
    """Turn sensitivity/scaling CSVs into compact thesis-ready LaTeX tables."""
    name = path.name
    selections: Dict[str, Sequence[str]] = {
        "scaling.csv": ["corpus_size", "gallery_kappa", "oser_accuracy", "fpir", "fnir", "mprisk__error_auroc", "holue__error_auroc", "max_similarity__error_auroc"],
        "mc_sensitivity.csv": ["M", "repeat", "gallery_kappa", "oser_accuracy", "fpir", "fnir", "mprisk_error_auroc", "holue_error_auroc", "reject_disagreement_vs_M0", "mprisk_spearman_vs_M0"],
        "kappa_root_sensitivity.csv": ["root_index", "gallery_kappa", "selected_by_validation", "oser_accuracy", "fpir", "fnir", "mprisk__error_auroc", "holue__error_auroc"],
        "posterior_sensitivity.csv": ["variant", "fitted_gallery_kappa", "oser_accuracy", "fpir", "fnir", "mprisk__error_auroc", "mprisk__aurc", "holue__error_auroc", "holue__aurc"],
    }
    if name not in selections:
        return None
    df = pd.read_csv(path)
    cols = [c for c in selections[name] if c in df.columns]
    if not cols:
        return None
    df = df[cols]
    first = cols[0]
    # Ensure the generic renderer has a textual first column.
    df[first] = df[first].astype(str)
    numeric_cols = [c for c in cols[1:] if pd.api.types.is_numeric_dtype(df[c])]
    directions = {c: ("low" if any(x in c for x in ["fpir", "fnir", "aurc", "disagreement"]) else "high") for c in numeric_cols}
    table_root = path.parent
    out_name = path.stem + "_table"
    _write_table(
        table_root, out_name, df,
        caption=f"{path.stem.replace('_', ' ').capitalize()} summary.",
        label=f"tab:{_slug(table_root.name)}_{_slug(path.stem)}",
        directions=directions,
        pretty_columns={c: c.replace("__", " ").replace("_", " ") for c in cols},
        cfg=cfg,
    )
    return table_root / "tables" / f"{out_name}.tex"


# -----------------------------------------------------------------------------
# Automatic report discovery
# -----------------------------------------------------------------------------


def generate_leaf_report(output_dir: str | Path, cfg: Any = None) -> Dict[str, Any]:
    root = Path(output_dir)
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        return {"output_dir": str(root), "generated": False, "reason": "summary.json missing"}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    experiment = str(summary.get("experiment", ""))
    csv_name = {
        "open_set_evidence_retrieval": "per_query.csv",
        "open_set_tool_routing": "per_example.csv",
        "ragtruth_hallucination_risk": "per_response.csv",
        "end_to_end_rag_incremental_risk": "per_response.csv",
    }.get(experiment)
    if csv_name is None or not (root / csv_name).is_file():
        return {"output_dir": str(root), "generated": False, "reason": f"unsupported or {csv_name} missing"}
    df = pd.read_csv(root / csv_name)
    prr_df = generate_rejection_curves(root, summary, df, cfg)
    generate_latex_tables(root, summary, prr_df, cfg)
    return {
        "output_dir": str(root),
        "experiment": experiment,
        "generated": True,
        "num_rows": int(len(df)),
        "num_prr_methods": int(len(prr_df)),
    }


def generate_modern_ai_reports(output_dir: str | Path, cfg: Any = None) -> Dict[str, Any]:
    """Generate curves/tables for a whole modern-AI output tree.

    This mirrors the older repository behavior: experiment execution immediately
    creates paper-facing figures plus CSV/LaTeX tables.  Recursive discovery also
    handles nested scaling, rewrite, MC, kappa-root, posterior-sensitivity, and
    smoke outputs without special-case runner code.
    """
    root = Path(output_dir)
    if not reporting_enabled(cfg):
        return {"enabled": False, "root": str(root), "reports": []}
    reports = []
    for summary_path in sorted(root.rglob("summary.json")):
        reports.append(generate_leaf_report(summary_path.parent, cfg))
    aggregate_tables = []
    for csv_path in sorted(root.rglob("*.csv")):
        if "rejection_curves" in csv_path.parts or "tables" in csv_path.parts:
            continue
        out = _aggregate_csv_table(csv_path, cfg)
        if out is not None:
            aggregate_tables.append(str(out))
    manifest = {
        "enabled": True,
        "root": str(root),
        "reports": reports,
        "aggregate_tables": aggregate_tables,
    }
    (root / "report_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
