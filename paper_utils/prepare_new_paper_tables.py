#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


# ---------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------


def to_plain(x: Any) -> Any:
    if OmegaConf.is_config(x):
        return OmegaConf.to_container(x, resolve=True)
    return x


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_or_empty(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        print(f"[warning] missing input table: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def is_number_like(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def latex_escape(s: Any) -> str:
    s = str(s)

    # Already contains LaTeX markup.
    if "\\" in s or "$" in s:
        return s

    replacements = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }

    for k, v in replacements.items():
        s = s.replace(k, v)

    return s


def latex_cell(x: Any) -> str:
    if x is None:
        return "--"

    if isinstance(x, float) and not np.isfinite(x):
        return "--"

    s = str(x)
    if s == "" or s.lower() == "nan":
        return "--"

    if "\\" in s or "$" in s:
        return s

    return latex_escape(s)


def format_float(x: Any, digits: int = 2, na: str = "--") -> str:
    try:
        x = float(x)
    except Exception:
        return na

    if not np.isfinite(x):
        return na

    x_round = np.round(x, digits)
    s = f"{x_round:.{digits}f}"

    # Remove trailing zeros but keep at least one digit after decimal if needed.
    if "." in s:
        s = s.rstrip("0").rstrip(".")

    if s == "-0":
        s = "0"

    return s


def format_pm(mean: Any, std: Any, digits: int = 2, na: str = "--") -> str:
    try:
        mean = float(mean)
        std = float(std)
    except Exception:
        return na

    if not np.isfinite(mean):
        return na

    if not np.isfinite(std):
        return format_float(mean, digits=digits, na=na)

    return (
        "$"
        + format_float(mean, digits=digits, na=na)
        + r"\pm"
        + format_float(std, digits=digits, na=na)
        + "$"
    )


def format_ci(delta: Any, low: Any, high: Any, digits: int = 2, na: str = "--") -> str:
    try:
        delta = float(delta)
        low = float(low)
        high = float(high)
    except Exception:
        return na

    if not np.isfinite(delta):
        return na

    return (
        "$"
        + format_float(delta, digits=digits)
        + r"\,["
        + format_float(low, digits=digits)
        + ","
        + format_float(high, digits=digits)
        + "]$"
    )


def get_pretty(cfg, category: str, name: Any) -> str:
    name = str(name)

    try:
        mapping = to_plain(getattr(cfg.pretty_name, category))
    except Exception:
        mapping = {}

    return str(mapping.get(name, name))


def get_metric_direction(cfg, col_name: str, default: str = "high") -> str:
    try:
        metric_order = to_plain(cfg.metric_order)
    except Exception:
        metric_order = {}

    return str(metric_order.get(col_name, default))


def match_value(series: pd.Series, value: Any) -> pd.Series:
    if isinstance(value, (list, tuple)):
        mask = np.zeros(len(series), dtype=bool)
        for v in value:
            mask |= match_value(series, v).values
        return pd.Series(mask, index=series.index)

    if is_number_like(value) and pd.api.types.is_numeric_dtype(series):
        return pd.Series(
            np.isclose(series.astype(float), float(value)), index=series.index
        )

    return series.astype(str) == str(value)


def select_df(df: pd.DataFrame, selectors: Dict[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    for col, value in selectors.items():
        if value is None:
            continue
        if col not in out.columns:
            print(f"[warning] selector column {col!r} missing in table")
            continue

        out = out[match_value(out[col], value)]

    return out


def first_or_mean(df: pd.DataFrame, value_col: str) -> float:
    if df.empty or value_col not in df.columns:
        return np.nan

    values = pd.to_numeric(df[value_col], errors="coerce").dropna().values
    if len(values) == 0:
        return np.nan

    return float(np.mean(values))


def best_second_per_column(
    numeric_df: pd.DataFrame,
    directions: Dict[str, str],
    digits: int,
    exclude_rows: Optional[List[str]] = None,
) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    if exclude_rows is None:
        exclude_rows = []

    out: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

    for col in numeric_df.columns:
        values = numeric_df[col].copy()

        for row_name in exclude_rows:
            if row_name in values.index:
                values = values.drop(row_name)

        values = pd.to_numeric(values, errors="coerce").dropna()
        if len(values) == 0:
            out[col] = (None, None)
            continue

        rounded = np.array([np.round(v, digits) for v in values.values])
        unique = np.unique(rounded)

        direction = directions.get(col, "high")

        if direction == "high":
            unique = unique[::-1]
        elif direction == "low":
            unique = unique
        else:
            raise ValueError(f"Unknown metric direction {direction!r} for {col}")

        best = float(unique[0]) if len(unique) >= 1 else None
        second = float(unique[1]) if len(unique) >= 2 else None
        out[col] = (best, second)

    return out


def format_highlighted_numeric_df(
    numeric_df: pd.DataFrame,
    directions: Optional[Dict[str, str]] = None,
    digits: int = 2,
    exclude_rows: Optional[List[str]] = None,
    highlight: bool = True,
) -> pd.DataFrame:
    if directions is None:
        directions = {c: "high" for c in numeric_df.columns}

    best_second = best_second_per_column(
        numeric_df,
        directions=directions,
        digits=digits,
        exclude_rows=exclude_rows,
    )

    display = pd.DataFrame(index=numeric_df.index)

    for col in numeric_df.columns:
        values = []
        best, second = best_second[col]

        for row_name, value in numeric_df[col].items():
            s = format_float(value, digits=digits)

            if highlight:
                try:
                    rounded = float(np.round(float(value), digits))
                except Exception:
                    values.append(s)
                    continue

                if best is not None and np.isfinite(rounded) and rounded == best:
                    s = r"\textbf{" + s + "}"
                elif second is not None and np.isfinite(rounded) and rounded == second:
                    s = r"\underline{" + s + "}"

            values.append(s)

        display[col] = values

    return display


def write_outputs(
    cfg,
    name: str,
    latex_code: str,
    numeric_df: Optional[pd.DataFrame] = None,
    display_df: Optional[pd.DataFrame] = None,
) -> None:
    exp_dir = Path(cfg.exp_dir)
    tex_dir = exp_dir / "tex"
    csv_dir = exp_dir / "csv"

    ensure_dir(tex_dir)
    ensure_dir(csv_dir)

    tex_path = tex_dir / f"{name}.tex"
    with tex_path.open("w", encoding="utf-8") as f:
        f.write(latex_code)

    print(f"[saved] {tex_path}")

    if numeric_df is not None:
        csv_path = csv_dir / f"{name}.csv"
        numeric_df.to_csv(csv_path)
        print(f"[saved] {csv_path}")

    if display_df is not None:
        csv_path = csv_dir / f"{name}_display.csv"
        display_df.to_csv(csv_path)
        print(f"[saved] {csv_path}")


def save_plot(fig, cfg, name: str) -> None:
    exp_dir = Path(cfg.exp_dir)
    fig_dir = exp_dir / "figures"
    ensure_dir(fig_dir)

    png_path = fig_dir / f"{name}.png"
    pdf_path = fig_dir / f"{name}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {png_path}")
    print(f"[saved] {pdf_path}")


def slugify_for_file(x: Any) -> str:
    x = str(x)
    for ch in [" ", "/", "\\", "+", ":", ";", ",", "(", ")", "[", "]", "{", "}"]:
        x = x.replace(ch, "_")
    while "__" in x:
        x = x.replace("__", "_")
    return x.strip("_")


def build_rejection_curve_figure(cfg, fcfg, name: str):
    """
    Draw rejection curves from all_rejection_curves.csv produced by
    mprisk_core_experiments.py.

    Expected columns:
      dataset, far, beta, method, prr_f1, fraction, f1_class, fpir, fnir, ...
    """
    input_path = Path(str(fcfg.input_path))
    df = read_csv_or_empty(input_path)

    if df.empty:
        print(f"[warning] no rejection-curve data for {name}")
        return

    df = select_df(
        df,
        {
            "dataset": fcfg.get("dataset"),
            "far": fcfg.get("far"),
            "beta": fcfg.get("beta"),
        },
    )

    if df.empty:
        print(f"[warning] empty rejection-curve selection for {name}")
        return

    metrics = list(fcfg.get("metrics", ["f1_class"]))
    methods = list(fcfg.get("methods", sorted(df["method"].unique())))

    metric_labels = {
        "f1_class": "$F_1$",
        "fpir": "FPIR",
        "fnir": "FNIR",
        "error_rate": "Error rate",
    }

    figsize = tuple(fcfg.get("figsize", [6.4, 4.8]))
    legend_fontsize = float(fcfg.get("legend_fontsize", 8))
    prr_in_legend = bool(fcfg.get("prr_in_legend", True))

    for metric in metrics:
        fig, ax = plt.subplots(figsize=figsize)

        for method in methods:
            sub = df[df["method"].astype(str) == str(method)].copy()
            if sub.empty or metric not in sub.columns:
                continue

            sub = sub.sort_values("fraction")

            prr = np.nan
            if "prr_f1" in sub.columns:
                prr_vals = pd.to_numeric(sub["prr_f1"], errors="coerce").dropna().values
                if len(prr_vals) > 0:
                    prr = float(prr_vals[0])

            if prr_in_legend and np.isfinite(prr):
                label = f"{get_pretty(cfg, 'model', method)}, PRR={prr:.2f}"
            else:
                label = get_pretty(cfg, "model", method)

            if method == "Random":
                style = {
                    "color": "gray",
                    "linestyle": "--",
                    "linewidth": 2.0,
                    "alpha": 0.8,
                }
            elif method == "Oracle":
                style = {
                    "color": "black",
                    "linestyle": "--",
                    "linewidth": 2.0,
                    "alpha": 0.8,
                }
            else:
                style = {
                    "linewidth": (
                        2.5
                        if "MPRisk" in str(method) or "HolUE" in str(method)
                        else 1.7
                    )
                }

            ax.plot(
                sub["fraction"].values,
                sub[metric].values,
                label=label,
                **style,
            )

        ax.set_xlabel("Filtered-out probe fraction")
        ax.set_ylabel(metric_labels.get(metric, metric))
        title = str(fcfg.get("title", ""))
        if title:
            ax.set_title(title)
        else:
            ax.set_title(
                f"{get_pretty(cfg, 'dataset', fcfg.get('dataset'))}, "
                f"FPIR={fcfg.get('far')}"
            )

        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=legend_fontsize)
        fig.tight_layout()

        save_plot(fig, cfg, f"{name}_{metric}")


def find_reliability_bin_file(
    input_dir: Path, dataset: str, method: str, far: Any, beta: Any
) -> Optional[Path]:
    method_slug = slugify_for_file(method)
    dataset_slug = slugify_for_file(dataset)

    patterns = [
        f"{dataset_slug}_{method_slug}_far_{far}_beta_{beta}.csv",
        f"{dataset_slug}*{method_slug}*far_{far}*beta_{beta}*.csv",
        f"*{dataset_slug}*{method_slug}*.csv",
    ]

    for pattern in patterns:
        matches = sorted(input_dir.glob(pattern))
        if len(matches) > 0:
            return matches[0]

    return None

def rebin_reliability(bin_df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """
    Merge fine-grained reliability bins into `n_bins` equal-width bins over [0, 1].

    Averages (mean_pred_error, empirical_error) are combined using count-weighting,
    while `count` is summed.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bin_df["left"].values + bin_df["right"].values)

    # Assign each original bin to a target coarse bin by its center.
    idx = np.clip(np.digitize(centers, edges) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        sub = bin_df[idx == b]
        total = sub["count"].sum()
        if total <= 0:
            continue
        w = sub["count"].values
        rows.append(
            {
                "left": edges[b],
                "right": edges[b + 1],
                "count": total,
                "mean_pred_error": np.average(sub["mean_pred_error"].values, weights=w),
                "empirical_error": np.average(sub["empirical_error"].values, weights=w),
            }
        )

    return pd.DataFrame(rows)

def build_calibration_diagram_figure(cfg, fcfg, name: str):
    """
    Draw calibration/reliability diagrams from bin CSV files produced by
    mprisk_diagnostics_experiments.py.

    Expected bin columns:
      left, right, count, mean_pred_error, empirical_error
    """
    input_dir = Path(str(fcfg.input_dir))
    dataset = str(fcfg.dataset)
    far = fcfg.far
    beta = fcfg.beta
    methods = list(fcfg.methods)

    figsize = tuple(fcfg.get("figsize", [5.2, 5.2]))
    n_bins = fcfg.get("n_bins", None)  # optional: reduce number of bins

    for method in methods:
        bin_path = find_reliability_bin_file(
            input_dir=input_dir,
            dataset=dataset,
            method=method,
            far=far,
            beta=beta,
        )

        if bin_path is None:
            print(
                f"[warning] reliability bins not found for "
                f"dataset={dataset}, method={method}, far={far}, beta={beta}"
            )
            continue

        bin_df = pd.read_csv(bin_path)
        bin_df = bin_df[bin_df["count"] > 0].copy()

        if bin_df.empty:
            print(f"[warning] empty reliability bins: {bin_path}")
            continue

        # Optionally merge into fewer, wider bins.
        if n_bins is not None and int(n_bins) < len(bin_df):
            bin_df = rebin_reliability(bin_df, int(n_bins))
            if bin_df.empty:
                print(f"[warning] empty reliability bins after rebin: {bin_path}")
                continue

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.5)

        widths = bin_df["right"].values - bin_df["left"].values
        centers = 0.5 * (bin_df["left"].values + bin_df["right"].values)

        ax.bar(
            centers,
            bin_df["empirical_error"].values,
            width=0.9 * widths,
            alpha=0.65,
            edgecolor="black",
            label="empirical error",
        )

        ax.scatter(
            bin_df["mean_pred_error"].values,
            bin_df["empirical_error"].values,
            c="red",
            s=35,
            zorder=5,
            label="bins",
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted error probability")
        ax.set_ylabel("Empirical error frequency")
        ax.set_title(
            f"{get_pretty(cfg, 'model', method)}, "
            f"{get_pretty(cfg, 'dataset', dataset)}, FPIR={far}"
        )
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=8)

        fig.tight_layout()

        save_plot(fig, cfg, f"{name}_{slugify_for_file(method)}")


# ---------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------


def latex_table_env_begin(tcfg) -> str:
    placement = str(tcfg.get("placement", "t"))

    if bool(tcfg.get("fix_table", False)):
        out = "\\begin{table}[H]\n"
    else:
        out = f"\\begin{{table}}[{placement}]\n"

    out += "\\centering\n"

    table_size = str(tcfg.get("table_size", "footnotesize"))
    if table_size:
        out += f"\\{table_size}\n"

    tabcolsep = tcfg.get("tabcolsep", None)
    if tabcolsep is not None:
        out += f"\\setlength\\tabcolsep{{{tabcolsep}pt}}\n"

    return out


def latex_table_env_end(tcfg) -> str:
    out = ""
    if tcfg.get("use_adjustbox", False):
        out += "\\end{adjustbox}\n"
    out += "\\end{table}\n"
    return out


def render_simple_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    tcfg,
    column_format: Optional[str] = None,
) -> str:
    if column_format is None:
        column_format = "l" + "c" * (len(df.columns) - 1)

    out = latex_table_env_begin(tcfg)
    out += f"\\caption{{{caption}}}\n"
    out += f"\\label{{{label}}}\n"

    if tcfg.get("use_adjustbox", False):
        width = str(tcfg.get("adjustbox_width", "\\textwidth"))
        out += f"\\begin{{adjustbox}}{{width={width}}}\n"

    out += f"\\begin{{tabular}}{{{column_format}}}\n"
    out += "\\toprule\n"

    out += " & ".join(latex_cell(c) for c in df.columns) + " \\\\\n"
    out += "\\midrule\n"

    for _, row in df.iterrows():
        out += " & ".join(latex_cell(v) for v in row.values) + " \\\\\n"

    out += "\\bottomrule\n"
    out += "\\end{tabular}\n"
    out += latex_table_env_end(tcfg)

    return out


def render_grouped_column_table(
    df: pd.DataFrame,
    groups: List[Tuple[str, int]],
    subheaders: List[str],
    caption: str,
    label: str,
    tcfg,
) -> str:
    column_count = sum(n for _, n in groups)
    column_format = "l" + "c" * column_count

    out = latex_table_env_begin(tcfg)
    out += f"\\caption{{{caption}}}\n"
    out += f"\\label{{{label}}}\n"

    if tcfg.get("use_adjustbox", False):
        width = str(tcfg.get("adjustbox_width", "\\textwidth"))
        out += f"\\begin{{adjustbox}}{{width={width}}}\n"

    out += f"\\begin{{tabular}}{{{column_format}}}\n"
    out += "\\toprule\n"

    first_header = ["Method"]
    for group_name, ncols in groups:
        first_header.append(
            f"\\multicolumn{{{ncols}}}{{c}}{{{latex_cell(group_name)}}}"
        )

    out += " & ".join(first_header) + " \\\\\n"

    out += " & " + " & ".join(latex_cell(x) for x in subheaders) + " \\\\\n"
    out += "\\midrule\n"

    for _, row in df.iterrows():
        out += " & ".join(latex_cell(v) for v in row.values) + " \\\\\n"

    out += "\\bottomrule\n"
    out += "\\end{tabular}\n"
    out += latex_table_env_end(tcfg)

    return out


# ---------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------


def build_main_prr_table(cfg, tcfg, name: str):
    df = read_csv_or_empty(tcfg.input_path)
    metric = str(tcfg.get("metric", "prr_f1"))
    beta = tcfg.get("beta", None)

    methods = list(tcfg.methods)
    datasets = list(tcfg.datasets)
    fars_by_dataset = to_plain(tcfg.fars)

    col_keys = []
    groups = []
    subheaders = []

    for dataset in datasets:
        fars = fars_by_dataset[dataset]
        groups.append((get_pretty(cfg, "dataset", dataset), len(fars)))

        for far in fars:
            key = f"{dataset}|{far}"
            col_keys.append(key)
            subheaders.append(f"${far}$")

    numeric = pd.DataFrame(index=methods, columns=col_keys, dtype=float)

    for method in methods:
        for dataset in datasets:
            for far in fars_by_dataset[dataset]:
                selected = select_df(
                    df,
                    {
                        "method": method,
                        "dataset": dataset,
                        "far": far,
                        "beta": beta,
                    },
                )
                numeric.loc[method, f"{dataset}|{far}"] = first_or_mean(
                    selected, metric
                )

    directions = {c: "high" for c in numeric.columns}
    display_numeric = format_highlighted_numeric_df(
        numeric,
        directions=directions,
        digits=int(tcfg.get("round_num", cfg.round_num)),
        exclude_rows=list(tcfg.get("exclude_from_best", [])),
        highlight=bool(tcfg.get("highlight_best", True)),
    )

    display = display_numeric.copy()
    display.insert(
        0,
        "Method",
        [get_pretty(cfg, "model", m) for m in display.index],
    )

    latex = render_grouped_column_table(
        display.reset_index(drop=True),
        groups=groups,
        subheaders=subheaders,
        caption=str(tcfg.caption),
        label=str(tcfg.label),
        tcfg=tcfg,
    )

    write_outputs(cfg, name, latex, numeric_df=numeric, display_df=display)


def build_error_type_detection_table(cfg, tcfg, name: str):
    df = read_csv_or_empty(tcfg.input_path)

    df = select_df(
        df,
        {
            "dataset": tcfg.get("dataset"),
            "far": tcfg.get("far"),
            "beta": tcfg.get("beta"),
        },
    )

    metric = str(tcfg.get("metric", "auroc"))
    methods = list(tcfg.methods)
    targets = list(tcfg.targets)

    numeric = pd.DataFrame(index=methods, columns=targets, dtype=float)

    for method in methods:
        for target in targets:
            selected = select_df(df, {"method": method, "target": target})
            numeric.loc[method, target] = first_or_mean(selected, metric)

    display_numeric = format_highlighted_numeric_df(
        numeric,
        directions={c: "high" for c in numeric.columns},
        digits=int(tcfg.get("round_num", cfg.round_num)),
        exclude_rows=list(tcfg.get("exclude_from_best", [])),
        highlight=bool(tcfg.get("highlight_best", True)),
    )

    display = display_numeric.copy()
    display.insert(
        0,
        "Method",
        [get_pretty(cfg, "model", m) for m in display.index],
    )
    display = display.rename(
        columns={target: get_pretty(cfg, "target", target) for target in targets}
    )

    latex = render_simple_table(
        display.reset_index(drop=True),
        caption=str(tcfg.caption),
        label=str(tcfg.label),
        tcfg=tcfg,
    )

    write_outputs(cfg, name, latex, numeric_df=numeric, display_df=display)


def build_component_ablation_table(cfg, tcfg, name: str):
    df = read_csv_or_empty(tcfg.input_path)

    df = select_df(
        df,
        {
            "dataset": tcfg.get("dataset"),
            "source_method": tcfg.get("source_method"),
            "far": tcfg.get("far"),
            "beta": tcfg.get("beta"),
        },
    )

    variants = list(tcfg.variants)
    metrics = list(tcfg.metrics)

    numeric = pd.DataFrame(index=variants, columns=metrics, dtype=float)

    for variant in variants:
        for metric in metrics:
            selected = select_df(df, {"variant": variant})
            numeric.loc[variant, metric] = first_or_mean(selected, metric)

    directions = {
        metric: str(
            tcfg.get("directions", {}).get(
                metric, get_metric_direction(cfg, metric, "high")
            )
        )
        for metric in metrics
    }

    display_numeric = format_highlighted_numeric_df(
        numeric,
        directions=directions,
        digits=int(tcfg.get("round_num", cfg.round_num)),
        exclude_rows=list(tcfg.get("exclude_from_best", [])),
        highlight=bool(tcfg.get("highlight_best", True)),
    )

    display = display_numeric.copy()
    display.insert(
        0,
        "Variant",
        [get_pretty(cfg, "variant", v) for v in display.index],
    )
    display = display.rename(
        columns={metric: get_pretty(cfg, "column", metric) for metric in metrics}
    )

    latex = render_simple_table(
        display.reset_index(drop=True),
        caption=str(tcfg.caption),
        label=str(tcfg.label),
        tcfg=tcfg,
    )

    write_outputs(cfg, name, latex, numeric_df=numeric, display_df=display)


def build_kl_inversion_table(cfg, tcfg, name: str):
    df = read_csv_or_empty(tcfg.input_path)

    df = select_df(
        df,
        {
            "dataset": tcfg.get("dataset"),
            "far": tcfg.get("far"),
            "beta": tcfg.get("beta"),
            "kl_reference": tcfg.get("kl_reference"),
        },
    )

    methods = list(tcfg.methods)
    metrics = list(tcfg.metrics)

    numeric = pd.DataFrame(index=methods, columns=metrics, dtype=float)

    for method in methods:
        for metric in metrics:
            selected = select_df(df, {"method": method})
            numeric.loc[method, metric] = first_or_mean(selected, metric)

    directions = {
        metric: str(
            tcfg.get("directions", {}).get(
                metric, get_metric_direction(cfg, metric, "high")
            )
        )
        for metric in metrics
    }

    display_numeric = format_highlighted_numeric_df(
        numeric,
        directions=directions,
        digits=int(tcfg.get("round_num", cfg.round_num)),
        exclude_rows=list(tcfg.get("exclude_from_best", [])),
        highlight=bool(tcfg.get("highlight_best", True)),
    )

    display = display_numeric.copy()
    display.insert(
        0,
        "Method",
        [get_pretty(cfg, "model", m) for m in display.index],
    )
    display = display.rename(columns={m: get_pretty(cfg, "column", m) for m in metrics})

    latex = render_simple_table(
        display.reset_index(drop=True),
        caption=str(tcfg.caption),
        label=str(tcfg.label),
        tcfg=tcfg,
    )

    write_outputs(cfg, name, latex, numeric_df=numeric, display_df=display)


def build_validation_size_table(cfg, tcfg, name: str):
    df = read_csv_or_empty(tcfg.input_path)

    df = select_df(
        df,
        {
            "dataset": tcfg.get("dataset"),
            "variant": tcfg.get("variant"),
            "far": tcfg.get("far"),
            "beta": tcfg.get("beta"),
        },
    )

    if df.empty:
        print(f"[warning] validation-size table {name} has no rows")
        return

    df = df.sort_values("validation_fraction")

    rows = []
    numeric_rows = []

    for _, row in df.iterrows():
        frac = float(row["validation_fraction"])

        test_prr = format_pm(
            row.get("test_prr_f1_mean", np.nan),
            row.get("test_prr_f1_std", np.nan),
            digits=int(tcfg.get("round_num", cfg.round_num)),
        )

        rows.append(
            {
                "Validation fraction": format_float(frac, digits=2),
                "Test PRR": test_prr,
                "$\\lambda_{FA}$": format_float(
                    row.get("lambda_fa_mean", np.nan), digits=2
                ),
                "$\\lambda_{ID}$": format_float(
                    row.get("lambda_id_mean", np.nan), digits=2
                ),
                "$\\lambda_{FR}$": format_float(
                    row.get("lambda_fr_mean", np.nan), digits=2
                ),
                "$\\lambda_{NS}$": format_float(
                    row.get("lambda_ns_mean", np.nan), digits=2
                ),
            }
        )

        numeric_rows.append(
            {
                "validation_fraction": frac,
                "test_prr_f1_mean": row.get("test_prr_f1_mean", np.nan),
                "test_prr_f1_std": row.get("test_prr_f1_std", np.nan),
                "lambda_fa_mean": row.get("lambda_fa_mean", np.nan),
                "lambda_id_mean": row.get("lambda_id_mean", np.nan),
                "lambda_fr_mean": row.get("lambda_fr_mean", np.nan),
                "lambda_ns_mean": row.get("lambda_ns_mean", np.nan),
            }
        )

    display = pd.DataFrame(rows)
    numeric = pd.DataFrame(numeric_rows).set_index("validation_fraction")

    latex = render_simple_table(
        display,
        caption=str(tcfg.caption),
        label=str(tcfg.label),
        tcfg=tcfg,
    )

    write_outputs(cfg, name, latex, numeric_df=numeric, display_df=display)


def build_operating_point_transfer_table(cfg, tcfg, name: str):
    df = read_csv_or_empty(tcfg.input_path)

    df = select_df(
        df,
        {
            "dataset": tcfg.get("dataset"),
            "variant": tcfg.get("variant"),
            "beta": tcfg.get("beta"),
        },
    )

    if df.empty:
        print(f"[warning] operating-point transfer table {name} has no rows")
        return

    metric = str(tcfg.get("metric", "test_prr_f1"))

    pivot = df.pivot_table(
        index="train_far",
        columns="eval_far",
        values=metric,
        aggfunc="mean",
    )

    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    directions = {c: "high" for c in pivot.columns}
    display_numeric = format_highlighted_numeric_df(
        pivot,
        directions=directions,
        digits=int(tcfg.get("round_num", cfg.round_num)),
        highlight=bool(tcfg.get("highlight_best", True)),
    )

    display = display_numeric.copy()
    display.insert(0, "Tune FPIR", [format_float(i, digits=2) for i in display.index])
    display = display.rename(
        columns={c: f"${format_float(c, digits=2)}$" for c in pivot.columns}
    )

    latex = render_simple_table(
        display.reset_index(drop=True),
        caption=str(tcfg.caption),
        label=str(tcfg.label),
        tcfg=tcfg,
    )

    write_outputs(cfg, name, latex, numeric_df=pivot, display_df=display)


def build_hyperparameter_sensitivity_table(cfg, tcfg, name: str):
    df = read_csv_or_empty(tcfg.input_path)

    df = select_df(
        df,
        {
            "dataset": tcfg.get("dataset"),
            "far": tcfg.get("far"),
            "beta": tcfg.get("beta"),
        },
    )

    if df.empty:
        print(f"[warning] hyperparameter table {name} has no rows")
        return

    if tcfg.get("sort_by", None) is not None:
        sort_by = str(tcfg.sort_by)
        ascending = str(tcfg.get("sort_direction", "high")) == "low"
        df = df.sort_values(sort_by, ascending=ascending)

    variants = list(df["variant"].astype(str).values)
    metrics = list(tcfg.metrics)

    numeric = pd.DataFrame(index=variants, columns=metrics, dtype=float)

    for variant in variants:
        selected = select_df(df, {"variant": variant})
        for metric in metrics:
            numeric.loc[variant, metric] = first_or_mean(selected, metric)

    directions = {
        metric: str(
            tcfg.get("directions", {}).get(
                metric, get_metric_direction(cfg, metric, "high")
            )
        )
        for metric in metrics
    }

    display_numeric = format_highlighted_numeric_df(
        numeric,
        directions=directions,
        digits=int(tcfg.get("round_num", cfg.round_num)),
        exclude_rows=list(tcfg.get("exclude_from_best", [])),
        highlight=bool(tcfg.get("highlight_best", True)),
    )

    display = display_numeric.copy()
    display.insert(
        0,
        "Variant",
        [get_pretty(cfg, "variant", v) for v in display.index],
    )
    display = display.rename(columns={m: get_pretty(cfg, "column", m) for m in metrics})

    latex = render_simple_table(
        display.reset_index(drop=True),
        caption=str(tcfg.caption),
        label=str(tcfg.label),
        tcfg=tcfg,
    )

    write_outputs(cfg, name, latex, numeric_df=numeric, display_df=display)


def build_mixed_prior_necessity_table(cfg, tcfg, name: str):
    df = read_csv_or_empty(tcfg.input_path)

    df = select_df(
        df,
        {
            "dataset": tcfg.get("dataset"),
            "source_method": tcfg.get("source_method"),
            "far": tcfg.get("far"),
            "beta": tcfg.get("beta"),
        },
    )

    if "prr_f1" in df.columns:
        df = df[df["prr_f1"].notna()]

    variants = list(tcfg.variants)
    metrics = list(tcfg.metrics)

    numeric = pd.DataFrame(index=variants, columns=metrics, dtype=float)

    for variant in variants:
        selected = select_df(df, {"variant": variant})
        for metric in metrics:
            numeric.loc[variant, metric] = first_or_mean(selected, metric)

    directions = {
        metric: str(
            tcfg.get("directions", {}).get(
                metric, get_metric_direction(cfg, metric, "high")
            )
        )
        for metric in metrics
    }

    display_numeric = format_highlighted_numeric_df(
        numeric,
        directions=directions,
        digits=int(tcfg.get("round_num", cfg.round_num)),
        exclude_rows=list(tcfg.get("exclude_from_best", [])),
        highlight=bool(tcfg.get("highlight_best", True)),
    )

    display = display_numeric.copy()
    display.insert(
        0,
        "Variant",
        [get_pretty(cfg, "variant", v) for v in display.index],
    )
    display = display.rename(columns={m: get_pretty(cfg, "column", m) for m in metrics})

    latex = render_simple_table(
        display.reset_index(drop=True),
        caption=str(tcfg.caption),
        label=str(tcfg.label),
        tcfg=tcfg,
    )

    write_outputs(cfg, name, latex, numeric_df=numeric, display_df=display)


def build_reliability_table(cfg, tcfg, name: str):
    df = read_csv_or_empty(tcfg.input_path)

    df = select_df(
        df,
        {
            "dataset": tcfg.get("dataset"),
            "far": tcfg.get("far"),
            "beta": tcfg.get("beta"),
        },
    )

    methods = list(tcfg.methods)
    metrics = list(tcfg.metrics)

    numeric = pd.DataFrame(index=methods, columns=metrics, dtype=float)

    for method in methods:
        selected = select_df(df, {"method": method})
        for metric in metrics:
            numeric.loc[method, metric] = first_or_mean(selected, metric)

    directions = {
        metric: str(
            tcfg.get("directions", {}).get(
                metric, get_metric_direction(cfg, metric, "high")
            )
        )
        for metric in metrics
    }

    display_numeric = format_highlighted_numeric_df(
        numeric,
        directions=directions,
        digits=int(tcfg.get("round_num", cfg.round_num)),
        exclude_rows=list(tcfg.get("exclude_from_best", [])),
        highlight=bool(tcfg.get("highlight_best", True)),
    )

    display = display_numeric.copy()
    display.insert(
        0,
        "Method",
        [get_pretty(cfg, "model", m) for m in display.index],
    )
    display = display.rename(columns={m: get_pretty(cfg, "column", m) for m in metrics})

    latex = render_simple_table(
        display.reset_index(drop=True),
        caption=str(tcfg.caption),
        label=str(tcfg.label),
        tcfg=tcfg,
    )

    write_outputs(cfg, name, latex, numeric_df=numeric, display_df=display)


def build_bootstrap_table(cfg, tcfg, name: str):
    df = read_csv_or_empty(tcfg.input_path)

    if df.empty:
        print(f"[warning] bootstrap table {name} has no rows")
        return

    selectors = to_plain(tcfg.get("selectors", {}))
    df = select_df(df, selectors)

    rows = []
    numeric_rows = []

    digits = int(tcfg.get("round_num", cfg.round_num))

    for _, row in df.iterrows():
        comparison = (
            get_pretty(cfg, "model", row["method_a"])
            + " $-$ "
            + get_pretty(cfg, "model", row["method_b"])
        )

        rows.append(
            {
                "Dataset": get_pretty(cfg, "dataset", row["dataset"]),
                "FPIR": f"${format_float(row['far'], digits=2)}$",
                "Comparison": comparison,
                "PRR A": format_float(row.get("prr_a_full", np.nan), digits=digits),
                "PRR B": format_float(row.get("prr_b_full", np.nan), digits=digits),
                "$\\Delta$ PRR [95\\% CI]": format_ci(
                    row.get("delta_full", np.nan),
                    row.get("ci95_low", np.nan),
                    row.get("ci95_high", np.nan),
                    digits=digits,
                ),
                "$p(\\Delta\\leq 0)$": format_float(
                    row.get("p_delta_le_0", np.nan), digits=3
                ),
            }
        )

        numeric_rows.append(row.to_dict())

    display = pd.DataFrame(rows)
    numeric = pd.DataFrame(numeric_rows)

    latex = render_simple_table(
        display,
        caption=str(tcfg.caption),
        label=str(tcfg.label),
        tcfg=tcfg,
    )

    write_outputs(cfg, name, latex, numeric_df=numeric, display_df=display)


def build_runtime_table(cfg, tcfg, name: str):
    df = read_csv_or_empty(tcfg.input_path)

    df = select_df(
        df,
        {
            "dataset": tcfg.get("dataset"),
            "far": tcfg.get("far"),
            "beta": tcfg.get("beta"),
        },
    )

    methods = list(tcfg.methods)
    metrics = list(tcfg.metrics)

    numeric = pd.DataFrame(index=methods, columns=metrics, dtype=float)

    for method in methods:
        selected = select_df(df, {"method": method})
        for metric in metrics:
            numeric.loc[method, metric] = first_or_mean(selected, metric)

    directions = {
        metric: str(
            tcfg.get("directions", {}).get(
                metric, get_metric_direction(cfg, metric, "low")
            )
        )
        for metric in metrics
    }

    display_numeric = format_highlighted_numeric_df(
        numeric,
        directions=directions,
        digits=int(tcfg.get("round_num", cfg.round_num)),
        exclude_rows=list(tcfg.get("exclude_from_best", [])),
        highlight=bool(tcfg.get("highlight_best", True)),
    )

    display = display_numeric.copy()
    display.insert(
        0,
        "Method",
        [get_pretty(cfg, "model", m) for m in display.index],
    )
    display = display.rename(columns={m: get_pretty(cfg, "column", m) for m in metrics})

    latex = render_simple_table(
        display.reset_index(drop=True),
        caption=str(tcfg.caption),
        label=str(tcfg.label),
        tcfg=tcfg,
    )

    write_outputs(cfg, name, latex, numeric_df=numeric, display_df=display)


BUILDERS = {
    "main_prr": build_main_prr_table,
    "error_type_detection": build_error_type_detection_table,
    "component_ablation": build_component_ablation_table,
    "kl_inversion": build_kl_inversion_table,
    "validation_size": build_validation_size_table,
    "operating_point_transfer": build_operating_point_transfer_table,
    "hyperparameter_sensitivity": build_hyperparameter_sensitivity_table,
    "mixed_prior_necessity": build_mixed_prior_necessity_table,
    "reliability": build_reliability_table,
    "bootstrap": build_bootstrap_table,
    "runtime": build_runtime_table,
}
FIGURE_BUILDERS = {
    "rejection_curve": build_rejection_curve_figure,
    "calibration_diagram": build_calibration_diagram_figure,
}


@hydra.main(
    config_path="/app/configs/latex_tables_new_paper",
    config_name="prepare_new_paper_tables",
    version_base="1.2",
)
def main(cfg):
    exp_dir = Path(cfg.exp_dir)
    ensure_dir(exp_dir)
    ensure_dir(exp_dir / "tex")
    ensure_dir(exp_dir / "csv")

    tables = to_plain(cfg.tables)

    for table_name, table_cfg_plain in tables.items():
        if not table_cfg_plain.get("enabled", True):
            print(f"[skip] table {table_name}")
            continue

        tcfg = OmegaConf.create(table_cfg_plain)
        table_type = str(tcfg.type)

        if table_type not in BUILDERS:
            raise ValueError(f"Unknown table type {table_type!r} for {table_name}")

        print("=" * 100)
        print(f"[build] {table_name} ({table_type})")
        print("=" * 100)

        BUILDERS[table_type](cfg, tcfg, table_name)
    if "figures" in cfg:
        figures = to_plain(cfg.figures)

        for fig_name, fig_cfg_plain in figures.items():
            if not fig_cfg_plain.get("enabled", True):
                print(f"[skip] figure {fig_name}")
                continue

            fcfg = OmegaConf.create(fig_cfg_plain)
            fig_type = str(fcfg.type)

            if fig_type not in FIGURE_BUILDERS:
                raise ValueError(f"Unknown figure type {fig_type!r} for {fig_name}")

            print("=" * 100)
            print(f"[figure] {fig_name} ({fig_type})")
            print("=" * 100)

            FIGURE_BUILDERS[fig_type](cfg, fcfg, fig_name)
    hydra_cfg = HydraConfig.get()
    print("\nAll requested tables are created.")
    print(f"Hydra job: {hydra_cfg.job.name}")
    print(f"Output directory: {Path(cfg.exp_dir).resolve()}")


if __name__ == "__main__":
    main()
