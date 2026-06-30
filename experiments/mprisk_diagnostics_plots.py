#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def maybe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        print(f"[skip] missing {path}")
        return None

    df = pd.read_csv(path)
    if len(df) == 0:
        print(f"[skip] empty {path}")
        return None

    return df


def save_fig(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path.with_suffix('.png')}")
    print(f"[saved] {out_path.with_suffix('.pdf')}")


def plot_mixed_prior_necessity(exp_dir: Path):
    path = exp_dir / "tables" / "mixed_prior_necessity.csv"
    df = maybe_read_csv(path)
    if df is None:
        return

    plot_dir = exp_dir / "plots" / "mixed_prior_necessity"

    # Only rows with PRR values, skip group-summary rows.
    df = df[df["prr_f1"].notna()].copy()
    if len(df) == 0:
        return

    preferred_order = [
        "collapsed_unknown_risk",
        "ordinary_risk_no_NS",
        "equal_full_risk",
        "MPRisk_current",
        "r_FA",
        "r_ID",
        "r_FR",
        "r_NS",
        "unknown_nonspecificity",
    ]

    group_cols = ["dataset", "source_method", "far", "beta"]

    for keys, sub in df.groupby(group_cols):
        dataset, source_method, far, beta = keys

        sub = sub.copy()
        sub["variant"] = pd.Categorical(
            sub["variant"],
            categories=preferred_order + sorted(set(sub["variant"]) - set(preferred_order)),
            ordered=True,
        )
        sub = sub.sort_values("variant")

        fig, ax = plt.subplots(figsize=(max(8.0, 0.55 * len(sub)), 4.8))

        ax.bar(
            np.arange(len(sub)),
            sub["prr_f1"].values,
            color="tab:blue",
            alpha=0.85,
        )

        ax.set_xticks(np.arange(len(sub)))
        ax.set_xticklabels(sub["variant"].astype(str).values, rotation=45, ha="right")
        ax.set_ylabel("PRR for F1 filtering")
        ax.set_title(f"Mixed-prior necessity: {dataset}, {source_method}, FPIR={far}")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

        for i, value in enumerate(sub["prr_f1"].values):
            ax.text(
                i,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )

        fig.tight_layout()

        out = plot_dir / f"{dataset}_{source_method}_far_{far}_beta_{beta}_prr"
        save_fig(fig, out)

        # Error-type AUROC plot.
        auroc_cols = [
            "any_error_auroc",
            "false_accept_auroc",
            "false_reject_auroc",
            "misidentification_auroc",
        ]
        if all(c in sub.columns for c in auroc_cols):
            fig, ax = plt.subplots(figsize=(max(9.0, 0.65 * len(sub)), 5.0))

            x = np.arange(len(sub))
            width = 0.20

            for offset, col in zip([-1.5, -0.5, 0.5, 1.5], auroc_cols):
                ax.bar(
                    x + offset * width,
                    sub[col].values,
                    width=width,
                    label=col.replace("_auroc", ""),
                    alpha=0.85,
                )

            ax.set_xticks(x)
            ax.set_xticklabels(sub["variant"].astype(str).values, rotation=45, ha="right")
            ax.set_ylabel("AUROC")
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"Error-type detection: {dataset}, {source_method}, FPIR={far}")
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)
            ax.legend(fontsize=8)

            fig.tight_layout()

            out = plot_dir / f"{dataset}_{source_method}_far_{far}_beta_{beta}_error_type_auroc"
            save_fig(fig, out)


def plot_ns_group_summary(exp_dir: Path):
    path = exp_dir / "tables" / "mixed_prior_necessity.csv"
    df = maybe_read_csv(path)
    if df is None:
        return

    df = df[df["variant"] == "r_NS_group_summary"].copy()
    if len(df) == 0:
        return

    plot_dir = exp_dir / "plots" / "mixed_prior_necessity"

    group_cols = ["dataset", "source_method", "far", "beta"]

    preferred_groups = [
        "false_reject",
        "true_reject",
        "false_accept",
        "misidentification",
        "correct",
    ]

    for keys, sub in df.groupby(group_cols):
        dataset, source_method, far, beta = keys

        sub = sub.copy()
        sub["group"] = pd.Categorical(
            sub["group"],
            categories=preferred_groups,
            ordered=True,
        )
        sub = sub.sort_values("group")

        fig, ax = plt.subplots(figsize=(7.0, 4.5))

        ax.bar(
            np.arange(len(sub)),
            sub["mean"].values,
            yerr=sub["std"].fillna(0.0).values,
            capsize=4,
            color="tab:orange",
            alpha=0.85,
        )

        ax.set_xticks(np.arange(len(sub)))
        ax.set_xticklabels(sub["group"].astype(str).values, rotation=30, ha="right")
        ax.set_ylabel("Mean $r_{NS}$")
        ax.set_title(f"Reject non-specificity by group: {dataset}, FPIR={far}")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

        fig.tight_layout()

        out = plot_dir / f"{dataset}_{source_method}_far_{far}_beta_{beta}_rns_by_group"
        save_fig(fig, out)


def plot_reliability_metrics(exp_dir: Path):
    path = exp_dir / "tables" / "reliability_metrics.csv"
    df = maybe_read_csv(path)
    if df is None:
        return

    plot_dir = exp_dir / "plots" / "reliability"

    metrics = ["ece", "brier", "nll", "error_auroc", "error_auprc"]
    metrics = [m for m in metrics if m in df.columns]

    group_cols = ["dataset", "far", "beta"]

    for keys, sub in df.groupby(group_cols):
        dataset, far, beta = keys

        for metric in metrics:
            sub_sorted = sub.sort_values(metric, ascending=("auroc" not in metric and "auprc" not in metric))

            fig, ax = plt.subplots(figsize=(6.5, 4.2))

            ax.bar(
                np.arange(len(sub_sorted)),
                sub_sorted[metric].values,
                color="tab:green",
                alpha=0.85,
            )

            ax.set_xticks(np.arange(len(sub_sorted)))
            ax.set_xticklabels(sub_sorted["method"].values, rotation=30, ha="right")
            ax.set_ylabel(metric.upper())
            ax.set_title(f"{metric.upper()}: {dataset}, FPIR={far}")
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)

            for i, value in enumerate(sub_sorted[metric].values):
                ax.text(
                    i,
                    value,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            fig.tight_layout()

            out = plot_dir / f"{dataset}_far_{far}_beta_{beta}_{metric}"
            save_fig(fig, out)


def plot_bootstrap(exp_dir: Path):
    path = exp_dir / "tables" / "bootstrap_prr_differences.csv"
    df = maybe_read_csv(path)
    if df is None:
        return

    plot_dir = exp_dir / "plots" / "bootstrap"

    if len(df) == 0:
        return

    labels = [
        f"{row.method_a} - {row.method_b}\n{row.dataset}, FPIR={row.far}"
        for _, row in df.iterrows()
    ]

    y = np.arange(len(df))
    delta = df["delta_full"].values
    ci_low = df["ci95_low"].values
    ci_high = df["ci95_high"].values

    xerr = np.vstack([delta - ci_low, ci_high - delta])

    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.55 * len(df))))

    ax.errorbar(
        delta,
        y,
        xerr=xerr,
        fmt="o",
        color="tab:blue",
        ecolor="black",
        elinewidth=1.5,
        capsize=4,
    )

    ax.axvline(0.0, color="red", linestyle="--", linewidth=1.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("PRR difference")
    ax.set_title("Paired bootstrap confidence intervals")
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)

    fig.tight_layout()

    out = plot_dir / "bootstrap_prr_differences"
    save_fig(fig, out)


def plot_runtime(exp_dir: Path):
    path = exp_dir / "tables" / "runtime_overhead.csv"
    df = maybe_read_csv(path)
    if df is None:
        return

    plot_dir = exp_dir / "plots" / "runtime"

    group_cols = ["dataset", "far", "beta"]

    for keys, sub in df.groupby(group_cols):
        dataset, far, beta = keys
        sub = sub.sort_values("elapsed_ms_per_probe")

        fig, ax = plt.subplots(figsize=(7.5, 4.2))

        ax.bar(
            np.arange(len(sub)),
            sub["elapsed_ms_per_probe"].values,
            color="tab:purple",
            alpha=0.85,
        )

        ax.set_xticks(np.arange(len(sub)))
        ax.set_xticklabels(sub["method"].values, rotation=35, ha="right")
        ax.set_ylabel("ms / probe")
        ax.set_title(f"Runtime overhead: {dataset}, FPIR={far}")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

        for i, value in enumerate(sub["elapsed_ms_per_probe"].values):
            ax.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

        fig.tight_layout()

        out = plot_dir / f"{dataset}_far_{far}_beta_{beta}_runtime"
        save_fig(fig, out)


def plot_quality_stress(exp_dir: Path):
    path = exp_dir / "tables" / "quality_stress.csv"
    df = maybe_read_csv(path)
    if df is None:
        return

    plot_dir = exp_dir / "plots" / "quality_stress"

    group_cols = ["dataset", "far", "beta", "score"]

    for keys, sub in df.groupby(group_cols):
        dataset, far, beta, score = keys
        sub = sub.sort_values("kappa_input_scale")

        fig, ax = plt.subplots(figsize=(6.8, 4.4))

        ax.plot(
            sub["kappa_input_scale"].values,
            sub["prr_f1"].values,
            marker="o",
            linewidth=2,
            label="PRR F1",
        )

        if "false_reject_auroc" in sub.columns:
            ax.plot(
                sub["kappa_input_scale"].values,
                sub["false_reject_auroc"].values,
                marker="s",
                linewidth=2,
                label="FR AUROC",
            )

        ax.set_xscale("log")
        ax.set_xlabel("kappa_input_scale")
        ax.set_ylabel("metric")
        ax.set_title(f"Quality stress: {dataset}, {score}, FPIR={far}")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()

        fig.tight_layout()

        out = plot_dir / f"{dataset}_far_{far}_beta_{beta}_{score}_quality_stress"
        save_fig(fig, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="outputs/experiments/mprisk_diagnostics",
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)

    plot_mixed_prior_necessity(exp_dir)
    plot_ns_group_summary(exp_dir)
    plot_reliability_metrics(exp_dir)
    plot_bootstrap(exp_dir)
    plot_runtime(exp_dir)
    plot_quality_stress(exp_dir)

    print(f"Plots saved under: {exp_dir / 'plots'}")


if __name__ == "__main__":
    main()