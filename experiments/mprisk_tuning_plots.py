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


def plot_validation_size(exp_dir: Path):
    path = exp_dir / "tables" / "validation_size_ablation_summary.csv"
    df = maybe_read_csv(path)
    if df is None:
        return

    plot_dir = exp_dir / "plots" / "validation_size"

    group_cols = ["dataset", "variant", "far", "beta"]
    for keys, sub in df.groupby(group_cols):
        dataset, variant, far, beta = keys
        sub = sub.sort_values("validation_fraction")

        x = sub["validation_fraction"].values
        y = sub["test_prr_f1_mean"].values
        yerr = sub["test_prr_f1_std"].fillna(0.0).values

        fig, ax = plt.subplots(figsize=(6.2, 4.3))
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=2,
            capsize=4,
        )

        ax.set_xscale("log")
        ax.set_xlabel("Validation fraction used for lambda tuning")
        ax.set_ylabel("Test PRR for F1 filtering")
        ax.set_title(f"{dataset}, {variant}, FPIR={far}, beta={beta}")
        ax.grid(True, linestyle="--", alpha=0.4)

        out = plot_dir / f"{dataset}_{variant}_far_{far}_beta_{beta}_validation_size"
        save_fig(fig, out)


def plot_operating_point_transfer(exp_dir: Path):
    path = exp_dir / "tables" / "operating_point_transfer.csv"
    df = maybe_read_csv(path)
    if df is None:
        return

    plot_dir = exp_dir / "plots" / "operating_point_transfer"

    group_cols = ["dataset", "variant", "beta"]
    for keys, sub in df.groupby(group_cols):
        dataset, variant, beta = keys

        pivot = sub.pivot_table(
            index="train_far",
            columns="eval_far",
            values="test_prr_f1",
            aggfunc="mean",
        )

        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(6.0, 4.8))
        im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")

        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels([str(c) for c in pivot.columns])
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels([str(i) for i in pivot.index])

        ax.set_xlabel("Evaluation FPIR")
        ax.set_ylabel("Lambda-tuning FPIR")
        ax.set_title(f"Operating-point transfer: {dataset}, {variant}, beta={beta}")

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.values[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value < np.nanmean(pivot.values) else "black",
                    fontsize=9,
                )

        fig.colorbar(im, ax=ax, label="Test PRR for F1 filtering")
        fig.tight_layout()

        out = plot_dir / f"{dataset}_{variant}_beta_{beta}_operating_point_transfer"
        save_fig(fig, out)


def plot_cross_dataset_transfer(exp_dir: Path):
    path = exp_dir / "tables" / "cross_dataset_transfer.csv"
    df = maybe_read_csv(path)
    if df is None:
        return

    plot_dir = exp_dir / "plots" / "cross_dataset_transfer"

    group_cols = ["variant", "far", "beta"]
    for keys, sub in df.groupby(group_cols):
        variant, far, beta = keys

        pivot = sub.pivot_table(
            index="source_dataset",
            columns="target_dataset",
            values="test_prr_f1",
            aggfunc="mean",
        )

        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")

        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels([str(c) for c in pivot.columns], rotation=35, ha="right")
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels([str(i) for i in pivot.index])

        ax.set_xlabel("Target test dataset")
        ax.set_ylabel("Source validation dataset")
        ax.set_title(f"Cross-dataset lambda transfer: {variant}, FPIR={far}, beta={beta}")

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.values[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value < np.nanmean(pivot.values) else "black",
                    fontsize=9,
                )

        fig.colorbar(im, ax=ax, label="Test PRR for F1 filtering")
        fig.tight_layout()

        out = plot_dir / f"{variant}_far_{far}_beta_{beta}_cross_dataset_transfer"
        save_fig(fig, out)


def plot_hyperparameter_sensitivity(exp_dir: Path):
    path = exp_dir / "tables" / "hyperparameter_sensitivity.csv"
    df = maybe_read_csv(path)
    if df is None:
        return

    plot_dir = exp_dir / "plots" / "hyperparameter_sensitivity"

    group_cols = ["dataset", "far", "beta"]
    for keys, sub in df.groupby(group_cols):
        dataset, far, beta = keys
        sub = sub.sort_values("test_prr_f1", ascending=False)

        fig, ax = plt.subplots(figsize=(max(7.5, 0.45 * len(sub)), 4.8))

        ax.bar(
            np.arange(len(sub)),
            sub["test_prr_f1"].values,
            color="tab:blue",
            alpha=0.85,
        )

        ax.set_xticks(np.arange(len(sub)))
        ax.set_xticklabels(sub["variant"].values, rotation=45, ha="right")

        ax.set_ylabel("Test PRR for F1 filtering")
        ax.set_title(f"Hyperparameter sensitivity: {dataset}, FPIR={far}, beta={beta}")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

        for i, value in enumerate(sub["test_prr_f1"].values):
            ax.text(
                i,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        fig.tight_layout()

        out = plot_dir / f"{dataset}_far_{far}_beta_{beta}_hyperparameter_sensitivity"
        save_fig(fig, out)

        # Also save lambda plot for the same group.
        lambda_cols = ["lambda_fa", "lambda_id", "lambda_fr", "lambda_ns"]
        if all(c in sub.columns for c in lambda_cols):
            fig, ax = plt.subplots(figsize=(max(8.0, 0.5 * len(sub)), 5.0))

            x = np.arange(len(sub))
            width = 0.2

            for offset, col in zip([-1.5, -0.5, 0.5, 1.5], lambda_cols):
                ax.bar(
                    x + offset * width,
                    sub[col].values,
                    width=width,
                    label=col,
                    alpha=0.85,
                )

            ax.set_xticks(x)
            ax.set_xticklabels(sub["variant"].values, rotation=45, ha="right")
            ax.set_ylabel("Tuned lambda value")
            ax.set_title(f"Tuned lambdas: {dataset}, FPIR={far}, beta={beta}")
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)
            ax.legend()

            fig.tight_layout()

            out = plot_dir / f"{dataset}_far_{far}_beta_{beta}_tuned_lambdas"
            save_fig(fig, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="outputs/experiments/mprisk_tuning",
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)

    plot_validation_size(exp_dir)
    plot_operating_point_transfer(exp_dir)
    plot_cross_dataset_transfer(exp_dir)
    plot_hyperparameter_sensitivity(exp_dir)

    print(f"Plots saved under: {exp_dir / 'plots'}")


if __name__ == "__main__":
    main()