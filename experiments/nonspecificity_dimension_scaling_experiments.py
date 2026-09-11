#!/usr/bin/env python3
"""Controlled study of dimension-normalized vMF non-specificity.

The dissertation defines

    N_d(kappa) = exp[-D_2(q_kappa || u_0)]

and considers the monotone transform

    Nbar_d(kappa) = N_d(kappa) ** (1 / (d - 1)).

Because the transform is monotone for every fixed embedding dimension d, it
cannot change within-dataset rankings or PRR.  The empirical question is
instead whether it makes the *absolute scale* more comparable across embedding
dimensions.

To avoid confounding dimension with a different physical concentration regime,
this script compares dimensions at matched mean resultant length
A_d(kappa).  For every target r in (0, 1), it solves A_d(kappa)=r and reports the
raw and dimension-normalized non-specificity.

Outputs
-------
<out_dir>/tables/dimension_stability.csv
    One row per (dimension, target A_d) pair.
<out_dir>/tables/dimension_stability_summary.csv
    Across-dimension scale spread at each matched A_d target.
<out_dir>/tables/ranking_invariance.csv
    Numerical sanity check that the root transform preserves ordering within
    each fixed dimension.
<out_dir>/study_manifest.json
    Full experiment settings.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the repository root importable when the script is launched as
# `python experiments/...py` without relying on an external PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import spearmanr

from evaluation.open_set_methods.kappa_utils import (
    vmf_mean_resultant_np,
    vmf_nonspecificity_np,
)


def solve_kappa_for_mean_resultant(target: float, d: int) -> float:
    """Solve A_d(kappa)=target for kappa >= 0."""
    target = float(target)
    d = int(d)
    if not (0.0 < target < 1.0):
        raise ValueError(f"target must lie in (0, 1), got {target}")
    if d < 2:
        raise ValueError(f"d must be >= 2, got {d}")

    def f(kappa: float) -> float:
        return float(vmf_mean_resultant_np(kappa, d=d)) - target

    lo = 0.0
    hi = max(1.0, float(d))
    while f(hi) < 0.0:
        hi *= 2.0
        if hi > 1e10:
            raise RuntimeError(
                f"Could not bracket A_d(kappa)={target} for d={d}; last hi={hi}"
            )
    return float(brentq(f, lo, hi, xtol=1e-11, rtol=1e-11, maxiter=300))


def root_nonspecificity(n_raw: np.ndarray | float, d: int) -> np.ndarray:
    n_raw = np.asarray(n_raw, dtype=np.float64)
    if np.any((n_raw <= 0.0) | (n_raw > 1.0) | (~np.isfinite(n_raw))):
        raise FloatingPointError("N_d(kappa) must be finite and in (0, 1]")
    # Work in log space to avoid unnecessary underflow/precision loss.
    return np.exp(np.log(n_raw) / float(d - 1))


def run_dimension_stability(dims: list[int], targets: list[float]) -> pd.DataFrame:
    rows = []
    for target in targets:
        for d in dims:
            kappa = solve_kappa_for_mean_resultant(target, d)
            achieved = float(vmf_mean_resultant_np(kappa, d=d))
            n_raw = float(vmf_nonspecificity_np(kappa, d=d))
            n_root = float(root_nonspecificity(n_raw, d=d))
            d2 = -float(np.log(n_raw))
            rows.append(
                {
                    "embedding_dim": int(d),
                    "target_mean_resultant": float(target),
                    "achieved_mean_resultant": achieved,
                    "kappa": kappa,
                    "N_raw": n_raw,
                    "N_root": n_root,
                    "renyi_D2": d2,
                    "renyi_D2_per_dimension": d2 / float(d - 1),
                    "log10_N_raw": float(np.log10(n_raw)),
                    "log10_N_root": float(np.log10(n_root)),
                }
            )
    return pd.DataFrame(rows)


def summarize_dimension_stability(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, g in df.groupby("target_mean_resultant", sort=True):
        raw = g["log10_N_raw"].to_numpy(dtype=np.float64)
        root = g["log10_N_root"].to_numpy(dtype=np.float64)
        raw_range = float(np.max(raw) - np.min(raw))
        root_range = float(np.max(root) - np.min(root))
        rows.append(
            {
                "target_mean_resultant": float(target),
                "num_dimensions": int(len(g)),
                "raw_log10_range": raw_range,
                "root_log10_range": root_range,
                "log10_range_reduction": raw_range - root_range,
                "relative_range_after_root": (
                    root_range / raw_range if raw_range > 0.0 else 0.0
                ),
                "raw_log10_std": float(np.std(raw)),
                "root_log10_std": float(np.std(root)),
                "root_reduces_dimension_spread": bool(root_range <= raw_range + 1e-12),
            }
        )
    return pd.DataFrame(rows)


def run_ranking_invariance(
    dims: list[int], kappa_min: float, kappa_max: float, num_points: int
) -> pd.DataFrame:
    kappas = np.geomspace(float(kappa_min), float(kappa_max), int(num_points))
    rows = []
    for d in dims:
        raw = vmf_nonspecificity_np(kappas, d=d).astype(np.float64)
        root = root_nonspecificity(raw, d=d)
        rho = float(spearmanr(raw, root).statistic)

        raw_order = np.argsort(raw, kind="mergesort")
        root_order = np.argsort(root, kind="mergesort")
        order_identical = bool(np.array_equal(raw_order, root_order))

        # The transform is strictly increasing on (0,1], so any discrepancy here
        # is a numerical bug rather than an empirical phenomenon.
        max_roundtrip_error = float(
            np.max(np.abs(np.power(root, d - 1) - raw))
        )
        rows.append(
            {
                "embedding_dim": int(d),
                "num_kappa_points": int(num_points),
                "spearman_raw_vs_root": rho,
                "ordering_identical": order_identical,
                "max_roundtrip_abs_error": max_roundtrip_error,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="outputs/experiments/nonspecificity_dimension_scaling",
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 768, 1024, 2048],
    )
    parser.add_argument(
        "--targets",
        type=float,
        nargs="+",
        default=[0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.98],
        help="Matched A_d(kappa) values used for the cross-dimension comparison.",
    )
    parser.add_argument("--kappa-min", type=float, default=1e-3)
    parser.add_argument("--kappa-max", type=float, default=1e5)
    parser.add_argument("--ranking-points", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dims = sorted(set(int(x) for x in args.dims))
    targets = sorted(set(float(x) for x in args.targets))

    out_dir = Path(args.out_dir)
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    stability = run_dimension_stability(dims, targets)
    summary = summarize_dimension_stability(stability)
    ranking = run_ranking_invariance(
        dims,
        kappa_min=args.kappa_min,
        kappa_max=args.kappa_max,
        num_points=args.ranking_points,
    )

    stability.to_csv(table_dir / "dimension_stability.csv", index=False)
    summary.to_csv(table_dir / "dimension_stability_summary.csv", index=False)
    ranking.to_csv(table_dir / "ranking_invariance.csv", index=False)

    manifest = {
        "study": "dimension-normalized vMF non-specificity",
        "formula_raw": "N_d(kappa)=exp(-D2(q_kappa||u0))",
        "formula_root": "Nbar_d(kappa)=N_d(kappa)^(1/(d-1))",
        "comparison_control": "matched mean resultant length A_d(kappa)",
        "dims": dims,
        "targets": targets,
        "ranking_kappa_min": float(args.kappa_min),
        "ranking_kappa_max": float(args.kappa_max),
        "ranking_points": int(args.ranking_points),
        "interpretation": {
            "ranking": (
                "Within a fixed dimension the root transform is strictly monotone; "
                "PRR/ranking cannot improve by construction."
            ),
            "scale": (
                "The empirical hypothesis is supported only if root_log10_range "
                "is systematically smaller than raw_log10_range at matched A_d."
            ),
        },
    }
    (out_dir / "study_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {table_dir / 'dimension_stability.csv'}")
    print(f"Wrote {table_dir / 'dimension_stability_summary.csv'}")
    print(f"Wrote {table_dir / 'ranking_invariance.csv'}")
    print(f"Wrote {out_dir / 'study_manifest.json'}")
    print("\nCross-dimension summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
