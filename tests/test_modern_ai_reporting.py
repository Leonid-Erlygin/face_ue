from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.modern_ai.reporting import (
    binary_rejection_curve,
    generate_leaf_report,
    make_fraction_grid,
    retrieval_rejection_curve,
)


def test_retrieval_rejection_curve_filters_errors_first():
    df = pd.DataFrame(
        {
            "known": [True, True, False, False],
            "rejected": [False, False, False, True],
            "correct_retrieval": [True, False, False, False],
            "any_error": [False, True, True, False],
        }
    )
    uncertainty = np.asarray([0.1, 0.9, 0.8, 0.2])
    curve = retrieval_rejection_curve(df, uncertainty, np.asarray([0.0, 0.5]))
    assert curve.loc[0, "error_rate"] == 0.5
    assert curve.loc[1, "error_rate"] == 0.0
    assert curve.loc[1, "oser_f1"] == 1.0


def test_binary_rejection_curve_matches_selective_risk_semantics():
    df = pd.DataFrame({"target": [0, 1, 0, 1]})
    uncertainty = np.asarray([0.1, 0.9, 0.2, 0.8])
    curve = binary_rejection_curve(df, uncertainty, np.asarray([0.0, 0.5]))
    assert curve.loc[0, "error_rate"] == 0.5
    assert curve.loc[1, "error_rate"] == 0.0
    assert curve.loc[1, "accuracy"] == 1.0


def test_generate_leaf_report_writes_curves_and_latex(tmp_path: Path):
    summary = {
        "experiment": "open_set_evidence_retrieval",
        "protocol": "toy",
        "fitted_gallery_kappa": 42.0,
        "open_set": {"oser_accuracy": 0.75, "oser_f1": 0.75, "fpir": 0.5, "fnir": 0.0},
        "ranking": {"recall@1": 0.9, "mrr": 0.95},
        "uncertainty_detection": {
            "max_similarity": {"any_error_auroc": 0.7, "any_error_auprc": 0.6, "aurc": 0.3},
            "mprisk": {"any_error_auroc": 0.9, "any_error_auprc": 0.8, "aurc": 0.1},
        },
        "calibration": {
            "max_similarity": {"brier": 0.2, "ece": 0.1, "nll": 0.5},
            "mprisk": {"brier": 0.1, "ece": 0.05, "nll": 0.3},
        },
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    pd.DataFrame(
        {
            "known": [True, True, False, False, True, False],
            "rejected": [False, False, False, True, False, True],
            "correct_retrieval": [True, False, False, False, True, False],
            "any_error": [False, True, True, False, False, False],
            "raw__max_similarity": [0.1, 0.8, 0.9, 0.2, 0.15, 0.25],
            "raw__mprisk": [0.1, 0.95, 0.85, 0.2, 0.12, 0.3],
        }
    ).to_csv(tmp_path / "per_query.csv", index=False)

    report = generate_leaf_report(
        tmp_path,
        {"reporting": {"rejection_fractions": [0.0, 0.5, 5]}},
    )
    assert report["generated"] is True
    assert (tmp_path / "rejection_curves" / "all_rejection_curves.csv").is_file()
    assert (tmp_path / "rejection_curves" / "oser_f1_rejection_curve.pdf").is_file()
    assert (tmp_path / "tables" / "uncertainty_metrics.tex").is_file()
    tex = (tmp_path / "tables" / "uncertainty_metrics.tex").read_text(encoding="utf-8")
    assert "MPRisk" in tex
    assert "\\textbf{0.9}" in tex


def test_fraction_grid_repository_triplet_convention():
    grid = make_fraction_grid([0.0, 0.5, 6])
    assert len(grid) == 6
    assert np.isclose(grid[0], 0.0)
    assert np.isclose(grid[-1], 0.5)
