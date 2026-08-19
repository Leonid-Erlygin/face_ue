from __future__ import annotations

import numpy as np
from scipy.special import ive
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def vmf_mean_resultant(kappa: np.ndarray, d: int) -> np.ndarray:
    """Stable A_d(kappa)=I_{d/2}(kappa)/I_{d/2-1}(kappa).

    The scaled-Bessel ratio is accurate in the normal operating range.  A
    dimension-aware closed-form approximation is used only when scipy
    underflows, which can happen for very small kappa in high dimensions.
    """
    k = np.maximum(np.asarray(kappa, dtype=np.float64), 1e-12)
    nu = d / 2.0 - 1.0
    den = ive(nu, k)
    num = ive(nu + 1.0, k)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    bad = ~np.isfinite(out)
    if np.any(bad):
        kb = k[bad]
        a = float(d - 1)
        # 2k / (a + sqrt(a^2 + 4k^2)) has the correct small- and
        # large-concentration limits and is numerically stable.
        out[bad] = (2.0 * kb) / (a + np.sqrt(a * a + 4.0 * kb * kb))
    return np.clip(out, 0.0, 1.0)


def _safe_auroc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=np.float64)
    ok = np.isfinite(score)
    y, score = y[ok], score[ok]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def summarize_scf_split(
    kappa: np.ndarray,
    true_center_cosine: np.ndarray,
    center_correct: np.ndarray,
    true_gallery_cosine: np.ndarray | None = None,
    gallery_correct: np.ndarray | None = None,
    *,
    embedding_dim: int,
) -> dict:
    """Summarize whether learned SCF concentration behaves as its loss implies.

    For KLDiracVMF the unconstrained optimum satisfies
        A_d(kappa_x) = mu_x^T w_y.
    Therefore the stationarity residual is a direct model diagnostic, not an
    ad-hoc confidence metric.
    """
    k = np.asarray(kappa, dtype=np.float64).reshape(-1)
    cos = np.asarray(true_center_cosine, dtype=np.float64).reshape(-1)
    correct = np.asarray(center_correct, dtype=bool).reshape(-1)
    if not (len(k) == len(cos) == len(correct)):
        raise ValueError("kappa/cosine/correct lengths differ")
    if len(k) == 0:
        raise ValueError("empty SCF diagnostic split")
    if np.any(~np.isfinite(k)) or np.any(k <= 0):
        raise FloatingPointError("SCF produced non-finite or non-positive kappa")

    logk = np.log(k)
    target = np.clip(cos, 0.0, 1.0)
    mean_resultant = vmf_mean_resultant(k, embedding_dim)
    residual = mean_resultant - target
    rho = spearmanr(logk, cos).statistic if len(k) > 1 else np.nan

    q = np.quantile(k, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    out = {
        "count": int(len(k)),
        "kappa": {
            "min": float(q[0]), "q05": float(q[1]), "q25": float(q[2]),
            "median": float(q[3]), "q75": float(q[4]), "q95": float(q[5]),
            "max": float(q[6]), "mean": float(np.mean(k)),
            "log_std": float(np.std(logk)),
        },
        "center": {
            "accuracy": float(np.mean(correct)),
            "true_cosine_mean": float(np.mean(cos)),
            "true_cosine_median": float(np.median(cos)),
            "spearman_log_kappa_vs_true_cosine": float(rho),
            "error_auroc_negative_log_kappa": _safe_auroc((~correct).astype(int), -logk),
        },
        "scf_stationarity": {
            "mean_A_d_kappa": float(np.mean(mean_resultant)),
            "mean_target_positive_cosine": float(np.mean(target)),
            "mean_signed_residual": float(np.mean(residual)),
            "mean_absolute_residual": float(np.mean(np.abs(residual))),
            "rmse_residual": float(np.sqrt(np.mean(residual ** 2))),
        },
        "kappa_by_center_outcome": {
            "correct_median": float(np.median(k[correct])) if np.any(correct) else np.nan,
            "wrong_median": float(np.median(k[~correct])) if np.any(~correct) else np.nan,
        },
    }

    if true_gallery_cosine is not None and gallery_correct is not None:
        gcos = np.asarray(true_gallery_cosine, dtype=np.float64).reshape(-1)
        gcorr = np.asarray(gallery_correct, dtype=bool).reshape(-1)
        grho = spearmanr(logk, gcos).statistic if len(k) > 1 else np.nan
        out["api_gallery"] = {
            "accuracy": float(np.mean(gcorr)),
            "true_cosine_mean": float(np.mean(gcos)),
            "true_cosine_median": float(np.median(gcos)),
            "spearman_log_kappa_vs_true_cosine": float(grho),
            "error_auroc_negative_log_kappa": _safe_auroc((~gcorr).astype(int), -logk),
            "kappa_correct_median": float(np.median(k[gcorr])) if np.any(gcorr) else np.nan,
            "kappa_wrong_median": float(np.median(k[~gcorr])) if np.any(~gcorr) else np.nan,
        }
    return out
