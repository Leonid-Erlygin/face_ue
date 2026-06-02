# evaluation/open_set_methods/kappa_utils.py
from __future__ import annotations

import math
import warnings
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import gammaln, ive

_EPS = 1e-12


def threshold_at_far(scores: np.ndarray, far: float) -> float:
    """
    Deterministic acceptance threshold for rule: accept iff score >= tau.
    Chooses approximately floor(far * N) out-of-gallery probes to be accepted.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    scores = scores[np.isfinite(scores)]

    if scores.size == 0:
        raise ValueError("Cannot compute FPIR threshold: no out-of-gallery scores.")

    far = float(far)
    if not (0.0 <= far <= 1.0):
        raise ValueError(f"far must be in [0, 1], got {far}")

    if far <= 0:
        return float(np.nextafter(scores.max(), np.inf))
    if far >= 1:
        return float(np.nextafter(scores.min(), -np.inf))

    n_accept = int(np.floor(far * scores.size))
    if n_accept <= 0:
        return float(np.nextafter(scores.max(), np.inf))

    idx = scores.size - n_accept
    tau = np.partition(scores, idx)[idx]
    return float(tau)


def log_uniform_density(d: int) -> float:
    return float(gammaln(d / 2.0) - math.log(2.0) - (d / 2.0) * math.log(math.pi))


def power_log_normalizer_np(kappa: np.ndarray | float, d: int) -> np.ndarray:
    k = np.asarray(kappa, dtype=np.float64)
    return (
        gammaln(d - 1.0 + k)
        + gammaln(d / 2.0 + k)
        + (k - 1.0) * math.log(2.0)
        - (d / 2.0) * math.log(math.pi)
        - gammaln(d - 1.0 + 2.0 * k)
    )


def vmf_log_normalizer_np(kappa: np.ndarray | float, d: int) -> np.ndarray:
    k = np.asarray(kappa, dtype=np.float64)
    k = np.maximum(k, _EPS)

    v = d / 2.0 - 1.0
    log_ive = np.log(ive(v, k)) + k

    out = (v * np.log(k)) - (d / 2.0) * math.log(2.0 * math.pi) - log_ive

    # Fallback for rare ive underflow.
    bad = ~np.isfinite(out)
    if np.any(bad):
        kk = k[bad]
        mu = 4.0 * v * v
        corr = 1.0 - (mu - 1.0) / (8.0 * kk)
        corr = np.maximum(corr, np.finfo(np.float64).tiny)
        log_iv_asympt = kk - 0.5 * np.log(2.0 * math.pi * kk) + np.log(corr)
        out[bad] = (
            v * np.log(kk)
            - (d / 2.0) * math.log(2.0 * math.pi)
            - log_iv_asympt
        )

    return out


def decision_margin_for_score(
    score: float,
    kappa: float,
    beta: float,
    K: int,
    d: int,
    class_model: str,
) -> float:
    """
    Positive margin means gallery class wins over out-of-gallery class.
    """
    score = float(np.clip(score, -1.0 + 1e-9, 1.0 - 1e-9))
    kappa = float(max(kappa, _EPS))

    if class_model == "power":
        log_like = power_log_normalizer_np(kappa, d) + kappa * np.log1p(score)
    elif class_model == "vMF":
        log_like = vmf_log_normalizer_np(kappa, d) + kappa * score
    else:
        raise ValueError(f"Unknown class_model={class_model}")

    log_gallery_prior = math.log((1.0 - beta) / K)
    log_oog_prior = math.log(beta)
    log_uniform = log_uniform_density(d)

    return float(log_gallery_prior + log_like - (log_oog_prior + log_uniform))


def solve_kappa_for_tau(
    tau: float,
    beta: float,
    K: int,
    d: int,
    class_model: str,
    kappa_low: float = 1.0,
    kappa_high: float = 1_000_000.0,
    grid_size: int = 256,
) -> float:
    """
    Deterministically choose gallery kappa so that the Bayesian reject/accept
    boundary matches cosine threshold tau.
    """
    if not (0.0 < beta < 1.0):
        raise ValueError(f"beta must be in (0, 1), got {beta}")

    tau = float(np.clip(tau, -1.0 + 1e-9, 1.0 - 1e-9))

    def f_log_kappa(log_k: float) -> float:
        return decision_margin_for_score(
            score=tau,
            kappa=float(np.exp(log_k)),
            beta=beta,
            K=K,
            d=d,
            class_model=class_model,
        )

    lo = math.log(kappa_low)
    hi = math.log(kappa_high)

    grid = np.linspace(lo, hi, grid_size)
    values = np.array([f_log_kappa(x) for x in grid], dtype=np.float64)
    finite = np.isfinite(values)

    if not np.any(finite):
        raise FloatingPointError("Could not compute finite kappa decision margins.")

    grid = grid[finite]
    values = values[finite]

    # Prefer exact sign-change root.
    for i in range(len(grid) - 1):
        if values[i] == 0:
            return float(np.exp(grid[i]))
        if values[i] * values[i + 1] < 0:
            root = brentq(f_log_kappa, grid[i], grid[i + 1], maxiter=100)
            return float(np.exp(root))

    # Otherwise minimize absolute residual.
    res = minimize_scalar(
        lambda x: abs(f_log_kappa(x)),
        bounds=(float(grid[0]), float(grid[-1])),
        method="bounded",
        options={"xatol": 1e-8},
    )
    kappa = float(np.exp(res.x))
    residual = abs(f_log_kappa(res.x))

    if residual > 1e-2:
        warnings.warn(
            f"Large residual while solving kappa: residual={residual:.4g}, "
            f"tau={tau:.6f}, beta={beta}, K={K}, model={class_model}",
            RuntimeWarning,
        )

    return kappa