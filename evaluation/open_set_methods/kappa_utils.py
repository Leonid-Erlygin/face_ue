# evaluation/open_set_methods/kappa_utils.py
from __future__ import annotations

import math
import warnings
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import gammaln, hyp0f1, ive

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
    """Stable log normalizer of a vMF density in ``d`` dimensions.

    ``scipy.special.ive`` is stable for the large concentrations used by SCF,
    but it can underflow for very small kappa at high Bessel order.  In that
    regime we use the exact hypergeometric identity

        C_d(k) = C_d(0) / 0F1(; d/2; k^2/4).

    A large-kappa asymptotic fallback is retained only for rare failures outside
    the small-kappa regime.
    """
    if int(d) < 2:
        raise ValueError(f"d must be >= 2, got {d}")

    original = np.asarray(kappa, dtype=np.float64)
    scalar = original.ndim == 0
    k = np.atleast_1d(original).astype(np.float64, copy=True)
    if np.any(~np.isfinite(k)) or np.any(k < 0):
        raise ValueError("kappa must contain finite non-negative values")

    out = np.empty_like(k)
    zero = k == 0.0
    out[zero] = log_uniform_density(int(d))

    positive = ~zero
    if np.any(positive):
        kp = k[positive]
        v = d / 2.0 - 1.0
        with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
            log_ive = np.log(ive(v, kp)) + kp
            vals = (
                v * np.log(kp)
                - (d / 2.0) * math.log(2.0 * math.pi)
                - log_ive
            )

        bad = ~np.isfinite(vals)
        if np.any(bad):
            kb = kp[bad]
            repaired = np.empty_like(kb)

            # Small/high-order regime: exact 0F1 representation.
            small = kb <= max(50.0, 0.25 * float(d))
            if np.any(small):
                h = hyp0f1(d / 2.0, (kb[small] ** 2) / 4.0)
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    repaired[small] = log_uniform_density(int(d)) - np.log(h)

            # Large-kappa fallback for any remaining numerical failures.
            if np.any(~small):
                kl = kb[~small]
                mu = 4.0 * v * v
                corr = 1.0 - (mu - 1.0) / (8.0 * kl)
                corr = np.maximum(corr, np.finfo(np.float64).tiny)
                log_iv_asympt = (
                    kl
                    - 0.5 * np.log(2.0 * math.pi * kl)
                    + np.log(corr)
                )
                repaired[~small] = (
                    v * np.log(kl)
                    - (d / 2.0) * math.log(2.0 * math.pi)
                    - log_iv_asympt
                )

            vals[bad] = repaired

        out[positive] = vals

    if scalar:
        return np.asarray(out[0])
    return out.reshape(original.shape)


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


def candidate_kappas_for_tau(
    tau: float,
    beta: float,
    K: int,
    d: int,
    class_model: str,
    kappa_low: float = 1.0,
    kappa_high: float = 1_000_000.0,
    grid_size: int = 512,
    residual_tol: float = 1e-6,
) -> list[float]:
    """Return every numerically distinct gallery-kappa root for a boundary.

    The power-spherical decision margin can be non-monotone in concentration,
    so FAR matching alone need not identify a unique kappa. Returning all roots
    lets an outer validation criterion choose among equally valid operating-point
    solutions instead of silently taking the first one.
    """
    if not (0.0 < beta < 1.0):
        raise ValueError(f"beta must be in (0, 1), got {beta}")
    tau = float(np.clip(tau, -1.0 + 1e-9, 1.0 - 1e-9))

    def f(log_k: float) -> float:
        return decision_margin_for_score(
            score=tau,
            kappa=float(np.exp(log_k)),
            beta=beta,
            K=K,
            d=d,
            class_model=class_model,
        )

    grid = np.linspace(math.log(kappa_low), math.log(kappa_high), int(grid_size))
    vals = np.asarray([f(x) for x in grid], dtype=np.float64)
    roots: list[float] = []
    for i in range(len(grid) - 1):
        a, b = vals[i], vals[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if abs(a) <= residual_tol:
            roots.append(float(np.exp(grid[i])))
        if a * b < 0:
            root = brentq(f, float(grid[i]), float(grid[i + 1]), maxiter=100)
            roots.append(float(np.exp(root)))
    if len(vals) and np.isfinite(vals[-1]) and abs(vals[-1]) <= residual_tol:
        roots.append(float(np.exp(grid[-1])))

    roots = sorted(roots)
    distinct: list[float] = []
    for root in roots:
        if not distinct or abs(math.log(root) - math.log(distinct[-1])) > 1e-5:
            distinct.append(root)
    return distinct


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



def vmf_mean_resultant_np(kappa: np.ndarray | float, d: int) -> np.ndarray:
    """Return the vMF mean-resultant length ``A_d(kappa)`` stably.

    For ``Z ~ vMF(mu, kappa)`` on the unit sphere,

        E[Z] = A_d(kappa) * mu,
        A_d(kappa) = I_{d/2}(kappa) / I_{d/2-1}(kappa).

    The ratio of exponentially scaled Bessel functions is stable over the
    concentration range used by SCF.  A dimension-aware approximation is used
    only for rare underflow cases.  Exact zeros are mapped to ``A_d(0)=0``.
    """
    if int(d) < 2:
        raise ValueError(f"d must be >= 2, got {d}")

    original = np.asarray(kappa, dtype=np.float64)
    scalar = original.ndim == 0
    k = np.atleast_1d(original).astype(np.float64, copy=True)
    if np.any(~np.isfinite(k)) or np.any(k < 0):
        raise ValueError("kappa must contain finite non-negative values")

    out = np.zeros_like(k)
    positive = k > 0
    if np.any(positive):
        kp = k[positive]
        nu = d / 2.0 - 1.0
        den = ive(nu, kp)
        num = ive(nu + 1.0, kp)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ratio = num / den

        # A_d(kappa) is strictly in (0, 1) for every finite kappa>0.  Treat
        # exact zero/one as numerical failures as well as NaN/Inf; scaled Bessel
        # evaluation can underflow to a misleading finite zero at high order.
        bad = (~np.isfinite(ratio)) | (ratio <= 0.0) | (ratio >= 1.0)
        if np.any(bad):
            kb = kp[bad]
            repaired = np.empty_like(kb)

            # In the small-kappa/high-order regime, evaluate the Bessel ratio
            # exactly through the hypergeometric representation
            #
            #   I_{nu+1}(k) / I_nu(k)
            #     = (k/d) * 0F1(; d/2+1; k^2/4) / 0F1(; d/2; k^2/4).
            #
            # This preserves the exact A_d(k) ~ k/d limit.
            small = kb <= max(50.0, 0.25 * float(d))
            if np.any(small):
                z = (kb[small] ** 2) / 4.0
                den_h = hyp0f1(d / 2.0, z)
                num_h = hyp0f1(d / 2.0 + 1.0, z)
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    repaired[small] = (kb[small] / float(d)) * (num_h / den_h)

            # Rare large-kappa fallback. This approximation has the correct
            # concentration limit A_d(k)->1 and is only used if SciPy's scaled
            # Bessel ratio itself failed.
            if np.any(~small):
                kl = kb[~small]
                a = float(d - 1)
                repaired[~small] = (2.0 * kl) / (
                    a + np.sqrt(a * a + 4.0 * kl * kl)
                )

            ratio[bad] = repaired

        out[positive] = np.clip(ratio, 0.0, 1.0)

    if scalar:
        return np.asarray(out[0])
    return out.reshape(original.shape)


def vmf_nonspecificity_np(kappa: np.ndarray | float, d: int) -> np.ndarray:
    """Normalized vMF non-specificity used by ``r_NS``.

    Computes

        N_d(kappa) = C_d(2 kappa) / (S_{d-1} C_d(kappa)^2)

    in log space.  ``N_d(0)=1`` and larger concentration produces smaller
    non-specificity.
    """
    original = np.asarray(kappa, dtype=np.float64)
    scalar = original.ndim == 0
    k = np.atleast_1d(original).astype(np.float64, copy=True)
    if np.any(~np.isfinite(k)) or np.any(k < 0):
        raise ValueError("kappa must contain finite non-negative values")

    log_surface = -log_uniform_density(int(d))
    k_safe = np.maximum(k, _EPS)
    log_c = vmf_log_normalizer_np(k_safe, d=int(d))
    log_c2 = vmf_log_normalizer_np(2.0 * k_safe, d=int(d))
    log_collision = log_surface + 2.0 * log_c - log_c2
    # The collision concentration is >= 1 analytically; clipping only protects
    # against round-off close to kappa=0.
    log_collision = np.clip(log_collision, 0.0, 700.0)
    out = np.exp(-log_collision)
    out[k == 0] = 1.0
    out = np.clip(out, 0.0, 1.0)

    if scalar:
        return np.asarray(out[0])
    return out.reshape(original.shape)


def vmf_kappa_scale_nll(
    scale: float,
    kappa: np.ndarray,
    true_cosine: np.ndarray,
    d: int,
) -> float:
    """Mean vMF negative log-likelihood for a global concentration scale.

    ``true_cosine`` is the cosine between the sample mean direction and an
    independently constructed true-class gallery/proxy direction.  Constants
    independent of ``scale`` are omitted only if they cancel; this function
    keeps the complete vMF log-normalizer term.
    """
    scale = float(scale)
    if scale < 0 or not np.isfinite(scale):
        return float("inf")

    k = np.asarray(kappa, dtype=np.float64).reshape(-1)
    c = np.asarray(true_cosine, dtype=np.float64).reshape(-1)
    if k.shape != c.shape:
        raise ValueError("kappa and true_cosine must have identical lengths")

    ok = np.isfinite(k) & np.isfinite(c) & (k > 0)
    if not np.any(ok):
        raise ValueError("No finite positive-kappa samples for scale calibration")

    k = k[ok]
    c = np.clip(c[ok], -1.0, 1.0)
    scaled = np.maximum(scale * k, _EPS)
    log_c = vmf_log_normalizer_np(scaled, d=int(d))
    loss = -log_c - scaled * c
    return float(np.mean(loss))


def fit_vmf_kappa_scale(
    kappa: np.ndarray,
    true_cosine: np.ndarray,
    d: int,
    *,
    max_scale: float = 1_000_000.0,
    min_return_scale: float = 1e-8,
) -> dict:
    """Fit one global SCF concentration scale from true-class geometry.

    The exact derivative of the vMF NLL with respect to ``s`` is

        sum_i kappa_i [A_d(s kappa_i) - c_i].

    It is monotone non-decreasing because ``A_d`` is increasing.  Therefore an
    interior optimum is unique.  Boundary solutions are reported explicitly.

    The fitted scale uses *class labels/proxies only*, never OSR error labels or
    an FPIR-specific objective, so the same scale can be reused across operating
    points.
    """
    if int(d) < 2:
        raise ValueError(f"d must be >= 2, got {d}")
    if not np.isfinite(max_scale) or max_scale <= 0:
        raise ValueError("max_scale must be finite and positive")

    k = np.asarray(kappa, dtype=np.float64).reshape(-1)
    c = np.asarray(true_cosine, dtype=np.float64).reshape(-1)
    if k.shape != c.shape:
        raise ValueError("kappa and true_cosine must have identical lengths")

    ok = np.isfinite(k) & np.isfinite(c) & (k > 0)
    if not np.any(ok):
        raise ValueError("No finite positive-kappa samples for scale calibration")

    k = k[ok]
    c = np.clip(c[ok], -1.0, 1.0)
    weight_sum = float(np.sum(k))

    def derivative(scale: float) -> float:
        a = vmf_mean_resultant_np(scale * k, d=int(d))
        # Normalize only for numerical scale; the zero is unchanged.
        return float(np.sum(k * (a - c)) / weight_sum)

    deriv_zero = float(-np.sum(k * c) / weight_sum)
    boundary = "interior"
    converged = True

    if deriv_zero >= 0.0:
        scale = 0.0
        boundary = "lower"
    else:
        hi = 1.0
        deriv_hi = derivative(hi)
        while deriv_hi < 0.0 and hi < max_scale:
            hi = min(max_scale, hi * 2.0)
            deriv_hi = derivative(hi)

        if deriv_hi < 0.0:
            scale = float(max_scale)
            boundary = "upper"
            converged = False
        else:
            scale = float(brentq(derivative, 0.0, hi, xtol=1e-12, rtol=1e-10))

    returned_scale = max(float(scale), float(min_return_scale))
    return {
        "scale": returned_scale,
        "raw_optimum_scale": float(scale),
        "boundary": boundary,
        "converged": bool(converged),
        "count": int(len(k)),
        "embedding_dim": int(d),
        "derivative_at_zero": float(deriv_zero),
        "derivative_at_scale": float(derivative(returned_scale)),
        "nll_scale_1": vmf_kappa_scale_nll(1.0, k, c, d=int(d)),
        "nll_fitted": vmf_kappa_scale_nll(returned_scale, k, c, d=int(d)),
    }
