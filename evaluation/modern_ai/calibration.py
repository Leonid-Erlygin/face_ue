from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def binary_nll(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.clip(np.asarray(p, dtype=np.float64).reshape(-1), 1e-8, 1.0 - 1e-8)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


class ConstantProbabilityCalibrator:
    def __init__(self, p: float):
        self.p = float(np.clip(p, 0.0, 1.0))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        n = np.asarray(x).shape[0]
        return np.full(n, self.p, dtype=np.float64)


class LogisticFeatureCalibrator:
    """Map arbitrary uncertainty features to P(error) on validation data."""

    def __init__(self, C: float = 1.0, max_iter: int = 2000):
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.model = None

    def fit(self, x: np.ndarray, y_error: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        y = np.asarray(y_error, dtype=int).reshape(-1)
        ok = np.all(np.isfinite(x), axis=1) & np.isfinite(y)
        if not np.any(ok):
            self.model = ConstantProbabilityCalibrator(float(np.mean(y)) if len(y) else 0.5)
            return self
        xx, yy = x[ok], y[ok]
        if len(np.unique(yy)) < 2:
            self.model = ConstantProbabilityCalibrator(float(np.mean(yy)))
            return self
        self.model = LogisticRegression(C=self.C, max_iter=self.max_iter, solver="lbfgs")
        self.model.fit(xx, yy)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Calibrator has not been fit.")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        if isinstance(self.model, ConstantProbabilityCalibrator):
            return self.model.predict_proba(x)
        out = np.full(len(x), np.nan, dtype=np.float64)
        ok = np.all(np.isfinite(x), axis=1)
        if np.any(ok):
            out[ok] = self.model.predict_proba(x[ok])[:, 1]
        return out


class IsotonicScoreCalibrator:
    """Monotone scalar calibration preserving the risk ranking."""

    def __init__(self, increasing: str | bool = True):
        self.increasing = increasing
        self.model = None
        self.constant = None

    def fit(self, score: np.ndarray, y_error: np.ndarray):
        score = np.asarray(score, dtype=np.float64).reshape(-1)
        y = np.asarray(y_error, dtype=np.float64).reshape(-1)
        ok = np.isfinite(score) & np.isfinite(y)
        if not np.any(ok):
            self.constant = float(np.mean(y)) if len(y) else 0.5
            return self
        ss, yy = score[ok], y[ok]
        if len(np.unique(yy)) < 2 or np.std(ss) < 1e-12:
            self.constant = float(np.mean(yy))
            return self
        self.model = IsotonicRegression(increasing=self.increasing, out_of_bounds="clip")
        self.model.fit(ss, yy)
        return self

    def predict_proba(self, score: np.ndarray) -> np.ndarray:
        score = np.asarray(score, dtype=np.float64).reshape(-1)
        if self.constant is not None:
            return np.full(len(score), self.constant, dtype=np.float64)
        if self.model is None:
            raise RuntimeError("Calibrator has not been fit.")
        out = np.full(len(score), np.nan, dtype=np.float64)
        ok = np.isfinite(score)
        if np.any(ok):
            out[ok] = np.asarray(self.model.predict(score[ok]), dtype=np.float64)
        return out


@dataclass
class ErrorCalibratorBundle:
    """Fair post-hoc calibration for retrieval uncertainty methods.

    All calibrators are fit on the same validation labels. HolUE is allowed its
    two native KL features; scalar methods use monotone isotonic calibration so
    calibration cannot improve their ranking metrics.
    """

    scalar: Dict[str, IsotonicScoreCalibrator]
    holue: LogisticFeatureCalibrator

    @classmethod
    def fit(
        cls,
        score_map: Mapping[str, np.ndarray],
        kl_1: np.ndarray,
        kl_2: np.ndarray,
        y_error: np.ndarray,
        exclude: Sequence[str] = ("holue",),
    ) -> "ErrorCalibratorBundle":
        scalar = {}
        for name, score in score_map.items():
            if name in set(exclude):
                continue
            scalar[name] = IsotonicScoreCalibrator(increasing=True).fit(score, y_error)
        holue = LogisticFeatureCalibrator().fit(
            np.column_stack([kl_1, kl_2]), y_error
        )
        return cls(scalar=scalar, holue=holue)

    def apply(
        self,
        score_map: Mapping[str, np.ndarray],
        kl_1: np.ndarray,
        kl_2: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        out = {}
        for name, calibrator in self.scalar.items():
            if name in score_map:
                out[name] = calibrator.predict_proba(score_map[name])
        out["holue"] = self.holue.predict_proba(np.column_stack([kl_1, kl_2]))
        return out
