import numpy as np

from typing import Any


class BernoulliVariance:
    def __call__(self, similarity: np.ndarray, probe_score: np.ndarray, tau) -> Any:
        s = probe_score
        # unc_score = -(s**2) + 2 * s * tau + 1 - 2 * tau
        unc_score = -np.abs(s - tau) + np.abs(1 - tau)  # - 0.5 rewrite this
        return unc_score


class RandomScore:
    def __call__(self, similarity: np.ndarray, probe_score: np.ndarray, tau) -> Any:
        unc_score = np.arange(probe_score.shape[0])
        np.random.shuffle(unc_score)
        return unc_score


class OracleScore:
    def __call__(self, similarity: np.ndarray, probe_score: np.ndarray, tau) -> Any:
        unc_score = np.arange(probe_score.shape[0])
        np.random.shuffle(unc_score)
        return unc_score
