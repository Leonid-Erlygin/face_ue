import numpy as np
from scipy.special import softmax

from typing import Any


class BernoulliVariance:
    def __call__(self, similarity: np.ndarray, probe_score: np.ndarray, tau) -> Any:
        s = probe_score
        conf_score = np.abs(s - tau)
        return -conf_score

class MaximumSoftmaxProbability:
    def __init__(self, T: float):
        self.T = T
    def __call__(self, similarity: np.ndarray, probe_score: np.ndarray, tau: float) -> Any:
        sims = np.column_stack([similarity, np.full(similarity.shape[0], tau)])
        sims /= self.T
        conf_score = np.max(softmax(sims, axis=-1), axis=-1)
        return 1 - conf_score

class RandomScore:
    def __call__(self, similarity: np.ndarray, probe_score: np.ndarray, tau) -> Any:
        unc_score = np.arange(probe_score.shape[0])
        rng = np.random.default_rng(1)
        rng.shuffle(unc_score)
        return unc_score


class OracleScore:
    def __call__(self, similarity: np.ndarray, probe_score: np.ndarray, tau) -> Any:
        unc_score = np.arange(probe_score.shape[0])
        rng = np.random.default_rng(1)
        rng.shuffle(unc_score)
        return unc_score
