from typing import List, Sequence, Tuple
from abc import ABC
import numpy as np

from evaluation.metrics import EvalMetricsT


class Abstract1NEval(ABC):
    def __call__(
        self,
        probe_feats: np.ndarray,
        probe_unc: np.ndarray,
        gallery_feats: np.ndarray,
        gallery_unc: np.ndarray,
        probe_ids: np.ndarray,
        gallery_ids: np.ndarray,
        fars: np.ndarray,
    ) -> EvalMetricsT:
        raise NotImplementedError
