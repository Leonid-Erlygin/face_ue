from typing import Tuple
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from sklearn.metrics import roc_auc_score, auc
from evaluation.metrics.base import Plot
from scipy.stats import rankdata
from .base import BaseMetric


class RejectAUROC(BaseMetric):
    def __init__(self, xs_range: Tuple) -> None:
        super().__init__()
        self.xs = np.linspace(*xs_range)

    def evaluate(
        self,
        probe_ids: np.ndarray,
        gallery_ids: np.ndarray,
        similarity_matrix: np.ndarray,
        confidence: np.ndarray,
    ) -> Plot:
        pred_indexes = np.argmax(similarity_matrix, axis=1)
        correct_prediction_mask = gallery_ids[pred_indexes] == probe_ids
        certainty = np.abs(confidence.mean() - confidence)
        ys = []
        for x in self.xs:
            thr = np.quantile(certainty, x)
            keep_mask = certainty > thr
            labels = correct_prediction_mask[keep_mask]
            scores = certainty[keep_mask]

            if np.any(labels) and np.any(~labels):
                score = roc_auc_score(labels, scores)
            else:
                score = np.nan

            ys.append(score)

        ys = np.array(ys)
        mask = ~np.isnan(ys)
        score = auc(self.xs[mask], ys[mask])
        return Plot(np.array(self.xs), np.array(ys), score=score)

    def setup_plt(self, fig: Figure, ax: Axes):
        ax.set_xlabel("Rejection rate")
        ax.set_ylabel("AUROC")
