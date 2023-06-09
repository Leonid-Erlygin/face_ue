from typing import Tuple
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from sklearn.metrics import auc, roc_curve, roc_auc_score
from scipy.special import softmax
from .base import BaseMetric, Plot


class DIRFAR(BaseMetric):
    def __init__(self, far_range: Tuple[int, int, int]) -> None:
        self.fars = 10 ** np.arange(far_range[0], far_range[1], 4.0 / far_range[2])
        self.fars = np.append(self.fars, 1)

    def setup_plt(self, fig: Figure, ax: Axes):
        ax.set_xlabel("False Alarm Rate")
        ax.set_xlim(0.0001, 1)
        ax.set_xscale("log")
        ax.set_ylabel("Detection & Identification Rate (%)")
        ax.set_ylim(0, 1)

        ax.grid(linestyle="--", linewidth=1)
        ax.legend(fontsize="x-small")
        fig.tight_layout()

    def evaluate(
        self,
        probe_ids,
        gallery_ids,
        similarity_matrix: np.ndarray,
        confidence: np.ndarray,
    ) -> Plot:
        gallery_ids_argsort = np.argsort(gallery_ids)
        is_seen = np.isin(probe_ids, gallery_ids)
        seen_sim: np.ndarray = similarity_matrix[is_seen]

        # Boolean mask (seen_probes, gallery_ids), 1 where the probe matches gallery sample
        pos_mask: np.ndarray = (
            probe_ids[is_seen, None] == gallery_ids[None, gallery_ids_argsort]
        )

        pos_sims = seen_sim[pos_mask]
        neg_sims = seen_sim[~pos_mask].reshape(*pos_sims.shape, -1)
        pos_score = confidence[is_seen]
        neg_score = confidence[~is_seen]
        non_gallery_sims = similarity_matrix[~is_seen]

        # see which test gallery images have higher closeness to true class in gallery than
        # to the wrong classes
        correct_pos_cond = pos_sims > np.max(neg_sims, axis=1)

        neg_score_sorted = np.sort(neg_score)[::-1]
        threshes, recalls = [], []
        for far in self.fars:
            # compute operating threshold τ, which gives neaded far
            thresh = neg_score_sorted[max(int((neg_score_sorted.shape[0]) * far) - 1, 0)]

            # compute DI rate at given operating threshold τ
            recall = (
                np.sum(np.logical_and(correct_pos_cond, pos_score > thresh))
                / pos_sims.shape[0]
            )
            threshes.append(thresh)
            recalls.append(recall)

        cmc_scores = list(zip(neg_sims, pos_sims.reshape(-1, 1))) + list(
            zip(non_gallery_sims, [None] * non_gallery_sims.shape[0])
        )


        xs = self.fars
        ys = np.array(recalls)

        return Plot(xs, ys, score=float(auc(xs, ys)))
