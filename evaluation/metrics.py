from typing import Any, List, Tuple
import numpy as np
from sklearn.metrics import top_k_accuracy_score
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from scipy import interpolate

EvalMetricsT = Tuple[int, int, int, List[float], List[float], List[Tuple[float, float]]]


class MeanDistanceReject:
    def __init__(self, metric_to_monitor: any) -> None:
        self.fractions = np.arange(0, 0.9, step=0.1)
        self.metric_to_monitor = metric_to_monitor

    def __call__(
        self,
        probe_ids: np.ndarray,
        gallery_ids: np.ndarray,
        similarity: np.ndarray,
        probe_score: np.ndarray,
    ) -> Any:
        mean_probe_score = np.mean(probe_score)
        unc_score = -np.abs(probe_score - mean_probe_score)

        unc_indexes = np.argsort(unc_score)
        aucs = {}
        for fraction in self.fractions:
            # drop worst fraction
            good_probes_idx = unc_indexes[: int((1 - fraction) * probe_ids.shape[0])]
            metric = self.metric_to_monitor(
                probe_ids=probe_ids[good_probes_idx],
                gallery_ids=gallery_ids,
                similarity=similarity[good_probes_idx],
                probe_score=probe_score[good_probes_idx],
            )
            for key, value in metric.items():
                if "recalls" in key:
                    rank = key.split("_")[1]
                    aucs[f"auc_{rank}_rank_mean_dist_unc"] = auc(
                        metric["fars"], metric[key]
                    )

        unc_metric = {"fractions": self.fractions}
        unc_metric.update(aucs)
        return unc_metric


class CMC:
    def __init__(self, top_n_ranks: List[int], display_ranks: List[int]) -> None:
        self.top_n_ranks = top_n_ranks
        self.display_ranks = display_ranks

    def __call__(
        self,
        probe_ids: np.ndarray,
        gallery_ids: np.ndarray,
        similarity: np.ndarray,
        probe_score: np.ndarray,
    ):
        gallery_ids_argsort = np.argsort(gallery_ids)
        gallery_ids = gallery_ids[gallery_ids_argsort]
        # if not labels_sorted:
        similarity = similarity[:, gallery_ids_argsort]

        # need to fix cmc computation
        cmc = []
        most_similar_classes = np.argsort(similarity, axis=1)[:, ::-1]
        for n in self.top_n_ranks:
            n_similar_classes = []
            for probe_similar_classes in most_similar_classes[:, :n]:
                n_similar_classes.append(gallery_ids[probe_similar_classes])
            correct_pos = []
            for pos_id, similar_classes in zip(probe_ids, n_similar_classes):
                correct_pos.append(np.isin([pos_id], similar_classes)[0])
            correct_pos = np.array(correct_pos)
            cmc.append(np.sum(correct_pos) / probe_ids.shape)
        cmc = np.array(cmc)
        metrics = {"ranks": self.top_n_ranks, "cmc": cmc}

        new_metrics = {}
        for n in self.display_ranks:
            new_metrics[f"final_cmc_at_rank_{n}"] = cmc[np.array(self.top_n_ranks) == n]
        metrics.update(new_metrics)
        return metrics


class TarFar:
    def __init__(self, far_range: List[int], display_fars: List[float]) -> None:
        self.fars = [
            10**ii for ii in np.arange(far_range[0], far_range[1], 4.0 / far_range[2])
        ] + [1]
        self.display_fars = display_fars

    def __call__(self, scores, labels):
        true_match_scores = scores[labels == 1]
        wrong_match_scores = scores[labels == 0]

        threshes, recalls = [], []
        wrong_match_scores_sorted = np.sort(wrong_match_scores)[::-1]
        for far in self.fars:
            thresh = wrong_match_scores_sorted[
                max(int((wrong_match_scores_sorted.shape[0]) * far) - 1, 0)
            ]
            recall = np.sum(true_match_scores > thresh) / true_match_scores.shape[0]
            threshes.append(thresh)
            recalls.append(recall)
        metrics = {
            "fars": self.fars,
            "recalls": np.array(recalls),
            "final_auc": auc(self.fars, np.array(recalls)),
        }
        new_metrics = {}
        f = interpolate.interp1d(metrics["fars"], metrics["recalls"])
        for far in self.display_fars:
            new_metrics[f"final_recall_at_far_{far}"] = f([far])[0]
        metrics.update(new_metrics)
        return metrics


class DetectionAndIdentificationRate:
    def __init__(
        self, top_n_ranks: List[int], far_range: List[int], display_fars: List[float]
    ) -> None:
        self.top_n_ranks = top_n_ranks
        self.fars = [
            10**ii for ii in np.arange(far_range[0], far_range[1], 4.0 / far_range[2])
        ] + [1]
        self.display_fars = display_fars

    def __call__(
        self,
        probe_ids: np.ndarray,
        gallery_ids: np.ndarray,
        similarity: np.ndarray,
        probe_score: np.ndarray,
        labels_sorted: bool = False,
    ) -> EvalMetricsT:
        """
        Computes Detection & identification rate for open set recognition
        Operating thresholds τ for rejecting imposter images are computed to match particular far in fars list
        We assume that for each test image, gallery class with highest similarity is selected as predicted class.
        See
        Handbook of Face Recognition
        https://link.springer.com/book/10.1007/978-0-85729-932-1
        page 554

        :param fars: List of false acceptance rates. Defines proportion of imposter test images,
            which gets wrongly classified as gallery image (i.e match score is above an operating threshold τ)
        :param probe_ids: List of true id's (or classes in general case) of test images
        :param gallery_ids: List of true id's (or classes in general case) of gallery images
        :param similarity: (probe_size, gallery_size) marix, which specifies closeness of all test images to each gallery class
        :param probe_score: specifies confinence that particular test image belongs to predicted class
            image's probe_score is less than operating threshold τ, then this image get rejected as imposter
        :param labels_sorted: specifies the order of labels in similarity matrix.
            If True, assumes the order is ascending, else assumes order is the same as in gallery_ids
        :return: Detection & identification (DI) rate at each FAR
        """
        gallery_ids_argsort = np.argsort(gallery_ids)
        gallery_ids = gallery_ids[gallery_ids_argsort]
        # if not labels_sorted:
        similarity = similarity[:, gallery_ids_argsort]

        is_seen = np.isin(probe_ids, gallery_ids)

        seen_sim: np.ndarray = similarity[is_seen]
        most_similar_classes = np.argsort(seen_sim, axis=1)[:, ::-1]
        seen_probe_ids = probe_ids[is_seen]

        pos_score = probe_score[is_seen]
        neg_score = probe_score[~is_seen]
        neg_score_sorted = np.sort(neg_score)[::-1]

        # pos_sims = seen_sim[pos_mask]
        # neg_sims = seen_sim[~pos_mask].reshape(*pos_sims.shape, -1)
        # pos_score = probe_score[is_seen]
        # neg_score = probe_score[~is_seen]
        # non_gallery_sims = similarity[~is_seen]
        #

        recalls = {}
        for rank in self.top_n_ranks:
            n_similar_classes = []
            for probe_similar_classes in most_similar_classes[:, :rank]:
                n_similar_classes.append(gallery_ids[probe_similar_classes])
            correct_pos = []
            for pos_id, similar_classes in zip(seen_probe_ids, n_similar_classes):
                correct_pos.append(np.isin([pos_id], similar_classes)[0])
            correct_pos = np.array(correct_pos)

            recall_values = []
            for far in self.fars:
                # compute operating threshold τ, which gives neaded far
                if len(neg_score_sorted) == 0:
                    thresh = -np.inf
                else:
                    thresh = neg_score_sorted[
                        max(int((neg_score_sorted.shape[0]) * far) - 1, 0)
                    ]

                # compute DI rate at given operating threshold τ
                recall = (
                    np.sum(np.logical_and(correct_pos, pos_score > thresh))
                    / seen_probe_ids.shape[0]
                )
                recall_values.append(recall)
            recall_values = np.array(recall_values)
            recall_name = f"top_{rank}_recalls"
            recalls[recall_name] = recall_values

        # metrics = dict(zip([f"top_{k}_count" for k in self.top_n_ranks], top_n_count))
        metrics = {}
        metrics.update({"fars": self.fars})
        metrics.update(recalls)

        # compute final metrics
        new_metrics = {}
        for key, value in metrics.items():
            if "top" in key:
                # compute auc
                rank = key.split("_")[1]
                new_metrics[f"final_AUC_{rank}_rank"] = auc(
                    metrics["fars"], metrics[key]
                )

                # compute fars
                # interpolate tar@far curve
                f = interpolate.interp1d(metrics["fars"], metrics[key])
                for far in self.display_fars:
                    new_metrics[f"final_top_{rank}_recall_at_far_{far}"] = f([far])[0]
        metrics.update(new_metrics)
        return metrics
