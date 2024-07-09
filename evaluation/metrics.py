from typing import Any, List, Tuple
import numpy as np
from sklearn.metrics import top_k_accuracy_score
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from scipy import interpolate
from pathlib import Path
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from utils.reliability_diagrams import _reliability_diagram_combined

EvalMetricsT = Tuple[int, int, int, List[float], List[float], List[Tuple[float, float]]]


class F1:
    @staticmethod
    def __call__(
        predicted_id: np.ndarray,
        was_rejected: np.ndarray,
        g_unique_ids: np.ndarray,
        probe_unique_ids: np.ndarray,
        predicted_unc: np.ndarray = None,
        method_name: str = None,
    ) -> dict:
        is_seen = np.isin(probe_unique_ids, g_unique_ids)
        similar_gallery_class = g_unique_ids[predicted_id[is_seen]]
        dir = np.mean(
            np.logical_and(
                probe_unique_ids[is_seen] == similar_gallery_class,
                was_rejected[is_seen] == False,
            )
        )
        far = np.mean(was_rejected[~is_seen] == False)
        result_metrics = {"osr_metric:f1": (2 * dir * (1 - far)) / (dir + (1 - far))}
        return result_metrics


class F1_classic:
    @staticmethod
    def __call__(
        predicted_id: np.ndarray,
        was_rejected: np.ndarray,
        g_unique_ids: np.ndarray,
        probe_unique_ids: np.ndarray,
        predicted_unc: np.ndarray = None,
        method_name: str = None,
    ) -> dict:
        # as in Towards Open Set Recognition paper
        is_seen = np.isin(probe_unique_ids, g_unique_ids)
        similar_gallery_class = g_unique_ids[predicted_id[is_seen]]
        tp = np.sum(
            np.logical_and(
                probe_unique_ids[is_seen] == similar_gallery_class,
                was_rejected[is_seen] == False,
            )
        )
        fp = np.sum(was_rejected[~is_seen] == False)
        precision = tp / (tp + fp)
        recall = tp / np.sum(is_seen)
        result_metrics = {
            "osr_metric:f1_class": (2 * precision * recall) / (precision + recall)
        }
        return result_metrics


class FrrFarIdent:
    def __call__(
        self,
        predicted_id: np.ndarray,
        was_rejected: np.ndarray,
        g_unique_ids: np.ndarray,
        probe_unique_ids: np.ndarray,
        predicted_unc: np.ndarray = None,
        method_name: str = None,
    ) -> dict:
        self.is_seen = np.isin(probe_unique_ids, g_unique_ids)

        self.false_reject = was_rejected[self.is_seen]
        true_accept = ~self.false_reject
        similar_gallery_class = g_unique_ids[predicted_id[self.is_seen]]

        false_ident = probe_unique_ids[self.is_seen] != similar_gallery_class
        true_ident = ~false_ident

        # errors
        self.true_accept_false_ident = np.logical_and(true_accept, false_ident)
        self.false_reject_false_ident = np.logical_and(self.false_reject, false_ident)
        self.false_reject_true_ident = np.logical_and(self.false_reject, true_ident)
        false_accept = was_rejected[~self.is_seen] == False
        # no error
        self.true_reject = was_rejected[~self.is_seen]
        self.true_accept_true_ident = np.logical_and(true_accept, true_ident)

        assert probe_unique_ids.shape[0] == np.sum(
            self.true_accept_false_ident
        ) + np.sum(self.false_reject_false_ident) + np.sum(
            self.false_reject_true_ident
        ) + np.sum(
            false_accept
        ) + np.sum(
            self.true_reject
        ) + np.sum(
            self.true_accept_true_ident
        )

        result_metrics = {
            "osr_metric:true_accept_false_ident": np.sum(self.true_accept_false_ident),
            "osr_metric:false_reject_false_ident": np.sum(
                self.false_reject_false_ident
            ),
            "osr_metric:false_reject_true_ident": np.sum(self.false_reject_true_ident),
            "osr_metric:false_ident": np.sum(false_ident),
            "osr_metric:false_reject": np.sum(self.false_reject_false_ident)
            + np.sum(self.false_reject_true_ident),
            "osr_metric:false_accept": np.sum(false_accept),
            "osr_metric:error_sum": np.sum(self.true_accept_false_ident)
            + np.sum(self.false_reject_false_ident)
            + np.sum(self.false_reject_true_ident)
            + np.sum(false_accept),
        }
        return result_metrics


class CalibrationPlot:
    def __init__(
        self,
        plot_save_dir: str,
        num_bins: int,
        draw_ece,
        draw_bin_importance,
        draw_averages,
        figsize,
        dpi=300,
    ) -> None:
        self.plot_save_dir = plot_save_dir
        self.num_bins = num_bins
        self.draw_ece = draw_ece
        self.draw_bin_importance = draw_bin_importance
        self.draw_averages = draw_averages
        self.figsize = figsize
        self.dpi = dpi

    def __call__(
        self,
        predicted_id: np.ndarray,
        was_rejected: np.ndarray,
        g_unique_ids: np.ndarray,
        probe_unique_ids: np.ndarray,
        predicted_unc: np.ndarray = None,
        method_name: str = None,
    ) -> dict:
        error_calc = FrrFarIdent()
        error_calc(predicted_id, was_rejected, g_unique_ids, probe_unique_ids)
        true_pred_label = np.zeros(probe_unique_ids.shape[0], dtype=bool)
        true_pred_label[error_calc.is_seen] = error_calc.true_accept_true_ident
        true_pred_label[~error_calc.is_seen] = error_calc.true_reject

        predicted_conf = -predicted_unc
        assert self.num_bins > 0

        bins = np.linspace(0.0, 1.0, self.num_bins + 1)
        indices = np.digitize(predicted_conf, bins, right=True)

        bin_accuracies = np.zeros(self.num_bins, dtype=np.float)
        bin_confidences = np.zeros(self.num_bins, dtype=np.float)
        bin_counts = np.zeros(self.num_bins, dtype=np.int)

        for b in range(self.num_bins):
            selected = np.where(indices == b + 1)[0]
            if len(selected) > 0:
                bin_accuracies[b] = np.mean(true_pred_label[selected])
                bin_confidences[b] = np.mean(predicted_conf[selected])
                bin_counts[b] = len(selected)

            avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
            avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

            gaps = np.abs(bin_accuracies - bin_confidences)
            ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
            mce = np.max(gaps)

        bin_data = {
            "accuracies": bin_accuracies,
            "confidences": bin_confidences,
            "counts": bin_counts,
            "bins": bins,
            "avg_accuracy": avg_acc,
            "avg_confidence": avg_conf,
            "expected_calibration_error": ece,
            "max_calibration_error": mce,
        }
        title = f"{method_name} calibration"
        fig = _reliability_diagram_combined(
            bin_data,
            self.draw_ece,
            self.draw_bin_importance,
            self.draw_averages,
            title,
            figsize=self.figsize,
            dpi=self.dpi,
            return_fig=True,
        )
        save_dir = Path(self.plot_save_dir)
        save_dir.mkdir(exist_ok=True)
        out_file = save_dir / f"{method_name}_calibration.png"
        fig.savefig(out_file, dpi=300)
        plt.close(fig)
        return {}


class ErrorDistribution:
    def __init__(self, plot_save_dir: str, value_types: List) -> None:
        self.plot_save_dir = plot_save_dir
        self.value_types = value_types

    def __call__(
        self,
        predicted_id: np.ndarray,
        was_rejected: np.ndarray,
        g_unique_ids: np.ndarray,
        probe_unique_ids: np.ndarray,
        predicted_unc: np.ndarray = None,
        method_name: str = None,
    ) -> dict:
        is_seen = np.isin(probe_unique_ids, g_unique_ids)

        false_reject = was_rejected[is_seen]
        true_accept = ~false_reject
        similar_gallery_class = g_unique_ids[predicted_id[is_seen]]

        false_ident = probe_unique_ids[is_seen] != similar_gallery_class
        true_ident = ~false_ident

        # errors
        true_accept_false_ident = np.logical_and(true_accept, false_ident)
        false_reject_false_ident = np.logical_and(false_reject, false_ident)
        false_reject_true_ident = np.logical_and(false_reject, true_ident)
        false_accept = was_rejected[~is_seen] == False
        # no error
        true_reject = was_rejected[~is_seen]
        true_accept_true_ident = np.logical_and(true_accept, true_ident)

        unc = np.array(
            [
                predicted_unc[~is_seen][false_accept],
                predicted_unc[is_seen][false_reject],
                predicted_unc[is_seen][true_accept_false_ident],
                predicted_unc[is_seen][false_reject_true_ident],
                predicted_unc[is_seen][true_accept_true_ident],
                predicted_unc[~is_seen][true_reject],
            ],
            dtype=object,
        )
        unc = np.concatenate(unc[self.value_types])
        log_scale = False
        if "MC" in method_name:
            unc += 1.0000000001
            log_scale = True
        elif "baseline" in method_name:
            # unc += 1.0000000001
            # unc = np.exp(np.exp(unc))
            log_scale = False
        # unc = np.log(unc)
        error_kind = np.array(
            [
                ["false accept"] * len(predicted_unc[~is_seen][false_accept]),
                ["false reject"] * len(predicted_unc[is_seen][false_reject]),
                ["true accept false ident"]
                * len(predicted_unc[is_seen][true_accept_false_ident]),
                ["false reject true ident"]
                * len(predicted_unc[is_seen][false_reject_true_ident]),
                ["true a&i"] * len(predicted_unc[is_seen][true_accept_true_ident]),
                ["true reject"] * len(predicted_unc[~is_seen][true_reject]),
            ],
            dtype=object,
        )
        error_kind = np.concatenate(error_kind[self.value_types])
        out_name = Path(self.plot_save_dir) / (method_name + ".png")
        data = pd.DataFrame({"unc": list(unc), "Error Kind": error_kind})
        sns.displot(
            data,
            kind="kde",
            x="unc",
            hue="Error Kind",
            log_scale=log_scale,
            common_norm=False,
        )
        plt.xlabel(f"{method_name} score")
        plt.savefig(out_name, dpi=300)
        return {}


class DirFar:
    @staticmethod
    def __call__(
        predicted_id: np.ndarray,
        was_rejected: np.ndarray,
        g_unique_ids: np.ndarray,
        probe_unique_ids: np.ndarray,
        predicted_unc: np.ndarray = None,
        method_name: str = None,
    ) -> dict:
        is_seen = np.isin(probe_unique_ids, g_unique_ids)
        similar_gallery_class = g_unique_ids[predicted_id[is_seen]]
        dir = np.mean(
            np.logical_and(
                probe_unique_ids[is_seen] == similar_gallery_class,
                was_rejected[is_seen] == False,
            )
        )
        far = np.mean(was_rejected[~is_seen] == False)
        result_metrics = {
            "osr_metric:dir": dir,
            "osr_metric:far": far,
        }
        return result_metrics


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
            new_metrics[f"final_cmc_at_rank_{n}"] = cmc[
                np.array(self.top_n_ranks) == n
            ][0][0]
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
        self,
        top_n_ranks: List[int],
        far_range: List[int],
        display_fars: List[float] = None,
    ) -> None:
        self.top_n_ranks = top_n_ranks
        self.fars = far_range
        # self.fars = [
        #     10**ii for ii in np.arange(far_range[0], far_range[1], 4.0 / far_range[2])
        # ] + [1]
        self.display_fars = far_range  # display_fars

    def __call__(
        self,
        probe_ids: np.ndarray,
        gallery_ids: np.ndarray,
        similarity: np.ndarray,
        probe_score: np.ndarray,
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

        # sort labels
        similarity = similarity[:, gallery_ids_argsort]

        is_seen = np.isin(probe_ids, gallery_ids)

        seen_sim: np.ndarray = similarity[is_seen]
        most_similar_classes = np.argsort(seen_sim, axis=1)[:, ::-1]
        seen_probe_ids = probe_ids[is_seen]

        pos_score = probe_score[is_seen]
        neg_score = probe_score[~is_seen]
        neg_score_sorted = np.sort(neg_score)[::-1]

        recalls = {}
        false_rejection_count = {}
        false_identification_count = {}
        for rank in self.top_n_ranks:
            n_similar_classes = gallery_ids[most_similar_classes[:, :rank]]
            correct_pos = np.any(
                seen_probe_ids[:, np.newaxis] == n_similar_classes, axis=1
            )
            recall_values = []
            false_rejection_values = []
            for far in self.fars:
                # compute operating threshold τ, which gives neaded far
                if len(neg_score_sorted) == 0:
                    thresh = -np.inf
                else:
                    thresh = neg_score_sorted[
                        max(int((neg_score_sorted.shape[0]) * far) - 1, 0)
                    ]
                print(f"FAR-{far}_thresh:{thresh}")
                # compute DI rate at given operating threshold τ
                recall = (
                    np.sum(np.logical_and(correct_pos, pos_score > thresh))
                    / seen_probe_ids.shape[0]
                )
                false_rejection_values.append(np.sum(pos_score < thresh))

                recall_values.append(recall)
            recall_values = np.array(recall_values)
            false_rejection_values = np.array(false_rejection_values)
            false_rejection_count[f"error_count:false-rejection-count_{rank}_rank"] = (
                false_rejection_values
            )
            false_identification_count[f"error_count:false-ident-count_{rank}_rank"] = (
                seen_probe_ids.shape[0] - np.sum(correct_pos)
            )
            recall_name = f"metric:recalls_{rank}_rank"
            recalls[recall_name] = recall_values
        metrics = {}
        metrics.update({"fars": self.fars})
        metrics.update(recalls)
        metrics.update(false_rejection_count)
        metrics.update(false_identification_count)
        # compute metrics
        new_metrics = {}
        for key, value in metrics.items():
            if "metric:recalls" in key:
                # compute auc
                rank = key.split("_")[-2]
                new_metrics[f"metric:AUC_{rank}_rank"] = auc(
                    metrics["fars"], metrics[key]
                )

                # compute fars
                # interpolate tar@far curve
                f = interpolate.interp1d(metrics["fars"], metrics[key])
                for far in self.display_fars:
                    recall = f([far])[0]
                    new_metrics[f"metric:recall-at-far_{far}_{rank}_rank"] = recall
            elif "error_count:false-rejection-count" in key:
                # interpolate tar@far curve
                rank = key.split("_")[-2]
                f = interpolate.interp1d(metrics["fars"], metrics[key])
                for far in self.display_fars:
                    rejection_count = f([far])[0]
                    new_metrics[
                        f"error_count::false-rejection-count-at-far_{far}_{rank}_rank"
                    ] = rejection_count
        metrics.update(new_metrics)
        return metrics
