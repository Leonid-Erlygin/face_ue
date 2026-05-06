import numpy as np
from .base_method import OpenSetMethod
from evaluation.open_set_methods.posterior_prob_based import PosteriorProb
import scipy
from scipy import interpolate
from evaluation.open_set_methods.posterior_prob_based import (
    prepare_calibration_dataset,
    PosteriorProbability,
)
from evaluation.metrics import FrrFarIdent
from evaluation.distance_functions.open_set_identification import CosineSim
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.special import softmax
from typing import Tuple


class SimilarityBasedPrediction(OpenSetMethod):
    def __init__(
        self,
        distance_function,
        acceptance_score,
        uncertainty_function,
        alpha: float,
        T: float = None,
        T_data_unc: float = None,
        far: float = None,
        calibration_set: bool = None,
        calibration_embs_name=None,
        calib_strategy="norm_val",
        oracle_predictions: bool = False,
        predictor=None,
    ) -> None:
        super().__init__()
        self.distance_function = distance_function
        self.far = far
        self.acceptance_score = acceptance_score
        self.uncertainty_function = uncertainty_function
        self.alpha = alpha
        self.T = T
        self.T_data_unc = T_data_unc
        self.calibration_set = calibration_set
        self.calibration_embs_name = calibration_embs_name
        self.oracle_predictions = oracle_predictions
        self.predictor = predictor
        if self.calibration_set is None:
            return
        self.calib_strategy = calib_strategy
        assert self.calib_strategy in ["norm_val", "norm_test"]

    def setup(
        self,
        probe_feats: np.ndarray,
        probe_unc: np.ndarray,
        gallery_feats: np.ndarray,
        gallery_unc: np.ndarray,
        g_unique_ids: np.ndarray,
        probe_unique_ids: np.ndarray,
        dataset_name: str,
    ):
        if self.far is None:
            raise ValueError
        self.g_unique_ids = g_unique_ids
        self.probe_unique_ids = probe_unique_ids

        probe_feats = probe_feats[:, np.newaxis, :]
        self.data_uncertainty = probe_unc

        similarity_matrix = self.distance_function(
            probe_feats,
            probe_unc,
            gallery_feats,
            gallery_unc,
        )
        self.similarity_matrix = np.mean(similarity_matrix, axis=1)
        self.probe_score = self.acceptance_score(self.similarity_matrix)

        is_seen = np.isin(probe_unique_ids, g_unique_ids)
        out_of_gallery_scores = self.probe_score[~is_seen]
        self.tau = np.sort(out_of_gallery_scores)[
            int(out_of_gallery_scores.shape[0] * (1 - self.far))
        ]
        if self.predictor is not None:
            assert self.predictor == "AccScore"
            # use cosine sim on pooled embeddings to perform predictions
            # 1. get default pool features
            gallery_feats_avg = gallery_feats[-6:, 1:]
            gallery_unc_avg = gallery_feats[-6:, 0]
            cosine_distance = CosineSim()
            similarity_matrix_avg = cosine_distance(
                probe_feats,
                probe_unc,
                gallery_feats_avg,
                gallery_unc_avg,
            )
            self.similarity_matrix_avg = np.mean(similarity_matrix_avg, axis=1)
            self.probe_score_avg = self.acceptance_score(self.similarity_matrix_avg)
            out_of_gallery_scores = self.probe_score_avg[~is_seen]
            self.tau_avg = np.sort(out_of_gallery_scores)[
                int(out_of_gallery_scores.shape[0] * (1 - self.far))
            ]

        if self.calibration_set is not None:
            self.gallery_pooled_templates_calib, self.probe_pooled_templates_calib = (
                prepare_calibration_dataset(
                    self.calibration_set, self.calibration_embs_name
                )
            )
            probe_feats_calib = self.probe_pooled_templates_calib["g1"][
                "template_pooled_features"
            ][:, np.newaxis, :]
            # probe_templates_feature,
            probe_unc_calib = self.probe_pooled_templates_calib["g1"][
                "template_pooled_data_unc"
            ]
            gallery_feats_calib = self.gallery_pooled_templates_calib["g1"][
                "template_pooled_features"
            ]
            gallery_unc_calib = self.gallery_pooled_templates_calib["g1"][
                "template_pooled_data_unc"
            ]

            self.similarity_matrix_calib = np.mean(
                self.distance_function(
                    probe_feats_calib,
                    probe_unc_calib,
                    gallery_feats_calib,
                    gallery_unc_calib,
                ),
                axis=1,
            )
            self.probe_score_calib = self.acceptance_score(self.similarity_matrix_calib)
            predicted_ids_calib = np.argmax(self.similarity_matrix_calib, axis=-1)

            # Get ground truth IDs for calibration probes
            true_ids_calib = self.probe_pooled_templates_calib["g1"][
                "template_subject_ids_sorted"
            ]

            # Calibrate T for MSP
            if hasattr(self.uncertainty_function, "T"):  # Is MSP or similar
                g_unique_ids = self.gallery_pooled_templates_calib["g1"][
                    "template_subject_ids_sorted"
                ]
                true_ids_calib = self.probe_pooled_templates_calib["g1"][
                    "template_subject_ids_sorted"
                ]

                # 1. Compute tau specifically for the calibration set to avoid test-set leakage
                is_seen_calib = np.isin(true_ids_calib, g_unique_ids)
                ood_scores_calib = self.probe_score_calib[~is_seen_calib]
                if len(ood_scores_calib) == 0:
                    raise ValueError(
                        "Calibration set contains no unseen probes for tau computation."
                    )
                tau_calib = np.sort(ood_scores_calib)[
                    int(ood_scores_calib.shape[0] * (1 - self.far))
                ]

                # 2. Run calibration with the correctly computed tau_calib
                T_opt, info = self.calibrate_msp_temperature(
                    similarity_calib=self.similarity_matrix_calib,
                    probe_score_calib=self.probe_score_calib,
                    tau_calib=tau_calib,
                    true_ids_calib=true_ids_calib,
                    g_unique_ids=g_unique_ids,
                    T_min=0.01,
                    T_max=50.0,
                    max_iter=100,
                    metric="auc",
                )
                self.uncertainty_function.T = T_opt
                print(
                    f"Calibrated MSP T: {T_opt:.3f} (AUC: {info['final_metric']:.4f})"
                )

    def predict(self):
        if self.predictor is not None:
            predict_id = np.argmax(self.similarity_matrix_avg, axis=-1)
            return predict_id, self.probe_score_avg < self.tau_avg
        predict_id = np.argmax(self.similarity_matrix, axis=-1)
        return predict_id, self.probe_score < self.tau

    def predict_uncertainty(self):
        if self.data_uncertainty.shape[1] == 1:
            # here self.data_uncertainty is scf concetration
            self.data_conf = self.data_uncertainty[:, 0]
        else:
            # pfe
            self.data_conf = 1 / scipy.stats.hmean(self.data_uncertainty, axis=1)
        if self.oracle_predictions:
            # compute true pred labels
            error_calc = FrrFarIdent()
            predicted_id = np.argmax(self.similarity_matrix, axis=-1)
            was_rejected = self.probe_score < self.tau
            error_calc(
                predicted_id, was_rejected, self.g_unique_ids, self.probe_unique_ids
            )
            true_pred_label = np.zeros(self.probe_unique_ids.shape[0], dtype=bool)
            true_pred_label[error_calc.is_seen] = error_calc.true_accept_true_ident
            true_pred_label[~error_calc.is_seen] = error_calc.true_reject
            unc = np.zeros(self.probe_unique_ids.shape[0])
            # false predictions with random priority
            false_pred_unc = np.arange(np.sum(~true_pred_label)) + 1
            rng = np.random.default_rng(1)
            rng.shuffle(false_pred_unc)
            unc[~true_pred_label] = false_pred_unc
            return unc
        else:
            unc = self.uncertainty_function(
                self.similarity_matrix, self.probe_score, self.tau
            )
            comb_conf = (-unc) * (1 - self.alpha) + self.data_conf * self.alpha
            return -comb_conf
        # if self.calibration_set is not None:
        #     # logistic calibration for scf confidence
        #     error_calc_calib = FrrFarIdent()
        #     predicted_id = np.argmax(self.similarity_matrix_calib, axis=-1)
        #     was_rejected = self.probe_score_calib < self.tau
        #     error_calc_calib(
        #         predicted_id,
        #         was_rejected,
        #         self.gallery_pooled_templates_calib["g1"][
        #             "template_subject_ids_sorted"
        #         ],
        #         self.probe_pooled_templates_calib["g1"]["template_subject_ids_sorted"],
        #     )
        #     true_pred_label = np.zeros(
        #         self.probe_pooled_templates_calib["g1"][
        #             "template_subject_ids_sorted"
        #         ].shape[0]
        #     )
        #     if self.calib_strategy == "norm_val":
        #         pass
        #     elif self.calib_strategy == "norm_test":
        #         pass
        #     else:
        #         raise ValueError
        #         if self.calibrate_by_false_reject:
        #             true_pred_label[~error_calc_calib.is_seen] = True
        #             true_pred_label[error_calc_calib.is_seen] = (
        #                 error_calc_calib.true_accept_true_ident
        #             )
        #         else:
        #             true_pred_label[error_calc_calib.is_seen] = (
        #                 error_calc_calib.true_accept_true_ident
        #             )
        #             true_pred_label[~error_calc_calib.is_seen] = (
        #                 error_calc_calib.true_reject
        #             )
        #         data_conf_calib = self.probe_pooled_templates_calib["g1"][
        #             "template_pooled_data_unc"
        #         ][:, 0]

        #         data_conf = PosteriorProbability.calibrate_scf_unc(
        #             self.data_conf,
        #             data_conf_calib,
        #             true_pred_label,
        #             verbose=False,
        #         )

        #         # calibration for baseline scores
        #         if self.alpha == 1:
        #             conf_norm = -unc
        #         else:
        #             unc_calib = self.uncertainty_function(
        #                 self.similarity_matrix_calib, self.probe_score_calib, self.tau
        #             )
        #             if self.beta_calib:
        #                 conf_norm = PosteriorProbability.beta_calib(
        #                     -unc, -unc_calib, true_pred_label
        #                 )
        #             else:
        #                 conf_norm = PosteriorProbability.calibrate_scf_unc(
        #                     -unc,
        #                     -unc_calib,
        #                     true_pred_label,
        #                     verbose=False,
        #                     scale_factor=1,
        #                 )
        # else:
        #     data_conf = self.data_conf
        #     conf_norm = -unc
        # comb_conf = conf_norm * (1 - self.alpha) + data_conf * self.alpha
        # return -comb_conf

    def calibrate_msp_temperature(
        self,
        similarity_calib: np.ndarray,
        probe_score_calib: np.ndarray,
        tau_calib: float,
        true_ids_calib: np.ndarray,
        g_unique_ids: np.ndarray,
        T_min: float = 0.1,
        T_max: float = 100.0,
        max_iter: int = 50,
        tol: float = 1e-3,
        metric: str = "auc",
    ) -> Tuple[float, dict]:
        n_gallery = similarity_calib.shape[1]

        # 1. Map true IDs to 0..n_gallery indices. Unseen (-1) -> n_gallery (reject class)
        id_to_idx = {gid: i for i, gid in enumerate(g_unique_ids)}
        true_indices = np.array(
            [id_to_idx.get(tid, n_gallery) for tid in true_ids_calib], dtype=int
        )

        # 2. Augment similarities with tau as the (n_gallery)-th class
        sims_aug = np.column_stack(
            [similarity_calib, np.full(similarity_calib.shape[0], tau_calib)]
        )

        # 3. Compute predictions in the unified space
        pred_ids_aug = np.argmax(sims_aug, axis=-1)

        # 4. Unified error mask (covers misID + false accepts automatically)
        is_error = pred_ids_aug != true_indices

        # 5. Objective function for search
        def objective(T: float) -> float:
            if T <= 1e-8:
                return 0.0 if metric == "auc" else -np.inf
            self.uncertainty_function.T = T
            msp_unc = self.uncertainty_function(similarity_calib, None, tau_calib)

            if len(np.unique(is_error)) < 2:
                return 0.5 if metric == "auc" else -np.inf

            if metric == "auc":
                return roc_auc_score(is_error, msp_unc)
            elif metric == "ap":
                return average_precision_score(is_error, msp_unc)
            else:
                raise ValueError("Use 'auc' or 'ap' for search")

        # 6. Ternary search (robust for unimodal T-vs-AUC curve)
        left, right = T_min, T_max
        for _ in range(max_iter):
            if right - left < tol:
                break
            m1 = left + (right - left) / 3
            m2 = right - (right - left) / 3
            if objective(m1) < objective(m2):
                left = m1
            else:
                right = m2

        T_opt = (left + right) / 2
        return T_opt, {
            "optimal_T": T_opt,
            "final_metric": objective(T_opt),
            "n_errors": int(is_error.sum()),
        }
