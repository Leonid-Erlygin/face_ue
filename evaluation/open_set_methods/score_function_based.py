import numpy as np
from .base_method import OpenSetMethod
from evaluation.open_set_methods.posterior_prob_based import PosteriorProb
from scipy import interpolate
from evaluation.open_set_methods.posterior_prob_based import (
    prepare_calibration_dataset,
    PosteriorProbability,
)
from evaluation.metrics import FrrFarIdent


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
        beta_calib: bool = True,
        oracle_predictions: bool = False,
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
        self.oracle_predictions = oracle_predictions
        if self.calibration_set is None:
            return
        self.calibrate_by_false_reject = False
        self.beta_calib = beta_calib
        self.gallery_pooled_templates_calib, self.probe_pooled_templates_calib = (
            prepare_calibration_dataset(calibration_set)
        )

    def setup(
        self,
        probe_feats: np.ndarray,
        probe_unc: np.ndarray,
        gallery_feats: np.ndarray,
        gallery_unc: np.ndarray,
        g_unique_ids: np.ndarray,
        probe_unique_ids: np.ndarray,
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
        if self.calibration_set is not None:
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

    def predict(self):
        predict_id = np.argmax(self.similarity_matrix, axis=-1)
        return predict_id, self.probe_score < self.tau

    def predict_uncertainty(self):
        if self.data_uncertainty.shape[1] == 1:
            # here self.data_uncertainty is scf concetration
            self.data_uncertainty = self.data_uncertainty[:, 0]
        else:
            raise NotImplemented
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
            np.random.shuffle(false_pred_unc)
            unc[~true_pred_label] = false_pred_unc

        else:
            unc = self.uncertainty_function(
                self.similarity_matrix, self.probe_score, self.tau
            )
        if self.calibration_set is not None:
            # logistic calibration for scf confidence
            error_calc = FrrFarIdent()
            predicted_id = np.argmax(self.similarity_matrix_calib, axis=-1)
            was_rejected = self.probe_score_calib < self.tau
            error_calc(
                predicted_id,
                was_rejected,
                self.gallery_pooled_templates_calib["g1"][
                    "template_subject_ids_sorted"
                ],
                self.probe_pooled_templates_calib["g1"]["template_subject_ids_sorted"],
            )
            true_pred_label = np.zeros(
                self.probe_pooled_templates_calib["g1"][
                    "template_subject_ids_sorted"
                ].shape[0]
            )
            if self.calibrate_by_false_reject:
                true_pred_label[~error_calc.is_seen] = True
                true_pred_label[error_calc.is_seen] = error_calc.true_accept_true_ident
            else:
                true_pred_label[error_calc.is_seen] = error_calc.true_accept_true_ident
                true_pred_label[~error_calc.is_seen] = error_calc.true_reject
            data_uncertainty_calib = self.probe_pooled_templates_calib["g1"][
                "template_pooled_data_unc"
            ][:, 0]
            data_conf = PosteriorProbability.calibrate_scf_unc(
                self.data_uncertainty,
                data_uncertainty_calib,
                true_pred_label,
                verbose=False,
            )

            # calibration for baseline scores

            unc_calib = self.uncertainty_function(
                self.similarity_matrix_calib, self.probe_score_calib, self.tau
            )
            if self.beta_calib:
                conf_norm = PosteriorProbability.beta_calib(
                    -unc, -unc_calib, true_pred_label
                )
            else:
                conf_norm = PosteriorProbability.calibrate_scf_unc(
                    -unc,
                    -unc_calib,
                    true_pred_label,
                    verbose=True,
                    scale_factor=1,
                )
        else:
            data_conf = self.data_uncertainty
            conf_norm = -unc
        comb_conf = conf_norm * (1 - self.alpha) + data_conf * self.alpha
        return -comb_conf
