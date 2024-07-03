import numpy as np
from .base_method import OpenSetMethod
from evaluation.open_set_methods.posterior_prob_based import PosteriorProb
from scipy import interpolate


class SimilarityBasedPrediction(OpenSetMethod):
    def __init__(
        self,
        distance_function,
        acceptance_score,
        uncertainty_function,
        alpha: float,
        T: float,
        T_data_unc: float,
        far: float = None,
    ) -> None:
        super().__init__()
        self.distance_function = distance_function
        self.far = far
        self.acceptance_score = acceptance_score
        self.uncertainty_function = uncertainty_function
        self.alpha = alpha
        self.T = T
        self.T_data_unc = T_data_unc

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

    def predict(self):
        predict_id = np.argmax(self.similarity_matrix, axis=-1)
        return predict_id, self.probe_score < self.tau

    def predict_uncertainty(self):
        if self.data_uncertainty.shape[1] == 1:
            # here self.data_uncertainty is scf concetration
            self.data_uncertainty = self.data_uncertainty[:, 0]
        else:
            raise NotImplemented
        unc = self.uncertainty_function(
            self.similarity_matrix, self.probe_score, self.tau
        )
        # unc_norm = (unc - np.min(unc)) / (np.max(unc) - np.min(unc))

        # min_kappa = 150
        # max_kappa = 2700
        # data_uncertainty_norm = (self.data_uncertainty - min_kappa) / (
        #     max_kappa - min_kappa
        # )
        # assert np.sum(data_uncertainty_norm < 0) == 0
        # data_uncertainty_norm = data_uncertainty
        # data_conf_norm = (data_uncertainty_norm) ** (1 / self.T_data_unc)

        # conf_norm = (-unc_norm + 1) ** (1 / self.T)
        conf_norm = -unc
        comb_conf = conf_norm * (1 - self.alpha)  # + data_conf_norm * self.alpha
        return -comb_conf
