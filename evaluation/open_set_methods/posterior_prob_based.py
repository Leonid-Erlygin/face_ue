import numpy as np
from .base_method import OpenSetMethod
from scipy.special import ive, hyp0f1, loggamma
from scipy.optimize import fsolve, minimize
from typing import List, Union, Any
from torch import nn
import torch


class PosteriorProbability(OpenSetMethod):
    def __init__(
        self,
        distance_function: Any,
        far: float,
        beta: float,
        uncertainty_type: str,
        alpha: float,
        aggregation: str,
        class_model: str,
        T: Union[float, List[float]],
        T_data_unc: float,
    ) -> None:
        super().__init__()
        self.distance_function = distance_function
        self.far = far
        self.beta = beta
        self.uncertainty_type = uncertainty_type
        self.alpha = alpha
        self.aggregation = aggregation
        self.all_classes_log_prob = None
        self.class_model = class_model
        self.C = 0.5
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
        """
        g_unique_ids and probe_unique_ids are needed to find kappa that gives certan self.far
        """
        probe_feats = probe_feats[:, np.newaxis, :]
        self.data_uncertainty = probe_unc

        similarity_matrix = torch.tensor(
            self.distance_function(
                probe_feats,
                probe_unc,
                gallery_feats,
                gallery_unc,
            )
        )
        T = self.T
        # find kappa
        is_seen = np.isin(probe_unique_ids, g_unique_ids)
        found_kappa = (
            fsolve(
                self.find_kappa_by_far,
                600.0 / 100,
                (
                    self.beta,
                    T,
                    self.class_model,
                    self.far,
                    is_seen,
                    similarity_matrix,
                ),
            )[0]
            * 100
        )
        print(f"Found kappa {np.round(found_kappa,4)} for far {self.far}")
        if self.class_model == "vMF_Power":
            raise ValueError
        else:
            self.posterior_prob = PosteriorProb(
                kappa=found_kappa,
                beta=self.beta,
                class_model=self.class_model,
                K=similarity_matrix.shape[-1],
            )
            self.all_classes_log_prob = (
                self.posterior_prob.compute_all_class_log_probabilities(
                    similarity_matrix, T
                )
            )
        self.all_classes_log_prob = torch.mean(self.all_classes_log_prob, dim=1).numpy()
        # assert np.all(self.all_classes_log_prob < 1e-10)

    @staticmethod
    def find_kappa_by_far(
        kappa: float,
        beta: float,
        T: float,
        class_model: str,
        target_far: float,
        is_seen: np.ndarray,
        similarity_matrix: torch.tensor,
    ):
        posterior_prob = PosteriorProb(
            kappa=kappa[0] * 100,
            beta=beta,
            class_model=class_model,
            K=similarity_matrix.shape[-1],
        )
        all_classes_log_prob = posterior_prob.compute_all_class_log_probabilities(
            similarity_matrix, T
        )
        all_classes_log_prob = torch.mean(all_classes_log_prob, dim=1).numpy()
        was_rejected = np.argmax(all_classes_log_prob, axis=-1) == (
            all_classes_log_prob.shape[-1] - 1
        )
        far = np.mean(was_rejected[~is_seen] == False)
        print(f"Found kappa {np.round(kappa[0]*100,4)} for far {far}")
        return np.abs(far - target_far)

    def get_class_log_probs(self, similarity_matrix: np.ndarray):
        self.setup(similarity_matrix)
        return self.all_classes_log_prob

    def predict(self):
        predict_id = np.argmax(self.all_classes_log_prob[:, :-1], axis=-1)
        return predict_id, np.argmax(self.all_classes_log_prob, axis=-1) == (
            self.all_classes_log_prob.shape[-1] - 1
        )

    def predict_uncertainty(self):
        if self.uncertainty_type == "maxprob":
            unc = -np.exp(np.max(self.all_classes_log_prob, axis=-1))
        elif self.uncertainty_type == "entr":
            unc = -np.sum(
                np.exp(self.all_classes_log_prob) * self.all_classes_log_prob, axis=-1
            )
        else:
            raise ValueError
        if self.data_uncertainty.shape[1] == 1:
            # here self.data_uncertainty is scf concetration
            self.data_uncertainty = self.data_uncertainty[:, 0]
        else:
            raise NotImplemented
        if self.data_uncertainty[0] == 0:
            # default pool
            return unc
        # min_kappa = 150
        min_kappa = 10
        max_kappa = 2700
        data_uncertainty_norm = (self.data_uncertainty - min_kappa) / (
            max_kappa - min_kappa
        )
        assert np.sum(data_uncertainty_norm < 0) == 0
        # data_uncertainty_norm = data_uncertainty
        data_conf_norm = (data_uncertainty_norm) ** (1 / self.T_data_unc)

        conf_norm = -unc
        if self.aggregation == "sum":
            comb_conf = conf_norm * (1 - self.alpha) + data_conf_norm * self.alpha
        elif self.aggregation == "product":
            comb_conf = (conf_norm ** (1 - self.alpha)) * (data_conf_norm**self.alpha)
        else:
            raise ValueError
        return -comb_conf


class PosteriorProb:
    def __init__(
        self,
        kappa: float,
        beta: float,
        class_model: str,
        K: int,
        d: int = 512,
    ) -> None:
        """
        Performes K+1 class classification, with K being number of gallery classed and
        K+1-th class is ood class.
        returns probabilities p(K+1|z) = \frac{p(z|c)p(c)}{p(z)} that test emb z belongs to reject class K+1
        where
        1. p(z|K+1) is uniform disribution on sphere
        2. p(z|c), c \leq K is von Mises-Fisher (vMF) distribution with koncentration k
        3. p(K+1) = beta and p(1)= ... =p(K) = (1 - beta)/K

        :param kappa: koncentration for von Mises-Fisher (vMF) distribution
        :param beta: prior probability of ood sample, p(K+1)
        """
        self.beta = beta
        self.n = d / 2
        self.K = K
        self.class_model = class_model
        self.log_prior = np.log(self.beta / ((1 - self.beta) / self.K))

        self.kappa = kappa

        self.log_uniform_dencity = (
            loggamma(self.n, dtype=np.float64) - np.log(2) - self.n * np.log(np.pi)
        )

        if self.class_model == "vMF":
            log_iv = np.log(ive(self.n - 1, self.kappa, dtype=np.float64)) + self.kappa
            self.alpha = hyp0f1(self.n, self.kappa**2 / 4, dtype=np.float64)
            self.log_normalizer = (
                (self.n - 1) * np.log(self.kappa) - self.n * np.log(2 * np.pi) - log_iv
            )
        elif self.class_model == "power":
            log_alpha_power = (
                loggamma(d / 2)
                + loggamma(d - 1 + 2 * self.kappa)
                - self.kappa * np.log(2)
                - loggamma(d - 1 + self.kappa)
                - loggamma(d / 2 + self.kappa)
            )
            self.alpha = np.exp(log_alpha_power)
            self.log_normalizer = (
                loggamma(d - 1 + self.kappa)
                + loggamma(d / 2 + self.kappa)
                + (self.kappa - 1) * np.log(2)
                - (d / 2) * np.log(np.pi)
                - loggamma(d - 1 + 2 * self.kappa)
            )
        else:
            raise ValueError

    def compute_log_z_prob(self, similarities: torch.tensor, T: torch.tensor):
        p_c = ((1 - self.beta) / self.K) ** (1 / T)
        if self.class_model == "vMF":
            logit_sum = (
                torch.sum(torch.exp(similarities * self.kappa * (1 / T)), dim=-1) * p_c
            )
        elif self.class_model == "power":
            logit_sum = (
                torch.sum((1 + similarities) ** (self.kappa * (1 / T)), dim=-1) * p_c
            )

        log_z_prob = (1 / T) * self.log_normalizer + torch.log(
            logit_sum + (self.alpha * self.beta) ** (1 / T)
        )
        return log_z_prob

    def compute_nll(
        self,
        T: torch.nn.Parameter,
        similarities: torch.tensor,
        true_label: torch.tensor,
    ):
        if type(T) == np.ndarray:
            T = torch.tensor(T, dtype=torch.float64)
        class_probs = self.compute_all_class_log_probabilities(similarities, T)[:, 0, :]
        loss = nn.NLLLoss()
        loss_value = loss(class_probs, true_label)
        return loss_value.item()

    def compute_all_class_log_probabilities(
        self, similarities: np.ndarray, T: float = 1
    ):
        log_z_prob = self.compute_log_z_prob(similarities, T)
        log_beta = np.log(self.beta)
        uniform_log_prob = (1 / T) * (self.log_uniform_dencity + log_beta) - log_z_prob

        # compute gallery classes log prob
        if self.class_model == "vMF":
            pz_c = self.kappa * similarities
        elif self.class_model == "power":
            pz_c = torch.log((1 + similarities)) * self.kappa
        gallery_log_probs = (1 / T) * (
            self.log_normalizer + pz_c + np.log((1 - self.beta) / self.K)
        ) - log_z_prob[..., np.newaxis]
        return torch.cat([gallery_log_probs, uniform_log_prob[..., np.newaxis]], dim=-1)
