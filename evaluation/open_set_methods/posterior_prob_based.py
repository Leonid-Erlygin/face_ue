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
        kappa: float,
        beta: float,
        uncertainty_type: str,
        alpha: float,
        aggregation: str,
        class_model: str,
        T: Union[float, List[float]],
        T_data_unc: float,
        kappa_is_tau: bool,
    ) -> None:
        super().__init__()
        self.distance_function = distance_function
        self.kappa = kappa
        self.beta = beta
        self.uncertainty_type = uncertainty_type
        self.alpha = alpha
        self.aggregation = aggregation
        self.all_classes_log_prob = None
        self.class_model = class_model
        self.C = 0.5
        self.T = T
        self.T_data_unc = T_data_unc
        self.kappa_is_tau = kappa_is_tau

    def setup(
        self,
        probe_feats: np.ndarray,
        probe_unc: np.ndarray,
        gallery_feats: np.ndarray,
        gallery_unc: np.ndarray,
    ):
        probe_feats = probe_feats[:, np.newaxis, :]
        self.data_uncertainty = probe_unc

        similarity_matrix = self.distance_function(
            probe_feats,
            probe_unc,
            gallery_feats,
            gallery_unc,
        )

        self.similarity_matrix = similarity_matrix
        if self.class_model == "vMF_Power":
            raise ValueError
            self.posterior_prob_vmf = PosteriorProb(
                kappa=self.kappa,
                beta=self.beta,
                class_model="vMF",
                K=similarity_matrix.shape[-1],
                kappa_is_tau=self.kappa_is_tau,
            )
            self.posterior_prob_power = PosteriorProb(
                kappa=self.kappa,
                beta=self.beta,
                class_model="power",
                K=similarity_matrix.shape[-1],
            )
            all_classes_log_prob_vmf = (
                self.posterior_prob_vmf.compute_all_class_log_probabilities(
                    self.similarity_matrix, self.T[0]
                )
            )
            all_classes_log_prob_power = (
                self.posterior_prob_power.compute_all_class_log_probabilities(
                    self.similarity_matrix, self.T[1]
                )
            )
            self.all_classes_log_prob = self.C * np.exp(all_classes_log_prob_vmf) + (
                1 - self.C
            ) * np.exp(all_classes_log_prob_power)
        else:
            self.posterior_prob = PosteriorProb(
                kappa=self.kappa,
                beta=self.beta,
                class_model=self.class_model,
                K=similarity_matrix.shape[-1],
                kappa_is_tau=self.kappa_is_tau,
            )
            self.all_classes_log_prob = (
                self.posterior_prob.compute_all_class_log_probabilities(
                    torch.tensor(self.similarity_matrix), self.T
                )
            )
        self.all_classes_log_prob = torch.mean(self.all_classes_log_prob, dim=1).numpy()
        # assert np.all(self.all_classes_log_prob < 1e-10)

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
        min_kappa = 150
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
        kappa_is_tau: bool = False,
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
        if kappa_is_tau:
            # in this case we need numericaly to find kappa by tau
            tau = kappa

            self.kappa = fsolve(
                self.compute_f_kappa, 200, (tau, self.log_prior, self.n)
            )[0]
            print(f"Tau {np.round(tau, 2)}, kappa {np.round(self.kappa, 2)}")
            # print(f'Error {self.compute_f_kappa(self.kappa, tau, self.log_prior, self.n)}')
        else:
            self.kappa = kappa

        self.log_uniform_dencity = (
            loggamma(self.n, dtype=np.float64) - np.log(2) - self.n * np.log(np.pi)
        )

        if self.class_model == "vMF":
            self.alpha = hyp0f1(self.n, self.kappa**2 / 4, dtype=np.float64)
            self.log_iv = (
                np.log(ive(self.n - 1, self.kappa, dtype=np.float64)) + self.kappa
            )
            self.log_normalizer = (
                (self.n - 1) * np.log(self.kappa)
                - self.n * np.log(2 * np.pi)
                - self.log_iv
            )
        elif self.class_model == "power":
            log_alpha_vmF = np.log(
                hyp0f1(self.n, self.kappa**2 / 4, dtype=np.float64)
            )

            shift = np.log(1 + (self.log_prior + log_alpha_vmF) / self.kappa)
            self.kappa_zero = fsolve(
                self.compute_f_kappa_zero, 6, (shift, self.log_prior, d)
            )[0]

            log_alpha_power = (
                loggamma(d / 2)
                + loggamma(d - 1 + 2 * self.kappa_zero)
                - self.kappa_zero * np.log(2)
                - loggamma(d - 1 + self.kappa_zero)
                - loggamma(d / 2 + self.kappa_zero)
            )
            self.alpha = np.exp(log_alpha_power)
            self.log_normalizer = (
                loggamma(d - 1 + self.kappa_zero)
                + loggamma(d / 2 + self.kappa_zero)
                + (self.kappa_zero - 1) * np.log(2)
                - (d / 2) * np.log(np.pi)
                - loggamma(d - 1 + 2 * self.kappa_zero)
            )
        else:
            raise ValueError

    @staticmethod
    def compute_f_kappa(kappa, tau, log_prior, n):
        log_alpha_vmF = np.log(hyp0f1(n, kappa**2 / 4, dtype=np.float64))
        return (log_prior + log_alpha_vmF) / kappa - tau

    @staticmethod
    def compute_f_kappa_zero(kappa_zero, shift, log_prior, d):
        log_alpha_zero = (
            loggamma(d / 2)
            + loggamma(d - 1 + 2 * kappa_zero)
            - kappa_zero * np.log(2)
            - loggamma(d - 1 + kappa_zero)
            - loggamma(d / 2 + kappa_zero)
        )
        return (log_prior + log_alpha_zero) / kappa_zero - shift

    def compute_log_z_prob(self, similarities: torch.tensor, T: torch.tensor):
        p_c = ((1 - self.beta) / self.K) ** (1 / T)
        if self.class_model == "vMF":
            logit_sum = (
                torch.sum(torch.exp(similarities * self.kappa * (1 / T)), dim=-1) * p_c
            )
        elif self.class_model == "power":
            logit_sum = (
                torch.sum((1 + similarities) ** (self.kappa_zero * (1 / T)), dim=-1)
                * p_c
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
            pz_c = torch.log((1 + similarities)) * self.kappa_zero
        gallery_log_probs = (1 / T) * (
            self.log_normalizer + pz_c + np.log((1 - self.beta) / self.K)
        ) - log_z_prob[..., np.newaxis]
        return torch.cat([gallery_log_probs, uniform_log_prob[..., np.newaxis]], dim=-1)
