from typing import Any


import torch
import numpy as np
from evaluation.samplers import VonMisesFisher


class GalleryParams(torch.nn.Module):
    def __init__(self, init_mean, init_kappa, init_T, train_T):
        super(GalleryParams, self).__init__()
        self.gallery_means = torch.nn.Parameter(torch.tensor(init_mean))
        self.gallery_kappas = torch.nn.Parameter(torch.tensor(init_kappa))
        if train_T:
            self.T = torch.nn.Parameter(torch.tensor(init_T))
        else:
            self.T = torch.tensor(init_T)
        # self.gallery_means = torch.nn.Parameter(torch.rand(3, 2, dtype=torch.float64))
        # self.gallery_kappas = torch.nn.Parameter(
        #     torch.rand(3, 1, dtype=torch.float64) * 10
        # )


class MonteCarloPredictiveProb:
    def __init__(
        self,
        M: int,
        gallery_prior: str,
        emb_unc_model: str,
        beta: float,
    ) -> None:
        """
        params:
        M -- number of MC samples
        gallery_prior -- model for p(z|c)
        emb_unc_model -- form of p(z|x)
        """
        self.M = M
        assert gallery_prior in ["power", "vMF"]
        assert emb_unc_model in ["vMF", "PFE"]
        if emb_unc_model == "vMF":
            self.sampler = VonMisesFisher(self.M)
        self.gallery_prior = gallery_prior
        self.beta = beta

    def __call__(
        self,
        mean: np.array,
        kappa: np.array,
        gallery_means: torch.nn.Parameter,
        gallery_kappas: torch.nn.Parameter,
        T: torch.nn.Parameter,
    ) -> Any:
        self.K = gallery_means.shape[0]
        # print(self.K)
        zs = torch.tensor(self.sampler(mean, kappa))
        d = torch.tensor([mean.shape[-1]])
        # print(zs.shape, gallery_means.shape)
        # print(zs, gallery_means)
        similarities = zs @ gallery_means.T
        # print(similarities.shape)
        # print(similarities)
        if self.gallery_prior == "power":
            log_m_c_power = (
                torch.special.gammaln(d - 1 + gallery_kappas)
                + torch.special.gammaln(d / 2 + gallery_kappas)
                + gallery_kappas * np.log(2)
                - torch.special.gammaln(d / 2)
                - torch.special.gammaln(d - 1 + 2 * gallery_kappas)
            )
            m_c_power = torch.exp(log_m_c_power)
            log_uniform_dencity = (
                torch.special.gammaln(d / 2) - np.log(2) - (d / 2) * np.log(np.pi)
            )
            log_normalizer = log_m_c_power + log_uniform_dencity
        # compute log z prob
        p_c = ((1 - self.beta) / self.K) ** (1 / T)
        # print(similarities.shape, gallery_kappas.shape, log_uniform.shape, m_c_power.shape)
        logit_sum = (
            torch.sum(
                (m_c_power[..., :, 0] ** (1 / T))
                * ((1 + similarities) ** (gallery_kappas[..., :, 0] * (1 / T))),
                dim=-1,
            )
            * p_c
        )
        log_z_prob = (1 / T) * log_uniform_dencity + torch.log(
            logit_sum + (self.beta) ** (1 / T)
        )

        log_beta = np.log(self.beta)
        # print(similarities.shape, gallery_kappas.shape)
        uniform_log_prob = (1 / T) * (log_uniform_dencity + log_beta) - log_z_prob

        # compute gallery classes log prob
        pz_c = (
            torch.log((1 + similarities)) * gallery_kappas[..., :, 0]
            + log_normalizer[..., :, 0]
        )
        # print(pz_c.shape, log_z_prob.shape)
        gallery_log_probs = (1 / T) * (
            pz_c + np.log((1 - self.beta) / self.K)
        ) - log_z_prob[..., np.newaxis]
        # print(uniform_log_prob.shape)
        log_probs = torch.cat(
            [gallery_log_probs, uniform_log_prob[..., np.newaxis]], dim=-1
        )
        # print(log_probs.shape)
        # print(torch.sum(log_probs, dim=-1))
        return log_probs
