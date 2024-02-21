from typing import Any


import torch
import numpy as np
from evaluation.samplers import VonMisesFisher


class GalleryMeans(torch.nn.Module):
    def __init__(self, init_means, device):
        super(GalleryMeans, self).__init__()
        self.gallery_means = torch.nn.Parameter(
            torch.tensor(init_means, dtype=torch.float64, device=device)
        )


class GalleryParams(torch.nn.Module):
    def __init__(self, init_mean, init_kappa, init_T, train_T, device):
        super(GalleryParams, self).__init__()
        self.gallery_means = torch.nn.Parameter(
            torch.tensor(init_mean, dtype=torch.float64, device=device)
        )


class MonteCarloPredictiveProb:
    def __init__(
        self,
        M: int,
        gallery_prior: str,
        emb_unc_model: str,
        beta: float,
        kappa_scale: float = 1.0,
        kappa_input_scale: float = 1.0,
        predict_T: float = 1.0,
        pred_uncertainty_type: str = "entropy",
    ) -> None:
        """
        params:
        M -- number of MC samples
        kappa_scale -- gallery unc multiplier
        gallery_prior -- model for p(z|c)
        emb_unc_model -- form of p(z|x)
        """
        self.M = M
        self.kappa_scale = kappa_scale
        self.kappa_input_scale = kappa_input_scale
        assert gallery_prior in ["power", "vMF"]
        assert emb_unc_model in ["vMF", "PFE"]
        if emb_unc_model == "vMF":
            self.sampler = VonMisesFisher(self.M)

        self.gallery_prior = gallery_prior
        self.beta = beta
        self.predict_T = predict_T
        self.pred_uncertainty_type = pred_uncertainty_type
        assert self.pred_uncertainty_type in ["entropy", "max_prob"]

    def setup(
        self,
        probe_feats: np.ndarray,
        probe_unc: np.ndarray,
        gallery_feats: np.ndarray,
        gallery_unc: np.ndarray,
    ):
        gallery_unc_scaled = gallery_unc * self.kappa_scale
        probe_unc_scaled = probe_unc * self.kappa_input_scale
        self.all_classes_log_prob = (
            self.compute_log_prob(
                probe_feats,
                probe_unc_scaled,
                gallery_feats,
                gallery_unc_scaled,
                self.predict_T,
            )
            .cpu()
            .detach()
            .numpy()
        )

    def predict(self):
        probs = np.exp(self.all_classes_log_prob)
        # probs = self.all_classes_log_prob
        self.mean_probs = np.mean(probs, axis=1)
        predict_id = np.argmax(self.mean_probs[:, :-1], axis=-1)
        return predict_id, np.argmax(self.mean_probs, axis=-1) == (
            self.mean_probs.shape[-1] - 1
        )

    def predict_uncertainty(self):
        if self.pred_uncertainty_type == "entropy":
            unc = -np.sum(self.mean_probs * np.log(self.mean_probs), axis=-1)
        elif self.pred_uncertainty_type == "max_prob":
            unc = -np.max(self.mean_probs, axis=-1)
        return unc

    def compute_log_prob(
        self,
        mean: np.array,
        kappa: np.array,
        gallery_means: torch.nn.Parameter,
        gallery_kappas: torch.nn.Parameter,
        T: torch.nn.Parameter,
    ) -> Any:
        if type(gallery_means) == np.ndarray:
            # inference
            gallery_means = torch.tensor(gallery_means)
            gallery_kappas = torch.tensor(gallery_kappas)
        self.K = gallery_means.shape[0]
        zs = torch.tensor(self.sampler(mean, kappa), device=gallery_means.device)
        d = torch.tensor([mean.shape[-1]], device=gallery_means.device)
        similarities = zs @ gallery_means.T
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
        uniform_log_prob = (1 / T) * (log_uniform_dencity + log_beta) - log_z_prob

        # compute gallery classes log prob
        pz_c = (
            torch.log((1 + similarities)) * gallery_kappas[..., :, 0]
            + log_normalizer[..., :, 0]
        )
        gallery_log_probs = (1 / T) * (
            pz_c + np.log((1 - self.beta) / self.K)
        ) - log_z_prob[..., np.newaxis]
        log_probs = torch.cat(
            [gallery_log_probs, uniform_log_prob[..., np.newaxis]], dim=-1
        )
        return log_probs
