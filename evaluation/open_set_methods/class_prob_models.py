from typing import Any


import torch
import numpy as np
from evaluation.samplers import VonMisesFisher
from scipy.optimize import fsolve, minimize


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


class MonteCarloPredictiveProbV2:
    def __init__(
        self,
        M: int,
        gallery_prior: str,
        emb_unc_model: str,
        beta: float,
        far: float,
        gallery_kappa: float = None,
        ood_kappa: float = None,
        kappa_scale: float = 1.0,
        kappa_input_scale: float = 1.0,
        predict_T: float = 1.0,
        pred_uncertainty_type: str = "entropy",
    ) -> None:
        """
        Here we assume several out-of-gallery class centers in order to enhance false reject recognition rate

        params:
        M -- number of MC samples
        kappa_scale -- gallery unc multiplier
        gallery_prior -- model for p(z|c)
        emb_unc_model -- form of p(z|x)
        """
        self.M = M
        self.gallery_kappa = gallery_kappa
        self.ood_kappa = ood_kappa
        self.kappa_scale = kappa_scale
        self.kappa_input_scale = kappa_input_scale
        assert gallery_prior in ["power", "vMF"]
        if gallery_prior == "vMF" or emb_unc_model == "power":
            raise NotImplementedError
        assert emb_unc_model in ["vMF", "power"]
        if emb_unc_model == "vMF":
            self.sampler = VonMisesFisher(self.M)

        self.gallery_prior = gallery_prior
        self.far = far
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
        g_unique_ids: np.ndarray = None,
        probe_unique_ids: np.ndarray = None,
    ):
        probe_unc_scaled = probe_unc * self.kappa_input_scale
        dtype = np.float64
        probe_feats = probe_feats.astype(dtype)
        probe_unc = probe_unc.astype(dtype)
        gallery_feats = gallery_feats.astype(dtype)
        gallery_unc = gallery_unc.astype(dtype)
        self.oog_classes_number = probe_feats.shape[-1] * 2
        gallery_unc_scaled = np.concatenate(
            [
                np.ones_like(gallery_unc) * self.gallery_kappa,
                np.ones((self.oog_classes_number, 1)) * self.ood_kappa,
            ]
        )
        self.mean_probs = (
            self.compute_mean_probs(
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
        if self.M != 0:
            # self.mean_probs_pred = None

            self.sampler = VonMisesFisher(0)
            self.beta = 0.99
            # self.beta = 0.594
            # gallery_unc_scaled = np.ones_like(gallery_unc) * self.gallery_kappa
            self.mean_probs_pred = (
                self.compute_mean_probs(
                    probe_feats,
                    probe_unc_scaled,
                    gallery_feats,
                    gallery_unc_scaled,
                    4,
                )
                .cpu()
                .detach()
                .numpy()
            )
        else:
            self.mean_probs_pred = None

    def predict(self):
        if self.mean_probs_pred is not None:
            predict_probs = self.mean_probs_pred
        else:
            predict_probs = self.mean_probs
        predict_id = np.argmax(predict_probs[:, : -self.oog_classes_number], axis=-1)
        return predict_id, np.argmax(predict_probs, axis=-1) >= (
            predict_probs.shape[-1] - self.oog_classes_number
        )

    def predict_uncertainty(self):
        # TODO: separate epistemic and aleatoric uncertainties
        if self.pred_uncertainty_type == "entropy":
            unc = -np.sum(self.mean_probs * np.log(self.mean_probs), axis=-1)
        elif self.pred_uncertainty_type == "max_prob":
            unc = -np.max(self.mean_probs, axis=-1)
        return unc

    def compute_mean_probs(
        self,
        mean: np.array,
        kappa: np.array,
        gallery_means: torch.nn.Parameter,
        class_kappas: torch.nn.Parameter,
        T: torch.nn.Parameter,
    ) -> Any:
        if type(gallery_means) == np.ndarray:
            # inference
            cuda = torch.device("cuda:0")
            gallery_means = torch.tensor(gallery_means, device=cuda)
            class_kappas = torch.tensor(class_kappas, device=cuda)
        # add out-of-gallery kappas
        self.K = gallery_means.shape[0]
        L = self.oog_classes_number
        zs = torch.tensor(self.sampler(mean, kappa), device=gallery_means.device)
        d = torch.tensor([mean.shape[-1]], device=gallery_means.device)
        similarities = torch.matmul(zs, gallery_means.T)
        similarities = torch.cat([similarities, zs, -zs], dim=-1)
        # compute log z prob
        p_c = ((1 - self.beta) / self.K) ** (1 / T)
        p_out = (self.beta / L) ** (1 / T)
        sim_to_power = torch.pow(
            torch.add(similarities, 1, out=similarities),
            (class_kappas[..., :, 0] * (1 / T)),
            out=similarities,
        )
        kappa_g = class_kappas[0, 0]
        kappa_o = class_kappas[-1, 0]
        alpha_galley_over_alpha_out = torch.exp(
            (1 / T)
            * (
                (kappa_g - kappa_o) * np.log(2.0)
                + torch.special.gammaln(d - 1 + kappa_o)
                + torch.special.gammaln((d - 1) / 2 + kappa_g)
                - torch.special.gammaln(d - 1 + kappa_g)
                - torch.special.gammaln((d - 1) / 2 + kappa_o)
            )
        )
        prior_class_probs = torch.tensor(
            [p_c] * self.K + [p_out * alpha_galley_over_alpha_out] * L, device=cuda
        )
        class_likelihoods = torch.multiply(
            sim_to_power, prior_class_probs[..., :], out=similarities
        )
        z_prob = torch.sum(class_likelihoods, dim=-1)
        gallery_probs = torch.divide(
            class_likelihoods,
            z_prob[..., np.newaxis],
            out=similarities,
        )
        mean_gallery_probs = torch.mean(gallery_probs, axis=1)
        return mean_gallery_probs


class MonteCarloPredictiveProb:
    def __init__(
        self,
        M: int,
        gallery_prior: str,
        emb_unc_model: str,
        beta: float,
        far: float,
        gallery_kappa: float = None,
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
        self.gallery_kappa = gallery_kappa
        self.kappa_scale = kappa_scale
        self.kappa_input_scale = kappa_input_scale
        assert gallery_prior in ["power", "vMF"]
        if gallery_prior == "vMF" or emb_unc_model == "power":
            raise NotImplementedError
        assert emb_unc_model in ["vMF", "power"]
        if emb_unc_model == "vMF":
            self.sampler = VonMisesFisher(self.M)

        self.gallery_prior = gallery_prior
        self.far = far
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
        g_unique_ids: np.ndarray = None,
        probe_unique_ids: np.ndarray = None,
    ):
        probe_unc_scaled = probe_unc * self.kappa_input_scale
        dtype = np.float64
        probe_feats = probe_feats.astype(dtype)
        probe_unc = probe_unc.astype(dtype)
        gallery_feats = gallery_feats.astype(dtype)
        gallery_unc = gallery_unc.astype(dtype)
        if g_unique_ids is not None and self.gallery_kappa == None:
            # find kappa
            is_seen = np.isin(probe_unique_ids, g_unique_ids)
            self.gallery_kappa = (
                minimize(
                    self.find_kappa_by_far,
                    493.125 / 100,
                    (
                        self,
                        probe_feats,
                        probe_unc_scaled,
                        gallery_feats,
                        gallery_unc,
                        self.predict_T,
                        self.far,
                        is_seen,
                    ),
                    method="Nelder-Mead",
                )[0]
                * 100
            )

        gallery_unc_scaled = np.ones_like(gallery_unc) * self.gallery_kappa

        self.mean_probs = (
            self.compute_mean_probs(
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
        if self.M != 0:
            # self.mean_probs_pred = None

            self.sampler = VonMisesFisher(0)
            self.beta = 0.5
            # self.beta = 0.594
            gallery_unc_scaled = np.ones_like(gallery_unc) * self.gallery_kappa
            self.mean_probs_pred = (
                self.compute_mean_probs(
                    probe_feats,
                    probe_unc_scaled,
                    gallery_feats,
                    gallery_unc_scaled,
                    4,
                )
                .cpu()
                .detach()
                .numpy()
            )
        else:
            self.mean_probs_pred = None

    @staticmethod
    def find_kappa_by_far(
        kappa,
        self,
        probe_feats,
        probe_unc_scaled,
        gallery_feats,
        gallery_unc,
        predict_T,
        target_far,
        is_seen,
    ):
        kappa = kappa[0]
        gallery_unc_scaled = np.ones_like(gallery_unc) * kappa * 100
        mean_probs = (
            self.compute_mean_probs(
                probe_feats,
                probe_unc_scaled,
                gallery_feats,
                gallery_unc_scaled,
                predict_T,
            )
            .cpu()
            .detach()
            .numpy()
        )
        was_rejected = np.argmax(mean_probs, axis=-1) == (mean_probs.shape[-1] - 1)
        far = np.mean(was_rejected[~is_seen] == False)
        print(f"Found kappa {np.round(kappa * 100,4)} for far {far}")
        return np.abs(far - target_far) / target_far

    def predict(self):
        if self.mean_probs_pred is not None:
            predict_probs = self.mean_probs_pred
        else:
            predict_probs = self.mean_probs

        predict_id = np.argmax(predict_probs[:, :-1], axis=-1)
        return predict_id, np.argmax(predict_probs, axis=-1) == (
            predict_probs.shape[-1] - 1
        )

    def predict_uncertainty(self):
        if self.pred_uncertainty_type == "entropy":
            unc = -np.sum(self.mean_probs * np.log(self.mean_probs), axis=-1)
        elif self.pred_uncertainty_type == "max_prob":
            unc = -np.max(self.mean_probs, axis=-1)
        return unc

    def compute_mean_probs(
        self,
        mean: np.array,
        kappa: np.array,
        gallery_means: torch.nn.Parameter,
        gallery_kappas: torch.nn.Parameter,
        T: torch.nn.Parameter,
    ) -> Any:
        # assert (
        #     mean.dtype == torch.float32
        #     and kappa.dtype == torch.float32
        #     and gallery_means.dtype == torch.float32
        #     and gallery_kappas.dtype == torch.float32
        # )
        if type(gallery_means) == np.ndarray:
            # inference
            cuda = torch.device("cuda:0")
            gallery_means = torch.tensor(gallery_means, device=cuda)
            gallery_kappas = torch.tensor(gallery_kappas, device=cuda)
        self.K = gallery_means.shape[0]
        zs = torch.tensor(self.sampler(mean, kappa), device=gallery_means.device)
        d = torch.tensor([mean.shape[-1]], device=gallery_means.device)
        similarities = torch.matmul(zs, gallery_means.T)
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
        assert self.gallery_prior == "power"
        # compute log z prob
        p_c = ((1 - self.beta) / self.K) ** (1 / T)
        sim_to_power = torch.pow(
            torch.add(similarities, 1, out=similarities),
            (gallery_kappas[..., :, 0] * (1 / T)),
            out=similarities,
        )
        logit_sum = (
            torch.sum(
                torch.mul(
                    sim_to_power, m_c_power[..., :, 0] ** (1 / T), out=similarities
                ),
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
        similarities = torch.matmul(zs, gallery_means.T, out=similarities)
        sim_to_power = torch.pow(
            torch.add(similarities, 1, out=similarities),
            (gallery_kappas[..., :, 0] * (1 / T)),
            out=similarities,
        )
        pz_c = torch.add(
            torch.log(sim_to_power, out=similarities),
            (1 / T) * log_normalizer[..., :, 0],
            out=similarities,
        )

        gallery_log_probs = torch.sub(
            torch.add(
                pz_c, (1 / T) * np.log((1 - self.beta) / self.K), out=similarities
            ),
            log_z_prob[..., np.newaxis],
            out=similarities,
        )
        gallery_probs = torch.exp(gallery_log_probs, out=similarities)
        uniform_probs = torch.exp(uniform_log_prob, out=uniform_log_prob)
        mean_gallery_probs = torch.mean(gallery_probs, axis=1)
        mean_uniform_probs = torch.mean(uniform_probs, axis=1)

        mean_probs = torch.cat(
            [mean_gallery_probs, mean_uniform_probs[..., np.newaxis]], dim=-1
        )

        return mean_probs
