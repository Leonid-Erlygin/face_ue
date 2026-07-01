from typing import Any


import torch

import numpy as np
from evaluation.samplers import VonMisesFisher
from scipy.optimize import fsolve, minimize
from scipy.special import ive, hyp0f1, loggamma
from evaluation.metrics import FrrFarIdent
from utils.golden_section import golden_selection_search
from evaluation.open_set_methods.score_function_based import SimilarityBasedPrediction
from evaluation.distance_functions.open_set_identification import CosineSim
from evaluation.confidence_functions import MaxSimilarity_confidence
from evaluation.open_set_methods.uncertainty_functions import BernoulliVariance
from evaluation.open_set_methods.calibration_utils import prepare_calibration_dataset
from pathlib import Path
import torch.nn.functional as F
from evaluation.open_set_methods.kappa_utils import (
    threshold_at_far,
    log_uniform_density,
    vmf_log_normalizer_np,
)

#     solve_kappa_for_tau,
#     ,
#     ,
# )


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


class FarLossCalc:
    def __init__(
        self,
        probe_feats,
        probe_unc_scaled,
        gallery_feats,
        gallery_unc,
        predict_T,
        target_far,
        is_seen,
        env,
        verbose=False,
    ) -> None:
        self.probe_feats = probe_feats
        self.probe_unc_scaled = probe_unc_scaled
        self.gallery_feats = gallery_feats
        self.gallery_unc = gallery_unc
        self.predict_T = predict_T
        self.target_far = target_far
        self.is_seen = is_seen
        self.env = env
        self.verbose = verbose

    def __call__(self, kappa: float) -> float:
        gallery_unc_scaled = np.ones_like(self.gallery_unc) * kappa
        out = self.env.compute_mean_probs_and_kl(
            self.probe_feats,
            self.probe_unc_scaled,
            self.gallery_feats,
            gallery_unc_scaled,
            self.predict_T,
        )
        mean_probs, kl_1, kl_2 = [x.cpu().detach().numpy() for x in out]

        oog_prob = 1 - np.sum(mean_probs, axis=-1, keepdims=True)
        all_prob = np.concatenate([mean_probs, oog_prob], axis=-1)
        was_rejected = np.argmax(all_prob, axis=-1) == (all_prob.shape[-1] - 1)
        far = np.mean(was_rejected[~self.is_seen] == False)
        if self.verbose:
            print(f"Found kappa {np.round(kappa,4)} for far {far}")
        return -np.abs(far - self.target_far) / self.target_far


class MonteCarloPredictiveProb:
    def __init__(
        self,
        gallery_prior: str,
        emb_unc_model: str,
        beta: float,
        far: float,
        M: int = 0,
        calibration_set=None,
        calibration_embs_name=None,
        calibration_transform=None,
        gallery_kappa: float = None,
        kappa_scale: float = 1.0,
        kappa_input_scale: float = 1.0,
        predict_T: float = 1.0,
        train_predict_T: bool = False,
        predict_T_lr: float = 1e-2,
        pred_uncertainty_type: str = "entropy",
        alpha: float = 0.5,
        log_dir: str = None,
        predictor=None,
    ) -> None:
        """
        params:
        M -- number of MC samples
        kappa_scale -- gallery unc multiplier
        gallery_prior -- model for p(z|c)
        emb_unc_model -- form of p(z|x)
        """
        self.M = M
        if not (0.0 < beta < 1.0):
            raise ValueError(f"beta must be in (0, 1), got {beta}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gallery_kappa = gallery_kappa
        self.kappa_scale = kappa_scale
        self.kappa_input_scale = kappa_input_scale
        self.predictor = predictor
        self.train_predict_T = train_predict_T
        self.predict_T_lr = predict_T_lr

        if self.train_predict_T:
            # Learnable scalar. float64 matches the rest of the function.
            self.predict_T = torch.nn.Parameter(
                torch.tensor(float(predict_T), dtype=torch.float64, device=self.device),
                requires_grad=True,
            )
        else:
            self.predict_T = predict_T
        if self.predictor is not None:
            assert predictor == "AccScore"
            self.predictor = SimilarityBasedPrediction(
                CosineSim(),
                MaxSimilarity_confidence(),
                BernoulliVariance(),
                alpha=0,
                calib_strategy="norm_val",
            )
        assert gallery_prior in ["power", "vMF"]
        if emb_unc_model == "power":
            raise NotImplementedError
        assert emb_unc_model in ["vMF", "power"]
        self.emb_unc_model = emb_unc_model
        if self.emb_unc_model == "vMF":
            self.sampler = VonMisesFisher(self.M)
        else:
            raise ValueError

        self.gallery_prior = gallery_prior
        self.far = far
        self.beta = beta
        self.pred_uncertainty_type = pred_uncertainty_type
        assert self.pred_uncertainty_type in ["entropy", "max_prob"]
        self.alpha = alpha
        self.log_dir = log_dir
        self.calibration_set = calibration_set
        self.calibration_embs_name = calibration_embs_name
        self.calibration_transform = calibration_transform
        self.kappa_high = 1000000
        self.kappa_low = 10
        self.eps = 1e-3
        self.max_iter = 100

    def setup(
        self,
        probe_feats: np.ndarray,
        probe_unc: np.ndarray,
        gallery_feats: np.ndarray,
        gallery_unc: np.ndarray,
        dataset_name: str,
        g_unique_ids: np.ndarray = None,
        probe_unique_ids: np.ndarray = None,
    ):
        if self.predictor is not None:
            self.predictor.far = self.far
            self.predictor.setup(
                probe_feats,
                probe_unc,
                gallery_feats,
                gallery_unc,
                g_unique_ids,
                probe_unique_ids,
                dataset_name,
            )
        dtype = np.float64
        probe_feats = probe_feats.astype(dtype)
        probe_unc = probe_unc.astype(dtype)
        probe_unc_scaled = probe_unc * self.kappa_input_scale
        gallery_feats = gallery_feats.astype(dtype)
        gallery_unc = gallery_unc.astype(dtype)
        self.g_unique_ids = g_unique_ids
        self.probe_unique_ids = probe_unique_ids
        self.dataset_name = dataset_name
        if g_unique_ids is not None and self.gallery_kappa == None:
            is_seen = np.isin(probe_unique_ids, g_unique_ids)
            max_scores = np.max(probe_feats @ gallery_feats.T, axis=1)
            tau = threshold_at_far(max_scores[~is_seen], self.far)

            far_loss_func = FarLossCalc(
                probe_feats,
                probe_unc_scaled,
                gallery_feats,
                gallery_unc,
                self.predict_T,
                self.far,
                is_seen,
                self,
                verbose=True,
            )
            self.gallery_kappa = golden_selection_search(
                self.kappa_high, self.kappa_low, self.eps, self.max_iter, far_loss_func
            )
            print(f"Found kappa {np.round(self.gallery_kappa,4)} for far {self.far}")

        gallery_unc_scaled = np.ones_like(gallery_unc) * self.gallery_kappa

        out = self.compute_mean_probs_and_kl(
            probe_feats,
            probe_unc_scaled,
            gallery_feats,
            gallery_unc_scaled,
            self.predict_T,
        )
        self.mean_probs, self.kl_1, self.kl_2 = [x.cpu().detach().numpy() for x in out]

        # get calibration set kl
        if self.calibration_set is not None:
            self.gallery_pooled_templates_calib, self.probe_pooled_templates_calib = (
                prepare_calibration_dataset(
                    self.calibration_set, self.calibration_embs_name
                )
            )
            self.calibration_transform = self.calibration_transform
            self.data_uncertainty_calib = self.probe_pooled_templates_calib["g1"][
                "template_pooled_data_unc"
            ]
            self.g_unique_ids_calib = self.gallery_pooled_templates_calib["g1"][
                "template_subject_ids_sorted"
            ]
            self.probe_unique_ids_calib = self.probe_pooled_templates_calib["g1"][
                "template_subject_ids_sorted"
            ]

            is_seen_calib = np.isin(
                self.probe_unique_ids_calib, self.g_unique_ids_calib
            )
            probe_feats_calib = self.probe_pooled_templates_calib["g1"][
                "template_pooled_features"
            ]
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

            probe_unc_calib_scaled = probe_unc_calib * self.kappa_input_scale

            far_loss_func_calib = FarLossCalc(
                probe_feats_calib,
                probe_unc_calib,
                gallery_feats_calib,
                gallery_unc_calib,
                self.predict_T,
                self.far,
                is_seen_calib,
                self,
                verbose=True,
            )

            calibratation_set_kappa = golden_selection_search(
                self.kappa_high,
                self.kappa_low,
                self.eps,
                self.max_iter,
                far_loss_func_calib,
                verbose=False,
            )

            gallery_unc_scaled_calib = (
                np.ones_like(gallery_unc_calib) * calibratation_set_kappa
            )
            probe_unc_calib_scaled = probe_unc_calib * self.kappa_input_scale
            out_calib = self.compute_mean_probs_and_kl(
                probe_feats_calib,
                probe_unc_calib_scaled,
                gallery_feats_calib,
                gallery_unc_scaled_calib,
                self.predict_T,
            )
            self.mean_probs_calib, self.kl_1_calib, self.kl_2_calib = [
                x.cpu().detach().numpy() for x in out_calib
            ]
            # calibrate

            # Buildground-truth labels for calibration loss (unchanged logic)
            predict_id_calib = np.argmax(self.mean_probs_calib, axis=-1)
            oog_prob = 1 - np.sum(self.mean_probs_calib, axis=-1, keepdims=True)
            all_prob = np.concatenate([self.mean_probs_calib, oog_prob], axis=-1)
            was_rejected_calib = np.argmax(all_prob, axis=-1) == (
                all_prob.shape[-1] - 1
            )
            error_calc = FrrFarIdent()
            error_calc(
                predict_id_calib,
                was_rejected_calib,
                self.g_unique_ids_calib,
                self.probe_unique_ids_calib,
            )

            if not self.train_predict_T:
                # --------- UNCHANGED original path ---------
                self.calibration_transform.train_calibration_parameters(
                    self.kl_1_calib,
                    self.kl_2_calib,
                    error_calc,
                    dataset_name=self.calibration_set.dataset_name,
                    far=self.far,
                )
            else:
                # --------- NEW: joint T + MLP path ---------
                # Cache tensors used by the closure (no NumPy round-trip inside the loop).
                pf = probe_feats_calib
                pu = probe_unc_calib_scaled
                gf = gallery_feats_calib
                gu = gallery_unc_scaled_calib

                def kl_forward_fn():
                    """Re-run the forward with the current value of self.predict_T.
                    Returns (kl_1, kl_2) as torch tensors carrying gradient w.r.t. T."""
                    _, kl1, kl2 = self.compute_mean_probs_and_kl(
                        pf, pu, gf, gu, self.predict_T
                    )
                    return kl1, kl2

                self.calibration_transform.train_calibration_parameters_joint(
                    kl_forward_fn=kl_forward_fn,
                    extra_params=[self.predict_T],
                    extra_params_lr=self.predict_T_lr,
                    error_calc=error_calc,
                    dataset_name=self.calibration_set.dataset_name,
                    far=self.far,
                )

                # After joint training, freeze T and refresh the stored KLs for test set
                # using the optimized T (so predict_uncertainty() is consistent).
                with torch.no_grad():
                    out = self.compute_mean_probs_and_kl(
                        probe_feats,
                        probe_unc_scaled,
                        gallery_feats,
                        gallery_unc_scaled,
                        self.predict_T,
                    )
                    self.mean_probs, self.kl_1, self.kl_2 = [
                        x.cpu().detach().numpy() for x in out
                    ]

                print(f"Learned predict_T = {float(self.predict_T.detach()):.4f}")

    def predict(self):
        if self.predictor is not None:
            predicted_id, was_rejected = self.predictor.predict()
            return predicted_id, was_rejected
        predict_probs = self.mean_probs
        predict_id = np.argmax(predict_probs, axis=-1)

        oog_prob = 1 - np.sum(predict_probs, axis=-1, keepdims=True)
        all_prob = np.concatenate([predict_probs, oog_prob], axis=-1)
        was_rejected = np.argmax(all_prob, axis=-1) == (all_prob.shape[-1] - 1)
        if self.log_dir is not None:
            # log error indicators and unc
            true_pred_label = np.zeros(self.probe_unique_ids.shape[0])
            error_calc = FrrFarIdent()
            error_calc(
                predict_id,
                was_rejected,
                self.g_unique_ids,
                self.probe_unique_ids,
            )
            true_pred_label[error_calc.is_seen] = error_calc.true_accept_true_ident
            true_pred_label[~error_calc.is_seen] = error_calc.true_reject
            np.savez(
                Path(self.log_dir) / f"kl_and_target_{self.predict_T}_M={self.M}.npz",
                kl_1=self.kl_1,
                kl_2=self.kl_2,
                true_pred_label=true_pred_label,
            )
        return predict_id, was_rejected

    def predict_uncertainty(self):
        if self.pred_uncertainty_type == "entropy":
            predict_id = np.argmax(self.mean_probs, axis=-1)
            oog_prob = 1 - np.sum(self.mean_probs, axis=-1, keepdims=True)
            all_prob = np.concatenate([self.mean_probs, oog_prob], axis=-1)
            was_rejected = np.argmax(all_prob, axis=-1) == (all_prob.shape[-1] - 1)
            true_pred_label = np.zeros(self.probe_unique_ids.shape[0])
            error_calc = FrrFarIdent()
            error_calc(
                predict_id,
                was_rejected,
                self.g_unique_ids,
                self.probe_unique_ids,
            )
            true_pred_label[error_calc.is_seen] = error_calc.true_accept_true_ident
            true_pred_label[~error_calc.is_seen] = error_calc.true_reject

            unc = self.calibration_transform.apply_calibration_transform(
                self.kl_1,
                self.kl_2,
                error_calc,
                dataset_name=self.dataset_name,
                far=self.far,
            )
        return unc

    def compute_mean_probs_and_kl(
        self,
        mean: np.ndarray,
        kappa: np.ndarray,
        gallery_means,
        gallery_kappas,
        T,
        return_mprisk_aux: bool = False,
    ) -> Any:
        """
        Stable log-space computation of:
          mean_gallery_probs, KL_1, KL_2

        Works for M=0 deterministic HolUE and for MC samples.
        """
        dtype = torch.float64
        device = self.device

        mean_np = np.asarray(mean, dtype=np.float64)
        kappa_np = np.asarray(kappa, dtype=np.float64)

        gallery_means_t = (
            gallery_means.to(device=device, dtype=dtype)
            if torch.is_tensor(gallery_means)
            else torch.as_tensor(gallery_means, device=device, dtype=dtype)
        )
        gallery_kappas_t = (
            gallery_kappas.to(device=device, dtype=dtype)
            if torch.is_tensor(gallery_kappas)
            else torch.as_tensor(gallery_kappas, device=device, dtype=dtype)
        )

        if not torch.is_tensor(T):
            T = torch.tensor(float(T), device=device, dtype=dtype)
        else:
            T = T.to(device=device, dtype=dtype)

        T = torch.clamp(T, min=1e-6)
        inv_T = 1.0 / T

        K = gallery_means_t.shape[0]
        d_int = int(mean_np.shape[-1])
        d = torch.tensor(float(d_int), device=device, dtype=dtype)

        self.K = K

        # Sample or use mean embedding.
        zs_np = self.sampler(mean_np, kappa_np)
        zs = torch.as_tensor(zs_np, device=device, dtype=dtype)

        # Defensive normalization.
        zs = F.normalize(zs, p=2.0, dim=-1)
        gallery_means_t = F.normalize(gallery_means_t, p=2.0, dim=-1)

        similarities = torch.matmul(zs, gallery_means_t.T)
        similarities = torch.clamp(similarities, -1.0 + 1e-9, 1.0 - 1e-9)

        log_uniform = torch.tensor(
            log_uniform_density(d_int), device=device, dtype=dtype
        )
        log_beta = torch.tensor(np.log(self.beta), device=device, dtype=dtype)
        log_gallery_prior = torch.tensor(
            np.log((1.0 - self.beta) / K), device=device, dtype=dtype
        )

        gk = gallery_kappas_t.reshape(-1).clamp_min(1e-12)

        if self.gallery_prior == "power":
            log_norm = (
                torch.lgamma(d - 1.0 + gk)
                + torch.lgamma(d / 2.0 + gk)
                + (gk - 1.0) * np.log(2.0)
                - (d / 2.0) * np.log(np.pi)
                - torch.lgamma(d - 1.0 + 2.0 * gk)
            )
            log_kernel = gk[None, None, :] * torch.log1p(similarities)

        elif self.gallery_prior == "vMF":
            log_norm_np = vmf_log_normalizer_np(gk.detach().cpu().numpy(), d=d_int)
            log_norm = torch.as_tensor(log_norm_np, device=device, dtype=dtype)
            log_kernel = gk[None, None, :] * similarities

        else:
            raise ValueError(f"Unknown gallery_prior={self.gallery_prior}")

        # Temperature-scaled unnormalized log posterior terms.
        log_gallery_terms = inv_T * (
            log_norm[None, None, :] + log_kernel + log_gallery_prior
        )
        log_oog_term = inv_T * (log_uniform + log_beta)

        log_gallery_sum = torch.logsumexp(log_gallery_terms, dim=-1)
        log_den = torch.logaddexp(log_gallery_sum, log_oog_term)

        gallery_log_probs = log_gallery_terms - log_den[..., None]
        gallery_probs = torch.exp(gallery_log_probs)
        mean_gallery_probs = torch.mean(gallery_probs, dim=1)
        # Local posterior probability of the continuous out-of-gallery component.
        # This is the aggregate probability of the mixed-prior unknown identity continuum
        # at a sampled embedding z.
        log_p0 = log_oog_term - log_den
        p0 = torch.exp(log_p0)
        mean_oog_prob = torch.mean(p0, dim=1)
        # KL_1 = sum p_T(c|x) log(p_T(c|x) / p(c))
        p_safe = mean_gallery_probs.clamp_min(1e-300)
        kl_1 = torch.sum(
            mean_gallery_probs * (torch.log(p_safe) - log_gallery_prior),
            dim=1,
        )

        # KL_2 for continuous OOG part.
        if self.emb_unc_model != "vMF":
            raise ValueError(f"Unsupported emb_unc_model={self.emb_unc_model}")

        mean_t = torch.as_tensor(mean_np, device=device, dtype=dtype)
        mean_t = F.normalize(mean_t, p=2.0, dim=-1)

        kappa_x_np = kappa_np[:, 0].astype(np.float64)
        kappa_x_np = np.maximum(kappa_x_np, 1e-12)
        log_norm_x_np = vmf_log_normalizer_np(kappa_x_np, d=d_int)

        kappa_x = torch.as_tensor(kappa_x_np, device=device, dtype=dtype)
        log_norm_x = torch.as_tensor(log_norm_x_np, device=device, dtype=dtype)

        sim_x = torch.sum(zs * mean_t[:, None, :], dim=-1)
        sim_x = torch.clamp(sim_x, -1.0 + 1e-9, 1.0 - 1e-9)

        log_p_z_given_x = log_norm_x[:, None] + kappa_x[:, None] * sim_x

        # p0 = torch.exp(log_oog_term - log_den)

        log_beta_over_sphere = log_beta + log_uniform
        log_arg = (inv_T - 1.0) * log_beta_over_sphere + log_p_z_given_x - log_den

        kl_2 = torch.mean(p0 * log_arg, dim=1)
        # ------------------------------------------------------------------
        # MPRisk auxiliary quantity: reject non-specificity.
        #
        # The mixed prior represents unknown identities as a continuum. Therefore,
        # even if the aggregate unknown probability pi0 is high, we can still ask:
        # is the posterior over unknown identities concentrated around a specific
        # unknown identity, or is it diffuse because the input sample is poor?
        #
        # For a rejected sample:
        #   high pi0 + diffuse unknown identity posterior => suspicious rejection.
        #
        # We use the inverse collision concentration of p(u | x, unknown).
        # In default analytic_vmf mode we use the exact collision concentration of
        # q(z | x) = vMF(mu_x, kappa_x):
        #
        #   C0 = S * integral q(z|x)^2 dz
        #      = S * C_d(kappa_x)^2 / C_d(2 kappa_x)
        #
        # and non_specificity = 1 / C0.
        #
        # This is intentionally tied to the mixed prior. If unknown were a single
        # collapsed class, this quantity would not exist.
        # ------------------------------------------------------------------
        if return_mprisk_aux:
            mode = getattr(self, "nonspecificity_mode", "analytic_vmf")
            log_surface_area = -log_uniform

            if mode == "analytic_vmf":
                # Exact vMF collision concentration of q(z|x).
                log_norm_2x_np = vmf_log_normalizer_np(2.0 * kappa_x_np, d=d_int)
                log_norm_2x = torch.as_tensor(
                    log_norm_2x_np, device=device, dtype=dtype
                )

                log_collision = log_surface_area + 2.0 * log_norm_x - log_norm_2x

            elif mode == "weighted_mc":
                # More literal but noisier estimate:
                # C0 = S / pi0^2 * E_q[ q(z|x) * p0(z)^2 ].
                #
                # Use this only with M > 1. With M=0 it degenerates to evaluating q at
                # its mean and overestimates concentration in high dimensions.
                log_mean_q_p0_sq = torch.logsumexp(
                    log_p_z_given_x + 2.0 * log_p0, dim=1
                ) - np.log(zs.shape[1])
                log_pi0 = torch.log(mean_oog_prob.clamp_min(1e-300))
                log_collision = log_surface_area + log_mean_q_p0_sq - 2.0 * log_pi0

            else:
                raise ValueError(
                    f"Unknown nonspecificity_mode={mode!r}. "
                    "Use 'analytic_vmf' or 'weighted_mc'."
                )

            # Mathematically collision >= 1. Clamp for numerical stability.
            log_collision = torch.clamp(log_collision, min=0.0, max=700.0)
            oog_nonspecificity = torch.exp(-log_collision)

        finite_tensors = [mean_gallery_probs, kl_1, kl_2]
        if return_mprisk_aux:
            finite_tensors.extend([mean_oog_prob, oog_nonspecificity])

        if any(not bool(torch.isfinite(t).all()) for t in finite_tensors):
            raise FloatingPointError(
                "Non-finite value in HolUE/MPRisk probability computation."
            )

        if return_mprisk_aux:
            return mean_gallery_probs, kl_1, kl_2, mean_oog_prob, oog_nonspecificity

        return mean_gallery_probs, kl_1, kl_2


class MPRiskPredictiveProb(MonteCarloPredictiveProb):
    """
    Mixed-Prior Bayes Risk for open-set recognition.

    This method replaces KL-based confidence with a posterior OSR risk.

    For an accepted probe predicted as gallery identity i:

        R = lambda_FA * pi_0
            + lambda_ID * (1 - pi_0 - pi_i)

    For a rejected probe:

        R = lambda_FR * (1 - pi_0)
            + lambda_NS * pi_0 * N_0

    where:
        pi_i  = posterior probability of gallery identity i,
        pi_0  = aggregate posterior probability of the unknown continuum,
        N_0   = reject non-specificity.

    The last term is the key mixed-prior correction. It prevents low-quality
    in-gallery samples that drift far from all gallery classes from being treated
    as confident true rejects. If the probe is rejected but the posterior over
    unknown identities is diffuse, the rejection is suspicious and uncertainty
    should be high.

    The class optionally calibrates two risk features:
        1. ordinary posterior OSR risk,
        2. mixed-prior non-specific reject penalty.

    Existing NNcalibration can be reused because it accepts two scalar features.
    """

    def __init__(
        self,
        gallery_prior: str,
        emb_unc_model: str,
        beta: float,
        far: float,
        M: int = 0,
        calibration_set=None,
        calibration_embs_name=None,
        calibration_transform=None,
        gallery_kappa: float = None,
        kappa_scale: float = 1.0,
        kappa_input_scale: float = 1.0,
        predict_T: float = 1.0,
        train_predict_T: bool = False,
        predict_T_lr: float = 1e-2,
        pred_uncertainty_type: str = "entropy",
        alpha: float = 0.5,
        log_dir: str = None,
        predictor=None,
        lambda_fa: float = 1.0,
        lambda_id: float = 1.0,
        lambda_fr: float = 1.0,
        lambda_ns: float = 1.0,
        nonspecificity_mode: str = "analytic_vmf",
        use_calibration: bool = True,
        tune_lambdas: bool = False,
        lambda_search_steps: int = 2048,
        lambda_search_log_low: float = -8.0,
        lambda_search_log_high: float = 8.0,
        lambda_tune_fraction_max: float = 0.5,
        lambda_tune_fraction_num: int = 20,
        lambda_tune_seed: int = 777,
        calibration_feature_mode: str = "scalar",
    ) -> None:
        if train_predict_T:
            raise NotImplementedError(
                "MPRiskPredictiveProb currently does not support "
                "joint training of predict_T. Set train_predict_T=False."
            )

        if predictor is not None:
            raise ValueError(
                "MPRiskPredictiveProb must score the same Bayesian decision "
                "whose posterior risk it computes. Do not use predictor='AccScore'."
            )

        super().__init__(
            gallery_prior=gallery_prior,
            emb_unc_model=emb_unc_model,
            beta=beta,
            far=far,
            M=M,
            calibration_set=calibration_set,
            calibration_embs_name=calibration_embs_name,
            calibration_transform=calibration_transform,
            gallery_kappa=gallery_kappa,
            kappa_scale=kappa_scale,
            kappa_input_scale=kappa_input_scale,
            predict_T=predict_T,
            train_predict_T=train_predict_T,
            predict_T_lr=predict_T_lr,
            pred_uncertainty_type=pred_uncertainty_type,
            alpha=alpha,
            log_dir=log_dir,
            predictor=predictor,
        )

        self.lambda_fa = float(lambda_fa)
        self.lambda_id = float(lambda_id)
        self.lambda_fr = float(lambda_fr)
        self.lambda_ns = float(lambda_ns)
        self.tune_lambdas = bool(tune_lambdas)
        self.lambda_search_steps = int(lambda_search_steps)
        self.lambda_search_log_low = float(lambda_search_log_low)
        self.lambda_search_log_high = float(lambda_search_log_high)
        self.lambda_tune_fraction_max = float(lambda_tune_fraction_max)
        self.lambda_tune_fraction_num = int(lambda_tune_fraction_num)
        self.lambda_tune_seed = int(lambda_tune_seed)
        if nonspecificity_mode not in ["analytic_vmf", "weighted_mc"]:
            raise ValueError(
                "nonspecificity_mode must be 'analytic_vmf' or 'weighted_mc'."
            )
        self.nonspecificity_mode = nonspecificity_mode

        self.use_calibration = bool(use_calibration)
        self._mprisk_calibration_trained = False
        calibration_feature_mode = str(calibration_feature_mode).strip()
        calibration_feature_mode = calibration_feature_mode.replace("-", "_").lower()

        if calibration_feature_mode not in ["scalar", "components"]:
            raise ValueError(
                "calibration_feature_mode must be either 'scalar' or 'components', "
                f"got {calibration_feature_mode!r}."
            )

        self.calibration_feature_mode = calibration_feature_mode

    def _compute_probs_aux(
        self,
        probe_feats: np.ndarray,
        probe_unc: np.ndarray,
        gallery_feats: np.ndarray,
        gallery_unc: np.ndarray,
        gallery_kappa: float,
    ):
        dtype = np.float64

        probe_feats = np.asarray(probe_feats, dtype=dtype)
        probe_unc = np.asarray(probe_unc, dtype=dtype)
        probe_unc_scaled = probe_unc * self.kappa_input_scale

        gallery_feats = np.asarray(gallery_feats, dtype=dtype)
        gallery_unc = np.asarray(gallery_unc, dtype=dtype)
        gallery_unc_scaled = np.ones_like(gallery_unc) * float(gallery_kappa)

        out = self.compute_mean_probs_and_kl(
            probe_feats,
            probe_unc_scaled,
            gallery_feats,
            gallery_unc_scaled,
            self.predict_T,
            return_mprisk_aux=True,
        )

        return [x.cpu().detach().numpy() for x in out]

    def _risk_components(
        self,
        mean_probs: np.ndarray,
        oog_prob: np.ndarray,
        oog_nonspecificity: np.ndarray,
    ) -> dict:
        mean_probs = np.asarray(mean_probs, dtype=np.float64)
        oog_prob = np.asarray(oog_prob, dtype=np.float64).reshape(-1)
        oog_nonspecificity = np.asarray(oog_nonspecificity, dtype=np.float64).reshape(
            -1
        )

        n, K = mean_probs.shape

        oog_prob = np.clip(oog_prob, 0.0, 1.0)
        oog_nonspecificity = np.clip(oog_nonspecificity, 0.0, 1.0)

        predicted_id = np.argmax(mean_probs, axis=-1)

        all_prob = np.concatenate([mean_probs, oog_prob[:, None]], axis=-1)
        was_rejected = np.argmax(all_prob, axis=-1) == K

        row_idx = np.arange(n)
        pi_hat = mean_probs[row_idx, predicted_id]

        other_known_prob = 1.0 - oog_prob - pi_hat
        other_known_prob = np.clip(other_known_prob, 0.0, 1.0)

        accepted = ~was_rejected
        rejected = was_rejected

        r_fa = np.zeros(n, dtype=np.float64)
        r_id = np.zeros(n, dtype=np.float64)
        r_fr = np.zeros(n, dtype=np.float64)
        r_ns = np.zeros(n, dtype=np.float64)

        # If accepted as gallery identity i:
        # false acceptance risk = probability of unknown.
        r_fa[accepted] = oog_prob[accepted]

        # If accepted as gallery identity i:
        # misidentification risk = probability of another known identity.
        r_id[accepted] = other_known_prob[accepted]

        # If rejected:
        # false rejection risk = probability of any known identity.
        r_fr[rejected] = 1.0 - oog_prob[rejected]

        # Mixed-prior correction:
        # high only when sample is rejected, unknown probability is high,
        # but the unknown identity posterior is diffuse.
        r_ns[rejected] = oog_prob[rejected] * oog_nonspecificity[rejected]

        components = {
            "predicted_id": predicted_id,
            "was_rejected": was_rejected,
            "r_fa": r_fa,
            "r_id": r_id,
            "r_fr": r_fr,
            "r_ns": r_ns,
        }

        ordinary_risk = (
            self.lambda_fa * r_fa + self.lambda_id * r_id + self.lambda_fr * r_fr
        )

        mixed_prior_penalty = r_ns

        mprisk = self._score_from_components(components)

        components.update(
            {
                "ordinary_risk": ordinary_risk,
                "mixed_prior_penalty": mixed_prior_penalty,
                "mprisk": mprisk,
            }
        )

        return components

        # mixed_prior_penalty = r_ns

        # mprisk = ordinary_risk + self.lambda_ns * mixed_prior_penalty

        # return {
        #     "predicted_id": predicted_id,
        #     "was_rejected": was_rejected,
        #     "r_fa": r_fa,
        #     "r_id": r_id,
        #     "r_fr": r_fr,
        #     "r_ns": r_ns,
        #     "ordinary_risk": ordinary_risk,
        #     "mixed_prior_penalty": mixed_prior_penalty,
        #     "mprisk": mprisk,
        # }

    def _store_test_risk_components(self, components: dict) -> None:
        self.predicted_id = components["predicted_id"]
        self.was_rejected = components["was_rejected"]

        self.r_fa = components["r_fa"]
        self.r_id = components["r_id"]
        self.r_fr = components["r_fr"]
        self.r_ns = components["r_ns"]

        self.risk_main = components["ordinary_risk"]
        self.risk_ns = components["mixed_prior_penalty"]
        self.mprisk = components["mprisk"]

    def _calibration_features_from_components(self, components: dict):
        """
        Select features passed to the calibration transform.

        scalar:
            use raw MPRisk only. This preserves the theoretically motivated
            MPRisk ordering after monotone calibration.

        components:
            use ordinary OSR risk and mixed-prior reject penalty separately.
            This is useful for ablation, but can hurt ranking if the calibrator
            is too flexible.
        """
        if self.calibration_feature_mode == "scalar":
            x1 = components["mprisk"]
            x2 = np.zeros_like(x1)
            return x1, x2

        if self.calibration_feature_mode == "components":
            x1 = components["ordinary_risk"]
            x2 = components["mixed_prior_penalty"]
            return x1, x2

        raise ValueError(
            f"Unknown calibration_feature_mode={self.calibration_feature_mode}"
        )

    def _test_calibration_features(self):
        if self.calibration_feature_mode == "scalar":
            x1 = self.mprisk
            x2 = np.zeros_like(x1)
            return x1, x2

        if self.calibration_feature_mode == "components":
            return self.risk_main, self.risk_ns

        raise ValueError(
            f"Unknown calibration_feature_mode={self.calibration_feature_mode}"
        )

    def _current_lambdas(self) -> np.ndarray:
        return np.array(
            [
                self.lambda_fa,
                self.lambda_id,
                self.lambda_fr,
                self.lambda_ns,
            ],
            dtype=np.float64,
        )

    def _set_lambdas(self, lambdas: np.ndarray) -> None:
        lambdas = np.asarray(lambdas, dtype=np.float64).reshape(4)
        lambdas = np.maximum(lambdas, 0.0)

        self.lambda_fa = float(lambdas[0])
        self.lambda_id = float(lambdas[1])
        self.lambda_fr = float(lambdas[2])
        self.lambda_ns = float(lambdas[3])

    def _score_from_components(
        self,
        components: dict,
        lambdas: np.ndarray = None,
    ) -> np.ndarray:
        if lambdas is None:
            lambdas = self._current_lambdas()

        lambdas = np.asarray(lambdas, dtype=np.float64).reshape(4)

        return (
            lambdas[0] * components["r_fa"]
            + lambdas[1] * components["r_id"]
            + lambdas[2] * components["r_fr"]
            + lambdas[3] * components["r_ns"]
        )

    @staticmethod
    def _safe_div(num: float, den: float, default: float = 0.0) -> float:
        den = float(den)
        if den == 0.0 or not np.isfinite(den):
            return default
        return float(num) / den

    def _f1_classic_for_subset(
        self,
        predicted_id: np.ndarray,
        was_rejected: np.ndarray,
        g_unique_ids: np.ndarray,
        probe_unique_ids: np.ndarray,
        subset_idx: np.ndarray,
    ) -> float:
        predicted_id = np.asarray(predicted_id)[subset_idx]
        was_rejected = np.asarray(was_rejected, dtype=bool)[subset_idx]
        probe_unique_ids = np.asarray(probe_unique_ids)[subset_idx]

        is_seen = np.isin(probe_unique_ids, g_unique_ids)

        tp = 0
        if np.any(is_seen):
            similar_gallery_class = g_unique_ids[predicted_id[is_seen]]
            tp = int(
                np.sum(
                    np.logical_and(
                        probe_unique_ids[is_seen] == similar_gallery_class,
                        was_rejected[is_seen] == False,
                    )
                )
            )

        fp = int(np.sum(was_rejected[~is_seen] == False))
        fn = int(np.sum(is_seen)) - tp

        precision = self._safe_div(tp, tp + fp)
        recall = self._safe_div(tp, tp + fn)
        f1 = self._safe_div(2.0 * precision * recall, precision + recall)

        return float(f1)

    def _correct_mask(
        self,
        predicted_id: np.ndarray,
        was_rejected: np.ndarray,
        g_unique_ids: np.ndarray,
        probe_unique_ids: np.ndarray,
    ) -> np.ndarray:
        predicted_id = np.asarray(predicted_id)
        was_rejected = np.asarray(was_rejected, dtype=bool)
        probe_unique_ids = np.asarray(probe_unique_ids)

        is_seen = np.isin(probe_unique_ids, g_unique_ids)

        correct = np.zeros(probe_unique_ids.shape[0], dtype=bool)

        if np.any(is_seen):
            similar_gallery_class = g_unique_ids[predicted_id[is_seen]]
            correct[is_seen] = np.logical_and(
                probe_unique_ids[is_seen] == similar_gallery_class,
                was_rejected[is_seen] == False,
            )

        correct[~is_seen] = was_rejected[~is_seen]

        return correct

    def _auc_f1_rejection_curve(
        self,
        uncertainty_score: np.ndarray,
        predicted_id: np.ndarray,
        was_rejected: np.ndarray,
        g_unique_ids: np.ndarray,
        probe_unique_ids: np.ndarray,
        fractions: np.ndarray,
    ) -> float:
        uncertainty_score = np.asarray(uncertainty_score, dtype=np.float64).reshape(-1)

        # Repository convention:
        # lower predicted_unc = more confident, kept first.
        order = np.argsort(uncertainty_score)

        n = uncertainty_score.shape[0]
        f1_values = []

        for fraction in fractions:
            keep_size = int((1.0 - float(fraction)) * n)
            keep_size = max(0, min(n, keep_size))

            keep_idx = order[:keep_size]

            f1 = self._f1_classic_for_subset(
                predicted_id=predicted_id,
                was_rejected=was_rejected,
                g_unique_ids=g_unique_ids,
                probe_unique_ids=probe_unique_ids,
                subset_idx=keep_idx,
            )
            f1_values.append(f1)

        return float(np.trapezoid(np.asarray(f1_values), fractions))

    def _validation_prr_for_score(
        self,
        uncertainty_score: np.ndarray,
        predicted_id: np.ndarray,
        was_rejected: np.ndarray,
        g_unique_ids: np.ndarray,
        probe_unique_ids: np.ndarray,
        fractions: np.ndarray,
        random_auc: float,
        oracle_auc: float,
    ) -> float:
        auc_value = self._auc_f1_rejection_curve(
            uncertainty_score=uncertainty_score,
            predicted_id=predicted_id,
            was_rejected=was_rejected,
            g_unique_ids=g_unique_ids,
            probe_unique_ids=probe_unique_ids,
            fractions=fractions,
        )

        denom = oracle_auc - random_auc
        if abs(denom) < 1e-12:
            return auc_value

        return float((auc_value - random_auc) / denom)

    def _build_lambda_candidates(self) -> list:
        rng = np.random.default_rng(
            self.lambda_tune_seed + int(10000 * float(self.far))
        )

        candidates = []

        # Current manually specified weights.
        candidates.append(self._current_lambdas())

        # Equal-cost baseline.
        candidates.append(np.ones(4, dtype=np.float64))

        # Single-component baselines.
        candidates.extend(
            [
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
                np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            ]
        )

        # Useful structured mixtures.
        candidates.extend(
            [
                np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64),
                np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
                np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float64),
                np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float64),
                np.array([0.0, 0.0, 1.0, 10.0], dtype=np.float64),
                np.array([1.0, 1.0, 1.0, 10.0], dtype=np.float64),
                np.array([1.0, 1.0, 1.0, 100.0], dtype=np.float64),
            ]
        )

        # Random log-uniform search.
        for _ in range(self.lambda_search_steps):
            log_lambdas = rng.uniform(
                self.lambda_search_log_low,
                self.lambda_search_log_high,
                size=4,
            )
            lambdas = np.exp(log_lambdas)

            # Scale is irrelevant for ranking, but normalize to avoid overflow.
            lambdas = lambdas / max(np.max(lambdas), 1e-12)

            candidates.append(lambdas.astype(np.float64))

        return candidates

    def _tune_lambdas_on_validation(
        self,
        calib_components: dict,
        g_unique_ids_calib: np.ndarray,
        probe_unique_ids_calib: np.ndarray,
    ) -> None:
        fractions = np.linspace(
            0.0,
            self.lambda_tune_fraction_max,
            self.lambda_tune_fraction_num,
        )

        predicted_id = calib_components["predicted_id"]
        was_rejected = calib_components["was_rejected"]

        correct = self._correct_mask(
            predicted_id=predicted_id,
            was_rejected=was_rejected,
            g_unique_ids=g_unique_ids_calib,
            probe_unique_ids=probe_unique_ids_calib,
        )
        is_error = ~correct

        rng = np.random.default_rng(self.lambda_tune_seed)

        # Oracle: errors get large uncertainty and are filtered first.
        oracle_score = is_error.astype(np.float64)
        oracle_score = oracle_score + 1e-9 * rng.random(len(oracle_score))

        # Random baseline.
        random_score = rng.random(len(oracle_score))

        oracle_auc = self._auc_f1_rejection_curve(
            uncertainty_score=oracle_score,
            predicted_id=predicted_id,
            was_rejected=was_rejected,
            g_unique_ids=g_unique_ids_calib,
            probe_unique_ids=probe_unique_ids_calib,
            fractions=fractions,
        )

        random_auc = self._auc_f1_rejection_curve(
            uncertainty_score=random_score,
            predicted_id=predicted_id,
            was_rejected=was_rejected,
            g_unique_ids=g_unique_ids_calib,
            probe_unique_ids=probe_unique_ids_calib,
            fractions=fractions,
        )

        best_lambdas = self._current_lambdas()
        best_score = -np.inf

        for lambdas in self._build_lambda_candidates():
            uncertainty_score = self._score_from_components(
                calib_components,
                lambdas=lambdas,
            )

            prr = self._validation_prr_for_score(
                uncertainty_score=uncertainty_score,
                predicted_id=predicted_id,
                was_rejected=was_rejected,
                g_unique_ids=g_unique_ids_calib,
                probe_unique_ids=probe_unique_ids_calib,
                fractions=fractions,
                random_auc=random_auc,
                oracle_auc=oracle_auc,
            )

            if prr > best_score:
                best_score = prr
                best_lambdas = lambdas.copy()

        self._set_lambdas(best_lambdas)

        print(
            "[MPRisk] Tuned lambdas on validation: "
            f"lambda_fa={self.lambda_fa:.6g}, "
            f"lambda_id={self.lambda_id:.6g}, "
            f"lambda_fr={self.lambda_fr:.6g}, "
            f"lambda_ns={self.lambda_ns:.6g}, "
            f"val_PRR={best_score:.4f}"
        )

    def setup(
        self,
        probe_feats: np.ndarray,
        probe_unc: np.ndarray,
        gallery_feats: np.ndarray,
        gallery_unc: np.ndarray,
        dataset_name: str,
        g_unique_ids: np.ndarray = None,
        probe_unique_ids: np.ndarray = None,
    ):
        """
        We reuse the parent setup only to:
          1. find gallery_kappa for the requested FPIR,
          2. initialize the Bayesian posterior model.

        Then we recompute posterior probabilities with MPRisk auxiliary terms.
        Calibration, if enabled, is MPRisk-specific and is performed below.
        """
        old_calibration_set = self.calibration_set
        old_calibration_transform = self.calibration_transform

        # Disable parent HolUE KL calibration path.
        self.calibration_set = None
        self.calibration_transform = None

        super().setup(
            probe_feats=probe_feats,
            probe_unc=probe_unc,
            gallery_feats=gallery_feats,
            gallery_unc=gallery_unc,
            g_unique_ids=g_unique_ids,
            probe_unique_ids=probe_unique_ids,
            dataset_name=dataset_name,
        )

        # Restore MPRisk calibration objects.
        self.calibration_set = old_calibration_set
        self.calibration_transform = old_calibration_transform

        self.g_unique_ids = g_unique_ids
        self.probe_unique_ids = probe_unique_ids
        self.dataset_name = dataset_name

        # Test-set probabilities and MPRisk auxiliaries.
        (
            self.mean_probs,
            self.kl_1,
            self.kl_2,
            self.oog_prob,
            self.oog_nonspecificity,
        ) = self._compute_probs_aux(
            probe_feats=probe_feats,
            probe_unc=probe_unc,
            gallery_feats=gallery_feats,
            gallery_unc=gallery_unc,
            gallery_kappa=self.gallery_kappa,
        )

        # Compute initial test components. They may be recomputed after lambda tuning.
        components = self._risk_components(
            self.mean_probs,
            self.oog_prob,
            self.oog_nonspecificity,
        )

        need_validation_protocol = self.calibration_set is not None and (
            self.tune_lambdas
            or (self.use_calibration and self.calibration_transform is not None)
        )

        calib_components = None

        if need_validation_protocol:
            self.gallery_pooled_templates_calib, self.probe_pooled_templates_calib = (
                prepare_calibration_dataset(
                    self.calibration_set,
                    self.calibration_embs_name,
                )
            )

            self.g_unique_ids_calib = self.gallery_pooled_templates_calib["g1"][
                "template_subject_ids_sorted"
            ]
            self.probe_unique_ids_calib = self.probe_pooled_templates_calib["g1"][
                "template_subject_ids_sorted"
            ]

            is_seen_calib = np.isin(
                self.probe_unique_ids_calib,
                self.g_unique_ids_calib,
            )

            probe_feats_calib = self.probe_pooled_templates_calib["g1"][
                "template_pooled_features"
            ]
            probe_unc_calib = self.probe_pooled_templates_calib["g1"][
                "template_pooled_data_unc"
            ]
            gallery_feats_calib = self.gallery_pooled_templates_calib["g1"][
                "template_pooled_features"
            ]
            gallery_unc_calib = self.gallery_pooled_templates_calib["g1"][
                "template_pooled_data_unc"
            ]

            probe_unc_calib_scaled = probe_unc_calib * self.kappa_input_scale

            far_loss_func_calib = FarLossCalc(
                probe_feats_calib,
                probe_unc_calib_scaled,
                gallery_feats_calib,
                gallery_unc_calib,
                self.predict_T,
                self.far,
                is_seen_calib,
                self,
                verbose=True,
            )

            calibration_set_kappa = golden_selection_search(
                self.kappa_high,
                self.kappa_low,
                self.eps,
                self.max_iter,
                far_loss_func_calib,
                verbose=False,
            )

            print(
                f"[MPRisk] Found calibration kappa "
                f"{np.round(calibration_set_kappa, 4)} for far {self.far}"
            )

            (
                self.mean_probs_calib,
                self.kl_1_calib,
                self.kl_2_calib,
                self.oog_prob_calib,
                self.oog_nonspecificity_calib,
            ) = self._compute_probs_aux(
                probe_feats=probe_feats_calib,
                probe_unc=probe_unc_calib,
                gallery_feats=gallery_feats_calib,
                gallery_unc=gallery_unc_calib,
                gallery_kappa=calibration_set_kappa,
            )

            calib_components = self._risk_components(
                self.mean_probs_calib,
                self.oog_prob_calib,
                self.oog_nonspecificity_calib,
            )
            if self.tune_lambdas:
                self._tune_lambdas_on_validation(
                    calib_components=calib_components,
                    g_unique_ids_calib=self.g_unique_ids_calib,
                    probe_unique_ids_calib=self.probe_unique_ids_calib,
                )

                # Recompute calibration and test components with tuned lambdas.
                calib_components = self._risk_components(
                    self.mean_probs_calib,
                    self.oog_prob_calib,
                    self.oog_nonspecificity_calib,
                )

                components = self._risk_components(
                    self.mean_probs,
                    self.oog_prob,
                    self.oog_nonspecificity,
                )
            if self.use_calibration:
                self.risk_main_calib = calib_components["ordinary_risk"]
                self.risk_ns_calib = calib_components["mixed_prior_penalty"]
                self.mprisk_calib = calib_components["mprisk"]

                calib_x1, calib_x2 = self._calibration_features_from_components(
                    calib_components
                )

                error_calc = FrrFarIdent()
                error_calc(
                    calib_components["predicted_id"],
                    calib_components["was_rejected"],
                    self.g_unique_ids_calib,
                    self.probe_unique_ids_calib,
                )

                self.calibration_transform.train_calibration_parameters(
                    calib_x1,
                    calib_x2,
                    error_calc,
                    dataset_name=self.calibration_set.dataset_name,
                    far=self.far,
                )

                self._mprisk_calibration_trained = True
        # Store final test components after optional lambda tuning.
        self._store_test_risk_components(components)
        if self.log_dir is not None:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            np.savez(
                Path(self.log_dir) / f"mprisk_components_far_{self.far}.npz",
                mean_probs=self.mean_probs,
                oog_prob=self.oog_prob,
                oog_nonspecificity=self.oog_nonspecificity,
                r_fa=self.r_fa,
                r_id=self.r_id,
                r_fr=self.r_fr,
                r_ns=self.r_ns,
                risk_main=self.risk_main,
                risk_ns=self.risk_ns,
                mprisk=self.mprisk,
            )

    def predict(self):
        return self.predicted_id, self.was_rejected

    def predict_uncertainty(self):
        if self.use_calibration and self._mprisk_calibration_trained:
            error_calc = FrrFarIdent()
            error_calc(
                self.predicted_id,
                self.was_rejected,
                self.g_unique_ids,
                self.probe_unique_ids,
            )

            # Existing calibrator returns -P(correct).
            # This is consistent with the repository convention:
            # lower values are kept first, higher values are filtered first.
            test_x1, test_x2 = self._test_calibration_features()

            unc = self.calibration_transform.apply_calibration_transform(
                test_x1,
                test_x2,
                error_calc,
                dataset_name=self.dataset_name,
                far=self.far,
            )
            return unc

        # Raw MPRisk: larger value = more uncertain.
        return self.mprisk
