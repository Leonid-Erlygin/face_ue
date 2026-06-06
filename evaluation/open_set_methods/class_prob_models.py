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
        self.kappa_high = 100000
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

        p0 = torch.exp(log_oog_term - log_den)

        log_beta_over_sphere = log_beta + log_uniform
        log_arg = (inv_T - 1.0) * log_beta_over_sphere + log_p_z_given_x - log_den

        kl_2 = torch.mean(p0 * log_arg, dim=1)

        if (
            not torch.isfinite(mean_gallery_probs).all()
            or not torch.isfinite(kl_1).all()
            or not torch.isfinite(kl_2).all()
        ):
            raise FloatingPointError(
                "Non-finite value in HolUE probability/KL computation."
            )

        return mean_gallery_probs, kl_1, kl_2
