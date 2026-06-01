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
        self.gallery_kappa = gallery_kappa
        self.kappa_scale = kappa_scale
        self.kappa_input_scale = kappa_input_scale
        self.predictor = predictor
        self.train_predict_T = train_predict_T
        self.predict_T_lr = predict_T_lr

        if self.train_predict_T:
            # Learnable scalar. float64 matches the rest of the function.
            self.predict_T = torch.nn.Parameter(
                torch.tensor(float(predict_T), dtype=torch.float64, device="cuda:0"),
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
        probe_unc_scaled = probe_unc * self.kappa_input_scale
        dtype = np.float64
        probe_feats = probe_feats.astype(dtype)
        probe_unc = probe_unc.astype(dtype)
        gallery_feats = gallery_feats.astype(dtype)
        gallery_unc = gallery_unc.astype(dtype)
        self.g_unique_ids = g_unique_ids
        self.probe_unique_ids = probe_unique_ids
        self.dataset_name = dataset_name
        if g_unique_ids is not None and self.gallery_kappa == None:
            # find kappa
            is_seen = np.isin(probe_unique_ids, g_unique_ids)
            kappa_low = 300
            kappa_high = 1000000
            max_iter = 50
            eps = 0.0005
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
                kappa_high, kappa_low, eps, max_iter, far_loss_func
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
            kappa_low = 200
            kappa_high = 1000000
            max_iter = 50
            eps = 0.0005
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

            calibratation_set_kappa = golden_selection_search(
                kappa_high, kappa_low, eps, max_iter, far_loss_func_calib, verbose=False
            )
            # print("hard coded kappa val!")
            # calibratation_set_kappa = self.gallery_kappa
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
        gallery_means,  # np.ndarray or torch.Tensor
        gallery_kappas,  # np.ndarray or torch.Tensor
        T,  # float OR torch.Tensor (possibly requires_grad)
    ) -> Any:
        gallery_kappas_np = (
            gallery_kappas
            if isinstance(gallery_kappas, np.ndarray)
            else gallery_kappas.detach().cpu().numpy()
        )

        cuda = torch.device("cuda:0")
        if isinstance(gallery_means, np.ndarray):
            gallery_means = torch.tensor(gallery_means, device=cuda)
        if isinstance(gallery_kappas, np.ndarray):
            gallery_kappas = torch.tensor(gallery_kappas, device=cuda)
        # Ensure T is a tensor on the right device/dtype (lets `1/T` etc. stay differentiable).
        if not torch.is_tensor(T):
            T = torch.tensor(
                float(T), dtype=gallery_means.dtype, device=gallery_means.device
            )
        else:
            T = T.to(dtype=gallery_means.dtype, device=gallery_means.device)

        inv_T = 1.0 / T

        self.K = gallery_means.shape[0]
        zs = torch.tensor(self.sampler(mean, kappa), device=gallery_means.device)
        d_np = mean.shape[-1]
        d = torch.tensor([d_np], device=gallery_means.device, dtype=gallery_means.dtype)

        # --- 1. Similarities -------------------------------------------------
        similarities = torch.matmul(zs, gallery_means.T)  # (N, M, K)

        log_uniform_dencity = (
            torch.special.gammaln(d / 2) - np.log(2) - (d / 2) * np.log(np.pi)
        )

        # --- 2. Prior constants (do NOT depend on T, kept in numpy) ---------
        if self.gallery_prior == "power":
            log_m_c = (
                torch.special.gammaln(d - 1 + gallery_kappas)
                + torch.special.gammaln(d / 2 + gallery_kappas)
                + gallery_kappas * np.log(2)
                - torch.special.gammaln(d / 2)
                - torch.special.gammaln(d - 1 + 2 * gallery_kappas)
            )
            log_normalizer = log_m_c + log_uniform_dencity
            pz_c_no_norm_log = torch.log(similarities + 1) * (
                gallery_kappas[..., :, 0] * inv_T
            )
        elif self.gallery_prior == "vMF":
            log_iv = (
                np.log(ive(d_np / 2 - 1, gallery_kappas_np, dtype=np.float64))
                + gallery_kappas_np
            )
            log_m_c_np = -np.log(
                hyp0f1(d_np / 2, gallery_kappas_np**2 / 4, dtype=np.float64)
            )
            log_normalizer_np = (
                (d_np / 2 - 1) * np.log(gallery_kappas_np)
                - d_np / 2 * np.log(2 * np.pi)
                - log_iv
            )
            log_m_c = torch.tensor(log_m_c_np, device=cuda)
            log_normalizer = torch.tensor(log_normalizer_np, device=cuda)
            pz_c_no_norm_log = similarities * (gallery_kappas[..., :, 0] * inv_T)
        else:
            raise ValueError

        # --- 3. log p(z) (temperature-scaled marginal) ----------------------
        p_c = ((1 - self.beta) / self.K) ** inv_T  # tensor if T tensor
        beta_T = self.beta**inv_T  # tensor if T tensor

        logit = pz_c_no_norm_log + log_m_c[..., :, 0] * inv_T
        logit_exp = torch.exp(logit)
        logit_sum = torch.sum(logit_exp, dim=-1) * p_c  # (N, M)

        log_p_c_T = inv_T * np.log((1.0 - self.beta) / self.K)
        log_beta_T = inv_T * np.log(self.beta)

        log_gallery_terms = pz_c_no_norm_log + log_m_c[..., :, 0] * inv_T + log_p_c_T
        log_gallery_sum = torch.logsumexp(log_gallery_terms, dim=-1)

        log_z_prob = inv_T * log_uniform_dencity + torch.logaddexp(
            log_gallery_sum,
            torch.as_tensor(
                log_beta_T, dtype=log_gallery_sum.dtype, device=log_gallery_sum.device
            ),
        )
        # --- 4. Gallery-class log-probs -------------------------------------
        # Note: `similarities` may have been algebraically reused above, but since
        # we never used `out=similarities`, it still holds zs @ gallery_means.T.
        if self.gallery_prior == "power":
            pz_c_no_norm_log2 = torch.log(similarities + 1) * (
                gallery_kappas[..., :, 0] * inv_T
            )
        else:  # vMF
            pz_c_no_norm_log2 = similarities * (gallery_kappas[..., :, 0] * inv_T)

        pz_c_log = pz_c_no_norm_log2 + inv_T * log_normalizer[..., :, 0]

        gallery_log_probs = (
            pz_c_log + inv_T * np.log((1 - self.beta) / self.K) - log_z_prob[..., None]
        )
        gallery_probs = torch.exp(gallery_log_probs)
        mean_gallery_probs = torch.mean(gallery_probs, dim=1)  # (N, K)

        # --- 5. KL_1 ---------------------------------------------------------
        # log p_c = (1/T) * log((1 - beta)/K)— differentiable in T
        log_p_c = torch.as_tensor(
            np.log((1.0 - self.beta) / self.K),
            dtype=mean_gallery_probs.dtype,
            device=mean_gallery_probs.device,
        )
        kl_1 = torch.sum(
            torch.xlogy(mean_gallery_probs, mean_gallery_probs)
            - mean_gallery_probs * log_p_c,
            dim=1,
        )

        # --- 6. KL_2 ---------------------------------------------------------
        if self.emb_unc_model == "vMF":
            d_scalar = d_np
            log_iv_x = (
                np.log(ive(d_scalar / 2 - 1, kappa[:, 0], dtype=np.float64))
                + kappa[:, 0]
            )
            log_normalizer_x = (
                (d_scalar / 2 - 1) * np.log(kappa[:, 0])
                - (d_scalar / 2) * np.log(2 * np.pi)
                - log_iv_x
            )
            log_normalizer_x = torch.tensor(
                log_normalizer_x, device=gallery_means.device
            )
            kappa_tensor = torch.tensor(kappa[:, 0], device=gallery_means.device)
            mean_tensor = torch.tensor(mean, device=gallery_means.device)

            similarities_x = torch.sum(zs * mean_tensor[:, None, :], dim=2)  # (N, M)
            log_p_z_given_x = (
                log_normalizer_x[:, None] + kappa_tensor[:, None] * similarities_x
            )

            p0 = beta_T / (logit_sum + beta_T)  # (N, M), differentiable in T
            log_beta_over_S = np.log(self.beta) + log_uniform_dencity.item()

            log_ratio = log_p_z_given_x - log_z_prob
            log_arg = (inv_T - 1.0) * log_beta_over_S + log_ratio

            kl_2 = torch.mean(p0 * log_arg, dim=1)
        else:
            raise ValueError

        # Safety check — only trip on real NaNs (avoid breaking autograd with .numpy())
        if torch.isnan(kl_1).any() or torch.isnan(kl_2).any():
            raise ValueError("NaN in KL computation")

        return mean_gallery_probs, kl_1, kl_2
