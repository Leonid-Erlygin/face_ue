import numpy as np
from .base_method import OpenSetMethod
from evaluation.metrics import FrrFarIdent
from evaluation.test_datasets import FaceRecogntionDataset
from scipy.special import iv

from evaluation.open_set_methods.calibration_utils import prepare_calibration_dataset
from scipy.special import ive, hyp0f1, loggamma
from pathlib import Path
from scipy.optimize import fsolve, minimize
from typing import List, Union, Any
from torch import nn
import torch
from utils.golden_section import golden_selection_search


class FarLossCalc:
    def __init__(
        self,
        beta: float,
        T: float,
        class_model: str,
        target_far: float,
        is_seen: np.ndarray,
        similarity_matrix: torch.tensor,
    ) -> None:
        self.beta = beta
        self.T = T
        self.class_model = class_model
        self.target_far = target_far
        self.is_seen = is_seen
        self.similarity_matrix = similarity_matrix

    def __call__(self, kappa: float) -> float:
        posterior_prob = PosteriorProb(
            kappa=kappa,
            beta=self.beta,
            class_model=self.class_model,
            K=self.similarity_matrix.shape[-1],
        )
        all_classes_log_prob = posterior_prob.compute_all_class_log_probabilities(
            self.similarity_matrix, self.T
        )
        # print(all_classes_log_prob.min())
        all_classes_log_prob = torch.mean(all_classes_log_prob, dim=1).numpy()
        was_rejected = np.argmax(all_classes_log_prob, axis=-1) == (
            all_classes_log_prob.shape[-1] - 1
        )
        far = np.mean(was_rejected[~self.is_seen] == False)
        # print(f"Found kappa {np.round(kappa,4)} for far {far}")
        return -np.abs(far - self.target_far) / self.target_far


class PosteriorProbability(OpenSetMethod):
    def __init__(
        self,
        distance_function: Any,
        beta: float,
        uncertainty_type: str,
        alpha: float,
        aggregation: str,
        class_model: str,
        predict_T: float,
        T_data_unc: float,
        far: float = None,
        gallery_kappa: float = None,
        calibrate_unc: bool = None,
        calibrate_by_false_reject: bool = False,
        calibrate_gallery_unc: str = None,
        calibration_set: FaceRecogntionDataset = None,
        calibration_embs_name=None,
        log_dir=None,
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
        self.predict_T = predict_T
        self.gallery_kappa = gallery_kappa
        self.T_data_unc = T_data_unc
        self.calibrate_unc = calibrate_unc
        self.calibrate_by_false_reject = calibrate_by_false_reject
        self.calibrate_gallery_unc = calibrate_gallery_unc
        self.calibration_set = calibration_set
        self.calibration_embs_name = calibration_embs_name
        self.log_dir = log_dir
        if calibration_set is None:
            return

    def setup(
        self,
        probe_feats: np.ndarray,
        probe_unc: np.ndarray,
        gallery_feats: np.ndarray,
        gallery_unc: np.ndarray,
        g_unique_ids: np.ndarray,
        probe_unique_ids: np.ndarray,
        dataset_name: str,
    ):
        """
        g_unique_ids and probe_unique_ids are needed to find kappa that gives certan self.far
        """
        if self.far is None:
            raise ValueError
        probe_feats = probe_feats[:, np.newaxis, :]
        self.data_uncertainty = probe_unc
        self.g_unique_ids = g_unique_ids
        self.probe_unique_ids = probe_unique_ids
        similarity_matrix = torch.tensor(
            self.distance_function(
                probe_feats,
                probe_unc,
                gallery_feats,
                gallery_unc,
            )
        )
        T = self.predict_T
        # find kappa
        is_seen = np.isin(probe_unique_ids, g_unique_ids)

        if self.gallery_kappa is None:
            kappa_low = 300
            kappa_high = 1000000
            max_iter = 50
            eps = 0.0005
            far_loss_func = FarLossCalc(
                self.beta, T, self.class_model, self.far, is_seen, similarity_matrix
            )
            self.gallery_kappa = golden_selection_search(
                kappa_high, kappa_low, eps, max_iter, far_loss_func, verbose=False
            )
            print(f"Found kappa {np.round(self.gallery_kappa,4)} for far {self.far}")
        if self.class_model == "vMF_Power":
            raise ValueError
        else:
            self.posterior_prob = PosteriorProb(
                kappa=self.gallery_kappa,
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

        # get calibration set log probs
        if self.calibration_set is not None:
            self.gallery_pooled_templates_calib, self.probe_pooled_templates_calib = (
                prepare_calibration_dataset(
                    self.calibration_set, self.calibration_embs_name
                )
            )
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

            similarity_matrix_calib = torch.tensor(
                self.distance_function(
                    probe_feats_calib,
                    probe_unc_calib,
                    gallery_feats_calib,
                    gallery_unc_calib,
                )
            )
            kappa_low = 300
            kappa_high = 1000000
            max_iter = 50
            eps = 0.0001
            far_loss_func = FarLossCalc(
                self.beta,
                T,
                self.class_model,
                self.far,
                is_seen_calib,
                similarity_matrix_calib,
            )
            calibratation_set_kappa = golden_selection_search(
                kappa_high, kappa_low, eps, max_iter, far_loss_func
            )
            print(
                f"Found kappa_calib {np.round(calibratation_set_kappa,4)} for far {self.far}"
            )
            posterior_prob_calib = PosteriorProb(
                kappa=calibratation_set_kappa,
                beta=self.beta,
                class_model=self.class_model,
                K=similarity_matrix_calib.shape[-1],
            )
            all_classes_log_prob_calib = (
                posterior_prob_calib.compute_all_class_log_probabilities(
                    similarity_matrix_calib, T
                )
            )
            self.all_classes_log_prob_calib = torch.mean(
                all_classes_log_prob_calib, dim=1
            ).numpy()

            self._calibrate_T(similarity_matrix_calib)
            T = self.predict_T
            # Recompute test & calib probabilities with optimal T
            self.all_classes_log_prob = (
                self.posterior_prob.compute_all_class_log_probabilities(
                    similarity_matrix, T
                )
            )
            self.all_classes_log_prob = torch.mean(
                self.all_classes_log_prob, dim=1
            ).numpy()

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
        # print(f"Found kappa {np.round(kappa[0]*100,4)} for far {far}")
        return -np.abs(far - target_far)

    def predict(self):
        self.predicted_id = np.argmax(self.all_classes_log_prob[:, :-1], axis=-1)
        self.was_rejected = np.argmax(self.all_classes_log_prob, axis=-1) == (
            self.all_classes_log_prob.shape[-1] - 1
        )
        return self.predicted_id, self.was_rejected

    def _calibrate_T(self, similarity_matrix_calib):
        """Find optimal T via ternary search on calibration set to maximize error-prediction AUC."""
        from sklearn.metrics import roc_auc_score
        import torch

        # 1. Compute ground-truth error labels using existing pipeline logic
        error_calc = FrrFarIdent()
        pred_ids = np.argmax(self.all_classes_log_prob_calib[:, :-1], axis=-1)
        was_rejected = (
            np.argmax(self.all_classes_log_prob_calib, axis=-1)
            == self.all_classes_log_prob_calib.shape[-1] - 1
        )
        error_calc(
            pred_ids, was_rejected, self.g_unique_ids_calib, self.probe_unique_ids_calib
        )

        true_pred_label = np.zeros(len(self.probe_unique_ids_calib), dtype=bool)
        true_pred_label[error_calc.is_seen] = error_calc.true_accept_true_ident
        true_pred_label[~error_calc.is_seen] = error_calc.true_reject
        is_error = ~true_pred_label

        # 2. Objective: AUC of uncertainty vs error at temperature T
        def objective(T):
            log_probs = self.posterior_prob.compute_all_class_log_probabilities(
                similarity_matrix_calib, T
            )
            log_probs = torch.mean(log_probs, dim=1).numpy()

            if self.uncertainty_type == "maxprob":
                unc = 1 - np.exp(np.max(log_probs, axis=-1))
            else:  # entropy
                probs = np.exp(log_probs)
                p_log_p = np.where(probs > 0, probs * np.log(probs), 0)
                unc = -np.sum(p_log_p, axis=-1)

            return roc_auc_score(is_error, unc) if len(np.unique(is_error)) > 1 else 0.5

        # 3. Ternary search (unimodal T-vs-AUC curve, analogous to kappa golden search)
        left, right, tol, max_iter = 1, 200.0, 1e-3, 100
        for _ in range(max_iter):
            if right - left < tol:
                break
            m1 = left + (right - left) / 3
            m2 = right - (right - left) / 3
            if objective(m1) < objective(m2):
                left = m1
            else:
                right = m2

        self.predict_T = (left + right) / 2
        print(
            f"[GalUE] Calibrated T={self.predict_T:.3f} | Error AUC={objective(self.predict_T):.4f}"
        )

    @staticmethod
    def train_calibration(
        confidence_score,
        true_pred_labels,
        prob_compute,
        params,
        lr,
        iter_num,
        verbose=False,
    ):
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.5)
        bce_loss = nn.BCELoss()
        confidence_score = torch.tensor(confidence_score, dtype=torch.float32)
        true_pred_labels = torch.tensor(true_pred_labels, dtype=torch.float32)
        for iter in range(iter_num):
            optimizer.zero_grad()
            probs = prob_compute(confidence_score, params)
            loss = bce_loss(probs, true_pred_labels)
            loss.backward()
            optimizer.step()
            param_values = [param.item() for param in params]
            if verbose:
                print(f"Iteration {iter}, Loss: {loss.item()} params: {param_values}")

    @staticmethod
    def beta_calib(conf_gallery, conf_gallery_calib, true_pred_label):
        a = torch.nn.Parameter(
            torch.tensor(1.0, dtype=torch.float64), requires_grad=True
        )
        b = torch.nn.Parameter(
            torch.tensor(1.0, dtype=torch.float64), requires_grad=True
        )
        c = torch.nn.Parameter(
            torch.tensor(1.0, dtype=torch.float64), requires_grad=True
        )

        def prob_compute(conf, params):
            logit = (
                params[0] * torch.log(conf)
                + params[1] * (-torch.log(1 - conf + 1e-40))
                + params[2]
            )
            return torch.special.expit(logit)

        PosteriorProbability.train_calibration(
            conf_gallery_calib,
            true_pred_label,
            prob_compute,
            [a, b, c],
            lr=0.01,
            iter_num=500,
            verbose=False,
        )
        conf_gallery = prob_compute(
            torch.tensor(conf_gallery, dtype=torch.float32),
            [a.data, b.data, c.data],
        ).numpy()
        return conf_gallery

    @staticmethod
    def calibrate_scf_unc(
        data_uncertainty,
        data_uncertainty_calib,
        true_pred_label,
        verbose=False,
        scale_factor: float = 500,
    ):

        m = torch.nn.Parameter(
            torch.tensor(0.5, dtype=torch.float64), requires_grad=True
        )
        gamma = torch.nn.Parameter(
            torch.tensor(1.0, dtype=torch.float64), requires_grad=True
        )
        # norm data unc to prevent saturation
        data_uncertainty_norm = data_uncertainty / scale_factor
        data_uncertainty_norm_calib = data_uncertainty_calib / scale_factor

        # train logistic calibration
        def prob_compute(conf, params):
            return torch.special.expit(params[1] * (conf - params[0]))

        PosteriorProbability.train_calibration(
            data_uncertainty_norm_calib,
            true_pred_label,
            prob_compute,
            [m, gamma],
            lr=0.01,
            iter_num=500,
            verbose=verbose,
        )
        data_conf = prob_compute(
            torch.tensor(data_uncertainty_norm, dtype=torch.float32),
            [m.data, gamma.data],
        ).numpy()
        return data_conf

    def predict_uncertainty(self):
        if self.uncertainty_type == "maxprob":
            conf_gallery = np.exp(np.max(self.all_classes_log_prob, axis=-1))
            # conf_gallery = np.max(self.all_classes_log_prob, axis=-1)
        elif self.uncertainty_type == "entr":
            conf_gallery = -(
                -np.sum(
                    np.exp(self.all_classes_log_prob) * self.all_classes_log_prob,
                    axis=-1,
                )
                + 1
            )
        else:
            raise ValueError
        if self.data_uncertainty.shape[1] == 1:
            # here self.data_uncertainty is scf concetration
            self.data_uncertainty = self.data_uncertainty[:, 0]
        else:
            raise NotImplemented

        if self.calibrate_unc:
            # logistic calibration for scf confidence
            error_calc = FrrFarIdent()
            predicted_id = np.argmax(self.all_classes_log_prob_calib[:, :-1], axis=-1)
            was_rejected = np.argmax(self.all_classes_log_prob_calib, axis=-1) == (
                self.all_classes_log_prob_calib.shape[-1] - 1
            )
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
            data_conf = self.calibrate_scf_unc(
                self.data_uncertainty,
                data_uncertainty_calib,
                true_pred_label,
                verbose=False,
            )
        else:
            data_conf = self.data_uncertainty
        if self.calibrate_gallery_unc is not None:
            assert self.uncertainty_type == "maxprob"
            # beta calibration for gallery confidence
            error_calc = FrrFarIdent()
            predicted_id = np.argmax(self.all_classes_log_prob_calib[:, :-1], axis=-1)
            was_rejected = np.argmax(self.all_classes_log_prob_calib, axis=-1) == (
                self.all_classes_log_prob_calib.shape[-1] - 1
            )
            error_calc(
                predicted_id,
                was_rejected,
                self.g_unique_ids_calib,
                self.probe_unique_ids_calib,
            )
            true_pred_label = np.zeros(self.probe_unique_ids_calib.shape[0])
            true_pred_label[error_calc.is_seen] = error_calc.true_accept_true_ident
            true_pred_label[~error_calc.is_seen] = error_calc.true_reject
            if self.calibrate_gallery_unc == "logistic":
                conf_gallery = np.max(self.all_classes_log_prob, axis=-1)
                conf_gallery_calib = np.exp(
                    np.max(self.all_classes_log_prob_calib, axis=-1)
                )
                conf_gallery = self.calibrate_scf_unc(
                    conf_gallery,
                    conf_gallery_calib,
                    true_pred_label,
                    verbose=False,
                )
            else:
                conf_gallery_calib = np.exp(
                    np.max(self.all_classes_log_prob_calib, axis=-1)
                )
                conf_gallery = self.beta_calib(
                    conf_gallery, conf_gallery_calib, true_pred_label
                )

        if self.aggregation == "sum":
            comb_conf = conf_gallery * (1 - self.alpha) + data_conf * self.alpha
        elif self.aggregation == "product":
            comb_conf = (conf_gallery ** (1 - self.alpha)) * (data_conf**self.alpha)
        else:
            raise ValueError
        if self.log_dir is not None:
            np.savez(
                Path(self.log_dir) / f"scf.npz",
                kappa=comb_conf,
            )
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
            self.log_alpha = np.log(self.alpha)
        elif self.class_model == "power":
            log_alpha_power = (
                loggamma(d / 2)
                + loggamma(d - 1 + 2 * self.kappa)
                - self.kappa * np.log(2)
                - loggamma(d - 1 + self.kappa)
                - loggamma(d / 2 + self.kappa)
            )
            # self.alpha = np.exp(log_alpha_power)
            self.log_alpha = log_alpha_power
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
            logit_sum + np.exp((1 / T) * (self.log_alpha + np.log(self.beta)))
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
