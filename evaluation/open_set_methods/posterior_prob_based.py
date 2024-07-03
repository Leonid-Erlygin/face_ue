import numpy as np
from .base_method import OpenSetMethod
from evaluation.metrics import FrrFarIdent
from evaluation.test_datasets import FaceRecogntioniDataset
from evaluation.template_pooling_strategies import PoolingDefault
from evaluation.embeddings import process_embeddings
from evaluation.face_recognition_test import Face_Fecognition_test

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
        all_classes_log_prob = torch.mean(all_classes_log_prob, dim=1).numpy()
        was_rejected = np.argmax(all_classes_log_prob, axis=-1) == (
            all_classes_log_prob.shape[-1] - 1
        )
        far = np.mean(was_rejected[~self.is_seen] == False)
        print(f"Found kappa {np.round(kappa,4)} for far {far}")
        return -np.abs(far - self.target_far)


class PosteriorProbability(OpenSetMethod):
    def __init__(
        self,
        distance_function: Any,
        beta: float,
        uncertainty_type: str,
        alpha: float,
        aggregation: str,
        class_model: str,
        T: Union[float, List[float]],
        T_data_unc: float,
        far: float = None,
        gallery_kappa: float = None,
        calibrate_unc: bool = False,
        calibrate_by_false_reject: bool = False,
        calibrate_gallery_unc: bool = False,
        calibration_set: FaceRecogntioniDataset = None,
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
        self.gallery_kappa = gallery_kappa
        self.T_data_unc = T_data_unc
        self.calibrate_unc = calibrate_unc
        self.calibrate_by_false_reject = calibrate_by_false_reject
        self.calibrate_gallery_unc = calibrate_gallery_unc
        self.calibration_set = calibration_set

        if calibration_set is None:
            return
        # prepare calibration set
        embeddings_path = (
            Path(calibration_set.dataset_path)
            / f"embeddings/scf_embs_{calibration_set.dataset_name}.npz"
        )
        aa = np.load(embeddings_path)

        embs = aa["embs"]
        unc = aa["unc"]

        image_input_feats = process_embeddings(
            embs,
            [],
            use_flip_test=False,
            use_norm_score=False,
            use_detector_score=False,
            face_scores=calibration_set.face_scores,
        )

        # pool using average pooling
        used_galleries = ["g1"]
        gallery_name = used_galleries[0]
        self.gallery_pooled_templates_calib = {
            gallery_name: {} for gallery_name in used_galleries
        }
        self.probe_pooled_templates_calib = {
            gallery_name: {} for gallery_name in used_galleries
        }
        template_subsets_path = (
            "/app/cache/template_cache_new"
            / Path("scf")
            / f"name_calib_template_subsets_PoolingDefault_{calibration_set.dataset_name}"
        )
        template_pool_path = (
            "/app/cache/template_cache_new"
            / Path("scf")
            / f"name_template_pool_gallery-PoolingDefault_probe-PoolingDefault_{calibration_set.dataset_name}"
        )
        template_subsets_path.mkdir(parents=True, exist_ok=True)
        template_pool_path.mkdir(parents=True, exist_ok=True)
        if (template_subsets_path / "probe.npz").is_file():
            data = np.load(template_subsets_path / "probe.npz")
            probe_features = data["probe_features"]
            probe_unc = data["probe_unc"]
            probe_templates_sorted = data["probe_templates_sorted"]
            probe_medias = data["probe_medias"]
            probe_subject_ids_sorted = data["probe_subject_ids_sorted"]
        else:
            (
                probe_features,
                probe_unc,
                probe_medias,
                probe_templates_sorted,
                probe_subject_ids_sorted,
            ) = Face_Fecognition_test.get_template_subsets(
                image_input_feats,
                unc,
                calibration_set.templates,
                calibration_set.medias,
                calibration_set.probe_ids,
                calibration_set.probe_templates,
            )
            np.savez(
                template_subsets_path / "probe.npz",
                probe_features=probe_features,
                probe_unc=probe_unc,
                probe_medias=probe_medias,
                probe_templates_sorted=probe_templates_sorted,
                probe_subject_ids_sorted=probe_subject_ids_sorted,
            )

        gallery_templates = getattr(self.calibration_set, f"{gallery_name}_templates")
        gallery_subject_ids = getattr(self.calibration_set, f"{gallery_name}_ids")
        if (template_subsets_path / f"gallery_{gallery_name}.npz").is_file():
            data = np.load(template_subsets_path / f"gallery_{gallery_name}.npz")
            gallery_features = data["gallery_features"]
            gallery_unc = data["gallery_unc"]
            gallery_medias = data["gallery_medias"]
            gallery_templates_sorted = data["gallery_templates_sorted"]
            gallery_subject_ids_sorted = data["gallery_subject_ids_sorted"]
        else:
            (
                gallery_features,
                gallery_unc,
                gallery_medias,
                gallery_templates_sorted,
                gallery_subject_ids_sorted,
            ) = Face_Fecognition_test.get_template_subsets(
                image_input_feats,
                unc,
                calibration_set.templates,
                calibration_set.medias,
                gallery_subject_ids,
                gallery_templates,
            )
            np.savez(
                template_subsets_path / f"gallery_{gallery_name}.npz",
                gallery_features=gallery_features,
                gallery_unc=gallery_unc,
                gallery_medias=gallery_medias,
                gallery_templates_sorted=gallery_templates_sorted,
                gallery_subject_ids_sorted=gallery_subject_ids_sorted,
            )
        kappa = np.exp(gallery_unc)
        average_pooling = PoolingDefault()
        pooled_data = average_pooling(
            gallery_features,
            kappa,
            gallery_templates_sorted,
            gallery_medias,
        )
        self.gallery_pooled_templates_calib[gallery_name] = {
            "template_pooled_features": pooled_data[0],
            "template_pooled_data_unc": pooled_data[1],
            "template_subject_ids_sorted": gallery_subject_ids_sorted,
        }

        # pool probe
        probe_kappa = np.exp(probe_unc)
        if (template_pool_path / f"probe_{gallery_name}.npz").is_file():
            data = np.load(template_pool_path / f"probe_{gallery_name}.npz")
            probe_pooled_data = (
                data["template_pooled_features"],
                data["template_pooled_data_unc"],
            )
        else:
            average_pooling = PoolingDefault()
            probe_pooled_data = average_pooling(
                probe_features,
                probe_kappa,
                probe_templates_sorted,
                probe_medias,
            )
            np.savez(
                template_pool_path / f"probe_{gallery_name}.npz",
                template_pooled_features=probe_pooled_data[0],
                template_pooled_data_unc=probe_pooled_data[1],
            )

        self.probe_pooled_templates_calib[gallery_name] = {
            "template_pooled_features": probe_pooled_data[0],
            "template_pooled_data_unc": probe_pooled_data[1],
            "template_subject_ids_sorted": probe_subject_ids_sorted,
        }

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
        T = self.T
        # find kappa
        is_seen = np.isin(probe_unique_ids, g_unique_ids)

        if self.gallery_kappa is None:
            kappa_low = 300
            kappa_high = 1800
            max_iter = 15
            eps = 0.005
            far_loss_func = FarLossCalc(
                self.beta, T, self.class_model, self.far, is_seen, similarity_matrix
            )
            self.gallery_kappa = golden_selection_search(
                kappa_high, kappa_low, eps, max_iter, far_loss_func
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
            kappa_high = 1800
            max_iter = 15
            eps = 0.005
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
        return -np.abs(far - target_far)

    def predict(self):
        self.predicted_id = np.argmax(self.all_classes_log_prob[:, :-1], axis=-1)
        self.was_rejected = np.argmax(self.all_classes_log_prob, axis=-1) == (
            self.all_classes_log_prob.shape[-1] - 1
        )
        return self.predicted_id, self.was_rejected

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
                self.g_unique_ids_calib,
                self.probe_unique_ids_calib,
            )
            true_pred_label = np.zeros(self.probe_unique_ids_calib.shape[0])
            if self.calibrate_by_false_reject:
                true_pred_label[~error_calc.is_seen] = True
                true_pred_label[error_calc.is_seen] = error_calc.true_accept_true_ident
            else:
                true_pred_label[error_calc.is_seen] = error_calc.true_accept_true_ident
                true_pred_label[~error_calc.is_seen] = error_calc.true_reject
            m = torch.nn.Parameter(
                torch.tensor(0.5, dtype=torch.float64), requires_grad=True
            )
            gamma = torch.nn.Parameter(
                torch.tensor(1.0, dtype=torch.float64), requires_grad=True
            )
            # norm data unc to prevent saturation
            data_uncertainty_norm = self.data_uncertainty / 500
            data_uncertainty_norm_calib = self.data_uncertainty_calib[:, 0] / 500

            # train logistic calibration
            def prob_compute(conf, params):
                return torch.special.expit(params[1] * (conf - params[0]))

            self.train_calibration(
                data_uncertainty_norm_calib,
                true_pred_label,
                prob_compute,
                [m, gamma],
                lr=0.01,
                iter_num=100,
                verbose=False,
            )
            data_conf = prob_compute(
                torch.tensor(data_uncertainty_norm, dtype=torch.float32),
                [m.data, gamma.data],
            ).numpy()
        else:
            data_conf = self.data_uncertainty
        if self.calibrate_gallery_unc:
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

            conf_gallery_calib = np.exp(
                np.max(self.all_classes_log_prob_calib, axis=-1)
            )
            self.train_calibration(
                conf_gallery_calib,
                true_pred_label,
                prob_compute,
                [a, b, c],
                lr=0.1,
                iter_num=100,
                verbose=False,
            )
            conf_gallery = prob_compute(
                torch.tensor(conf_gallery, dtype=torch.float32),
                [a.data, b.data, c.data],
            ).numpy()

        if self.aggregation == "sum":
            comb_conf = conf_gallery * (1 - self.alpha) + data_conf * self.alpha
        elif self.aggregation == "product":
            comb_conf = (conf_gallery ** (1 - self.alpha)) * (data_conf**self.alpha)
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
