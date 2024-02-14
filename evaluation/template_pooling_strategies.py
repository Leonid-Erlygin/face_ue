from typing import Tuple
from abc import ABC
from typing import Any
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize
import scipy
import geotorch
from scipy.special import ive, hyp0f1, loggamma
from evaluation.open_set_methods.class_prob_models import GalleryParams
import torch


class AbstractTemplatePooling(ABC):
    def __call__(
        self,
        img_feats: np.ndarray,
        raw_unc: np.ndarray,
        templates: np.ndarray,
        medias: np.ndarray,
        choose_templates: np.ndarray,
        choose_ids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


class PoolingDefault(AbstractTemplatePooling):
    def __call__(
        self,
        img_feats: np.ndarray,
        raw_unc: np.ndarray,
        templates: np.ndarray,
        medias: np.ndarray,
    ):
        # templates = np.sort(templates) # WAS THIS A PROBLEM?
        ## here we assume that after default pooling uncertainty are not used
        unique_templates, indices = np.unique(templates, return_index=True)
        # unique_templates, indices = np.unique(choose_templates, return_index=True)
        # unique_subjectids = choose_ids[indices]

        template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))
        for count_template, uqt in tqdm(
            enumerate(unique_templates),
            "Extract template feature",
            total=len(unique_templates),
        ):
            (ind_t,) = np.where(templates == uqt)
            face_norm_feats = img_feats[ind_t]
            face_medias = medias[ind_t]
            unique_medias, unique_media_counts = np.unique(
                face_medias, return_counts=True
            )
            media_norm_feats = []
            for u, ct in zip(unique_medias, unique_media_counts):
                (ind_m,) = np.where(face_medias == u)
                if ct == 1:
                    media_norm_feats += [face_norm_feats[ind_m]]
                else:  # image features from the same video will be aggregated into one feature
                    media_norm_feats += [
                        np.mean(face_norm_feats[ind_m], 0, keepdims=True)
                    ]
            media_norm_feats = np.concatenate(media_norm_feats)
            template_feats[count_template] = np.sum(media_norm_feats, axis=0)

        template_norm_feats = normalize(template_feats)
        return (
            template_norm_feats,
            np.zeros((len(unique_templates), 1)),
        )


class PoolingMonteCarlo(AbstractTemplatePooling):
    def __init__(self, probability_model: Any, train_T: bool) -> None:
        super().__init__()
        self.probability_model = probability_model
        self.train_T = train_T

    def __call__(
        self,
        img_featues: np.ndarray,
        kappa: np.ndarray,
        template_ids: np.ndarray,
        medias: np.ndarray,
    ):
        unique_templates, indices, counts = np.unique(
            template_ids, return_index=True, return_counts=True
        )
        # initialize class mean
        init_means = np.array(
            [
                np.mean(img_featues[np.where(template_ids == uqt)], axis=0)
                for uqt in unique_templates
            ]
        )
        init_kappa = [400.0]
        init_kappas = [init_kappa] * len(unique_templates)
        init_T = 1.0

        # train mean and concentration
        gallery_params = GalleryParams(
            init_means, init_kappas, init_T, train_T=self.train_T
        )
        target_classes = []
        for c, count in enumerate(counts):
            target_classes.extend([c] * count)

        target_classes = torch.tensor(target_classes)
        geotorch.sphere(gallery_params, "gallery_means")
        optimizer = torch.optim.Adam(gallery_params.parameters(), lr=0.001)

        num_steps = 450

        nll_loss = torch.nn.NLLLoss()

        for iter in range(num_steps):
            optimizer.zero_grad()
            # compute nll loss
            log_probs = self.probability_model(
                img_featues,
                kappa,
                gallery_params.gallery_means,
                gallery_params.gallery_kappas,
                gallery_params.T,
            )[:, :, :-1]
            probs = torch.exp(log_probs)
            mean_probs = torch.mean(probs, axis=1)
            log_probs_new = torch.log(mean_probs)
            loss = nll_loss(log_probs_new, target_classes)
            print(gallery_params.gallery_means)
            print(torch.norm(gallery_params.gallery_means, dim=-1))
            print(gallery_params.gallery_kappas)
            print(torch.max(mean_probs))
            print(log_probs_new)
            print(f"Iteration {iter}, Loss: {loss.item()}")
            loss.backward()
            optimizer.step()

        # return template_norm_feats, templates_kappa


class PoolingSCF(AbstractTemplatePooling):
    def __call__(
        self,
        img_feats: np.ndarray,
        kappa: np.ndarray,
        templates: np.ndarray,
        medias: np.ndarray,
    ):
        # templates = np.sort(templates)
        unique_templates, indices = np.unique(
            templates, return_index=True
        )  # unique_templates, indices = np.unique(choose_templates, return_index=True)
        # unique_subjectids = choose_ids[indices]

        template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))
        templates_kappa = np.zeros((len(unique_templates), kappa.shape[1]))

        for count_template, uqt in tqdm(
            enumerate(unique_templates),
            "Extract template feature",
            total=len(unique_templates),
        ):
            (ind_t,) = np.where(templates == uqt)
            face_norm_feats = img_feats[ind_t]
            conf_template = kappa[ind_t]
            face_medias = medias[ind_t]
            unique_medias, unique_media_counts = np.unique(
                face_medias, return_counts=True
            )
            media_norm_feats = []
            kappa_in_template = []
            for u, ct in zip(unique_medias, unique_media_counts):
                (ind_m,) = np.where(face_medias == u)
                if ct == 1:
                    media_norm_feats += [face_norm_feats[ind_m]]
                    kappa_in_template += [conf_template[ind_m]]
                else:  # image features from the same video will be aggregated into one feature
                    kappa_in_template += [
                        np.mean(conf_template[ind_m], 0, keepdims=True)
                    ]
                    media_norm_feats += [
                        np.sum(
                            face_norm_feats[ind_m] * conf_template[ind_m],
                            axis=0,
                            keepdims=True,
                        )
                        / np.sum(conf_template[ind_m])
                    ]
            media_norm_feats = np.concatenate(media_norm_feats)
            kappa_in_template = np.concatenate(kappa_in_template)

            template_feats[count_template] = np.sum(
                media_norm_feats * kappa_in_template, axis=0
            ) / np.sum(kappa_in_template)
            final_kappa_in_template = np.mean(kappa_in_template, axis=0)

            templates_kappa[count_template] = final_kappa_in_template

        template_norm_feats = normalize(template_feats)
        return template_norm_feats, templates_kappa


class PoolingProb(AbstractTemplatePooling):
    def __call__(
        self,
        img_feats: np.ndarray,
        conf: np.ndarray,
        data_conf: np.ndarray,
        templates: np.ndarray,
        medias: np.ndarray,
    ):
        # here we aggregate probe templates using conf and also return mean data conf to get pure aggregated scf conf
        templates = np.sort(templates)
        assert templates.shape[0] == img_feats.shape[0]
        assert templates.shape[0] == conf.shape[0]
        assert templates.shape[0] == data_conf.shape[0]
        assert templates.shape[0] == medias.shape[0]

        unique_templates, indices = np.unique(templates, return_index=True)
        conf = conf[:, np.newaxis]
        assert conf.shape == data_conf.shape

        template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))
        template_feats_data_conf = np.zeros((len(unique_templates), img_feats.shape[1]))
        templates_conf = np.zeros((len(unique_templates), conf.shape[1]))
        templates_data_conf = np.zeros((len(unique_templates), data_conf.shape[1]))

        for count_template, uqt in tqdm(
            enumerate(unique_templates),
            "Extract template feature",
            total=len(unique_templates),
        ):
            (ind_t,) = np.where(templates == uqt)
            face_norm_feats = img_feats[ind_t]
            data_conf_template = data_conf[ind_t]
            conf_template = conf[ind_t]
            face_medias = medias[ind_t]
            unique_medias, unique_media_counts = np.unique(
                face_medias, return_counts=True
            )
            media_norm_feats = []
            media_norm_feats_data_conf = []
            conf_in_template = []
            data_conf_in_template = []

            for u, ct in zip(unique_medias, unique_media_counts):
                (ind_m,) = np.where(face_medias == u)
                if ct == 1:
                    media_norm_feats += [face_norm_feats[ind_m]]
                    media_norm_feats_data_conf += [face_norm_feats[ind_m]]
                    conf_in_template += [conf_template[ind_m]]
                    data_conf_in_template += [data_conf_template[ind_m]]
                else:  # image features from the same video will be aggregated into one feature
                    conf_in_template += [
                        np.mean(conf_template[ind_m], 0, keepdims=True)
                    ]
                    data_conf_in_template += [
                        np.mean(data_conf_template[ind_m], 0, keepdims=True)
                    ]

                    media_norm_feats += [
                        np.sum(
                            face_norm_feats[ind_m] * conf_template[ind_m],
                            axis=0,
                            keepdims=True,
                        )
                        / np.sum(conf_template[ind_m])
                    ]
                    media_norm_feats_data_conf += [
                        np.sum(
                            face_norm_feats[ind_m] * data_conf_template[ind_m],
                            axis=0,
                            keepdims=True,
                        )
                        / np.sum(data_conf_template[ind_m])
                    ]
            media_norm_feats = np.concatenate(media_norm_feats)
            media_norm_feats_data_conf = np.concatenate(media_norm_feats_data_conf)
            data_conf_in_template = np.concatenate(data_conf_in_template)
            conf_in_template = np.concatenate(conf_in_template)

            template_feats[count_template] = np.sum(
                media_norm_feats * conf_in_template, axis=0
            ) / np.sum(conf_in_template)

            template_feats_data_conf[count_template] = np.sum(
                media_norm_feats_data_conf * data_conf_in_template, axis=0
            ) / np.sum(data_conf_in_template)

            final_data_conf_in_template = np.mean(data_conf_in_template, axis=0)

            templates_data_conf[count_template] = final_data_conf_in_template

        template_norm_feats = normalize(template_feats)
        return template_norm_feats, templates_data_conf


class PoolingPFEHarmonicMean(AbstractTemplatePooling):
    def __call__(
        self,
        img_feats: np.ndarray,
        raw_unc: np.ndarray,
        templates: np.ndarray,
        medias: np.ndarray,
    ):
        unique_templates, indices = np.unique(templates, return_index=True)
        # unique_templates, indices = np.unique(choose_templates, return_index=True)
        # unique_subjectids = choose_ids[indices]

        # compute harmonic mean of unc
        # raise NotImplemented
        # need to use aggregation as in Eqn. (6-7) and min variance pool, when media type is the same
        # across pooled images
        sigma_sq = np.exp(raw_unc)
        conf = 1 / scipy.stats.hmean(sigma_sq, axis=1, keepdims=True)

        template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))
        templates_sigma_sq = np.zeros((len(unique_templates), sigma_sq.shape[1]))

        for count_template, uqt in tqdm(
            enumerate(unique_templates),
            "Extract template feature",
            total=len(unique_templates),
        ):
            (ind_t,) = np.where(templates == uqt)
            face_norm_feats = img_feats[ind_t]
            conf_template = conf[ind_t]
            raw_sigma_sq_in_template = sigma_sq[ind_t]
            face_medias = medias[ind_t]
            unique_medias, unique_media_counts = np.unique(
                face_medias, return_counts=True
            )
            media_norm_feats = []
            conf_in_template = []
            sigma_sq_in_template = []
            for u, ct in zip(unique_medias, unique_media_counts):
                (ind_m,) = np.where(face_medias == u)
                if ct == 1:
                    media_norm_feats += [face_norm_feats[ind_m]]
                    conf_in_template += [conf_template[ind_m]]
                    sigma_sq_in_template += [raw_sigma_sq_in_template[ind_m]]
                else:  # image features from the same video will be aggregated into one feature
                    media_var = raw_sigma_sq_in_template[ind_m]
                    result_media_variance = np.min(media_var, 0, keepdims=True)
                    # result_media_variance = 1 / np.sum(1 / media_var, axis=0, keepdims=True)
                    sigma_sq_in_template += [result_media_variance]

                    conf_in_template += [
                        np.mean(conf_template[ind_m], 0, keepdims=True)
                    ]
                    media_norm_feats += [
                        np.sum(
                            face_norm_feats[ind_m] * conf_template[ind_m],
                            axis=0,
                            keepdims=True,
                        )
                        / np.sum(conf_template[ind_m])
                    ]

            media_norm_feats = np.concatenate(media_norm_feats)
            conf_in_template = np.concatenate(conf_in_template)

            template_feats[count_template] = np.sum(
                media_norm_feats * conf_in_template, axis=0
            ) / np.sum(conf_in_template)

            sigma_sq_in_template = np.concatenate(sigma_sq_in_template)
            pfe_template_variance = 1 / np.sum(
                1 / sigma_sq_in_template, axis=0
            )  # Eqn. (7) https://ieeexplore.ieee.org/document/9008376
            assert False, "check this aggregation"
            templates_sigma_sq[count_template] = pfe_template_variance

        template_norm_feats = normalize(template_feats)
        return (
            template_norm_feats,
            templates_sigma_sq,
            unique_templates,
        )


class PoolingPFE(AbstractTemplatePooling):
    def __call__(
        self,
        img_feats: np.ndarray,
        raw_unc: np.ndarray,
        templates: np.ndarray,
        medias: np.ndarray,
    ):
        unique_templates, indices = np.unique(templates, return_index=True)
        # unique_templates, indices = np.unique(choose_templates, return_index=True)
        # unique_subjectids = choose_ids[indices]

        # compute harmonic mean of unc
        # raise NotImplemented
        # need to use aggregation as in Eqn. (6-7) and min variance pool, when media type is the same
        # across pooled images
        sigma_sq = np.exp(raw_unc)

        # template_feats = np.zeros((len(unique_templates), img_feats.shape[1]), dtype=img_feats.dtype)
        template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))
        templates_sigma_sq = np.zeros((len(unique_templates), raw_unc.shape[1]))
        for count_template, uqt in tqdm(
            enumerate(unique_templates),
            "Extract template feature",
            total=len(unique_templates),
        ):
            (ind_t,) = np.where(templates == uqt)
            face_norm_feats = img_feats[ind_t]
            raw_sigma_sq_in_template = sigma_sq[ind_t]
            face_medias = medias[ind_t]
            unique_medias, unique_media_counts = np.unique(
                face_medias, return_counts=True
            )
            media_norm_feats = []
            sigma_sq_in_template = []
            for u, ct in zip(unique_medias, unique_media_counts):
                (ind_m,) = np.where(face_medias == u)
                if ct == 1:
                    media_norm_feats += [face_norm_feats[ind_m]]
                    sigma_sq_in_template += [raw_sigma_sq_in_template[ind_m]]
                else:  # image features from the same video will be aggregated into one feature
                    # here we pool variance by taking minimum value
                    media_var = raw_sigma_sq_in_template[ind_m]
                    result_media_variance = np.min(media_var, 0, keepdims=True)
                    # result_media_variance = 1 / np.sum(1 / media_var, axis=0, keepdims=True)
                    sigma_sq_in_template += [result_media_variance]
                    media_norm_feats += [
                        np.sum(
                            (face_norm_feats[ind_m] / media_var),
                            axis=0,
                            keepdims=True,
                        )
                        * result_media_variance
                    ]
            media_norm_feats = np.concatenate(media_norm_feats)
            sigma_sq_in_template = np.concatenate(sigma_sq_in_template)

            pfe_template_variance = 1 / np.sum(
                1 / sigma_sq_in_template, axis=0
            )  # Eqn. (7) https://ieeexplore.ieee.org/document/9008376
            template_feats[count_template] = (
                np.sum(media_norm_feats / sigma_sq_in_template, axis=0)  # Eqn. (6)
                * pfe_template_variance
            )
            final_template_conf = pfe_template_variance

            templates_sigma_sq[count_template] = final_template_conf

        template_norm_feats = normalize(template_feats)
        return (
            template_norm_feats,
            templates_sigma_sq,
            unique_templates,
        )
