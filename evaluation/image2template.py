import numpy as np
import tqdm
from sklearn.preprocessing import normalize


def image2template_feature(
    img_feats,
    raw_unc,
    templates,
    medias,
    choose_templates,
    choose_ids,
    conf_pool: bool,
    unc_type: str,
):
    if choose_templates is not None:  # 1:N
        unique_templates, indices = np.unique(choose_templates, return_index=True)
        unique_subjectids = choose_ids[indices]
    else:  # 1:1
        unique_templates = np.unique(templates)
        unique_subjectids = None
    if unc_type == "pfe":
        # compute harmonic mean of unc
        # raise NotImplemented
        # need to use aggregation as in Eqn. (6-7) and min variance pool, when media type is the same
        # across pooled images
        sigma_sq = np.exp(raw_unc)
        # conf = 1 / scipy.stats.hmean(raw_unc, axis=1)
        conf = sigma_sq
    elif unc_type == "scf":
        conf = np.exp(raw_unc)
    else:
        raise ValueError
    # template_feats = np.zeros((len(unique_templates), img_feats.shape[1]), dtype=img_feats.dtype)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))
    templates_conf = np.zeros((len(unique_templates), raw_unc.shape[1]))
    for count_template, uqt in tqdm(
        enumerate(unique_templates),
        "Extract template feature",
        total=len(unique_templates),
    ):
        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        conf_template = conf[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
        media_norm_feats = []
        template_conf = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
                template_conf += [conf_template[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                if conf_pool:
                    if unc_type == "scf":
                        template_conf += [
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
                    elif unc_type == "pfe":
                        # here we pool variance by taking minimum value
                        media_var = conf_template[ind_m]
                        result_media_variance = np.min(media_var, 0, keepdims=True)
                        # result_media_variance = 1 / np.sum(1 / media_var, axis=0, keepdims=True)
                        template_conf += [result_media_variance]
                        media_norm_feats += [
                            np.sum(
                                (face_norm_feats[ind_m] / media_var),
                                axis=0,
                                keepdims=True,
                            )
                            * result_media_variance
                        ]
                    else:
                        raise ValueError
                else:
                    media_norm_feats += [
                        np.mean(face_norm_feats[ind_m], 0, keepdims=True)
                    ]
        media_norm_feats = np.concatenate(media_norm_feats)
        template_conf = np.concatenate(template_conf)

        if conf_pool:
            if unc_type == "scf":
                template_feats[count_template] = np.sum(
                    media_norm_feats * template_conf[:, np.newaxis], axis=0
                ) / np.sum(template_conf)
                final_template_conf = np.mean(template_conf, axis=0)
            elif unc_type == "pfe":
                pfe_template_variance = 1 / np.sum(
                    1 / template_conf, axis=0
                )  # Eqn. (7) https://ieeexplore.ieee.org/document/9008376
                template_feats[count_template] = (
                    np.sum(media_norm_feats / template_conf, axis=0)  # Eqn. (6)
                    * pfe_template_variance
                )
                final_template_conf = pfe_template_variance
            else:
                raise ValueError

            templates_conf[count_template] = (
                final_template_conf  # np.mean(template_conf, axis=0)
            )
        else:
            template_feats[count_template] = np.sum(media_norm_feats, axis=0)

    template_norm_feats = normalize(template_feats)
    return template_norm_feats, templates_conf, unique_templates, unique_subjectids
