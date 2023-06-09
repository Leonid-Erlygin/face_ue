from dataclasses import dataclass
from functools import partial
from typing import Any, Literal
import numpy as np
from pathlib import Path
from joblib import Memory

from .tools.extract import extract_IJB_data_11, extract_gallery_prob_data
from .tools.embeddings import get_embeddings, process_embeddings
from ..template_pooling_strategies import AbstractTemplatePooling


@dataclass
class Query1N:
    probe_feats: np.ndarray
    probe_unc: np.ndarray
    gallery_feats: np.ndarray
    gallery_unc: np.ndarray
    probe_ids: np.ndarray
    gallery_ids: np.ndarray

    @property
    def p(self):
        return self.probe_ids.shape[0]

    @property
    def g(self):
        return self.gallery_ids.shape[0]

    @property
    def d(self):
        return self.probe_feats.shape[1]

    @property
    def G(self):
        return self.gallery_feats

    @property
    def P(self):
        return self.probe_feats

    @property
    def dG(self):
        return self.gallery_unc

    @property
    def dP(self):
        return self.probe_unc


@Memory("/app/cache/dataloader/").cache
def dataloader_1N_function(
    data_path: str,
    subset: Literal["IJBB", "IJBC"],
    features: Literal["scf", "pfe"],
    template_pooling: AbstractTemplatePooling,
    use_detector_score: bool,
    gallery_idx: int,
):
    restore_embs = f"/app/cache/features/{features}_ijb_embs_{subset}.npz"

    (
        templates,
        medias,
        p1,
        p2,
        label,
        img_names,
        landmarks,
        face_scores,
    ) = extract_IJB_data_11(data_path, subset)

    print(">>>> Reload embeddings from:", restore_embs)
    aa = np.load(restore_embs)

    if "embs" in aa and "unc" in aa:
        embs = aa["embs"]
        embs_f = []
        unc = aa["unc"]
    else:
        print("ERROR: %s NOT containing embs / unc" % restore_embs)
        exit(1)
    print(">>>> Done.")
    data_path, subset = data_path, subset
    face_scores = face_scores.astype(embs.dtype)

    use_detector_score = use_detector_score
    (
        templates,
        medias,
        p1,
        p2,
        label,
        img_names,
        landmarks,
        face_scores,
    ) = extract_IJB_data_11(data_path, subset)

    (
        g1_templates,
        g1_ids,
        g2_templates,
        g2_ids,
        probe_mixed_templates,
        probe_mixed_ids,
    ) = extract_gallery_prob_data(data_path, subset)

    img_input_feats = process_embeddings(
        embs,
        embs_f,
        use_flip_test=False,
        use_norm_score=False,
        use_detector_score=use_detector_score,
        face_scores=face_scores,
    )

    # calculate probe features
    (
        probe_mixed_templates_feature,
        probe_template_unc,
        probe_mixed_unique_templates,
        probe_mixed_unique_subject_ids,
    ) = template_pooling(
        img_input_feats,
        unc,
        templates,
        medias,
        probe_mixed_templates,
        probe_mixed_ids,
    )

    templates, ids = [[g1_templates, g1_ids], [g2_templates, g2_ids]][gallery_idx]

    (
        templates_feature,
        template_unc,
        unique_templates,
        unique_ids,
    ) = template_pooling(
        img_input_feats,
        unc,
        templates,
        medias,
        templates,
        ids,
    )

    return Query1N(
        probe_mixed_templates_feature,
        probe_template_unc,
        templates_feature,
        template_unc,
        probe_mixed_unique_subject_ids,
        unique_ids,
    )


class DataLoader1N:
    def __init__(
        self,
        data_path,
        subset,
        features,
        template_pooling: AbstractTemplatePooling,
        use_detector_score: bool,
        use_two_galleries: bool,
    ) -> None:
        self.caller = partial(
            dataloader_1N_function,
            data_path,
            subset,
            features,
            template_pooling,
            use_detector_score
        )
        self.use_two_galleries = use_two_galleries
        
        self.name = "@".join(["subset", "features", template_pooling.name])

    def __iter__(self):
        for gallery_idx in [0, 1]:
            yield self.caller(gallery_idx)
            
            if not self.use_two_galleries:
                break
