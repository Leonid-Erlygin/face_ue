from dataclasses import dataclass
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


class Dataloader1N:
    def __init__(
        self,
        data_path: str,
        subset,
        force_reload,
        restore_embs,
        template_pooling_strategy: AbstractTemplatePooling,
        use_detector_score,
        use_two_galleries,
        recompute_template_pooling,
        features,
    ):
        self.use_two_galleries = use_two_galleries
        self.recompute_template_pooling = recompute_template_pooling
        self.features = features
        (
            templates,
            medias,
            p1,
            p2,
            label,
            img_names,
            landmarks,
            face_scores,
        ) = extract_IJB_data_11(data_path, subset, force_reload=force_reload)
        
        print(">>>> Reload embeddings from:", restore_embs)
        aa = np.load(restore_embs)

        if "embs" in aa and "unc" in aa:
            self.embs = aa["embs"]
            self.embs_f = []
            self.unc = aa["unc"]
        else:
            print("ERROR: %s NOT containing embs / unc" % restore_embs)
            exit(1)
        print(">>>> Done.")
        self.data_path, self.subset, self.force_reload = data_path, subset, force_reload
        self.templates, self.medias, self.p1, self.p2, self.label = (
            templates,
            medias,
            p1,
            p2,
            label,
        )
        self.face_scores = face_scores.astype(self.embs.dtype)

        memory = Memory(Path("/app/cache/template_cache"))
        self.template_pooling_strategy: AbstractTemplatePooling = memory.cache(template_pooling_strategy.__call__) # type: ignore
        self.use_detector_score = use_detector_score

    @property
    def name(self):
        return "@".join([self.template_pooling_strategy.name, self.subset])

    def load_1N_query(self, npoints=100):
        (
            g1_templates,
            g1_ids,
            g2_templates,
            g2_ids,
            probe_mixed_templates,
            probe_mixed_ids,
        ) = extract_gallery_prob_data(
            self.data_path, self.subset, force_reload=self.force_reload
        )

        img_input_feats = process_embeddings(
            self.embs,
            self.embs_f,
            use_flip_test=False,
            use_norm_score=False,
            use_detector_score=self.use_detector_score,
            face_scores=self.face_scores,
        )

        # calculate probe features
        (
            probe_mixed_templates_feature,
            probe_template_unc,
            probe_mixed_unique_templates,
            probe_mixed_unique_subject_ids,
        ) = self.template_pooling_strategy(
            img_input_feats,
            self.unc,
            self.templates,
            self.medias,
            probe_mixed_templates,
            probe_mixed_ids,
        )

        (
            g1_templates_feature,
            g1_template_unc,
            g1_unique_templates,
            g1_unique_ids,
        ) = self.template_pooling_strategy(
            img_input_feats,
            self.unc,
            self.templates,
            self.medias,
            g1_templates,
            g1_ids,
        )

        print(f"{g1_templates_feature.shape=}")  # (1772, 512)
        
        yield Query1N(
            probe_mixed_templates_feature,
            probe_template_unc,
            g1_templates_feature,
            g1_template_unc,
            probe_mixed_unique_subject_ids,
            g1_unique_ids,
        )

        if self.use_two_galleries:
            (
                g2_templates_feature,
                g2_template_unc,
                g2_unique_templates,
                g2_unique_ids,
            ) = self.template_pooling_strategy(
                img_input_feats,
                self.unc,
                self.templates,
                self.medias,
                g2_templates,
                g2_ids,
            )

            print(f"{g2_templates_feature.shape=}")  # (1759, 512)
        
            yield Query1N(
                probe_mixed_templates_feature,
                probe_template_unc,
                g2_templates_feature,
                g2_template_unc,
                probe_mixed_unique_subject_ids,
                g2_unique_ids,
            )
