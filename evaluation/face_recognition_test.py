import numpy as np
from pathlib import Path


from .embeddings import process_embeddings
from .image2template import image2template_feature
from .template_pooling_strategies import AbstractTemplatePooling
from .eval_functions.open_set_identification.abc import Abstract1NEval
from .test_datasets import FaceRecogntioniDataset


class Face_Fecognition_test:
    def __init__(
        self,
        evaluation_function: Abstract1NEval,
        test_dataset: FaceRecogntioniDataset,
        embeddings_path: str,
        template_pooling_strategy: AbstractTemplatePooling,
        use_detector_score,
        use_two_galleries,
        recompute_template_pooling,
        open_set_identification_metrics,
        closed_set_identification_metrics,
        verification_metrics,
        open_set_uncertainty_metrics,
    ):
        self.use_two_galleries = use_two_galleries
        self.test_dataset = test_dataset
        self.recompute_template_pooling = recompute_template_pooling
        self.open_set_identification_metrics = open_set_identification_metrics
        self.closed_set_identification_metrics = closed_set_identification_metrics
        self.verification_metrics = verification_metrics
        self.open_set_uncertainty_metrics = open_set_uncertainty_metrics

        print(">>>> Reload embeddings from:", embeddings_path)
        aa = np.load(embeddings_path)
        self.embeddings_path = embeddings_path
        self.embs = aa["embs"]
        self.embs_f = []
        self.unc = aa["unc"]

        if self.test_dataset.face_scores is not None:
            self.test_dataset.face_scores = self.test_dataset.face_scores.astype(
                self.embs.dtype
            )
        self.evaluation_function = evaluation_function
        self.template_pooling_strategy = template_pooling_strategy

        self.use_detector_score = use_detector_score

        # process embeddings
        self.image_input_feats = process_embeddings(
            self.embs,
            self.embs_f,
            use_flip_test=False,
            use_norm_score=False,
            use_detector_score=self.use_detector_score,
            face_scores=self.test_dataset.face_scores,
        )
        # pool templates

        self.pool_templates(cache_dir="/app/cache/template_cache")

    def pool_templates(self, cache_dir: str):
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        class_name = self.template_pooling_strategy.__class__.__name__
        pooled_templates_path = (
            cache_dir
            / f"template_pool_{class_name}_det_score_{str(self.use_detector_score)}_{self.test_dataset.dataset_name}.npz"
        )
        if pooled_templates_path.is_file() and self.recompute_template_pooling is False:
            pooled_data = np.load(pooled_templates_path)
            self.template_pooled_emb = pooled_data["template_pooled_emb"]
            self.template_pooled_unc = pooled_data["template_pooled_unc"]
            self.template_ids = pooled_data["template_ids"]
        else:
            (
                self.template_pooled_emb,
                self.template_pooled_unc,
                self.template_ids,
            ) = self.template_pooling_strategy(
                self.image_input_feats,
                self.unc,
                self.test_dataset.templates,
                self.test_dataset.medias,
            )
            np.savez(
                pooled_templates_path,
                template_pooled_emb=self.template_pooled_emb,
                template_pooled_unc=self.template_pooled_unc,
                template_ids=self.template_ids,
            )

    def get_template_subsets(
        self, choose_templates: np.ndarray, choose_ids: np.ndarray
    ):
        unique_templates, indices = np.unique(choose_templates, return_index=True)
        unique_subjectids = choose_ids[indices]

        templates_feature = np.zeros(
            (len(unique_templates), self.template_pooled_emb.shape[1])
        )
        template_unc = np.zeros(
            (len(unique_templates), self.template_pooled_unc.shape[1])
        )

        for count_template, uqt in enumerate(unique_templates):
            (ind_t,) = np.where(self.template_ids == uqt)
            templates_feature[count_template] = self.template_pooled_emb[ind_t]
            template_unc[count_template] = self.template_pooled_unc[ind_t]
        return templates_feature, template_unc, unique_subjectids

    def run_model_test_verification(
        self,
    ):
        scores = self.evaluation_function(
            self.template_pooled_emb,
            self.template_pooled_unc,
            self.template_ids,
            self.test_dataset.p1,
            self.test_dataset.p2,
        )

        metrics = {}
        for metric in self.verification_metrics:
            metrics.update(
                metric(
                    scores=scores,
                    labels=self.test_dataset.label,
                )
            )
        return metrics

    def run_model_test_closedset_identification(self):
        (
            g1_templates_feature,
            g1_template_unc,
            g1_unique_ids,
        ) = self.get_template_subsets(
            self.test_dataset.g1_templates, self.test_dataset.g1_ids
        )
        (
            probe_templates_feature,
            probe_template_unc,
            probe_unique_ids,
        ) = self.get_template_subsets(
            self.test_dataset.probe_templates, self.test_dataset.probe_ids
        )
        is_seen_g1 = np.isin(probe_unique_ids, g1_unique_ids)

        similarity, probe_score = self.evaluation_function(
            probe_templates_feature[is_seen_g1],
            probe_template_unc[is_seen_g1],
            g1_templates_feature,
            g1_template_unc,
        )

        metrics = {}
        for metric in self.closed_set_identification_metrics:
            metrics.update(
                metric(
                    probe_unique_ids[is_seen_g1],
                    g1_unique_ids,
                    similarity,
                    probe_score,
                )
            )

        if self.use_two_galleries and self.test_dataset.g2_templates.shape != ():
            (
                g2_templates_feature,
                g2_template_unc,
                g2_unique_ids,
            ) = self.get_template_subsets(
                self.test_dataset.g2_templates, self.test_dataset.g2_ids
            )
            print("g2_templates_feature:", g2_templates_feature.shape)  # (1759, 512)
            print(">>>> Gallery 2")
            is_seen_g2 = np.isin(probe_unique_ids, g2_unique_ids)

            similarity, probe_score = self.evaluation_function(
                probe_templates_feature[is_seen_g2],
                probe_template_unc[is_seen_g2],
                g2_templates_feature,
                g2_template_unc,
            )
            g2_metrics = {}
            for metric in self.closed_set_identification_metrics:
                g2_metrics.update(
                    metric(
                        probe_unique_ids[is_seen_g2],
                        g2_unique_ids,
                        similarity,
                        probe_score,
                    )
                )
            for key in g2_metrics.keys():
                if "cmc" in key:
                    metrics[key] = (metrics[key] + g2_metrics[key]) / 2

        return metrics

    def run_model_test_openset_identification(self):
        (
            g1_templates_feature,
            g1_template_unc,
            g1_unique_ids,
        ) = self.get_template_subsets(
            self.test_dataset.g1_templates, self.test_dataset.g1_ids
        )
        print("g1_templates_feature:", g1_templates_feature.shape)  # (1772, 512)

        (
            probe_templates_feature,
            probe_template_unc,
            probe_unique_ids,
        ) = self.get_template_subsets(
            self.test_dataset.probe_templates, self.test_dataset.probe_ids
        )
        print("probe_templates_feature:", probe_templates_feature.shape)  # (19593, 512)

        print("probe_unique_ids:", probe_unique_ids.shape)  # (19593,)

        print(">>>> Gallery 1")

        similarity, probe_score = self.evaluation_function(
            probe_templates_feature,
            probe_template_unc,
            g1_templates_feature,
            g1_template_unc,
        )

        # recognition metrics
        metrics = {}
        for metric in self.open_set_identification_metrics:
            metrics.update(
                metric(
                    probe_ids=probe_unique_ids,
                    gallery_ids=g1_unique_ids,
                    similarity=similarity,
                    probe_score=probe_score,
                )
            )

        # uncertainty metrics
        unc_metrics = {}
        for unc_metric in self.open_set_uncertainty_metrics:
            unc_metrics.update(
                unc_metric(
                    probe_ids=probe_unique_ids,
                    gallery_ids=g1_unique_ids,
                    similarity=similarity,
                    probe_score=probe_score,
                )
            )

        if self.use_two_galleries and self.test_dataset.g2_templates.shape != ():
            (
                g2_templates_feature,
                g2_template_unc,
                g2_unique_ids,
            ) = self.get_template_subsets(
                self.test_dataset.g2_templates, self.test_dataset.g2_ids
            )
            print("g2_templates_feature:", g2_templates_feature.shape)  # (1759, 512)
            print(">>>> Gallery 2")
            similarity, probe_score = self.evaluation_function(
                probe_templates_feature,
                probe_template_unc,
                g2_templates_feature,
                g2_template_unc,
            )
            g2_metrics = {}
            for metric in self.open_set_identification_metrics:
                g2_metrics.update(
                    metric(
                        probe_ids=probe_unique_ids,
                        gallery_ids=g2_unique_ids,
                        similarity=similarity,
                        probe_score=probe_score,
                    )
                )
            # uncertainty metrics
            g2_unc_metrics = {}
            for unc_metric in self.open_set_uncertainty_metrics:
                g2_unc_metrics.update(
                    unc_metric(
                        probe_ids=probe_unique_ids,
                        gallery_ids=g2_unique_ids,
                        similarity=similarity,
                        probe_score=probe_score,
                    )
                )
            for key in g2_metrics.keys():
                if "recalls" in key or "AUC" in key or "top" in key:
                    metrics[key] = (metrics[key] + g2_metrics[key]) / 2
            for key in g2_unc_metrics.keys():
                if "final_auc" in key:
                    unc_metrics[key] = (unc_metrics[key] + g2_unc_metrics[key]) / 2

        else:
            is_seen = np.isin(probe_unique_ids, g1_unique_ids)

        return metrics, unc_metrics
