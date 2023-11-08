import numpy as np
from pathlib import Path
import warnings

from .embeddings import process_embeddings
from .image2template import image2template_feature
from .template_pooling_strategies import AbstractTemplatePooling
from .distance_functions.open_set_identification.abc import Abstract1NEval
from .test_datasets import FaceRecogntioniDataset


class Face_Fecognition_test:
    def __init__(
        self,
        task_type: str,
        method_name: str,
        recognition_method,
        sampler,
        distance_function: Abstract1NEval,
        test_dataset: FaceRecogntioniDataset,
        embeddings_path: str,
        gallery_template_pooling_strategy: AbstractTemplatePooling,
        probe_template_pooling_strategy: AbstractTemplatePooling,
        use_detector_score: bool,
        use_two_galleries: bool,
        recompute_template_pooling: bool,
        recognition_metrics: dict,
        uncertainty_metrics: dict,
    ):
        self.task_type = task_type
        self.method_name = method_name
        self.recognition_method = recognition_method
        self.use_two_galleries = use_two_galleries
        self.test_dataset = test_dataset
        self.recompute_template_pooling = recompute_template_pooling
        self.recognition_metrics = recognition_metrics
        self.uncertainty_metrics = uncertainty_metrics
        self.sampler = sampler
        self.distance_function = distance_function
        self.gallery_template_pooling_strategy = gallery_template_pooling_strategy
        self.probe_template_pooling_strategy = probe_template_pooling_strategy
        self.use_detector_score = use_detector_score

        # load nn embeddings
        aa = np.load(embeddings_path)
        self.embeddings_path = embeddings_path
        self.embs = aa["embs"]
        self.embs_f = []
        self.unc = aa["unc"]
        if self.test_dataset.face_scores is not None:
            self.test_dataset.face_scores = self.test_dataset.face_scores.astype(
                self.embs.dtype
            )

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

        self.pool_templates(cache_dir="/app/cache/template_cache_new")

        assert self.image_input_feats.shape[0] == self.unc.shape[0]
        assert self.image_input_feats.shape[0] == self.test_dataset.medias.shape[0]

    def pool_templates(self, cache_dir: str):
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # pooled_templates_path = (
        #     cache_dir
        #     / f"template_pool_{class_name}_det_score_{str(self.use_detector_score)}_{self.test_dataset.dataset_name}.npz"
        # )
        template_subsets_path = (
            cache_dir
            / f"template_subsets_{self.probe_template_pooling_strategy.__class__.__name__}_{self.test_dataset.dataset_name}"
        )
        template_subsets_path.mkdir(parents=True, exist_ok=True)
        pooled_templates_path = 1
        if self.recompute_template_pooling is False and pooled_templates_path.is_file():
            pooled_data = np.load(pooled_templates_path)
            self.template_pooled_emb = pooled_data["template_pooled_emb"]
            self.template_pooled_unc = pooled_data["template_pooled_unc"]
            self.template_ids = pooled_data["template_ids"]
        else:
            print("Pooling embeddings...")
            # first pool gallery templates
            # then use them to poll probe templates
            # probe templates should be pooled using appropriate gallery

            # 1. Pool each gallery separetly using gallery pooling strategy
            # 2. Probe templates shoold be pooled 2 ways: a) against gallery_1, b) against gallery_2
            # during recognition appropriate pooling of probe templates should be used, when testing
            # against two galleries
            used_galleries = ["g1"]
            if (
                self.use_two_galleries
                and self.test_dataset is not None
                and self.test_dataset.g2_templates.shape != ()
            ):
                used_galleries += ["g2"]

            self.gallery_pooled_templates = {
                gallery_name: {} for gallery_name in used_galleries
            }
            self.probe_pooled_templates = {
                gallery_name: {} for gallery_name in used_galleries
            }

            if (template_subsets_path / "probe.npz").is_file():
                data = np.load(template_subsets_path / "probe.npz")
                probe_features = data["probe_features"]
                probe_unc = data["probe_unc"]
                probe_medias = data["probe_medias"]
                probe_subject_ids_sorted = data["probe_subject_ids_sorted"]
            else:
                (
                    probe_features,
                    probe_unc,
                    probe_medias,
                    probe_subject_ids_sorted,
                ) = self.get_template_subsets(
                    self.test_dataset.probe_ids,
                    self.test_dataset.probe_templates,
                )
                np.savez(
                    template_subsets_path / "probe.npz",
                    probe_features=probe_features,
                    probe_unc=probe_unc,
                    probe_medias=probe_medias,
                    probe_subject_ids_sorted=probe_subject_ids_sorted,
                )
            assert probe_unc.shape[1] == 1  # working with scf unc
            probe_kappa = np.exp(probe_unc)

            for gallery_name in used_galleries:
                gallery_templates = getattr(
                    self.test_dataset, f"{gallery_name}_templates"
                )
                gallery_subject_ids = getattr(self.test_dataset, f"{gallery_name}_ids")
                if (template_subsets_path / f"gallery_{gallery_name}.npz").is_file():
                    data = np.load(
                        template_subsets_path / f"gallery_{gallery_name}.npz"
                    )
                    gallery_features = data["gallery_features"]
                    gallery_unc = data["gallery_unc"]
                    gallery_medias = data["gallery_medias"]
                    gallery_subject_ids_sorted = data["gallery_subject_ids_sorted"]
                else:
                    (
                        gallery_features,
                        gallery_unc,
                        gallery_medias,
                        gallery_subject_ids_sorted,
                    ) = self.get_template_subsets(
                        gallery_subject_ids, gallery_templates
                    )
                    np.savez(
                        template_subsets_path / f"gallery_{gallery_name}.npz",
                        gallery_features=gallery_features,
                        gallery_unc=gallery_unc,
                        gallery_medias=gallery_medias,
                        gallery_subject_ids_sorted=gallery_subject_ids_sorted,
                    )
                # 1. pool selected gallery templates
                assert gallery_unc.shape[1] == 1  # working with scf unc
                kappa = np.exp(gallery_unc)
                pooled_data = self.gallery_template_pooling_strategy(
                    gallery_features, kappa, gallery_templates, gallery_medias
                )
                self.gallery_pooled_templates[gallery_name] = {
                    "template_pooled_features": pooled_data[0],
                    "template_pooled_data_unc": pooled_data[1],
                    "template_unique_ids": pooled_data[2],
                    "template_subject_ids_sorted": gallery_subject_ids_sorted,
                }

                # 2. pool probe templates using 'gallery_name' gallery
                if (
                    "PoolingProb"
                    in self.probe_template_pooling_strategy.__class__.__name__
                ):
                    print("Computing similarity...")
                    similarity = self.distance_function(
                        probe_features[:, np.newaxis, :],
                        probe_kappa,
                        self.gallery_pooled_templates[gallery_name][
                            "template_pooled_features"
                        ],
                        self.gallery_pooled_templates[gallery_name][
                            "template_pooled_data_unc"
                        ],
                    )

                    self.recognition_method.setup(similarity)
                    predicted_unc = self.recognition_method.predict_uncertainty(
                        probe_kappa
                    )
                    probe_pooled_data = self.probe_template_pooling_strategy(
                        probe_features,
                        -predicted_unc,
                        probe_kappa,
                        self.test_dataset.probe_templates,
                        probe_medias,
                    )

                else:
                    probe_pooled_data = self.probe_template_pooling_strategy(
                        probe_features,
                        probe_kappa,
                        self.test_dataset.probe_templates,
                        probe_medias,
                    )

                self.probe_pooled_templates[gallery_name] = {
                    "template_pooled_features": probe_pooled_data[0],
                    "template_pooled_data_unc": probe_pooled_data[1],
                    "template_unique_ids": probe_pooled_data[2],
                    "template_subject_ids_sorted": probe_subject_ids_sorted,
                }

            # np.savez(
            #     pooled_templates_path,
            #     template_pooled_emb=self.template_pooled_emb,
            #     template_pooled_unc=self.template_pooled_unc,
            #     template_ids=self.template_ids,
            # )

    def get_template_subsets(
        self,
        subject_ids: np.ndarray,
        choose_templates: np.ndarray,
    ):
        assert subject_ids.shape[0] == choose_templates.shape[0]
        choose_templates_sort_id = np.argsort(choose_templates)
        choose_templates = choose_templates[choose_templates_sort_id]
        subject_ids_sorted = subject_ids[choose_templates_sort_id]
        unique_templates, indices = np.unique(choose_templates, return_index=True)
        unique_subject_ids = subject_ids_sorted[indices]
        templates_emb_subset = []
        template_uncertainty_subset = []
        medias_subset = []
        for uqt in unique_templates:
            ind_t = self.test_dataset.templates == uqt
            templates_emb_subset.append(self.image_input_feats[ind_t])
            template_uncertainty_subset.append(self.unc[ind_t])
            medias_subset.append(self.test_dataset.medias[ind_t])
        templates_emb_subset = np.concatenate(templates_emb_subset, axis=0)
        template_uncertainty_subset = np.concatenate(
            template_uncertainty_subset, axis=0
        )
        medias_subset = np.concatenate(medias_subset, axis=0)

        return (
            templates_emb_subset,
            template_uncertainty_subset,
            medias_subset,
            unique_subject_ids,
        )

    def predict_and_compute_metrics(self):
        return getattr(self, f"run_model_test_{self.task_type}")()

    def run_model_test_open_set_identification(self):
        used_galleries = ["g1"]
        if (
            self.use_two_galleries
            and self.test_dataset is not None
            and self.test_dataset.g2_templates.shape != ()
        ):
            used_galleries += ["g2"]

        metrics = {gallery: {} for gallery in used_galleries}
        unc_metrics = {gallery: {} for gallery in used_galleries}

        for gallery_name in used_galleries:
            # sample probe feature vectors
            probe_templates_feature = self.sampler(
                self.probe_pooled_templates[gallery_name]["template_pooled_features"],
                self.probe_pooled_templates[gallery_name]["template_pooled_data_unc"],
            )

            similarity = self.distance_function(
                probe_templates_feature,
                self.probe_pooled_templates[gallery_name]["template_pooled_data_unc"],
                self.gallery_pooled_templates[gallery_name]["template_pooled_features"],
                self.gallery_pooled_templates[gallery_name]["template_pooled_data_unc"],
            )

            # setup osr method and predict
            self.recognition_method.setup(similarity)
            predicted_id, was_rejected = self.recognition_method.predict()
            predicted_unc = self.recognition_method.predict_uncertainty(
                self.probe_pooled_templates[gallery_name][
                    "template_pooled_data_unc"
                ]  # need to use data uncertainty only
            )

            for metric in self.recognition_metrics[self.task_type]:
                metrics[gallery_name].update(
                    metric(
                        predicted_id=predicted_id,
                        was_rejected=was_rejected,
                        g_unique_ids=self.gallery_pooled_templates[gallery_name][
                            "template_subject_ids_sorted"
                        ],
                        probe_unique_ids=self.probe_pooled_templates[gallery_name][
                            "template_subject_ids_sorted"
                        ],
                    )
                )

            # compute uncertainty metrics

            for unc_metric in self.uncertainty_metrics[self.task_type]:
                unc_metrics[gallery_name].update(
                    unc_metric(
                        predicted_id=predicted_id,
                        was_rejected=was_rejected,
                        g_unique_ids=self.gallery_pooled_templates[gallery_name][
                            "template_subject_ids_sorted"
                        ],
                        probe_unique_ids=self.probe_pooled_templates[gallery_name][
                            "template_subject_ids_sorted"
                        ],
                        predicted_unc=predicted_unc,
                    )
                )

        # aggregate metrics over two galleries
        if len(used_galleries) == 2:
            result_metrics = {}
            result_unc_metrics = {}
            for key in metrics[used_galleries[1]].keys():
                if "osr_metric:" in key:
                    result_metrics[key] = (
                        metrics[used_galleries[0]][key]
                        + metrics[used_galleries[1]][key]
                    ) / 2
                else:
                    result_metrics[key] = metrics[used_galleries[0]][key]
            for key in unc_metrics[used_galleries[1]].keys():
                if "osr_unc_metric:" in key:
                    result_unc_metrics[key] = (
                        unc_metrics[used_galleries[0]][key]
                        + unc_metrics[used_galleries[1]][key]
                    ) / 2
                else:
                    result_unc_metrics[key] = unc_metrics[used_galleries[1]][key]
        else:
            result_metrics = metrics[used_galleries[0]]
            result_unc_metrics = unc_metrics[used_galleries[0]]

        return result_metrics, result_unc_metrics

    def run_model_test_verification(
        self,
    ):
        scores = self.distance_function(
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

    def run_model_test_closed_set_identification(self):
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

        similarity, probe_score = self.distance_function(
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

            similarity, probe_score = self.distance_function(
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
