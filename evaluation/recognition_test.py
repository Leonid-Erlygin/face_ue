import numpy as np
from pathlib import Path
from tqdm import tqdm

from .embeddings import process_embeddings
from .test_datasets import FaceRecogntionDataset
from .embedding_utils import get_template_subsets

def _decode_uncertainty(self, unc: np.ndarray) -> np.ndarray:
    name = self.embedding_type.lower()

    # SCF stores log kappa.
    if "scf" in name:
        return np.exp(unc)

    # PFE stores log sigma^2.
    if "pfe" in name:
        return np.exp(unc)

    # ScaleFace writer currently stores scale itself.
    if "scaleface" in name or "scale" in name:
        return unc

    # Safe default for older saved SCF/PFE-like files.
    return np.exp(unc)

class Recognition_test:
    def __init__(
        self,
        task_type: str,
        method_name: str,
        pretty_name: str,
        recognition_method,
        test_dataset: FaceRecogntionDataset,
        embedding_type: str,
        embeddings_path: str,
        use_two_galleries: bool,
        recompute_template_pooling: bool,
        recognition_metrics: dict,
        uncertainty_metrics: dict,
        gallery_template_pooling_strategy,
        use_detector_score: bool = False,
        probe_template_pooling_strategy=None,
    ):
        self.task_type = task_type
        self.method_name = method_name
        self.pretty_name = pretty_name
        self.recognition_method = recognition_method
        self.use_two_galleries = use_two_galleries
        self.test_dataset = test_dataset
        self.embedding_type = embedding_type
        self.recompute_template_pooling = recompute_template_pooling
        self.recognition_metrics = recognition_metrics
        self.uncertainty_metrics = uncertainty_metrics
        self.gallery_template_pooling_strategy = gallery_template_pooling_strategy
        self.probe_template_pooling_strategy = probe_template_pooling_strategy
        self.use_detector_score = use_detector_score

        # load nn embeddings
        aa = np.load(embeddings_path)
        self.embeddings_path = embeddings_path
        self.embs = aa["embs"]
        self.unc = aa["unc"]
        if self.test_dataset.face_scores is not None:
            self.test_dataset.face_scores = self.test_dataset.face_scores.astype(
                self.embs.dtype
            )

        # process embeddings
        self.image_input_feats = process_embeddings(
            self.embs,
            [],
            use_flip_test=False,
            use_norm_score=False,
            use_detector_score=self.use_detector_score,
            face_scores=self.test_dataset.face_scores,
        )
        # pool templates

        if self.task_type == "open_set_identification":
            self.pool_templates_osfr(cache_dir="/app/cache/template_cache_new")
        elif self.task_type == "verification":
            self.pool_templates_verification(
                cache_dir="/app/cache/template_cache_verif"
            )

        assert self.image_input_feats.shape[0] == self.unc.shape[0]
        assert self.image_input_feats.shape[0] == self.test_dataset.medias.shape[0]

    def pool_templates_verification(self, cache_dir: str):
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        template_pool_path = (
            cache_dir
            / Path(self.embedding_type)
            / f"template_pool-{self.gallery_template_pooling_strategy.__class__.__name__}_{self.test_dataset.dataset_name}"
        )
        template_pool_path.mkdir(parents=True, exist_ok=True)
        if (
            template_pool_path / f"pool.npz"
        ).is_file() and self.recompute_template_pooling is False:
            data = np.load(template_pool_path / f"pool.npz")
            pooled_data = (
                data["template_pooled_features"],
                data["template_pooled_data_unc"],
            )
            template_ids = data["template_ids"]
        else:
            unc = self._decode_uncertainty(self.unc)
            pooled_data = self.gallery_template_pooling_strategy(
                self.image_input_feats,
                unc,
                self.test_dataset.templates,
                self.test_dataset.medias,
            )

            template_ids = np.unique(self.test_dataset.templates)

            np.savez(
                template_pool_path / f"pool.npz",
                template_pooled_features=pooled_data[0],
                template_pooled_data_unc=pooled_data[1],
                template_ids=template_ids,
            )

        self.template_pooled_emb = pooled_data[0]
        self.template_pooled_unc = pooled_data[1]
        self.template_ids = template_ids

    def pool_templates_osfr(self, cache_dir: str):
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        template_subsets_path = (
            cache_dir
            / Path(self.embedding_type)
            / f"template_subsets_{self.probe_template_pooling_strategy.__class__.__name__}_{self.test_dataset.dataset_name}_score-norm_{self.use_detector_score}"
        )
        template_pool_path = (
            cache_dir
            / Path(self.embedding_type)
            / f"template_pool_gallery-{self.gallery_template_pooling_strategy.__class__.__name__}_probe-{self.probe_template_pooling_strategy.__class__.__name__}_{self.test_dataset.dataset_name}"
        )

        similarity_matrix_path = template_subsets_path / "sim_matrix"
        similarity_matrix_path.mkdir(parents=True, exist_ok=True)
        template_subsets_path.mkdir(parents=True, exist_ok=True)
        template_pool_path.mkdir(parents=True, exist_ok=True)

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
            ) = get_template_subsets(
                self.image_input_feats,
                self.unc,
                self.test_dataset.templates,
                self.test_dataset.medias,
                self.test_dataset.probe_ids,
                self.test_dataset.probe_templates,
            )
            np.savez(
                template_subsets_path / "probe.npz",
                probe_features=probe_features,
                probe_unc=probe_unc,
                probe_medias=probe_medias,
                probe_templates_sorted=probe_templates_sorted,
                probe_subject_ids_sorted=probe_subject_ids_sorted,
            )

        probe_unc = self._decode_uncertainty(probe_unc)

        for gallery_name in used_galleries:
            gallery_templates = getattr(self.test_dataset, f"{gallery_name}_templates")
            gallery_subject_ids = getattr(self.test_dataset, f"{gallery_name}_ids")
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
                ) = get_template_subsets(
                    self.image_input_feats,
                    self.unc,
                    self.test_dataset.templates,
                    self.test_dataset.medias,
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
            # 1. pool selected gallery templates
            # assert gallery_unc.shape[1] == 1  # working with scf unc
            gallery_unc = self._decode_uncertainty(gallery_unc)
            if (
                template_pool_path / f"gallery_{gallery_name}.npz"
            ).is_file() and self.recompute_template_pooling is False:
                data = np.load(template_pool_path / f"gallery_{gallery_name}.npz")
                pooled_data = (
                    data["template_pooled_features"],
                    data["template_pooled_data_unc"],
                )
            else:
                pooled_data = self.gallery_template_pooling_strategy(
                    gallery_features,
                    gallery_unc,
                    gallery_templates_sorted,
                    gallery_medias,
                )
                np.savez(
                    template_pool_path / f"gallery_{gallery_name}.npz",
                    template_pooled_features=pooled_data[0],
                    template_pooled_data_unc=pooled_data[1],
                )
            self.gallery_pooled_templates[gallery_name] = {
                "template_pooled_features": pooled_data[0],
                "template_pooled_data_unc": pooled_data[1],
                "template_subject_ids_sorted": gallery_subject_ids_sorted,
            }

            # 2. pool probe templates using 'gallery_name' gallery
            if "PoolingProb" in self.probe_template_pooling_strategy.__class__.__name__:
                if (template_pool_path / f"probe_{gallery_name}.npz").is_file():
                    data = np.load(template_pool_path / f"probe_{gallery_name}.npz")
                    probe_pooled_data = (
                        data["template_pooled_features"],
                        data["template_pooled_data_unc"],
                    )
                else:
                    self.recognition_method.setup(
                        probe_features,
                        probe_unc,
                        self.gallery_pooled_templates[gallery_name][
                            "template_pooled_features"
                        ],
                        self.gallery_pooled_templates[gallery_name][
                            "template_pooled_data_unc"
                        ],
                        g_unique_ids=self.gallery_pooled_templates[gallery_name][
                            "template_subject_ids_sorted"
                        ],
                        probe_unique_ids=self.test_dataset.probe_ids,
                    )
                    predicted_id, was_rejected = self.recognition_method.predict()

                    predicted_unc = self.recognition_method.predict_uncertainty()
                    for metric in self.recognition_metrics[self.task_type]:
                        metric(
                            predicted_id=predicted_id,
                            was_rejected=was_rejected,
                            g_unique_ids=self.gallery_pooled_templates[gallery_name][
                                "template_subject_ids_sorted"
                            ],
                            probe_unique_ids=self.test_dataset.probe_ids,
                            predicted_unc=predicted_unc,
                            method_name=self.pretty_name + "_bf-pool",
                            far=self.recognition_method.far,
                        )
                    probe_pooled_data = self.probe_template_pooling_strategy(
                        probe_features,
                        -predicted_unc,
                        probe_unc,
                        probe_templates_sorted,
                        probe_medias,
                    )
                    self.recognition_method.gallery_kappa = None

            else:
                # log scf pool as it is not changing
                if (template_pool_path / f"probe_{gallery_name}.npz").is_file():
                    data = np.load(template_pool_path / f"probe_{gallery_name}.npz")
                    probe_pooled_data = (
                        data["template_pooled_features"],
                        data["template_pooled_data_unc"],
                    )
                else:
                    probe_pooled_data = self.probe_template_pooling_strategy(
                        probe_features,
                        probe_unc,
                        probe_templates_sorted,
                        probe_medias,
                    )
                    np.savez(
                        template_pool_path / f"probe_{gallery_name}.npz",
                        template_pooled_features=probe_pooled_data[0],
                        template_pooled_data_unc=probe_pooled_data[1],
                    )

            self.probe_pooled_templates[gallery_name] = {
                "template_pooled_features": probe_pooled_data[0],
                "template_pooled_data_unc": probe_pooled_data[1],
                "template_subject_ids_sorted": probe_subject_ids_sorted,
            }

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

            # setup osr method and predict
            self.recognition_method.setup(
                self.probe_pooled_templates[gallery_name][
                    "template_pooled_features"
                ],  # probe_templates_feature,
                self.probe_pooled_templates[gallery_name]["template_pooled_data_unc"],
                self.gallery_pooled_templates[gallery_name]["template_pooled_features"],
                self.gallery_pooled_templates[gallery_name]["template_pooled_data_unc"],
                g_unique_ids=self.gallery_pooled_templates[gallery_name][
                    "template_subject_ids_sorted"
                ],
                probe_unique_ids=self.probe_pooled_templates[gallery_name][
                    "template_subject_ids_sorted"
                ],
                dataset_name=self.test_dataset.dataset_name,
            )
            predicted_id, was_rejected = self.recognition_method.predict()
            predicted_unc = self.recognition_method.predict_uncertainty()
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
                        predicted_unc=predicted_unc,
                        method_name=self.pretty_name,
                        far=self.recognition_method.far,
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

        return result_metrics, result_unc_metrics, predicted_unc

    def run_model_test_verification(
        self,
    ):
        scores, predicted_unc = self.recognition_method(
            self.template_pooled_emb,
            self.template_pooled_unc,
            self.template_ids,
            self.test_dataset.p1,
            self.test_dataset.p2,
        )

        metrics = {}
        for metric in self.recognition_metrics[self.task_type]:
            metrics.update(
                metric(
                    scores=scores,
                    labels=self.test_dataset.label,
                )
            )
        unc_metrics = {}
        # compute uncertainty metrics
        for unc_metric in self.uncertainty_metrics[self.task_type]:
            unc_metrics.update(
                unc_metric(
                    scores=scores,
                    labels=self.test_dataset.label,
                    predicted_unc=predicted_unc[:, 0],
                )
            )
        return metrics, unc_metrics, predicted_unc

    def run_model_test_closed_set_identification(self):
        (
            g1_templates_feature,
            g1_template_unc,
            g1_unique_ids,
        ) = get_template_subsets(
            self.test_dataset.g1_templates, self.test_dataset.g1_ids
        )
        (
            probe_templates_feature,
            probe_template_unc,
            probe_unique_ids,
        ) = get_template_subsets(
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
            ) = get_template_subsets(
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
