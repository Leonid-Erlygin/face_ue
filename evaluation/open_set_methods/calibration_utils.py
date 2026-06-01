from evaluation.embeddings import process_embeddings
from pathlib import Path
import numpy as np
from evaluation.embedding_utils import get_template_subsets
CACHE_ROOT = Path("/app/cache/template_cache_new")

def prepare_calibration_dataset(calibration_set, embs_name):
    # prepare calibration set
    embeddings_path = (
        Path(calibration_set.dataset_path)
        / f"embeddings/{embs_name}_embs_{calibration_set.dataset_name}.npz"
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
    gallery_pooled_templates_calib = {
        gallery_name: {} for gallery_name in used_galleries
    }
    probe_pooled_templates_calib = {gallery_name: {} for gallery_name in used_galleries}
    template_subsets_path = (
    CACHE_ROOT
    / embs_name
    / f"calib_template_subsets_PoolingDefault_{calibration_set.dataset_name}"
)

    template_pool_path = (
        CACHE_ROOT
        / embs_name
        / f"calib_template_pool_gallery-PoolingDefault_probe-PoolingDefault_{calibration_set.dataset_name}"
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
        ) = get_template_subsets(
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

    gallery_templates = getattr(calibration_set, f"{gallery_name}_templates")
    gallery_subject_ids = getattr(calibration_set, f"{gallery_name}_ids")
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
    from evaluation.template_pooling_strategies import PoolingDefault

    average_pooling = PoolingDefault()
    pooled_data = average_pooling(
        gallery_features,
        kappa,
        gallery_templates_sorted,
        gallery_medias,
    )
    gallery_pooled_templates_calib[gallery_name] = {
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

    probe_pooled_templates_calib[gallery_name] = {
        "template_pooled_features": probe_pooled_data[0],
        "template_pooled_data_unc": probe_pooled_data[1],
        "template_subject_ids_sorted": probe_subject_ids_sorted,
    }
    return gallery_pooled_templates_calib, probe_pooled_templates_calib
