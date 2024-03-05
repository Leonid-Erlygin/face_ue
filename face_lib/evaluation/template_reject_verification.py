"""
The script to compare template-image verification protocol, based on IJB-C
It could calculate the PFE, magface, Scale results
Though it requires preprocessed with face detector images
  and pretrained with scripts/train checkpoints
For the options see scripts/eval/template_reject_verification.sh

optimal setups for methods:
magface - mean aggregation, cosine distance, image ue value
PFE - pfe aggregation, mls distance, image ue value
scale - mean aggregation, cosine distance, image ue value
"""

# rewrite the list
# check that pfe ue bigger for verify single images
# correct mls sigma aggregation?


import os
import sys
import numpy as np
from datetime import datetime
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.metrics import auc
from pathlib import Path
import pickle
from tqdm import tqdm

import hydra

path = str(Path(__file__).parent.parent.parent.absolute())
sys.path.insert(0, path)
from face_lib.datasets import IJBDataset, IJBATest, IJBCTemplates
from face_lib.utils import cfg as cfg_utils
import face_lib.evaluation.plots as plots
from face_lib.evaluation.utils import (
    get_required_models,
    get_distance_uncertainty_funcs,
)
from face_lib.evaluation.feature_extractors import (
    extract_features_uncertainties_from_list,
)
from face_lib.evaluation.reject_verification import get_rejected_tar_far
import face_lib.probability.likelihoods as likelihoods
import face_lib.probability.samplers as samplers
from face_lib.probability.utils import compute_probalities

from face_lib.evaluation.aggregation import aggregate_templates


@hydra.main(
    config_path=str(Path(".").resolve() / "configs/hydra"),
    config_name=Path(__file__).stem + "_pfe",
    version_base="1.2",
)
def eval_template_reject_verification(cfg):
    rejected_portions = np.linspace(*cfg.rejected_portions)
    FARs = list(map(float, cfg.FARs))

    # Setup the plots
    if rejected_portions is None:
        rejected_portions = [
            0.0,
        ]
    if FARs is None:
        FARs = [
            0.0,
        ]

    all_results = OrderedDict()
    n_figures = len(cfg.methods)
    distance_fig, distance_axes = None, [None] * n_figures
    uncertainty_fig, uncertainty_axes = None, [None] * n_figures

    distance_fig, distance_axes = plt.subplots(
        nrows=1, ncols=n_figures, figsize=(9 * n_figures, 8)
    )
    uncertainty_fig, uncertainty_axes = plt.subplots(
        nrows=1, ncols=n_figures, figsize=(9 * n_figures, 8)
    )
    if n_figures == 1:
        distance_axes = [distance_axes]
        uncertainty_axes = [uncertainty_axes]

    for (
        method,
        distance_ax,
        uncertainty_ax,
    ) in zip(cfg.methods, distance_axes, uncertainty_axes):
        fusion_name = method.uncertainty_backbone_name
        distance_name = method.distance_name
        uncertainty_name = method.uncertainty_name
        print(f"==={fusion_name} {distance_name} {uncertainty_name} ===")

        # returns features and uncertainties for a list of images
        (
            image_paths,
            feature_dict,
            uncertainty_dict,
            classifier,
            device,
        ) = get_image_embeddings(cfg, method)

        tester = IJBCTemplates(image_paths, feature_dict, uncertainty_dict)
        tester.init_proto(cfg.protocol_path)

        distance_func, uncertainty_func = get_distance_uncertainty_funcs(
            distance_name=distance_name,
            uncertainty_name=uncertainty_name,
            classifier=classifier,
            device=device,
            distaces_batch_size=cfg.distaces_batch_size,
        )

        if cfg.equal_uncertainty_enroll:
            aggregate_templates(tester.enroll_templates(), fusion_name, cfg.norm_mean)
            aggregate_templates(tester.verification_templates(), "first", cfg.norm_mean)
        else:
            aggregate_templates(tester.all_templates(), fusion_name)

        if distance_name == "prob-distance" or uncertainty_name in [
            "prob-unc-pair",
            "prob-unc",
            "entropy-unc",
        ]:
            set_probability_based_uncertainty(
                tester, cfg, method, fusion_name, distance_name, uncertainty_name
            )

        (
            feat_1,
            feat_2,
            unc_1,
            unc_2,
            label_vec,
        ) = tester.get_features_uncertainties_labels()

        result_table = get_rejected_tar_far(
            feat_1,
            feat_2,
            unc_1,
            unc_2,
            label_vec,
            distance_func=distance_func,
            pair_uncertainty_func=uncertainty_func,
            uncertainty_mode=method.uncertainty_mode,
            FARs=FARs,
            distance_ax=distance_ax,
            uncertainty_ax=uncertainty_ax,
            rejected_portions=rejected_portions,
            equal_uncertainty_enroll=cfg.equal_uncertainty_enroll,
        )

        # delete arrays to prevent memory leak
        del feat_1
        del feat_2
        del unc_1
        del unc_2
        del label_vec

        distance_ax.set_title(f"{distance_name} {uncertainty_name}")
        uncertainty_ax.set_title(f"{distance_name} {uncertainty_name}")

        if "likelihood" in method.keys():
            likelihood = (
                method.likelihood.name.replace("_", "-")
                + "_"
                + str(method.likelihood.args)
            )
        else:
            likelihood = ""
        if "sampler" in method.keys():
            sampler = (
                method.sampler.name.replace("_", "-") + "_" + str(method.sampler.args)
            )
        else:
            sampler = ""
        all_results[
            (
                method.uncertainty_backbone_name,
                uncertainty_name,
                likelihood,
                sampler,
            )
        ] = result_table

    res_AUCs = OrderedDict()
    for method, table in all_results.items():
        res_AUCs[method] = {
            far: auc(rejected_portions, TARs) for far, TARs in table.items()
        }

    save_plots(
        cfg, all_results, res_AUCs, rejected_portions, distance_fig, uncertainty_fig
    )


def set_probability_based_uncertainty(
    tester, cfg, method, fusion_name, distance_name, uncertainty_name
):
    # cache probability matrix
    prob_cache_path = (
        Path(cfg.cache_dir)
        / f"{fusion_name}_{method.sampler.name}_{str(method.sampler.args)}_{method.likelihood.name.replace('_','-')}_{str(method.likelihood.args)}_probabilities.npy"
    )
    if prob_cache_path.is_file() and cfg.debug is False:
        print("Using cached probabily matrix")
        probabilities = np.load(prob_cache_path)
    else:
        print("Computing probabilities")
        sampler = getattr(samplers, method.sampler.name)(**method.sampler.args)
        likelihood = getattr(likelihoods, method.likelihood.name)(
            **method.likelihood.args
        )
        apply_softmax = (
            True if method.likelihood.name not in cfg.likelihoods_no_softmax else False
        )
        probabilities = compute_probalities(
            tester, sampler, likelihood, True, 0, apply_softmax=apply_softmax
        )
        np.save(prob_cache_path, probabilities)

    if distance_name == "prob-distance":
        # set probability distances
        print("Setting prob distance")
        for i, verif_template in tqdm(enumerate(tester.verification_templates())):
            new_mu = {}
            for j, enroll_template in enumerate(tester.enroll_templates()):
                new_mu[enroll_template.template_id] = probabilities[i, j]
            verif_template.mu = new_mu
        for t in tester.enroll_templates():
            t.mu = t.template_id
    if uncertainty_name in ["prob-unc-pair", "entropy-unc", "prob-unc"]:
        print("Setting pair prob uncertainty")
        enroll_templates_ids = []
        for t in tester.enroll_templates():
            t.sigma_sq = t.template_id
            enroll_templates_ids.append(t.template_id)
        for i, verif_template in tqdm(enumerate(tester.verification_templates())):
            verif_template.sigma_sq = np.array(
                [enroll_templates_ids, probabilities[i, :]]
            )


def get_image_embeddings(cfg, method):
    device = torch.device("cuda:" + str(cfg.device_id))

    # Setup the data
    if cfg.protocol != "ijbc":
        raise ValueError('Unkown protocol. Only accept "ijbc" at the moment.')

    testset = IJBDataset(cfg.dataset_path)
    image_paths = testset["abspath"].values
    short_paths = ["/".join(Path(p).parts[-2:]) for p in image_paths]

    features_path = (
        Path(cfg.cache_dir)
        / "features"
        / f"{method.uncertainty_backbone_name}_features.pickle"
    )
    uncertainty_path = (
        Path(cfg.cache_dir)
        / "features"
        / f"{method.uncertainty_backbone_name}_uncertainty.pickle"
    )

    if features_path.is_file() and uncertainty_path.is_file():
        with open(features_path, "rb") as f:
            feature_dict = pickle.load(f)
        with open(uncertainty_path, "rb") as f:
            uncertainty_dict = pickle.load(f)
    else:
        print("Calculating")
        if cfg.uncertainty_strategy == "magface":
            raise ValueError("Can't compute magface here")

        if cfg.uncertainty_strategy == "scale_finetuned":
            strategy = "scale"
        else:
            strategy = cfg.uncertainty_strategy

        model_args = cfg_utils.load_config(cfg.config_path)
        checkpoint = torch.load(cfg.checkpoint_path, map_location=device)

        (
            backbone,
            head,
            discriminator,
            classifier,
            scale_predictor,
            uncertainty_model,
        ) = get_required_models(
            checkpoint=checkpoint, args=cfg, model_args=model_args, device=device
        )
        features, uncertainties = extract_features_uncertainties_from_list(
            backbone,
            head,
            image_paths=image_paths,
            uncertainty_strategy=strategy,
            batch_size=cfg.batch_size,
            discriminator=discriminator,
            scale_predictor=scale_predictor,
            uncertainty_model=uncertainty_model,
            device=device,
            verbose=cfg.verbose,
        )
        feature_dict = {p: feature for p, feature in zip(short_paths, features)}
        with open(features_path, "wb") as f:
            pickle.dump(feature_dict, f)
        uncertainty_dict = {p: scale for p, scale in zip(short_paths, uncertainties)}
        with open(uncertainty_path, "wb") as f:
            pickle.dump(uncertainty_dict, f)
    return image_paths, feature_dict, uncertainty_dict, None, device


def save_plots(
    cfg, all_results, res_AUCs, rejected_portions, distance_fig, uncertainty_fig
):
    for name, aucs in res_AUCs.items():
        print(name)
        for FAR, AUC in aucs.items():
            print(f"\tFAR={round(FAR, 5)} TAR_AUC : {round(AUC, 5)}")

    for name, result_table in all_results.items():
        title = "Template" + "\n".join(name)
        save_to_path = os.path.join(
            cfg.exp_dir,
            "_".join(name) + ".jpg",
        )

        plots.plot_rejected_TAR_FAR(
            result_table, rejected_portions, title, save_to_path
        )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plots.plot_TAR_FAR_different_methods(
        all_results,
        rejected_portions,
        res_AUCs,
        title="Template reject verification",
        save_figs_path=os.path.join(cfg.exp_dir, f"all_methods_{timestamp}.jpg"),
    )

    distance_fig.savefig(
        os.path.join(cfg.exp_dir, f"distance_dist_{timestamp}.jpg"), dpi=400
    )
    uncertainty_fig.savefig(
        os.path.join(cfg.exp_dir, f"uncertainry_dist_{timestamp}.jpg"), dpi=400
    )

    setup_name = {True: "single", False: "full"}[cfg.equal_uncertainty_enroll]

    torch.save(
        all_results,
        os.path.join(
            cfg.exp_dir,
            f"table_{timestamp}_{setup_name}.pt",
        ),
    )


if __name__ == "__main__":
    eval_template_reject_verification()
