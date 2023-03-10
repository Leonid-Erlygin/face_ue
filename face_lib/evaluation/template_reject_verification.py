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
from collections import defaultdict, OrderedDict
from sklearn.metrics import auc
from pathlib import Path
import pickle
from scipy.special import softmax

path = str(Path(__file__).parent.parent.parent.absolute())
sys.path.insert(0, path)
from face_lib.datasets import IJBDataset, IJBATest, IJBCTemplates
from face_lib.utils import cfg
import face_lib.evaluation.plots as plots
from face_lib.evaluation.utils import get_required_models, get_distance_uncertainty_funcs
from face_lib.evaluation.feature_extractors import get_features_uncertainties_labels
from face_lib.evaluation.feature_extractors import extract_features_uncertainties_from_list
from face_lib.evaluation.reject_verification import get_rejected_tar_far

from face_lib.evaluation import name_to_distance_func, l2_normalize
from face_lib.evaluation.aggregation import aggregate_PFE, aggregate_min, aggregate_softmax
from face_lib.evaluation.argument_parser import verify_arguments_template_reject_verification, parse_cli_arguments


def aggregate_templates(templates, method):
    for t in templates:
        if method == 'first':
            t.mu = l2_normalize(t.features[0])
            t.sigma_sq = t.sigmas[0]
        elif method == 'PFE':
            t.mu, t.sigma_sq = aggregate_PFE(t.features, sigma_sq=t.sigmas)
        elif method == 'mean':
            t.mu = l2_normalize(np.mean(t.features, axis=0))
            t.sigma_sq = np.mean(t.sigmas, axis=0)
        elif method == 'stat-mean':
            t.mu = l2_normalize(np.mean(t.features, axis=0))
            t.sigma_sq = np.mean(t.sigmas, axis=0) * (len(t.sigmas))**0.5
        elif method == 'argmax':
            idx = np.argmax(t.sigmas)
            t.mu = t.features[idx]
            t.sigma_sq = t.sigmas[idx]
        elif method == 'stat-softmax':
            weights = softmax(t.sigmas[:, 0])
            t.mu = l2_normalize(np.dot(weights, t.features))
            t.sigma_sq = np.dot(weights, t.sigmas) * len(t.sigmas)**0.5
        elif method.startswith('softmax'):
            parts = method.split('-')
            if len(parts) == 1:
                temperature = 1.0
            else:
                temperature = float(parts[1])
            weights = softmax(t.sigmas[:, 0] / temperature)
            t.mu = l2_normalize(np.dot(weights, t.features))
            t.sigma_sq = np.dot(weights, t.sigmas)
        elif method == 'weighted':
            mu = l2_normalize(t.features)
            weights = t.sigmas[:, 0]
            weights = weights / np.sum(weights)
            t.mu = l2_normalize(np.dot(weights, mu))
            t.sigma_sq = np.dot(weights, t.sigmas)
        elif method.startswith('weighted-softmax'):
            parts = method.split('-')
            if len(parts) == 2:
                temperature = 1.0
            else:
                temperature = float(parts[2])
            weights = t.sigmas[:, 0]
            weights = weights / np.sum(weights)
            t.mu = l2_normalize(np.dot(weights, t.features))
            weights = softmax(t.sigmas[:, 0] / temperature)
            t.sigma_sq = np.dot(weights, t.sigmas)
        elif method == 'weighted':
            mu = l2_normalize(t.features)
            weights = t.sigmas[:, 0]
            weights = weights / np.sum(weights)
            t.mu = l2_normalize(np.dot(weights, mu))
            t.sigma_sq = np.dot(weights, t.sigmas)
            pass
        else:
            raise ValueError(f"Wrong aggregate method {method}")


def eval_template_reject_verification(
    backbone,
    dataset_path,
    protocol="ijbc",
    protocol_path=".",
    uncertainty_strategy="head",
    uncertainty_mode="uncertainty",
    batch_size=64,
    distaces_batch_size=None,
    rejected_portions=None,
    FARs=None,
    fusions_distances_uncertainties=None,
    head=None,
    discriminator=None,
    classifier=None,
    scale_predictor=None,
    save_fig_path=None,
    device=torch.device("cpu"),
    verbose=False,
    uncertainty_model=None,
    cached_embeddings=False,
    equal_uncertainty_enroll=False
):

    # Setup the plots
    if rejected_portions is None:
        rejected_portions = [0.0,]
    if FARs is None:
        FARs = [0.0,]

    all_results = OrderedDict()
    n_figures = len(fusions_distances_uncertainties)
    distance_fig, distance_axes = None, [None] * n_figures
    uncertainty_fig, uncertainty_axes = None, [None] * n_figures
    if save_fig_path is not None:
        distance_fig, distance_axes = plt.subplots(
            nrows=1, ncols=n_figures,
            figsize=(9 * n_figures, 8))
        uncertainty_fig, uncertainty_axes = plt.subplots(
            nrows=1, ncols=n_figures,
            figsize=(9 * n_figures, 8))

    # Setup the data
    if protocol != "ijbc":
        raise ValueError('Unkown protocol. Only accept "ijbc" at the moment.')

    testset = IJBDataset(dataset_path)
    image_paths = testset["abspath"].values
    short_paths = ["/".join(Path(p).parts[-2:]) for p in image_paths]

    # returns features and uncertainties for a list of images
    if cached_embeddings:
        with open(Path(save_fig_path) / f'{uncertainty_strategy}_features.pickle', 'rb') as f:
            feature_dict = pickle.load(f)
        with open(Path(save_fig_path) / f'{uncertainty_strategy}_uncertainty.pickle', 'rb') as f:
            uncertainty_dict = pickle.load(f)
    else:
        print('Calculating')
        if uncertainty_strategy == 'magface':
            raise ValueError("Can't compute magface here")

        if uncertainty_strategy == 'scale_finetuned':
            strategy = 'scale'
        else:
            strategy = uncertainty_strategy

        features, uncertainties = extract_features_uncertainties_from_list(
            backbone,
            head,
            image_paths=image_paths,
            uncertainty_strategy=strategy,
            batch_size=batch_size,
            discriminator=discriminator,
            scale_predictor=scale_predictor,
            uncertainty_model=uncertainty_model,
            device=device,
            verbose=verbose,
        )
        feature_dict = {p: feature for p, feature in zip(short_paths, features)}
        with open(Path(save_fig_path) / f'{uncertainty_strategy}_features.pickle', 'wb') as f:
            pickle.dump(feature_dict, f)
        uncertainty_dict = {p: scale for p, scale in zip(short_paths, uncertainties)}
        with open(Path(save_fig_path) / f'{uncertainty_strategy}_uncertainty.pickle', 'wb') as f:
            pickle.dump(uncertainty_dict, f)

    tester = IJBCTemplates(image_paths, feature_dict, uncertainty_dict)
    tester.init_proto(protocol_path)

    prev_fusion_name = None
    for (fusion_name, distance_name, uncertainty_name), distance_ax, uncertainty_ax in \
            zip(fusions_distances_uncertainties, distance_axes, uncertainty_axes):
        print(f"==={fusion_name} {distance_name} {uncertainty_name} ===")

        distance_func, uncertainty_func = get_distance_uncertainty_funcs(
            distance_name=distance_name,
            uncertainty_name=uncertainty_name,
            classifier=classifier,
            device=device,
            distaces_batch_size=distaces_batch_size,
        )

        if fusion_name != prev_fusion_name:
            if equal_uncertainty_enroll:
                aggregate_templates(tester.enroll_templates(), fusion_name)
                aggregate_templates(tester.verification_templates(), 'first')
            else:
                aggregate_templates(tester.all_templates(), fusion_name)

        feat_1, feat_2, unc_1, unc_2, label_vec = \
            tester.get_features_uncertainties_labels()

        print('shapes')
        print(feat_1.shape, feat_2.shape, unc_1.shape, unc_2.shape, label_vec.shape)

        result_table = get_rejected_tar_far(
            feat_1,
            feat_2,
            unc_1,
            unc_2,
            label_vec,
            distance_func=distance_func,
            pair_uncertainty_func=uncertainty_func,
            uncertainty_mode=uncertainty_mode,
            FARs=FARs,
            distance_ax=distance_ax,
            uncertainty_ax=uncertainty_ax,
            rejected_portions=rejected_portions,
            equal_uncertainty_enroll=equal_uncertainty_enroll
        )

        # delete arrays to prevent memory leak
        del(feat_1)
        del(feat_2)
        del(unc_1)
        del(unc_2)
        del(label_vec)

        if save_fig_path is not None:
            distance_ax.set_title(f"{distance_name} {uncertainty_name}")
            uncertainty_ax.set_title(f"{distance_name} {uncertainty_name}")

        all_results[(fusion_name, distance_name, uncertainty_name)] = result_table
        prev_fusion_name = fusion_name

    res_AUCs = OrderedDict()
    for method, table in all_results.items():
        res_AUCs[method] = {
            far: auc(rejected_portions, TARs) for far, TARs in table.items()
        }

    for (fusion_name, distance_name, uncertainty_name), aucs in res_AUCs.items():
        print(fusion_name, distance_name, uncertainty_name)
        for FAR, AUC in aucs.items():
            print(f"\tFAR={round(FAR, 5)} TAR_AUC : {round(AUC, 5)}")

    if save_fig_path:
        for (fusion_name, distance_name, uncertainty_name), result_table in all_results.items():
            title = "Template" + distance_name + " " + uncertainty_name
            save_to_path = (
                os.path.join(save_fig_path, fusion_name + '_' + distance_name + "_" + uncertainty_name + ".jpg")
            )
            if save_fig_path:
                plots.plot_rejected_TAR_FAR(result_table, rejected_portions, title, save_to_path)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plots.plot_TAR_FAR_different_methods(
            all_results,
            rejected_portions,
            res_AUCs,
            title="Template reject verification",
            save_figs_path=os.path.join(save_fig_path, f"all_methods_{uncertainty_strategy}_{timestamp}.jpg")
        )
        # plt.show()

        distance_fig.savefig(os.path.join(save_fig_path, f"distance_dist_{timestamp}.jpg"), dpi=400)
        uncertainty_fig.savefig(os.path.join(save_fig_path, f"uncertainry_dist_{timestamp}.jpg"), dpi=400)

        setup_name = {
            True: 'single', False: 'full'
        }[equal_uncertainty_enroll]

        torch.save(all_results, os.path.join(save_fig_path, f"table_{uncertainty_strategy}_{timestamp}_{setup_name}.pt"))


def main():
    args = parse_cli_arguments()
    args = verify_arguments_template_reject_verification(args)
    print(args)

    if os.path.isdir(args.save_fig_path) and not args.save_fig_path.endswith("test"):
        raise RuntimeError("Directory exists")
    else:
        os.makedirs(args.save_fig_path, exist_ok=True)

    device = torch.device("cuda:" + str(args.device_id))

    model_args = cfg.load_config(args.config_path)
    print(model_args)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    backbone, head, discriminator, classifier, scale_predictor, uncertainty_model = \
        get_required_models(checkpoint=checkpoint, args=args, model_args=model_args, device=device)

    rejected_portions = np.linspace(*args.rejected_portions)
    FARs = list(map(float, args.FARs))
    fusions_distances_uncertainties = list(
        map(lambda x: x.split("_"), args.fusion_distance_uncertainty_metrics)
    )

    eval_template_reject_verification(
        backbone,
        dataset_path=args.dataset_path,
        protocol=args.protocol,
        protocol_path=args.protocol_path,
        uncertainty_strategy=args.uncertainty_strategy,
        uncertainty_mode=args.uncertainty_mode,
        batch_size=args.batch_size,
        distaces_batch_size=args.distaces_batch_size,
        rejected_portions=rejected_portions,
        FARs=FARs,
        fusions_distances_uncertainties=fusions_distances_uncertainties,
        head=head,
        discriminator=discriminator,
        classifier=classifier,
        scale_predictor=scale_predictor,
        save_fig_path=args.save_fig_path,
        device=device,
        verbose=args.verbose,
        uncertainty_model=uncertainty_model,
        cached_embeddings=args.cached_embeddings,
        equal_uncertainty_enroll=args.equal_uncertainty_enroll
    )


if __name__ == "__main__":
    main()
