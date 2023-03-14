import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from path import Path
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from sklearn.metrics import auc

path = str(Path(__file__).parent.parent.parent.abspath())
sys.path.insert(0, path)

from face_lib.utils import cfg
import face_lib.utils.metrics as metrics
import face_lib.evaluation.plots as plots
from face_lib.evaluation.argument_parser import (
    parse_cli_arguments,
    verify_arguments_reject_verification,
)
from face_lib.evaluation.feature_extractors import get_features_uncertainties_labels
from face_lib.evaluation.utils import (
    get_required_models,
    get_distance_uncertainty_funcs,
    extract_statistics,
)


def eval_reject_verification(
    backbone,
    head,
    dataset_path,
    pairs_table_path,
    uncertainty_strategy="head",
    uncertainty_mode="uncertainty",
    batch_size=64,
    distaces_batch_size=None,
    rejected_portions=None,
    FARs=None,
    distances_uncertainties=None,
    discriminator=None,
    classifier=None,
    scale_predictor=None,
    uncertainty_model=None,
    precalculated_path=None,
    save_fig_path=None,
    device=torch.device("cpu"),
    verbose=False,
    val_pairs_table_path=None,
):
    if rejected_portions is None:
        rejected_portions = [
            0.0,
        ]
    if FARs is None:
        FARs = [
            0.0,
        ]

    mu_1, mu_2, sigma_sq_1, sigma_sq_2, label_vec = get_features_uncertainties_labels(
        backbone,
        head,
        dataset_path,
        pairs_table_path,
        uncertainty_strategy=uncertainty_strategy,
        batch_size=batch_size,
        verbose=verbose,
        discriminator=discriminator,
        scale_predictor=scale_predictor,
        uncertainty_model=uncertainty_model,
        precalculated_path=precalculated_path,
    )

    val_statistics = None
    if val_pairs_table_path is not None:
        val_data = get_features_uncertainties_labels(
            backbone,
            head,
            dataset_path,
            val_pairs_table_path,
            uncertainty_strategy=uncertainty_strategy,
            batch_size=batch_size,
            verbose=verbose,
            discriminator=discriminator,
            scale_predictor=scale_predictor,
            uncertainty_model=uncertainty_model,
            precalculated_path=precalculated_path,
        )
        val_statistics = extract_statistics(val_data)

    print("Mu_1 :", mu_1.shape, mu_1.dtype)
    print("Mu_2 :", mu_2.shape, mu_2.dtype)
    print("sigma_sq_1 :", sigma_sq_1.shape, sigma_sq_1.dtype)
    print("sigma_sq_2 :", sigma_sq_2.shape, sigma_sq_2.dtype)
    print("labels :", label_vec.shape, label_vec.dtype)

    distance_fig, distance_axes = None, [None] * len(distances_uncertainties)
    uncertainty_fig, uncertainty_axes = None, [None] * len(distances_uncertainties)
    if save_fig_path is not None:
        distance_fig, distance_axes = plt.subplots(
            nrows=1,
            ncols=len(distances_uncertainties),
            figsize=(9 * len(distances_uncertainties), 8),
        )
        uncertainty_fig, uncertainty_axes = plt.subplots(
            nrows=1,
            ncols=len(distances_uncertainties),
            figsize=(9 * len(distances_uncertainties), 8),
        )
    if not hasattr(distance_axes, "__iter__"):
        distance_axes = (distance_axes,)
        uncertainty_axes = (uncertainty_axes,)

    all_results = OrderedDict()

    for (distance_name, uncertainty_name), distance_ax, uncertainty_ax in zip(
        distances_uncertainties, distance_axes, uncertainty_axes
    ):
        print(f"=== {distance_name} {uncertainty_name} ===")

        distance_func, uncertainty_func = get_distance_uncertainty_funcs(
            distance_name=distance_name,
            uncertainty_name=uncertainty_name,
            classifier=classifier,
            device=device,
            distaces_batch_size=distaces_batch_size,
            val_statistics=val_statistics,
        )

        result_table = get_rejected_tar_far(
            mu_1,
            mu_2,
            sigma_sq_1,
            sigma_sq_2,
            label_vec,
            distance_func=distance_func,
            pair_uncertainty_func=uncertainty_func,
            uncertainty_mode=uncertainty_mode,
            FARs=FARs,
            distance_ax=distance_ax,
            uncertainty_ax=uncertainty_ax,
            rejected_portions=rejected_portions,
        )

        if save_fig_path is not None:
            distance_ax.set_title(f"{distance_name} {uncertainty_name}")
            uncertainty_ax.set_title(f"{distance_name} {uncertainty_name}")

        all_results[(distance_name, uncertainty_name)] = result_table

    res_AUCs = OrderedDict()
    for method, table in all_results.items():
        res_AUCs[method] = {
            far: auc(rejected_portions, TARs) for far, TARs in table.items()
        }

    for (distance_name, uncertainty_name), aucs in res_AUCs.items():
        print(distance_name, uncertainty_name)
        for FAR, AUC in aucs.items():
            print(f"\tFAR={round(FAR, 5)} TAR_AUC : {round(AUC, 5)}")

    if save_fig_path:
        for (distance_name, uncertainty_name), result_table in all_results.items():
            title = (
                pairs_table_path.split("/")[-1][-4]
                + " "
                + distance_name
                + " "
                + uncertainty_name
            )
            save_to_path = os.path.join(
                save_fig_path, distance_name + "_" + uncertainty_name + ".jpg"
            )
            if save_fig_path:
                plots.plot_rejected_TAR_FAR(
                    result_table, rejected_portions, title, save_to_path
                )

        plots.plot_TAR_FAR_different_methods(
            all_results,
            rejected_portions,
            res_AUCs,
            title=pairs_table_path.split("/")[-1][:-4],
            save_figs_path=os.path.join(save_fig_path, "all_methods.jpg"),
        )

        distance_fig.savefig(os.path.join(save_fig_path, "distance_dist.jpg"), dpi=400)
        uncertainty_fig.savefig(
            os.path.join(save_fig_path, "uncertainry_dist.jpg"), dpi=400
        )

        torch.save(all_results, os.path.join(save_fig_path, "table.pt"))


def get_rejected_tar_far(
    mu_1,
    mu_2,
    sigma_sq_1,
    sigma_sq_2,
    label_vec,
    distance_func,
    pair_uncertainty_func,
    FARs,
    uncertainty_mode="uncertainty",
    distance_ax=None,
    uncertainty_ax=None,
    rejected_portions=None,
    equal_uncertainty_enroll=False,
):
    # If something's broken, uncomment the line below

    # score_vec = force_compare(distance_func)(mu_1, mu_2, sigma_sq_1, sigma_sq_2)
    score_vec = distance_func(mu_1, mu_2, sigma_sq_1, sigma_sq_2)

    # if equal_uncertainty_enroll:
    #     sigma_sq_1 = np.ones_like(sigma_sq_1)

    uncertainty_vec = pair_uncertainty_func(mu_1, mu_2, sigma_sq_1, sigma_sq_2)
    if isinstance(uncertainty_vec, tuple):
        uncertainty_vec, positive = uncertainty_vec
    else:
        positive = None

    result_table = defaultdict(list)
    result_fars = defaultdict(list)

    if positive is None:
        sorted_indices = uncertainty_vec.argsort()
        score_vec = score_vec[sorted_indices]
        label_vec = label_vec[sorted_indices]
        uncertainty_vec = uncertainty_vec[sorted_indices]
        assert score_vec.shape == label_vec.shape

        if uncertainty_mode == "uncertainty":
            pass
        elif uncertainty_mode == "confidence":
            score_vec, label_vec, uncertainty_vec = (
                score_vec[::-1],
                label_vec[::-1],
                uncertainty_vec[::-1],
            )
        else:
            raise RuntimeError("Don't know this type uncertainty mode")

        
        for rejected_portion in tqdm(rejected_portions):
            cur_len = int(score_vec.shape[0] * (1 - rejected_portion))
            tars, fars, thresholds = metrics.ROC(
                score_vec[:cur_len], label_vec[:cur_len], FARs=FARs
            )
            for far, tar in zip(FARs, tars):
                result_table[far].append(tar)
            for wanted_far, real_far in zip(FARs, fars):
                result_fars[wanted_far].append(real_far)
    else:
        
        negative = np.invert(positive)
        sorted_indices_positive = uncertainty_vec[positive].argsort()
        sorted_indices_negative = uncertainty_vec[negative].argsort()
        print(f'Using separate thresholds for {len(sorted_indices_negative)} negative and {len(sorted_indices_positive)} positive pair')

        if uncertainty_mode == "uncertainty":
            pass
        elif uncertainty_mode == "confidence":
            sorted_indices_positive = sorted_indices_positive[::-1]
            sorted_indices_negative = sorted_indices_negative[::-1]
        else:
            raise RuntimeError("Don't know this type uncertainty mode")
        
        uncertainty_vec_positive = uncertainty_vec[positive][sorted_indices_positive]
        score_vec_positive = score_vec[positive][sorted_indices_positive]
        label_vec_positive = label_vec[positive][sorted_indices_positive]

        uncertainty_vec_negative = uncertainty_vec[negative][sorted_indices_negative]
        score_vec_negative = score_vec[negative][sorted_indices_negative]
        label_vec_negative = label_vec[negative][sorted_indices_negative]

        for rejected_portion in tqdm(rejected_portions):
            cur_len_positive = int(score_vec_positive.shape[0] * (1 - rejected_portion))
            cur_len_negative = int(score_vec_negative.shape[0] * (1 - rejected_portion))

            score_vec_slice = np.concatenate([score_vec_positive[:cur_len_positive], score_vec_negative[:cur_len_negative]])
            label_vec_slice = np.concatenate([label_vec_positive[:cur_len_positive], label_vec_negative[:cur_len_negative]])
            tars, fars, thresholds = metrics.ROC(
                score_vec_slice, label_vec_slice, FARs=FARs
            )
            for far, tar in zip(FARs, tars):
                result_table[far].append(tar)
            for wanted_far, real_far in zip(FARs, fars):
                result_fars[wanted_far].append(real_far)


    plots.plot_distribution(
        score_vec,
        label_vec,
        xlabel_name="Distances",
        ylabel_name="Amount",
        ax=distance_ax,
    )

    plots.plot_distribution(
        uncertainty_vec,
        label_vec,
        xlabel_name="Uncertainties",
        ylabel_name="Amount",
        ax=uncertainty_ax,
    )

    return result_table


if __name__ == "__main__":
    # args = parse_args_reject_verification()

    args = parse_cli_arguments()
    args = verify_arguments_reject_verification(args)

    if os.path.isdir(args.save_fig_path) and not args.save_fig_path.endswith("test"):
        raise RuntimeError("Directory exists")
    else:
        os.makedirs(args.save_fig_path, exist_ok=True)

    device = torch.device("cuda:" + str(args.device_id))
    model_args = cfg.load_config(args.config_path)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    (
        backbone,
        head,
        discriminator,
        classifier,
        scale_predictor,
        uncertainty_model,
    ) = get_required_models(
        checkpoint=checkpoint, args=args, model_args=model_args, device=device
    )

    rejected_portions = np.linspace(*args.rejected_portions)
    distances_uncertainties = list(
        map(lambda x: x.split("_"), args.distance_uncertainty_metrics)
    )

    eval_reject_verification(
        backbone,
        head,
        args.dataset_path,
        args.pairs_table_path,
        uncertainty_strategy=args.uncertainty_strategy,
        uncertainty_mode=args.uncertainty_mode,
        batch_size=args.batch_size,
        distaces_batch_size=args.distances_batch_size,
        rejected_portions=rejected_portions,
        FARs=args.FARs,
        distances_uncertainties=distances_uncertainties,
        discriminator=discriminator,
        classifier=classifier,
        scale_predictor=scale_predictor,
        precalculated_path=args.precalculated_path,
        uncertainty_model=uncertainty_model,
        save_fig_path=args.save_fig_path,
        device=device,
        verbose=args.verbose,
        val_pairs_table_path=args.val_pairs_table_path,
    )
