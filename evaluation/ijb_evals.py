#!/usr/bin/env python3
import os
import numpy as np
from pathlib import Path
import hydra
from hydra.utils import instantiate
import sys
import pandas as pd
from sklearn.metrics import roc_curve, auc

from evaluation.face_recognition_test import Face_Fecognition_test
from evaluation.visualize import (
    plot_dir_far_scores,
    plot_tar_far_scores,
    plot_cmc_scores,
    plot_rejection_scores,
)

path = str(Path(__file__).parent.parent.absolute())
sys.path.insert(1, path)


def instantiate_list(query_list):
    return [instantiate(value) for value in query_list]


def get_args_string(d):
    args = []
    for key, value in d.items():
        args.append(f"{key}:{value}")
    return "-".join(args)


def create_open_set_ident_recognition_metric_table(
    recognition_result_dict: dict, pretty_names:dict,
) -> pd.DataFrame:

    column_names = ['pretty_name', 'model_name']
    data_rows = []
    for i, model_name in enumerate(recognition_result_dict.keys()):
        metrics = []
        for metric_key, metric_value in recognition_result_dict[model_name].items():
            if "final" in metric_key:
                if i==0:
                    column_names.append(metric_key)
                metrics.append(metric_value)
        data_rows.append([pretty_names[model_name], model_name] + metrics)
    # recognition_result_dict
    df = pd.DataFrame(data_rows, columns=column_names)
    return df

def create_open_set_ident_plots(recognition_result_dict: dict, out_dir: Path, pretty_names: dict):
    metric_names = []
    for _, metric in recognition_result_dict.items():
        for key in metric.keys():
            if 'recalls' in key:
                metric_names.append(key)
        break
    for metric_name in metric_names:
        rank = metric_name.split('_')[1]
        model_names = []
        scores = []
        for model_name, metrics in recognition_result_dict.items():
            model_names.append(pretty_names[model_name])
            scores.append((metrics['fars'], metrics[metric_name]))

        fig = plot_dir_far_scores(scores=scores, names=model_names, y_label=f"Rank {rank} Detection & Identification Rate")
        fig.savefig(out_dir / f"rank_{rank}_di_far_plot.png", dpi=300)


def create_open_set_ident_uncertainty_metric_table(
    uncertainty_result_dict: dict,
) -> pd.DataFrame:
    pass


@hydra.main(
    config_path=str(
        Path(__file__).resolve().parents[1] / "configs/uncertainty_benchmark"
    ),
    config_name=Path(__file__).stem,
    version_base="1.2",
)
def main(cfg):
    # instantiate
    open_set_identification_metrics = instantiate_list(
        cfg.open_set_identification_metrics
    )
    open_set_uncertainty_metrics = instantiate_list(cfg.open_set_uncertainty_metrics)
    closed_set_identification_metrics = instantiate_list(
        cfg.closed_set_identification_metrics
    )
    verification_metrics = instantiate_list(cfg.verification_metrics)

    test_dataset = instantiate(cfg.test_dataset)

    verif_scores, verif_names = [], []
    open_set_ident_scores, open_set_ident_names = [], []
    closed_set_ident_scores, closed_set_ident_names = [], []

    open_set_ident_rejection_scores, open_set_ident_rejection_names = [], []

    # create result dirs:
    dataset_name = cfg.test_dataset.dataset_name
    open_set_identification_result_dir = (
        Path(cfg.exp_dir) / dataset_name / "open_set_identification"
    )
    open_set_identification_result_dir.mkdir(exist_ok=True, parents=True)

    closed_set_identification_result_dir = (
        Path(cfg.exp_dir) / dataset_name / "closed_set_identification"
    )
    closed_set_identification_result_dir.mkdir(exist_ok=True, parents=True)

    verification_result_dir = Path(cfg.exp_dir) / dataset_name / "verification"
    verification_result_dir.mkdir(exist_ok=True, parents=True)

    # create result tables place holders
    open_set_recognition_result_metrics = {}
    open_set_uncertainty_result_metrics = {}

    closed_set_recognition_result_metrics = {}
    closed_set_uncertainty_result_metrics = {}

    verification_recognition_result_metrics = {}
    verification_uncertainty_result_metrics = {}

    open_set_ident_pretty_names = {}

    # define methods

    methods = []
    method_types = []
    if 'open_set_identification_methods' in cfg:
        methods+=cfg.open_set_identification_methods
        method_types+=["open_set_identification"] * len(
        cfg.open_set_identification_methods
         )
    if 'closed_set_identification_methods' in cfg:
        methods += cfg.closed_set_identification_methods
        method_types += ["closed_set_identification"] * len(cfg.closed_set_identification_methods)
    if 'verification_methods' in cfg:
        methods+= cfg.verification_methods
        method_types += ["verification"] * len(cfg.verification_methods)

    for method, method_type in zip(methods, method_types):
        evaluation_function = instantiate(method.evaluation_function)

        template_pooling = instantiate(method.template_pooling_strategy)
        tt = Face_Fecognition_test(
            evaluation_function=evaluation_function,
            test_dataset=test_dataset,
            embeddings_path=method.embeddings_path,
            template_pooling_strategy=template_pooling,
            use_detector_score=method.use_detector_score,
            use_two_galleries=cfg.use_two_galleries,
            recompute_template_pooling=cfg.recompute_template_pooling,
            open_set_identification_metrics=open_set_identification_metrics,
            closed_set_identification_metrics=closed_set_identification_metrics,
            verification_metrics=verification_metrics,
            open_set_uncertainty_metrics=open_set_uncertainty_metrics,
        )

        if method_type == "open_set_identification":  # 1:N test
            # introduce method name that fully defines method features

            method_name_parts = []
            method_name_parts.append(
                f"pooling-with-{template_pooling.__class__.__name__}"
            )
            method_name_parts.append(f"use-det-score-{method.use_detector_score}")
            method_name_parts.append(
                f"eval-with-{evaluation_function.__class__.__name__}"
            )
            confidence_function = evaluation_function.__dict__["confidence_function"]
            evaluation_function_args = dict(evaluation_function.__dict__)
            evaluation_function_args.pop("confidence_function")
            eval_args = get_args_string(evaluation_function_args)
            if len(eval_args) != 0:
                method_name_parts.append(f"eval-args-{eval_args}")
            method_name_parts.append(
                f"conf-func-{confidence_function.__class__.__name__}"
            )
            conf_args = get_args_string(confidence_function.__dict__)
            if len(conf_args) != 0:
                method_name_parts.append(f"conf-args-{conf_args}")

            method_name = "_".join(method_name_parts)
            print(method_name)
            # run recognition and uncertainty metric computation
            (
                open_set_identification_metric_values,
                open_set_uncertainty_metric_values,
            ) = tt.run_model_test_openset_identification()
            open_set_recognition_result_metrics[
                method_name
            ] = open_set_identification_metric_values

            open_set_uncertainty_result_metrics[
                method_name
            ] = open_set_uncertainty_metric_values

            open_set_ident_pretty_names.update({method_name: method.pretty_name})


        elif method_type == "verification":  # Basic 1:1 N0D1F1 test
            continue
            verification_metric_values = tt.run_model_test_verification()
            verif_far = verification_metric_values["fars"]
            verif_scores.append([verif_far, verification_metric_values["recalls"]])
            verif_names.append(save_name)
        elif method_type == "closed_set_identification":
            continue
            closed_set_identification_metric_values = (
                tt.run_model_test_closedset_identification()
            )
            closed_set_ident_scores.append(
                [
                    closed_set_identification_metric_values["ranks"],
                    closed_set_identification_metric_values["cmc"],
                ]
            )
            closed_set_ident_names.append(save_name)
        else:
            raise ValueError
    # open set identif metric table
    df = create_open_set_ident_recognition_metric_table(open_set_recognition_result_metrics, open_set_ident_pretty_names)
    df.to_csv(open_set_identification_result_dir / 'open_set_recognition.csv', index=False)
    # identif plot
    create_open_set_ident_plots(open_set_recognition_result_metrics, open_set_identification_result_dir, open_set_ident_pretty_names)
    

    # # verif plot

    # fig_verif = plot_tar_far_scores(scores=verif_scores, names=verif_names)
    # fig_verif.savefig(Path(cfg.exp_dir) / "tar_far_plot.png", dpi=300)
    # print("Plot verif path:")
    # print(str(Path(cfg.exp_dir) / "tar_far_plot.png"))

    # # cmc plot

    # fig_verif = plot_cmc_scores(
    #     scores=closed_set_ident_scores, names=closed_set_ident_names
    # )
    # fig_verif.savefig(Path(cfg.exp_dir) / "cmc_plot.png", dpi=300)
    # print("Plot closed ident path:")
    # print(str(Path(cfg.exp_dir) / "cmc_plot.png"))

    # # rejection plot

    # fig_rejection = plot_rejection_scores(
    #     scores=open_set_ident_rejection_scores, names=open_set_ident_rejection_names
    # )
    # fig_rejection.savefig(Path(cfg.exp_dir) / "rejection_plot.png", dpi=300)
    # print("Plot open ident rejection path:")
    # print(str(Path(cfg.exp_dir) / "rejection_plot.png"))


if __name__ == "__main__":
    main()
