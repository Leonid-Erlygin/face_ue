#!/usr/bin/env python3
# from evaluation.uncertainty_metrics import DisposeBasedOnUnc


# from uncertainty_metrics import DisposeBasedOnUnc
from pathlib import Path
import hydra
import omegaconf
from omegaconf import OmegaConf
from hydra.utils import instantiate
import numpy as np
from itertools import product
from evaluation.face_recognition_test import Face_Fecognition_test

# from face_recognition_test import Face_Fecognition_test
import pandas as pd
import matplotlib.pyplot as plt
from evaluation.visualize import (
    plot_dir_far_scores,
    plot_cmc_scores,
    plot_tar_far_scores,
    plot_rejection_scores,
)


def instantiate_list(query_list):
    return [instantiate(value) for value in query_list]


def get_args_string(d):
    args = []
    for key, value in d.items():
        args.append(f"{key}:{value}")
    return "-".join(args)


def create_method_name(method, template_pooling, recognition_method):
    method_name_parts = []
    method_name_parts.append(f"pooling-with-{template_pooling.__class__.__name__}")
    method_name_parts.append(f"use-det-score-{method.use_detector_score}")
    method_name_parts.append(f"osr-method-{recognition_method.__class__.__name__}")
    method_name = "_".join(method_name_parts)
    return method_name


def init_methods(cfg):
    methods = []
    method_types = []
    if "open_set_identification_methods" in cfg:
        methods += cfg.open_set_identification_methods
        method_types += ["open_set_identification"] * len(
            cfg.open_set_identification_methods
        )
    if "closed_set_identification_methods" in cfg:
        methods += cfg.closed_set_identification_methods
        method_types += ["closed_set_identification"] * len(
            cfg.closed_set_identification_methods
        )
    if "verification_methods" in cfg:
        methods += cfg.verification_methods
        method_types += ["verification"] * len(cfg.verification_methods)
    return methods, method_types


def multiply_methods(cfg, methods, method_task_type):
    assert len(list(set(method_task_type))) == 1  # only osr
    if "tau_list" not in cfg:
        return methods, method_task_type
    new_methods = []
    for method in methods:
        if "kappa_is_tau" in method.recognition_method:
            assert method.recognition_method.kappa_is_tau

            if type(method.recognition_method.T) != omegaconf.listconfig.ListConfig:
                method.recognition_method.T = [method.recognition_method.T] * len(
                    method.recognition_method.kappa
                )
            if (
                type(method.recognition_method.T_data_unc)
                != omegaconf.listconfig.ListConfig
            ):
                method.recognition_method.T_data_unc = [
                    method.recognition_method.T_data_unc
                ] * len(method.recognition_method.kappa)
            for tau, T, T_data_unc in zip(
                method.recognition_method.kappa,
                method.recognition_method.T,
                method.recognition_method.T_data_unc,
            ):
                new_method = method.copy()
                new_method.recognition_method.kappa = tau
                new_method.recognition_method.T = T
                new_method.recognition_method.T_data_unc = T_data_unc
                new_method.pretty_name = (
                    method.pretty_name
                    # + f"_tau-{np.round(tau, 2)}_T-{np.round(T, 2)}_T_data-{np.round(T_data_unc, 2)}"
                )
                new_methods.append(new_method)
        else:
            new_methods.append(method)
    return new_methods, [method_task_type[0]] * len(new_methods)


@hydra.main(
    config_path=str(
        Path(__file__).resolve().parents[1] / "configs/uncertainty_benchmark"
    ),
    config_name=Path(__file__).stem,
    version_base="1.2",
)
def main(cfg):
    # 0. Define methods
    methods, method_task_type = init_methods(cfg)
    tasks_names = list(set(method_task_type))

    # instantiate metrics
    recognition_metrics = {
        task: instantiate_list(getattr(cfg, f"{task}_metrics")) for task in tasks_names
    }
    uncertainty_metrics = {
        task: instantiate_list(getattr(cfg, f"{task}_uncertainty_metrics"))
        for task in tasks_names
    }

    # instantiate datasets
    test_datasets = instantiate_list(cfg.test_datasets)
    dataset_names = [test_dataset.dataset_name for test_dataset in test_datasets]
    # create result dictionary
    metric_values = {
        (task, dataset_name): {"recognition": {}, "uncertainty": {}}
        for task, dataset_name in product(tasks_names, dataset_names)
    }
    unc_values = {dataset_name: {} for dataset_name in dataset_names}

    # create pretty name map
    pretty_names = {task: {} for task in tasks_names}

    # 1. Compute recognition and uncertaity metrics
    for (method, task_type), far, test_dataset in product(
        zip(methods, method_task_type), cfg.far_list, test_datasets
    ):
        dataset_name = test_dataset.dataset_name

        # instantiate method
        gallery_template_pooling_strategy = instantiate(
            method.gallery_template_pooling_strategy
        )
        if hasattr(method, "probe_template_pooling_strategy"):
            probe_template_pooling_strategy = instantiate(
                method.probe_template_pooling_strategy
            )
        else:
            probe_template_pooling_strategy = None
        recognition_method = instantiate(method.recognition_method)
        # set far value
        recognition_method.far = far
        # create unique method name
        if cfg.create_pool_plot is False:
            method_name = (
                create_method_name(
                    method,
                    gallery_template_pooling_strategy,
                    recognition_method,
                )
                + f"_{method.pretty_name}"
            )
        else:
            method_name = (
                create_method_name(
                    method,
                    gallery_template_pooling_strategy,
                    recognition_method,
                )
                + f"_{method.pretty_name}"
            )
        print(method_name)
        pretty_names[task_type][method_name] = method.pretty_name
        embeddings_path = (
            Path(test_dataset.dataset_path)
            / f"embeddings/{method.embeddings}_embs_{dataset_name}.npz"
        )
        # create tester
        tt = Face_Fecognition_test(
            task_type=task_type,
            method_name=method_name,
            pretty_name=method.pretty_name,
            recognition_method=recognition_method,
            test_dataset=test_dataset,
            embedding_type=method.embeddings,
            embeddings_path=embeddings_path,
            gallery_template_pooling_strategy=gallery_template_pooling_strategy,
            probe_template_pooling_strategy=probe_template_pooling_strategy,
            use_detector_score=method.use_detector_score,
            use_two_galleries=cfg.use_two_galleries,
            recompute_template_pooling=cfg.recompute_template_pooling,
            recognition_metrics=recognition_metrics,
            uncertainty_metrics=uncertainty_metrics,
        )

        (
            recognition_metric_values,
            uncertainty_metric_values,
            predicted_unc,
        ) = tt.predict_and_compute_metrics()
        if task_type == "verification":
            metric_values[(task_type, dataset_name)]["recognition"][
                method_name
            ] = recognition_metric_values
            metric_values[(task_type, dataset_name)]["uncertainty"][
                method_name
            ] = uncertainty_metric_values
        else:
            metric_values[(task_type, dataset_name)]["recognition"][
                (method_name, far)
            ] = recognition_metric_values
            metric_values[(task_type, dataset_name)]["uncertainty"][
                (method_name, far)
            ] = uncertainty_metric_values
            unc_values[dataset_name][(method_name, far)] = predicted_unc

    # 2. Create plots and tables

    # load metric name converter
    metric_pretty_names = OmegaConf.load(cfg.metric_pretty_name_path)

    for task_type, dataset_name in metric_values:
        # create output dir
        out_dir = Path(cfg.exp_dir) / str(task_type) / str(dataset_name)
        out_dir.mkdir(parents=True, exist_ok=True)

        metric_names = []
        model_names = []
        recognition_metric_names = []
        for model_name, metric in metric_values[(task_type, dataset_name)][
            "uncertainty"
        ].items():
            for key in metric:
                if "unc_metric" in key:
                    metric_names.append(key)
                    model_names.append(model_name)
            break
        for model_name, metric in metric_values[(task_type, dataset_name)][
            "recognition"
        ].items():
            for key in metric:
                if "TAR@FAR" in key:
                    recognition_metric_names.append(key)
            break
        fractions = next(
            iter(metric_values[(task_type, dataset_name)]["uncertainty"].items())
        )[1]["fractions"]

        if task_type == "verification":
            names = []
            scores = []
            for model_name, metric in metric_values[(task_type, dataset_name)][
                "recognition"
            ].items():
                names.append(pretty_names[task_type][model_name])
                scores.append([metric["fars"], metric["recalls"]])
            fig = plot_tar_far_scores(scores, names)
            fig.savefig(
                out_dir / f"tar_far.png",
                dpi=300,
            )
            plt.close(fig)
            # create rejection plots

            for metric_name in metric_names:
                scores = []
                model_names = []
                data_rows = []
                for method_name, metrics in metric_values[(task_type, dataset_name)][
                    "uncertainty"
                ].items():
                    if "random" in pretty_names[task_type][method_name]:
                        random_area = metrics["fractions"][-1] * np.mean(
                            metrics[metric_name]
                        )
                        far_to_random_oracle_areas[far][0] = random_area
                        if cfg.display_oracle_curve is False:
                            continue
                    elif "oracle" in pretty_names[task_type][method_name]:
                        oracle_area = metrics["fractions"][-1] * np.mean(
                            metrics[metric_name]
                        )
                        far_to_random_oracle_areas[far][1] = oracle_area
                        if cfg.display_oracle_curve is False:
                            continue
                    model_names.append(pretty_names[task_type][method_name])
                    scores.append((metrics["fractions"], metrics[metric_name]))
                    data_rows.append(
                        [pretty_names[task_type][method_name], *metrics[metric_name]]
                    )

                metric_pretty_name = metric_pretty_names[metric_name.split(":")[-1]]
                if isinstance(metric_pretty_name, str):
                    metric_pretty_name = [metric_pretty_name]
                metric_pretty_name = " ".join(metric_pretty_name)

                auc_at_far_data_frames = []
                aggr_filter_tables_dir = out_dir / "filter_tabels"
                aggr_filter_tables_dir.mkdir(parents=True, exist_ok=True)

                filter_plots_dir = out_dir / "filter_plots"
                filter_tables_dir = out_dir / "filter_tabels"
                filter_plots_dir.mkdir(parents=True, exist_ok=True)
                filter_tables_dir.mkdir(parents=True, exist_ok=True)

                fig, rejection_metric_values = plot_rejection_scores(
                    scores=scores,
                    names=model_names,
                    y_label=f"{metric_pretty_name}",
                    # random_area=far_to_random_oracle_areas[far][0],
                    # oracle_area=far_to_random_oracle_areas[far][1],
                )
                fig.savefig(
                    filter_plots_dir / f"{metric_name.split(':')[-1]}_filtering.png",
                    dpi=300,
                )
                plt.close(fig)

                # # save filter table
                # rejection_df = pd.DataFrame(data_rows, columns=column_names)
                # rejection_df.to_csv(
                #     filter_tables_dir / f'{metric_name.split(":")[-1]}_filtering.csv'
                # )
                # # save auc table

                # auc_at_far_data_frames.append(
                #     pd.DataFrame(
                #         {
                #             "models": model_names,
                #             f"AUCS": rejection_metric_values,
                #         }
                #     )
                # )

            # create recognition tables
            data_rows = []
            column_names = []
            for method_name, metrics in metric_values[(task_type, dataset_name)][
                "recognition"
            ].items():

                data_rows.append(
                    [
                        pretty_names[task_type][method_name],
                        *[
                            metrics[metric_name]
                            for metric_name in recognition_metric_names
                        ],
                    ]
                )
            recognition_df = pd.DataFrame(
                data_rows, columns=["models"] + recognition_metric_names
            )
            recognition_df.to_csv(out_dir / f"tar@fars.csv")
            continue

        # create rejection plots
        # fraction_data_rows = {frac: [] for frac in fractions}
        # fraction_column_names = ["models"] + [
        #     metric_name.split(":")[-1] for metric_name in metric_names
        # ]
        column_names = ["models", *[str(np.round(frac, 4)) for frac in fractions]]

        for metric_name in metric_names:
            far_to_model_names = {far: [] for far in cfg.far_list}
            far_to_scores = {far: [] for far in cfg.far_list}
            far_to_data_rows = {far: [] for far in cfg.far_list}
            far_to_random_oracle_areas = {far: [None, None] for far in cfg.far_list}
            for (method_name, far), metrics in metric_values[(task_type, dataset_name)][
                "uncertainty"
            ].items():
                if "random" in pretty_names[task_type][method_name]:
                    random_area = metrics["fractions"][-1] * np.mean(
                        metrics[metric_name]
                    )
                    far_to_random_oracle_areas[far][0] = random_area
                    if cfg.display_oracle_curve is False:
                        continue
                elif "oracle" in pretty_names[task_type][method_name]:
                    oracle_area = metrics["fractions"][-1] * np.mean(
                        metrics[metric_name]
                    )
                    far_to_random_oracle_areas[far][1] = oracle_area
                    if cfg.display_oracle_curve is False:
                        continue
                far_to_model_names[far].append(pretty_names[task_type][method_name])
                far_to_scores[far].append((metrics["fractions"], metrics[metric_name]))
                far_to_data_rows[far].append(
                    [pretty_names[task_type][method_name], *metrics[metric_name]]
                )

            metric_pretty_name = metric_pretty_names[metric_name.split(":")[-1]]
            if isinstance(metric_pretty_name, str):
                metric_pretty_name = [metric_pretty_name]
            metric_pretty_name = " ".join(metric_pretty_name)

            auc_at_far_data_frames = []
            aggr_filter_tables_dir = out_dir / "filter_tabels"
            aggr_filter_tables_dir.mkdir(parents=True, exist_ok=True)
            for far in far_to_scores:
                filter_plots_dir = out_dir / "filter_plots" / str(far)
                filter_tables_dir = out_dir / "filter_tabels" / str(far)
                filter_plots_dir.mkdir(parents=True, exist_ok=True)
                filter_tables_dir.mkdir(parents=True, exist_ok=True)

                fig, rejection_metric_values = plot_rejection_scores(
                    scores=far_to_scores[far],
                    names=far_to_model_names[far],
                    y_label=f"{metric_pretty_name}",
                    random_area=far_to_random_oracle_areas[far][0],
                    oracle_area=far_to_random_oracle_areas[far][1],
                )
                fig.savefig(
                    filter_plots_dir / f"{metric_name.split(':')[-1]}_filtering.png",
                    dpi=300,
                )
                plt.close(fig)

                # save filter table
                rejection_df = pd.DataFrame(far_to_data_rows[far], columns=column_names)
                rejection_df.to_csv(
                    filter_tables_dir / f'{metric_name.split(":")[-1]}_filtering.csv'
                )
                # save auc table

                auc_at_far_data_frames.append(
                    pd.DataFrame(
                        {
                            "models": far_to_model_names[far],
                            f"FAR={far}": rejection_metric_values,
                        }
                    )
                )
            for i in range(len(auc_at_far_data_frames) - 1):
                auc_at_far_data_frames[0] = pd.merge(
                    auc_at_far_data_frames[0],
                    auc_at_far_data_frames[i + 1],
                    on="models",
                )
            auc_at_far_data_frames[0].to_csv(
                aggr_filter_tables_dir
                / f'{metric_name.split(":")[-1]}_prr_filtering.csv'
            )
            continue
            # save trans auc table
            new_auc_df_lines = []
            new_auc_df_columns = ["models"] + list(auc_df.columns[3:-1])
            aucs = []
            model_names = []
            for far, df in auc_df.groupby("far"):
                new_auc_df_columns.append(f"FAR={far}")
                aucs.append(list(df["auc"]))
                model_names = list(df["models"])
            aucs = np.array(aucs).T
            aucs = list(aucs)
            for model_name, auc in zip(model_names, aucs):
                new_auc_df_lines.append([model_name, *auc.tolist()])
            new_auc_df = pd.DataFrame(new_auc_df_lines, columns=new_auc_df_columns)
            new_auc_df.to_csv(
                out_table_dir / f'{metric_name.split(":")[-1]}_aucs_pretty.csv'
            )

        # for frac, data_rows in fraction_data_rows.items():
        #     frac_rejection_df = pd.DataFrame(data_rows, columns=fraction_column_names)
        #     frac_rejection_df.to_csv(
        #         out_table_fractions_dir
        #         / f'{str(np.round(frac, 4)).ljust(6, "0")}_frac_rejection.csv'
        #     )
        if cfg.create_pool_plot:
            # create pool plot
            tables_path = out_dir / "tabels"
            far_table = pd.read_csv(tables_path / "far_rejection.csv")
            dir_table = pd.read_csv(tables_path / "dir_rejection.csv")

            y_label = "Detection $&$ Identification rate"
            model_to_points = {}
            for model_name, far_value, dir_value in zip(
                far_table.models, far_table["0.0"], dir_table["0.0"]
            ):
                if model_name in model_to_points:
                    model_to_points[model_name].append([far_value, dir_value])
                else:
                    model_to_points[model_name] = [[far_value, dir_value]]
            names = []
            scores = []
            for model_name, points in model_to_points.items():
                names.append(model_name)
                scores.append(np.array(points).T)
            scores = np.array(scores)

            fig = plot_dir_far_scores(scores, names, y_label, marker=".")
            fig.savefig(out_dir / f"pool.png", dpi=300)

            plt.close(fig)

    print(cfg.exp_dir)


if __name__ == "__main__":
    main()
