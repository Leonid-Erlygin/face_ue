import pandas as pd
import numpy as np
from pathlib import Path
import hydra
from omegaconf import OmegaConf


def compute_best_values(table, metric_order):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]

    newdf = table.select_dtypes(include=numerics)
    best_values = {}
    for column_name in newdf.columns:
        if column_name not in metric_order or metric_order[column_name] == "high":
            sorted_values = np.sort(table[column_name].values)[::-1]
        elif metric_order[column_name] == "low":
            sorted_values = np.sort(table[column_name].values)
        else:
            raise ValueError
        best_values[column_name] = (sorted_values[0], sorted_values[1])
    return best_values


def create_table_head(result_latex_code, caption, table_lable, cfg):
    used_columns = cfg.used_columns
    column_pretty_name = cfg.pretty_name.column
    if cfg.fix_table:
        result_latex_code += "\\begin{table}[H]\n"
    else:
        result_latex_code += "\\begin{table}\n"
    if cfg.use_scriptsize:
        result_latex_code += "\\scriptsize\n"
    # result_latex_code += "\\caption{" + caption + "}\n"
    # result_latex_code += "\\label{" + table_lable + "}\n"
    if cfg.use_adjustbox:
        result_latex_code += "\\begin{adjustbox}{width=0.5\\textwidth}\n"
    result_latex_code += (
        "\\begin{tabular}{" + "l" + "c" * (len(used_columns) - 1) + "}\n"
    )
    result_latex_code += "\\toprule\n"

    next_column_index = 1
    for raw_column_name in used_columns:
        pretty_name = column_pretty_name[raw_column_name]
        if isinstance(pretty_name, str):
            pretty_name = [pretty_name]
        result_latex_code += (
            "\\begin{tabular}{c}\n" + "\\\\\n".join(pretty_name) + "\n\\end{tabular}"
        )
        if next_column_index < len(used_columns):
            result_latex_code += "&\n"
        else:
            result_latex_code += "\\\\\n"
            result_latex_code += "\midrule\n"
        next_column_index += 1
    return result_latex_code


def create_table_body(result_latex_code, cfg):
    all_metric_values = pd.read_csv(cfg.metric_table_path)

    # draw table

    # next_ds_index = 1
    best_values = compute_best_values(
        all_metric_values[cfg.used_columns], cfg.metric_order
    )
    for row_index, (_, row) in enumerate(all_metric_values.iterrows()):
        for column_index, column_name in enumerate(cfg.used_columns):
            if column_name == "models":
                result_latex_code += cfg.pretty_name.model[row[column_name]] + " & "
            elif column_name != "models":
                # metric value
                metric_value = row[column_name]
                if metric_value == best_values[column_name][0]:
                    # best value
                    result_latex_code += (
                        "\\textbf{" + str(np.round(metric_value, cfg.round_num)) + "} "
                    )
                elif metric_value == best_values[column_name][1]:
                    # second best value
                    result_latex_code += (
                        "\\underline{"
                        + str(np.round(metric_value, cfg.round_num))
                        + "} "
                    )
                else:
                    result_latex_code += (
                        f" {str(np.round(metric_value, cfg.round_num))} "
                    )
                if column_index < len(cfg.used_columns) - 1:
                    result_latex_code += "& "
                else:
                    # end of row
                    result_latex_code += "\\\\\n"
    return result_latex_code


def create_table_tail(result_latex_code, caption, table_lable, cfg):
    result_latex_code += "\\bottomrule\n"
    result_latex_code += "\\end{tabular}\n"
    if cfg.use_adjustbox:
        result_latex_code += "\\end{adjustbox}\n"
    result_latex_code += "\\caption{" + caption + "}\n"
    result_latex_code += "\\label{" + table_lable + "}\n"
    result_latex_code += "\\end{table}\n"

    return result_latex_code


@hydra.main(
    config_path="/app/configs/latex_tables",
    config_name=Path(__file__).stem,
    version_base="1.2",
)
def run(cfg):
    result_latex_code = """"""
    if "{dataset_name}" in cfg.caption:
        caption = cfg.caption.format(
            dataset_name=cfg.dataset, task=cfg.pretty_name.task[cfg.task]
        )
    else:
        caption = cfg.caption
    if "{dataset}" in cfg.metric_table_path:
        cfg.metric_table_path = cfg.metric_table_path.format(dataset=cfg.dataset)
    cfg.used_columns = OmegaConf.to_container(cfg.used_columns_dict)[cfg.task][
        cfg.dataset
    ]
    result_latex_code = create_table_head(
        result_latex_code,
        caption,
        cfg.table_lable,
        cfg,
    )

    result_latex_code = create_table_body(result_latex_code, cfg)

    result_latex_code = create_table_tail(
        result_latex_code, caption, cfg.table_lable, cfg
    )

    # save result
    with open(Path(cfg.exp_dir) / "table.tex", "w") as fd:
        fd.write(result_latex_code)
    print("Out file:")
    print(str(Path(cfg.exp_dir) / "table.tex"))


if __name__ == "__main__":
    run()
