#!/usr/bin/env python3
import os
from typing import List
import numpy as np
from pathlib import Path
import hydra
from hydra.utils import instantiate
import sys
import os

from evaluation.dataloaders.data1N import DataLoader1N
from evaluation.confidence_functions import AbstractConfidence
from evaluation.metrics.base import BaseMetric
from evaluation.similarity_functions.base import BaseSimilarity

path = str(Path(__file__).parent.parent.absolute())
sys.path.insert(1, path)


@hydra.main(
    config_path=str(Path(__file__).resolve().parents[1] / "configs/uncertainty_benchmark"),
    config_name=Path(__file__).stem,
    version_base="1.2",
)
def main(cfg):
    metrics: List[BaseMetric] = []
    for metric in cfg.metrics:
        metric: BaseMetric = instantiate(metric)
        metrics.append(metric)

    for method in cfg.open_set_recognition_methods:
        similarity_function: BaseSimilarity = instantiate(method.similarity_function)
        confidence_function: AbstractConfidence = instantiate(method.confidence_function)
        dataloader: DataLoader1N = instantiate(method.dataloader)

        method_name = '@'.join([similarity_function.name, confidence_function.name])
        for i, query in enumerate(dataloader):
            similarity = similarity_function(query)
            confidence = confidence_function(similarity)
            for metric in metrics:
                metric(query, similarity, confidence, method_name)
            
            np.savez(os.path.join(cfg.exp_dir, f'{method_name}@{dataloader.name}_gallery{i}.npz'), similarity, confidence)

    for metric in metrics:
        plot_path = os.path.join(cfg.exp_dir, metric.name + '.png')
        metric.plot().savefig(plot_path, dpi=300)
        print(f"Plot path for {metric.name}:", plot_path)


if __name__ == "__main__":
    main()
