import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
from .base import BaseSimilarity
from ..dataloaders import Query1N


class TcmNN(BaseSimilarity):
    def __init__(self, number_of_nearest_neighbors, scale, p_value_cache_path) -> None:
        """
        implemets knn based open-set face identification algorithm.
        See
        https://ieeexplore.ieee.org/document/1512050
        and
        https://link.springer.com/chapter/10.1007/3-540-36755-1_32
        """
        self.number_of_nearest_neighbors = number_of_nearest_neighbors
        self.scale = scale  # needed because we have |D_i^y| = 1 and |D_i^{-y}|!=1
        self.p_value_cache_path = Path(p_value_cache_path)

    def __call__(self, query: Query1N):
        # 1. compute distances from each gallery class to other gallery classes
        # here each class has exact one feature vector

        gallery_distance_matrix = -np.dot(query.G, query.G.T) + 1  # (3531, 3531)
        D_minus_y = np.sort(gallery_distance_matrix, axis=1)[
            :, 1 : self.number_of_nearest_neighbors + 1
        ]
        D_minus_sum = np.sum(D_minus_y, axis=1)
        # 2. compute distances from each probe feature to all gallery classes
        probe_gallery_distance_matrix = -np.dot(query.P, query.G.T) + 1  # (19593, 3531)
        probe_gallery_distance_matrix_sorted = np.argsort(
            probe_gallery_distance_matrix, axis=1
        )[:, : self.number_of_nearest_neighbors + 1]

        # 3. calculate strangeness of gallery samples depending on class of probe sample

        probe_p_values = []
        gallery_strangeness = np.zeros(shape=(query.G.shape[0], query.G.shape[0]))
        gallery_strangeness += self.scale / D_minus_sum[np.newaxis, :]
        l = query.G.shape[0]
        # here is error as we do not account for just added probe 'probe_index'
        # while computing strangeness for non probe class gallery classes
        cache_path = Path(
            self.p_value_cache_path
            / f"k_{self.number_of_nearest_neighbors}_scale_{self.scale}_gallery_size_{l}.npy"
        )
        if cache_path.is_file():
            probe_p_values = np.load(cache_path)
        else:
            for probe_index in tqdm(range(query.P.shape[0])):
                np.fill_diagonal(
                    gallery_strangeness,
                    probe_gallery_distance_matrix[probe_index] / D_minus_sum,
                )
                other_class_distance_sum = []
                default_sum = np.sum(
                    probe_gallery_distance_matrix_sorted[probe_index][:-1]
                )

                for gallery_id in range(query.G.shape[0]):
                    if gallery_id in probe_gallery_distance_matrix_sorted[probe_index]:
                        a = probe_gallery_distance_matrix_sorted[probe_index].copy()
                        a[np.where(a == gallery_id)] = 0
                        other_class_distance_sum.append(np.sum(a))
                    else:
                        other_class_distance_sum.append(default_sum)
                probe_strangeness = probe_gallery_distance_matrix[
                    probe_index
                ] / np.array(other_class_distance_sum)
                # Eq. (8) https://ieeexplore.ieee.org/document/1512050

                p_values = 1 / (l + 1) + (np.sum(gallery_strangeness, axis=1)) / (
                    (l + 1) * probe_strangeness
                )
                probe_p_values.append(p_values)
            probe_p_values = np.array(probe_p_values)  # (19593, 1772)
            np.save(cache_path, probe_p_values)

        return probe_p_values
