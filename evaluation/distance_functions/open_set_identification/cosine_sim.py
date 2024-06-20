import numpy as np


from evaluation.distance_functions.open_set_identification.abc import Abstract1NEval
from evaluation.confidence_functions import AbstractConfidence
from evaluation.distance_functions.distance_functions import CosineSimDistance


class CosineSim(Abstract1NEval):
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        probe_feats,
        probe_unc,
        gallery_feats,
        gallery_unc,
    ):
        """
        probe_feats: n x num_z_samples x 512
        """
        compute_cosine_sim = CosineSimDistance()
        probe_feats = np.moveaxis(probe_feats, 2, 1)  # n x 512 x num_z_samples
        similarity = compute_cosine_sim(probe_feats, gallery_feats)
        assert similarity.shape[0] == probe_feats.shape[0]
        similarity = np.moveaxis(similarity, 2, 1)  # n x num_z_samples x K
        return similarity
