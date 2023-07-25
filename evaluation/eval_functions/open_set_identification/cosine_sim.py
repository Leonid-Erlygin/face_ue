import numpy as np


from evaluation.eval_functions.open_set_identification.abc import Abstract1NEval
from evaluation.confidence_functions import AbstractConfidence
from evaluation.eval_functions.distaince_functions import CosineSimDistance

from scipy.special import softmax

class CosineSim(Abstract1NEval):
    def __init__(self, confidence_function: AbstractConfidence) -> None:
        self.confidence_function = confidence_function

    def __call__(
        self,
        probe_feats,
        probe_unc,
        gallery_feats,
        gallery_unc,
    ):
        print(
            "probe_feats: %s, gallery_feats: %s"
            % (probe_feats.shape, gallery_feats.shape)
        )
        compute_cosine_sim = CosineSimDistance()
        similarity = compute_cosine_sim(
            probe_feats, gallery_feats
        )  # np.dot(probe_feats, gallery_feats.T)  # (19593, 1772)
        similarity = (similarity + 1)/2
        probe_score = self.confidence_function(similarity)

        return similarity, probe_score
