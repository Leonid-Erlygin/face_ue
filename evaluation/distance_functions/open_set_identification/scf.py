import numpy as np

# from evaluation.metrics import compute_detection_and_identification_rate
from evaluation.distance_functions.open_set_identification.abc import Abstract1NEval
from evaluation.confidence_functions import AbstractConfidence


from evaluation.distance_functions.distance_functions import ScfSim


class SCF(Abstract1NEval):
    def __init__(
        self,
        k_shift: float,
        cosine_pred: bool,
    ) -> None:
        """
        Implements SCF mutual “likelihood” of distributions belonging to the same person (sharing the same latent code)

        https://ieeexplore.ieee.org/document/9577756
        Eq. (13)
        """
        self.k_shift = k_shift
        self.cosine_pred = cosine_pred

    def __call__(
        self,
        probe_feats,
        probe_unc,
        gallery_feats,
        gallery_unc,
    ):
        # print(
        #     "probe_feats: %s, gallery_feats: %s"
        #     % (probe_feats.shape, gallery_feats.shape)
        # )
        compute_scf_sim = ScfSim()
        gallery_unc = gallery_unc + self.k_shift
        probe_unc = probe_unc + self.k_shift

        scf_similarity = compute_scf_sim(
            probe_feats, gallery_feats, gallery_unc, probe_unc
        )

        if self.cosine_pred:
            similarity = np.dot(probe_feats, gallery_feats.T)
        else:
            similarity = scf_similarity

        return similarity
