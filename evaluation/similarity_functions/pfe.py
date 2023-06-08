from pathlib import Path

import numexpr as ne
import numpy as np
from tqdm import tqdm
from joblib import Memory
from .base import BaseSimilarity
from ..dataloader.data1N import Query1N


@Memory("/app/cache/pfe_cache").cache
def compute_pfe(
    similarity,
    probe_feats,
    probe_sigma_sq,
    gallery_feats,
    gallery_sigma_sq,
):
    sigma_sq_sum = ne.evaluate("probe_sigma_sq + gallery_sigma_sq")
    slice = ne.evaluate(
        "(probe_feats - gallery_feats)**2 / sigma_sq_sum + log(sigma_sq_sum)"
    )
    slice_sum = ne.evaluate("sum(slice, axis=2)")
    return ne.evaluate("slice_sum + similarity")


class PFE(BaseSimilarity):
    def __init__(self, variance_scale: float) -> None:
        """
        Implements PFE “likelihood” of distributions belonging to the same person (sharing the same latent code)

        https://ieeexplore.ieee.org/document/9008376
        Eq. (3)
        """
        self.variance_scale = variance_scale

    def compute(
        self,
        query: Query1N,
    ):
        similarity = np.dot(query.P, query.G.T)  # (19593, 1772)

        # compute pfe likelihood

        chunck_size = 128
        pfe_similarity = np.stack(
            [
                compute_pfe(
                    similarity[..., chk],
                    query.P[..., chk],
                    query.dP[:, None, chk] * self.variance_scale,
                    query.G[..., chk],
                    query.dG[None, :, chk] * self.variance_scale,
                )
                for chk in tqdm(np.array_split(np.arange(query.d), query.d // chunck_size))
            ],
            axis=-1,
        )

        return pfe_similarity
