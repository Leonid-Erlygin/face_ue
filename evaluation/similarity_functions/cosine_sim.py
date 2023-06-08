import numpy as np
from .base import BaseSimilarity
from ..dataloader.data1N import Query1N


class CosineSim(BaseSimilarity):
    def compute(
        self,
        query: Query1N,
    ):
        similarity = np.dot(query.P, query.G.T)  # (19593, 1772)

        # Compute Detection & identification rate for open set recognition
        return similarity