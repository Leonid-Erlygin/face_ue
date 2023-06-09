from typing import Any, List, Sequence, Tuple
import numpy as np

from ..dataloaders import Query1N

class BaseSimilarity:
    similarity_sorted: bool = False 
    """Whether the method sorts similarity matrix columns in respect to gallery ids. Defaults to False"""

    @property
    def name(self) -> str:
        """Method name to display in plots

        :return: the name of this method, containing all its hyperparameters
        :rtype: str
        """
        return self.__class__.__name__

    def compute(
        self,
        query: Query1N
    ) -> np.ndarray:
        """Put your similarity matrix computation here

        :param query: query from the dataset
        :type query: Query1N
        :return: similarity_matrix
        :rtype: np.ndarray
        """
        raise NotImplementedError

    def __call__(self, query: Query1N) -> np.ndarray:
        """Calls compute, and sorts similarity matrix if neccessary. Most cases, you don't need to override this.

        :param query: query from the dataset
        :type query: Query1N
        :return: similarity_matrix
        :rtype: np.ndarray
        """
        similarity_matrix = self.compute(query)
        assert similarity_matrix.shape[0] == query.p and similarity_matrix.shape[1] == query.g
        if not self.similarity_sorted:
            similarity_matrix = similarity_matrix[:, np.argsort(query.gallery_ids)]

        return similarity_matrix
