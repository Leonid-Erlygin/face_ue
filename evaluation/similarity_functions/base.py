from typing import Any, List, Sequence, Tuple
import numpy as np

from ..dataloaders import Query1N

class BaseSimilarity:
    """Whether the method sorts similarity matrix columns in respect to gallery ids. Defaults to False"""

    @property
    def name(self) -> str:
        """Method name to display in plots

        :return: the name of this method, containing all its hyperparameters
        :rtype: str
        """
        return self.__class__.__name__

    def __call__(
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
