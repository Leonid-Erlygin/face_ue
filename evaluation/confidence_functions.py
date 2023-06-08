from typing import Any
import numpy as np
from scipy.special import softmax

from abc import ABC

class AbstractConfidence(ABC):
    def __call__(self, similarity_matrix) -> Any:
        raise NotImplementedError
    
    @property
    def name(self):
        return self.__class__.__name__

class NAC_confidence(AbstractConfidence):
    def __init__(self, k: int, s: float, shift: float, normalize: bool) -> None:
        """
        Implemetns Neighborhood Aware Cosine (NAC) that computes
        similarity based on neighborhood information

        https://arxiv.org/pdf/2301.01922.pdf

        :param k: for kNN
        :param s: scale (=1/T)
        """
        self.k = k  # 15 is a good value
        self.s = s  # 1 is a good value
        self.shift = shift
        self.normalize = normalize

    def __call__(self, similarity_matrix: np.ndarray) -> Any:
        """
        :param similarity: (probe_size, gallery_size) marix, which specifies closeness of all test images to each gallery class
        :return probe_score: specifies confinence that particular test image belongs to predicted class
            image's probe_score is less than operating threshold τ, then this image get rejected as imposter
        """
        if self.normalize:
            similarity_matrix = (similarity_matrix - np.mean(similarity_matrix, axis=1, keepdims=True))/ np.std(similarity_matrix, axis=1, keepdims=True)
        top_k_logits = np.sort(similarity_matrix, axis=1)[:, -self.k :]
        # if self.normalize:
        #     top_k_logits = (
        #         top_k_logits - np.mean(top_k_logits, axis=1, keepdims=True)
        #     ) / np.std(top_k_logits, axis=1, keepdims=True)
        return softmax((top_k_logits) * self.s, axis=1)[:, -1]


class MaxSimilarity_confidence(AbstractConfidence):
    def __init__(self, foo: str) -> None:
        """
        Returns confidence for each test image of being gallery (known class) sample
        Here we take similarity to most similar class in gallery as confidence measure
        """
        self.foo = foo

    def __call__(self, similarity_matrix: np.ndarray) -> Any:
        """
        :param similarity: (probe_size, gallery_size) marix, which specifies closeness of all test images to each gallery class
        :return probe_score: specifies confinence that particular test image belongs to predicted class
            image's probe_score is less than operating threshold τ, then this image get rejected as imposter
        """
        return np.max(similarity_matrix, axis=1)
