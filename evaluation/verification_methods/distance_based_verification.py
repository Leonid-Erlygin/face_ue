from typing import Any
import numpy as np
from tqdm import tqdm


class VerifEval:
    def __init__(self, distance_function) -> None:
        self.distance_function = distance_function

    def __call__(
        self,
        template_pooled_emb: np.ndarray,
        template_pooled_unc: np.ndarray,
        unique_templates: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
    ) -> Any:
        template2id = np.zeros(max(unique_templates) + 1, dtype=int)
        template2id[unique_templates] = np.arange(len(unique_templates))
        batch_size = 50000
        steps = int(np.ceil(len(p1) / batch_size))
        scores = []
        for id in tqdm(range(steps), "Verification"):
            feat1 = template_pooled_emb[
                template2id[p1[id * batch_size : (id + 1) * batch_size]]
            ]
            feat2 = template_pooled_emb[
                template2id[p2[id * batch_size : (id + 1) * batch_size]]
            ]

            unc1 = template_pooled_unc[
                template2id[p1[id * batch_size : (id + 1) * batch_size]]
            ]
            unc2 = template_pooled_unc[
                template2id[p2[id * batch_size : (id + 1) * batch_size]]
            ]
            scores.extend(self.distance_function(feat1, feat2, unc1, unc2))
        return np.array(scores)
