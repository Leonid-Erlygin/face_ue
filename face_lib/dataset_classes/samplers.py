from torch.utils.data.sampler import Sampler
import numpy as np


class UniformBatchSampler(Sampler):
    r"""Yield a mini-batch of indices.

    Args:
        cosine_sim_path: Path to computer cosine similarities of each sample to its class center
        batch_size: Size of mini-batch.
    """

    def __init__(self, cosine_sim_path, batch_size):
        self.batch_size = batch_size
        cosine_sim = np.load(cosine_sim_path)
        self.sorted_id_map = np.argsort(cosine_sim)
        sorted_cosine_sim = np.sort(cosine_sim)
        coarseness = 300
        derivative_cosine_sim = np.gradient(sorted_cosine_sim[::coarseness])
        self.sorted_id_scores = np.array(
            [
                derivative_cosine_sim[i // coarseness]
                for i in range(len(self.sorted_id_map))
            ]
        )
        self.sorted_id_scores = self.sorted_id_scores / np.sum(self.sorted_id_scores)
        self.rng = np.random.default_rng(776)
        self.ids = np.arange(self.sorted_id_scores.shape[0])

    def __iter__(self):
        # implement logic of sampling here
        for _ in range(int(len(self.sorted_id_scores) // self.batch_size)):
            sorted_batch_ids = self.rng.choice(
                self.ids,
                self.batch_size,
                p=self.sorted_id_scores,
            )
            uniform_batch_ids = self.sorted_id_map[sorted_batch_ids]
            yield uniform_batch_ids
        # uniform_batch_cosine = cosine_sim[uniform_batch_ids]
        # batch = []
        # for i, item in enumerate(self.data):
        #     batch.append(i)

    def __len__(self):
        return int(len(self.sorted_id_scores) // self.batch_size)
