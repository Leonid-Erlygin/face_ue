from torch.utils.data.sampler import Sampler
import numpy as np


class UniformBatchSamplerWithBins(Sampler):
    r"""Yield a mini-batch of indices.

    Args:
        cosine_sim_path: Path to computer cosine similarities of each sample to its class center
        batch_size: Size of mini-batch.
    """

    def __init__(self, cosine_sim_path, batch_size, cosine_border_values: list):
        self.batch_size = batch_size
        cosine_sim = np.load(cosine_sim_path)
        self.sorted_id_map = np.argsort(cosine_sim)
        sorted_cosine_sim = np.sort(cosine_sim)
        n_bins = len(cosine_border_values)
        self.bin_sample_size = batch_size / n_bins
        self.id_borders = []
        id_widths = []
        prev_border = 0
        for cosine_border_values in cosine_border_values:
            id_border = np.searchsorted(sorted_cosine_sim, cosine_border_values)
            self.id_borders.append(id_border)
            id_widths.append(id_border - prev_border)
            prev_border = id_border
        assert np.searchsorted(sorted_cosine_sim, 1) == sorted_cosine_sim.shape[0]

        self.rng = np.random.default_rng(776)
        self.ids = np.arange(sorted_cosine_sim.shape[0])

    def __iter__(self):
        # implement logic of sampling here
        for _ in range(int(len(self.ids) // self.batch_size)):
            sorted_batch_ids = []
            prev_id = 0
            for id_border in self.id_borders:
                sorted_batch_ids.append(
                    self.rng.choice(
                        np.arange(prev_id, id_border),
                        int(self.bin_sample_size),
                    )
                )
                prev_id = id_border
            sorted_batch_ids = np.concatenate(sorted_batch_ids)
            uniform_batch_ids = self.sorted_id_map[sorted_batch_ids]
            yield uniform_batch_ids

    def __len__(self):
        return int(len(self.ids) // self.batch_size)


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
