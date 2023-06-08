import numpy as np
from ..dataloader.data1N import Query1N
from .base import BaseSimilarity


def compute_scf_sim(mu_ij: np.ndarray, X_unc: np.ndarray, Y_unc: np.ndarray, d: int):
    from scipy.special import ive

    X_unc = X_unc[None, :, 0]
    k_i_times_k_j = Y_unc * X_unc
    k_ij = np.sqrt(Y_unc**2 + X_unc**2 + mu_ij * k_i_times_k_j)

    log_iv_i = np.log(1e-6 + ive(d / 2 - 1, Y_unc, dtype=Y_unc.dtype)) + Y_unc
    log_iv_j = np.log(1e-6 + ive(d / 2 - 1, X_unc, dtype=X_unc.dtype)) + X_unc
    log_iv_ij = np.log(1e-6 + ive(d / 2 - 1, k_ij, dtype=k_ij.dtype)) + k_ij

    scf_similarity = (
        (d / 2 - 1) * (np.log(Y_unc) + np.log(X_unc) - np.log(k_ij))  # type: ignore
        - (log_iv_i + log_iv_j - log_iv_ij)
        - d / 2 * np.log(2 * np.pi)
        - d * np.log(64)
    )

    return scf_similarity


class SCF(BaseSimilarity):
    def __init__(self, k_shift: float, use_cosine_sim_match: bool) -> None:
        """
        Implements SCF mutual “likelihood” of distributions belonging to the same person (sharing the same latent code)

        https://ieeexplore.ieee.org/document/9577756
        Eq. (13)
        """
        self.k_shift = k_shift
        self.use_cosine_sim_match = use_cosine_sim_match

    def compute(
        self,
        query: Query1N,
    ):
        mu_ij = 2 * np.dot(query.P, query.G.T)
        scf_similarity = compute_scf_sim(
            mu_ij, query.dG + self.k_shift, query.dP + self.k_shift, query.d
        )

        if self.use_cosine_sim_match:
            similarity = mu_ij / 2
        else:
            similarity = scf_similarity

        return similarity
