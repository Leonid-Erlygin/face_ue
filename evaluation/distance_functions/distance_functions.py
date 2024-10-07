from tqdm import tqdm
import numexpr as ne
import numpy as np

from scipy.special import ive


def compute_pfe(
    pfe_similarity,
    chunck_slice,
    probe_feats,
    probe_sigma_sq,
    gallery_feats,
    gallery_sigma_sq,
):
    probe_sigma_sq_slice = probe_sigma_sq[:, :, chunck_slice]
    gallery_sigma_sq_slice = gallery_sigma_sq[:, :, chunck_slice]
    probe_feats_slice = probe_feats[:, :, chunck_slice]
    gallery_feats_slice = gallery_feats[:, :, chunck_slice]
    sigma_sq_sum = ne.evaluate("probe_sigma_sq_slice + gallery_sigma_sq_slice")
    slice = ne.evaluate(
        "(probe_feats_slice - gallery_feats_slice)**2 / sigma_sq_sum + log(sigma_sq_sum)"
    )
    slice_sum = ne.evaluate("sum(slice, axis=2)")
    ne.evaluate("slice_sum + pfe_similarity", out=pfe_similarity)


class CosineSimPairwise:
    def __init__(self):
        self.nothing = 0

    @staticmethod
    def __call__(
        X_1: np.ndarray,
        X_2: np.ndarray,
        X_unc: np.ndarray = None,
        Y_unc: np.ndarray = None,
    ):
        return np.sum(X_1 * X_2, axis=1)


class CosineSimDistance:
    @staticmethod
    def __call__(
        X_1: np.ndarray,
        X_2: np.ndarray,
        X_unc: np.ndarray = None,
        Y_unc: np.ndarray = None,
    ):
        return X_2 @ X_1


class PfeSimPairwise:
    def __init__(self, variance_scale: float) -> None:
        self.variance_scale = variance_scale

    def __call__(
        self,
        X_1: np.ndarray,
        X_2: np.ndarray,
        X_unc: np.ndarray = None,
        Y_unc: np.ndarray = None,
    ):
        X_unc = X_unc * self.variance_scale
        Y_unc = Y_unc * self.variance_scale
        assert X_unc.shape[1] == 512
        unc_ii = X_unc + Y_unc

        pfe_similarity = -0.5 * np.sum(
            (X_1 - X_2) ** 2 / unc_ii + np.log(unc_ii), axis=1
        )
        return pfe_similarity
        # X = X[:, np.newaxis, :]
        # X_unc = X_unc[:, np.newaxis, :]

        # Y = Y[np.newaxis, :, :]
        # Y_unc = Y_unc[np.newaxis, :, :]
        # pfe_similarity = np.zeros((X.shape[0], Y.shape[0]))

        # chunck_size = 128
        # for d in tqdm(range(X.shape[2] // chunck_size)):
        #     self.compute_pfe(
        #         pfe_similarity,
        #         slice(d * chunck_size, (d + 1) * chunck_size),
        #         X,
        #         X_unc,
        #         Y,
        #         Y_unc,
        #     )
        # pfe_similarity = -0.5 * pfe_similarity
        # if pfe_cache_path is not None:
        #     np.save(pfe_cache_path, pfe_similarity)
        # return pfe_similarity


class ScfSimPairwise:
    def __init__(self, k_shift: float) -> None:
        self.k_shift = k_shift

    def __call__(
        self,
        X_1: np.ndarray,
        X_2: np.ndarray,
        X_unc: np.ndarray = None,
        Y_unc: np.ndarray = None,
    ):
        mu_ii = np.sum(X_1 * X_2, axis=1)

        d = X_1.shape[1]
        assert X_unc.shape[1] == 1
        X_unc = X_unc[:, 0] + self.k_shift
        Y_unc = Y_unc[:, 0] + self.k_shift
        k_i_times_k_prime_i = Y_unc * X_unc
        k_ii = np.sqrt(Y_unc**2 + X_unc**2 + 2 * mu_ii * k_i_times_k_prime_i)

        log_iv_i = np.log(1e-6 + ive(d / 2 - 1, Y_unc, dtype=Y_unc.dtype)) + Y_unc
        log_iv_prime_i = np.log(1e-6 + ive(d / 2 - 1, X_unc, dtype=X_unc.dtype)) + X_unc
        log_iv_ii = np.log(1e-6 + ive(d / 2 - 1, k_ii, dtype=k_ii.dtype)) + k_ii

        scf_similarity = (
            (d / 2 - 1) * (np.log(Y_unc) + np.log(X_unc) - np.log(k_ii))  # type: ignore
            - (log_iv_i + log_iv_prime_i - log_iv_ii)
            - d / 2 * np.log(2 * np.pi)
            - d * np.log(64)
        )

        return scf_similarity


class ScfSim:
    @staticmethod
    def __call__(
        X_1: np.ndarray, X_2: np.ndarray, X_unc: np.ndarray, Y_unc: np.ndarray
    ):
        d = X_1.shape[1]
        mu_ij = 2 * np.dot(X_1, X_2.T)
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


class PfeSim:
    def __call__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        X_unc: np.ndarray,
        Y_unc: np.ndarray,
        pfe_cache_path=None,
    ):
        X = X
        X_unc = X_unc

        Y = Y
        Y_unc = Y_unc
        pfe_similarity = np.zeros((X.shape[0], Y.shape[0]))

        chunck_size = 128
        assert X.shape[2] % chunck_size == 0
        for d in tqdm(range(X.shape[2] // chunck_size)):
            compute_pfe(
                pfe_similarity,
                slice(d * chunck_size, (d + 1) * chunck_size),
                X,
                X_unc,
                Y,
                Y_unc,
            )
        pfe_similarity = -0.5 * pfe_similarity
        if pfe_cache_path is not None:
            np.save(pfe_cache_path, pfe_similarity)
        return pfe_similarity
