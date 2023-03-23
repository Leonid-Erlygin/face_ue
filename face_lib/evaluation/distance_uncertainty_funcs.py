import numpy as np
from tqdm import tqdm
import scipy


def prob_distance(mu_1, mu_2, sigma_sq_1, sigma_sq_2):
    # mu_1 - enroll templates
    return [mu_2[i][mu_1[i]] for i in range(len(mu_1))]


def prob_unc_pair(mu_1, mu_2, sigma_sq_1, sigma_sq_2):
    # sigma_sq_2 - verif templates
    enroll_ids = sigma_sq_2[0][0]
    enroll_id_to_idx = {int(enroll_ids[int(i)]): int(i) for i in range(len(enroll_ids))}

    sigma_sq_2_array = np.array(sigma_sq_2)[:, 1, :]

    sigma_sq_2_argmax = np.argmax(sigma_sq_2_array, axis=1)
    sigma_sq_2_max_ids = [int(enroll_ids[i]) for i in sigma_sq_2_argmax]
    confindences = []
    positive = []
    for i in tqdm(range(len(sigma_sq_2))):
        p_ij = sigma_sq_2_array[i][enroll_id_to_idx[sigma_sq_1[i]]]
        if sigma_sq_1[i] == sigma_sq_2_max_ids[i]:
            conf = p_ij
            positive.append(True)
        else:
            conf = 1 - p_ij
            positive.append(False)
        confindences.append(conf)

    return np.array(confindences), np.array(positive)


def entropy_unc(mu_1, mu_2, sigma_sq_1, sigma_sq_2):
    uncertainties = []
    for verif_template in sigma_sq_2:
        probabilities = verif_template[1, :]
        uncertainties.append(scipy.stats.entropy(probabilities))
    return np.array(uncertainties)


def prob_unc(mu_1, mu_2, sigma_sq_1, sigma_sq_2):
    # sigma_sq_2 - verif templates
    confindences = []
    for verif_template in sigma_sq_2:
        confindences.append(np.max(verif_template[1, :]))

    return np.array(confindences)


def harmonic_mean(x, axis: int = -1):
    x_sum = ((x ** (-1)).mean(axis=axis)) ** (-1)
    return x_sum


def l2_normalize(x, axis=None, eps=1e-8):
    x = x / (eps + np.linalg.norm(x, axis=axis, keepdims=True))
    return x


def cosine_similarity(x1, x2):
    x1, x2 = np.array(x1), np.array(x2)
    x1, x2 = l2_normalize(x1, axis=1), l2_normalize(x2, axis=1)
    return np.sum(x1 * x2, axis=1)


def centered_cosine_similarity(x1, x2):
    cos_sim = cosine_similarity(x1, x2)
    cos_sim -= cos_sim.mean()
    cos_sim /= cos_sim.std()
    return cos_sim


def biased_cosine_similarity(x1, x2, bias):
    cos_sim = cosine_similarity(x1, x2)
    cos_sim -= bias
    return cos_sim


def pair_euc_score(x1, x2, unc1=None, unc2=None):
    x1, x2 = np.array(x1), np.array(x2)
    dist = np.sum(np.square(x1 - x2), axis=1)
    return -dist


def pair_cosine_score(x1, x2, unc1=None, unc2=None):
    return cosine_similarity(x1, x2)


def pair_centered_cosine_score(x1, x2, unc1=None, unc2=None):
    return centered_cosine_similarity(x1, x2)


def pair_biased_cosine_score(x1, x2, unc1=None, unc2=None, bias=None):
    return biased_cosine_similarity(x1, x2, bias=bias)


def pair_MLS_score(x1, x2, sigma_sq1=None, sigma_sq2=None):
    if sigma_sq1 is None:
        x1, x2 = np.array(x1), np.array(x2)
        assert (
            sigma_sq2 is None
        ), "either pass in concated features, or mu, sigma_sq for both!"
        D = int(x1.shape[1] / 2)
        mu1, sigma_sq1 = x1[:, :D], x1[:, D:]
        mu2, sigma_sq2 = x2[:, :D], x2[:, D:]
    else:
        mu1, mu2 = np.array(x1), np.array(x2)
        sigma_sq1, sigma_sq2 = np.array(sigma_sq1), np.array(sigma_sq2)
        mu1, mu2 = l2_normalize(mu1, axis=1), l2_normalize(mu2, axis=1)
        # mu1, mu2 = x1, x2
    sigma_sq_mutual = sigma_sq1 + sigma_sq2
    dist = np.sum(
        np.square(mu1 - mu2) / sigma_sq_mutual + np.log(sigma_sq_mutual), axis=1
    )
    return -dist


def pair_scale_mul_cosine_score(x1, x2, scale1=None, scale2=None):
    scale1, scale2 = scale1.squeeze(axis=1), scale2.squeeze(axis=1)
    dist = cosine_similarity(x1, x2)
    dist = dist * scale1 * scale2
    return dist


def pair_scale_harmonic_cosine_score(x1, x2, scale1=None, scale2=None):
    scale1, scale2 = scale1.squeeze(axis=1), scale2.squeeze(axis=1)
    dist = cosine_similarity(x1, x2)
    dist = dist * scale1 * scale2 / (scale1 + scale2)
    return dist


def pair_sqrt_scale_mul_cosine_score(x1, x2, scale1=None, scale2=None):
    scale1, scale2 = np.sqrt(scale1.squeeze(axis=1)), np.sqrt(scale2.squeeze(axis=1))
    dist = cosine_similarity(x1, x2)
    dist = dist * scale1 * scale2
    return dist


def pair_sqrt_scale_harmonic_cosine_score(x1, x2, scale1=None, scale2=None):
    scale1, scale2 = np.sqrt(scale1.squeeze(axis=1)), np.sqrt(scale2.squeeze(axis=1))
    dist = cosine_similarity(x1, x2)
    dist = dist * scale1 * scale2 / (scale1 + scale2)
    return dist


def pair_scale_mul_centered_cosine_score(x1, x2, scale1=None, scale2=None):
    scale1, scale2 = scale1.squeeze(axis=1), scale2.squeeze(axis=1)
    dist = centered_cosine_similarity(x1, x2)
    dist = dist * scale1 * scale2
    return dist


def pair_scale_harmonic_centered_cosine_score(x1, x2, scale1=None, scale2=None):
    scale1, scale2 = scale1.squeeze(axis=1), scale2.squeeze(axis=1)
    dist = centered_cosine_similarity(x1, x2)
    dist = dist * scale1 * scale2 / (scale1 + scale2)
    return dist


def pair_sqrt_scale_mul_centered_cosine_score(x1, x2, scale1=None, scale2=None):
    scale1, scale2 = np.sqrt(scale1.squeeze(axis=1)), np.sqrt(scale2.squeeze(axis=1))
    dist = centered_cosine_similarity(x1, x2)
    dist = dist * scale1 * scale2
    return dist


def pair_sqrt_scale_harmonic_centered_cosine_score(x1, x2, scale1=None, scale2=None):
    scale1, scale2 = np.sqrt(scale1.squeeze(axis=1)), np.sqrt(scale2.squeeze(axis=1))
    dist = centered_cosine_similarity(x1, x2)
    dist = dist * scale1 * scale2 / (scale1 + scale2)
    return dist


def pair_scale_mul_biased_cosine_score(x1, x2, scale1=None, scale2=None, bias=None):
    scale1, scale2 = scale1.squeeze(axis=1), scale2.squeeze(axis=1)
    dist = biased_cosine_similarity(x1, x2, bias=bias)
    dist = dist * scale1 * scale2
    return dist


def pair_scale_harmonic_biased_cosine_score(
    x1, x2, scale1=None, scale2=None, bias=None
):
    scale1, scale2 = scale1.squeeze(axis=1), scale2.squeeze(axis=1)
    dist = biased_cosine_similarity(x1, x2, bias=bias)
    dist = dist * scale1 * scale2 / (scale1 + scale2)
    return dist


def pair_sqrt_scale_mul_biased_cosine_score(
    x1, x2, scale1=None, scale2=None, bias=None
):
    scale1, scale2 = np.sqrt(scale1.squeeze(axis=1)), np.sqrt(scale2.squeeze(axis=1))
    dist = biased_cosine_similarity(x1, x2, bias=bias)
    dist = dist * scale1 * scale2
    return dist


def pair_sqrt_scale_harmonic_biased_cosine_score(
    x1, x2, scale1=None, scale2=None, bias=None
):
    scale1, scale2 = np.sqrt(scale1.squeeze(axis=1)), np.sqrt(scale2.squeeze(axis=1))
    dist = biased_cosine_similarity(x1, x2, bias=bias)
    dist = dist * scale1 * scale2 / (scale1 + scale2)
    return dist


def pair_pfe_mul_biased_cosine_score(x1, x2, scale1=None, scale2=None, bias=None):
    scale1, scale2 = 1 / harmonic_mean(scale1), 1 / harmonic_mean(scale2)
    dist = biased_cosine_similarity(x1, x2, bias=bias)
    dist = dist * scale1 * scale2
    return dist


def pair_pfe_harmonic_biased_cosine_score(x1, x2, scale1=None, scale2=None, bias=None):
    scale1, scale2 = 1 / harmonic_mean(scale1), 1 / harmonic_mean(scale2)
    dist = biased_cosine_similarity(x1, x2, bias=bias)
    dist = dist * scale1 * scale2 / (scale1 + scale2)
    return dist


def pair_sqrt_pfe_mul_biased_cosine_score(x1, x2, scale1=None, scale2=None, bias=None):
    scale1, scale2 = 1 / harmonic_mean(scale1), 1 / harmonic_mean(scale2)
    scale1, scale2 = np.sqrt(scale1), np.sqrt(scale2)
    dist = biased_cosine_similarity(x1, x2, bias=bias)
    dist = dist * scale1 * scale2
    return dist


def pair_sqrt_pfe_harmonic_biased_cosine_score(
    x1, x2, scale1=None, scale2=None, bias=None
):
    scale1, scale2 = 1 / harmonic_mean(scale1), 1 / harmonic_mean(scale2)
    scale1, scale2 = np.sqrt(scale1), np.sqrt(scale2)
    dist = biased_cosine_similarity(x1, x2, bias=bias)
    dist = dist * scale1 * scale2 / (scale1 + scale2)
    return dist


def pair_uncertainty_sum(mu_1, mu_2, sigma_sq_1, sigma_sq_2):
    sigma_sq_1 = np.array(sigma_sq_1)
    sigma_sq_2 = np.array(sigma_sq_2)
    print(sigma_sq_1.shape, sigma_sq_2.shape)
    if len(sigma_sq_1.shape) == 1:
        sigma_sq_1 = sigma_sq_1[:, np.newaxis]
        sigma_sq_2 = sigma_sq_2[:, np.newaxis]
    exit()
    return np.sum(sigma_sq_1, axis=1) + np.sum(sigma_sq_2, axis=1)


def pair_uncertainty_squared_sum(mu_1, mu_2, sigma_sq_1, sigma_sq_2):
    return sigma_sq_1.sum(axis=1) ** 2 + sigma_sq_2.sum(axis=1) ** 2


def pair_uncertainty_mul(mu_1, mu_2, sigma_sq_1, sigma_sq_2):
    return sigma_sq_1.prod(axis=1) * sigma_sq_2.prod(axis=1)


def pair_uncertainty_harmonic_sum(mu_1, mu_2, sigma_sq_1, sigma_sq_2):
    return harmonic_mean(sigma_sq_1, axis=1) + harmonic_mean(sigma_sq_2, axis=1)


def pair_uncertainty_harmonic_mul(mu_1, mu_2, sigma_sq_1, sigma_sq_2):
    return harmonic_mean(sigma_sq_1, axis=1) * harmonic_mean(sigma_sq_2, axis=1)


def pair_uncertainty_concatenated_harmonic(mu_1, mu_2, sigma_sq_1, sigma_sq_2):
    return harmonic_mean(
        np.concatenate(
            (
                sigma_sq_1,
                sigma_sq_2,
            ),
            axis=1,
        ),
        axis=1,
    )


def pair_uncertainty_squared_harmonic(mu_1, mu_2, uncertainty_1, uncertainty_2):
    return harmonic_mean(
        np.concatenate(
            (
                uncertainty_1**2,
                uncertainty_2**2,
            ),
            axis=1,
        ),
        axis=1,
    )


def pair_uncertainty_cosine_analytic(mu_1, mu_2, sigma_sq_1, sigma_sq_2):
    return (
        sigma_sq_1 * sigma_sq_2 + (mu_1**2) * sigma_sq_2 + (mu_2**2) * sigma_sq_1
    ).sum(axis=1)


def pair_uncertainty_min(mu_1, mu_2, sigma_sq_1, sigma_sq_2):
    return np.min(np.stack([sigma_sq_1.sum(axis=1), sigma_sq_2.sum(axis=1)]), axis=0)


def pair_uncertainty_similarity(mu_1, mu_2, sigma_sq_1, sigma_sq_2):
    return cosine_similarity(mu_1, mu_2)


def pair_uncertainty_perfect(mu_1, mu_2, label_1, label_2):
    assert np.allclose(label_1, label_2)

    cosines = cosine_similarity(mu_1, mu_2)
    labels = label_1 * 2 - 1
    margins = labels * (cosines - cosines.mean(axis=0))
    return margins


# def get_scale_confidences(feat_1, feat_2, unc_1, unc_2):
#     unc_1, unc_2 = unc_1.squeeze(axis=1), unc_2.squeeze(axis=1)
#     return feat_1, feat_2, unc_1, unc_2
#
# get_norm_confidences = get_scale_confidences
#
# def get_PFE_confidences(feat_1, feat_2, unc_1, unc_2):
#     unc_1, unc_2 = 1 / harmonic_mean(unc_1), 1 / harmonic_mean(unc_2)
#     return feat_1, feat_2, unc_1, unc_2
#
#
#
# name_to_uncertainty = {
#     "scale": get_scale_confidences,
#     "norm": get_norm_confidences,
#     "PFE": get_PFE_confidences,
# }
#
# def create_distance_function(sqrt=False, confidences="norm", )
