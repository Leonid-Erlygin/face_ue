from typing import Any
import numpy as np
import warnings
from multiprocessing import Pool

np.seterr(all="warn")


def random_VMF(mu, kappa, size=None):
    """
    Von Mises-Fisher distribution sampler with
    mean direction mu and concentration kappa.
    Source:https://hal.science/hal-04004568
    """
    # parse input parameters
    kappa = kappa[0]
    n = 1 if size is None else np.product(size)
    shape = () if size is None else tuple(np.ravel(size))
    mu = np.asarray(mu)
    with np.errstate(divide="raise"):
        try:
            mu_norm = mu / np.linalg.norm(mu)
        except Exception as e:
            test_0 = 1
            test = 1
            print("Caught it!")
    if np.any(np.isnan(mu_norm)):
        test_0 = 1
        print("Caught it!")
    (d,) = mu_norm.shape
    # z component:radial samples perpendicular to mu
    z = np.random.normal(0, 1, (n, d))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    z = z - (z @ mu_norm[:, None]) * mu_norm[None, :]
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    # sample angles ( in cos and sin form )
    cos = _random_VMF_cos(d, kappa, n)
    sin = np.sqrt(1 - cos**2)
    # combine angles with the z component
    x = z * sin[:, None] + cos[:, None] * mu_norm[None, :]
    return x.reshape((*shape, d))[np.newaxis, :, :]


def _random_VMF_cos(d: int, kappa: float, n: int):
    """
    Generate n iid samples t with density function given by
    p(t)=someConstant*(1-t**2)**((d-2)/2)*exp(kappa*t)
    """
    # b = Eq . 4 of https :// doi . org / 10 . 1080 / 0 3 6 1 0 9 1 9 4 0 8 8 1 3 1 6 1
    b = (d - 1) / (2 * kappa + (4 * kappa**2 + (d - 1) ** 2) ** 0.5)
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (d - 1) * np.log(1 - x0**2)
    found = 0
    out = []
    while found < n:
        m = min(n, int((n - found) * 1.5))
        z = np.random.beta((d - 1) / 2, (d - 1) / 2, size=m)
        t = (1 - (1 + b) * z) / (1 - (1 - b) * z)
        test = kappa * t + (d - 1) * np.log(1 - x0 * t) - c
        accept = test >= -np.random.exponential(size=m)
        out.append(t[accept])
        found += len(out[-1])
    return np.concatenate(out)[:n]


class VonMisesFisher:
    def __init__(self, num_samples: int, num_workers=21) -> None:
        self.num_samples = num_samples
        self.num_workers = num_workers

    def __call__(self, feature_mean: np.ndarray, kappas: np.ndarray) -> Any:
        if self.num_samples > 0:
            sample_list = []
            with Pool(self.num_workers) as p:
                sample_list = p.starmap(
                    random_VMF,
                    zip(feature_mean, kappas, [self.num_samples] * len(feature_mean)),
                )
            # for mu, kappa in zip(feature_mean, kappas):
            #     samples = random_VMF(mu, kappa=kappa[0], size=self.num_samples)[
            #         np.newaxis, :, :
            #     ]
            #     sample_list.append(samples)
            return np.concatenate(sample_list, axis=0)
        else:
            return feature_mean[:, np.newaxis, :]
