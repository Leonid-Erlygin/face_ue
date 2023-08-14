from typing import Any
import numpy as np


def random_VMF(mu, kappa, size=None):
    """
    Von Mises-Fisher distribution sampler with
    mean direction mu and concentration kappa.
    Source:https://hal.science/hal-04004568
    """
    # parse input parameters
    n = 1 if size is None else np.product(size)
    shape = () if size is None else tuple(np.ravel(size))
    mu = np.asarray(mu)
    mu = mu / np.linalg.norm(mu)
    (d,) = mu.shape
    # z component:radial samples perpendicular to mu
    z = np.random.normal(0, 1, (n, d))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    z = z - (z @ mu[:, None]) * mu[None, :]
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    # sample angles ( in cos and sin form )
    cos = _random_VMF_cos(d, kappa, n)
    sin = np.sqrt(1 - cos**2)
    # combine angles with the z component
    x = z * sin[:, None] + cos[:, None] * mu[None, :]
    return x.reshape((*shape, d))


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
    def __init__(self, num_samples: int) -> None:
        self.num_samples = num_samples

    def __call__(self, feature_mean: np.ndarray, kappas: np.ndarray) -> Any:
        if self.num_samples > 0:
            sample_list = []
            for mu, kappa in zip(feature_mean, kappas):
                samples = random_VMF(mu, kappa=kappa, size=self.num_samples)[
                    np.newaxis, :, :
                ]
                sample_list.append(samples)
            return np.concatenate(sample_list, axis=0)
        else:
            return feature_mean[:, np.newaxis, :]
