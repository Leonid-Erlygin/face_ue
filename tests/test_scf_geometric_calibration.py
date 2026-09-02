import numpy as np

from evaluation.open_set_methods.class_prob_models import MonteCarloPredictiveProb
from evaluation.open_set_methods.kappa_utils import (
    fit_vmf_kappa_scale,
    log_uniform_density,
    vmf_log_normalizer_np,
    vmf_mean_resultant_np,
    vmf_nonspecificity_np,
)


def test_vmf_mean_resultant_zero_monotone_and_bounded():
    k = np.asarray([0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0])
    a = vmf_mean_resultant_np(k, d=512)
    assert a[0] == 0.0
    assert np.all(np.isfinite(a))
    assert np.all((a >= 0.0) & (a <= 1.0))
    assert np.all(np.diff(a) > 0.0)


def test_global_geometric_scale_is_recovered_from_exact_stationarity_targets():
    d = 128
    raw_kappa = np.asarray([10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0])
    true_scale = 2.75
    cosine = vmf_mean_resultant_np(true_scale * raw_kappa, d=d)
    fit = fit_vmf_kappa_scale(raw_kappa, cosine, d=d)
    assert fit["boundary"] == "interior"
    assert fit["converged"]
    assert np.isclose(fit["scale"], true_scale, rtol=1e-6, atol=1e-7)
    assert fit["nll_fitted"] <= fit["nll_scale_1"]


def test_vmf_nonspecificity_decreases_with_kappa():
    k = np.asarray([0.0, 1.0, 10.0, 100.0, 1000.0])
    n = vmf_nonspecificity_np(k, d=64)
    assert np.isclose(n[0], 1.0)
    assert np.all((n > 0.0) & (n <= 1.0))
    assert np.all(np.diff(n) < 0.0)



def test_vmf_log_normalizer_handles_zero_and_scalar_high_dimension():
    d = 512
    at_zero = vmf_log_normalizer_np(0.0, d=d)
    assert np.isfinite(at_zero)
    assert np.isclose(float(at_zero), log_uniform_density(d), rtol=0.0, atol=1e-12)

    values = vmf_log_normalizer_np(np.asarray([0.0, 1e-8, 1.0, 1000.0]), d=d)
    assert np.all(np.isfinite(values))
    assert np.isclose(values[0], log_uniform_density(d), rtol=0.0, atol=1e-12)


def test_geometric_scale_lower_boundary_for_nonpositive_alignment():
    fit = fit_vmf_kappa_scale(
        np.asarray([10.0, 20.0, 30.0]),
        np.asarray([0.0, -0.1, -0.2]),
        d=64,
    )
    assert fit["boundary"] == "lower"
    assert fit["raw_optimum_scale"] == 0.0
    assert fit["scale"] > 0.0

def test_holue_without_supervised_calibrator_returns_negative_raw_kl_sum():
    method = MonteCarloPredictiveProb.__new__(MonteCarloPredictiveProb)
    method.pred_uncertainty_type = "entropy"
    method.calibration_transform = None
    method.kl_1 = np.asarray([1.0, 2.0, 3.0])
    method.kl_2 = np.asarray([0.25, -0.5, 1.0])
    got = method.predict_uncertainty()
    expected = -(method.kl_1 + method.kl_2)
    assert np.allclose(got, expected)


def test_m0_geometric_kappa_scaling_changes_kl2_not_osr_posterior():
    rng = np.random.default_rng(7)
    d = 16
    probes = rng.normal(size=(5, d))
    probes /= np.linalg.norm(probes, axis=1, keepdims=True)
    gallery = rng.normal(size=(3, d))
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
    kappa = np.asarray([[20.0], [30.0], [40.0], [50.0], [60.0]])
    gallery_kappa = np.full((3, 1), 25.0)

    method = MonteCarloPredictiveProb(
        gallery_prior="power",
        emb_unc_model="vMF",
        beta=0.5,
        far=0.1,
        M=0,
        calibration_set=None,
        calibration_transform=None,
        predict_T=20.0,
    )

    p1, kl11, kl21 = [
        x.detach().cpu().numpy()
        for x in method.compute_mean_probs_and_kl(
            probes, kappa, gallery, gallery_kappa, 20.0
        )
    ]
    p2, kl12, kl22 = [
        x.detach().cpu().numpy()
        for x in method.compute_mean_probs_and_kl(
            probes, 3.0 * kappa, gallery, gallery_kappa, 20.0
        )
    ]

    assert np.allclose(p1, p2, rtol=0.0, atol=1e-12)
    assert np.allclose(kl11, kl12, rtol=0.0, atol=1e-12)
    assert not np.allclose(kl21, kl22)
