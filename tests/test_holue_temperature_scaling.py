import numpy as np

from evaluation.open_set_methods.class_prob_models import MonteCarloPredictiveProb


def _toy_inputs(seed=17, n_probe=7, n_gallery=4, d=12):
    rng = np.random.default_rng(seed)
    probes = rng.normal(size=(n_probe, d))
    probes /= np.linalg.norm(probes, axis=1, keepdims=True)
    gallery = rng.normal(size=(n_gallery, d))
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
    probe_kappa = np.linspace(15.0, 75.0, n_probe)[:, None]
    gallery_kappa = np.full((n_gallery, 1), 35.0)
    return probes, probe_kappa, gallery, gallery_kappa


def _method(mode="normalized"):
    return MonteCarloPredictiveProb(
        gallery_prior="power",
        emb_unc_model="vMF",
        beta=0.5,
        far=0.1,
        M=0,
        calibration_set=None,
        calibration_transform=None,
        predict_T=20.0,
        temperature_scaling_mode=mode,
    )


def _as_numpy_dict(d):
    return {k: v.detach().cpu().numpy() for k, v in d.items()}


def test_normalized_temperature_scaling_has_unit_mixed_mass():
    probes, probe_kappa, gallery, gallery_kappa = _toy_inputs()
    method = _method("normalized")

    for T in [0.5, 1.0, 2.0, 7.0, 20.0, 50.0]:
        diag = _as_numpy_dict(
            method.compute_temperature_scaling_diagnostics(
                probes, probe_kappa, gallery, gallery_kappa, T
            )
        )
        assert np.allclose(
            diag["normalized_total_mass"], 1.0, rtol=0.0, atol=1e-12
        )


def test_accepted_paper_oog_formula_is_not_normalized_for_t_not_one():
    probes, probe_kappa, gallery, gallery_kappa = _toy_inputs()
    method = _method("normalized")

    diag_t1 = _as_numpy_dict(
        method.compute_temperature_scaling_diagnostics(
            probes, probe_kappa, gallery, gallery_kappa, 1.0
        )
    )
    assert np.allclose(diag_t1["legacy_total_mass"], 1.0, rtol=0.0, atol=1e-12)

    diag_t20 = _as_numpy_dict(
        method.compute_temperature_scaling_diagnostics(
            probes, probe_kappa, gallery, gallery_kappa, 20.0
        )
    )
    defect = np.abs(diag_t20["legacy_total_mass"] - 1.0)
    assert np.max(defect) > 1e-3


def test_normalized_and_legacy_formulas_coincide_at_t_one():
    probes, probe_kappa, gallery, gallery_kappa = _toy_inputs()
    normalized = _method("normalized")
    legacy = _method("legacy_paper")

    out_n = normalized.compute_mean_probs_and_kl(
        probes, probe_kappa, gallery, gallery_kappa, 1.0
    )
    out_l = legacy.compute_mean_probs_and_kl(
        probes, probe_kappa, gallery, gallery_kappa, 1.0
    )

    for a, b in zip(out_n, out_l):
        assert np.allclose(
            a.detach().cpu().numpy(), b.detach().cpu().numpy(), rtol=0.0, atol=1e-12
        )


def test_only_continuous_kl_term_changes_between_scaling_formulas():
    probes, probe_kappa, gallery, gallery_kappa = _toy_inputs()
    normalized = _method("normalized")
    legacy = _method("legacy_paper")

    p_n, kl1_n, kl2_n = normalized.compute_mean_probs_and_kl(
        probes, probe_kappa, gallery, gallery_kappa, 20.0
    )
    p_l, kl1_l, kl2_l = legacy.compute_mean_probs_and_kl(
        probes, probe_kappa, gallery, gallery_kappa, 20.0
    )

    assert np.allclose(
        p_n.detach().cpu().numpy(), p_l.detach().cpu().numpy(), rtol=0.0, atol=1e-12
    )
    assert np.allclose(
        kl1_n.detach().cpu().numpy(), kl1_l.detach().cpu().numpy(), rtol=0.0, atol=1e-12
    )
    assert not np.allclose(kl2_n.detach().cpu().numpy(), kl2_l.detach().cpu().numpy())
