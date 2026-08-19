import numpy as np

from evaluation.modern_ai.scf_diagnostics import summarize_scf_split, vmf_mean_resultant


def test_vmf_mean_resultant_is_monotone_and_bounded():
    k = np.asarray([1.0, 10.0, 100.0, 1000.0, 10000.0])
    a = vmf_mean_resultant(k, 768)
    assert np.all(np.isfinite(a))
    assert np.all((a >= 0) & (a <= 1))
    assert np.all(np.diff(a) > 0)


def test_scf_diagnostic_rewards_kappa_that_tracks_alignment():
    cosine = np.asarray([0.2, 0.4, 0.6, 0.8, 0.9, 0.95])
    # A broad increasing concentration sequence is enough to test ordering;
    # stationarity itself is reported rather than hard-coded in this unit test.
    kappa = np.asarray([100, 300, 600, 1400, 3000, 7000.0])
    correct = np.asarray([False, False, True, True, True, True])
    out = summarize_scf_split(
        kappa, cosine, correct,
        true_gallery_cosine=cosine - 0.03,
        gallery_correct=correct,
        embedding_dim=768,
    )
    assert out["center"]["spearman_log_kappa_vs_true_cosine"] > 0.99
    assert out["center"]["error_auroc_negative_log_kappa"] > 0.99
    assert out["kappa"]["log_std"] > 0
    assert "mean_absolute_residual" in out["scf_stationarity"]
