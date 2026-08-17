import numpy as np

from evaluation.modern_ai.embedders import HashingTextEmbedder
from evaluation.modern_ai.query_uncertainty import query_embeddings_and_kappa


def test_rewrite_kappa_can_hold_original_mean_fixed():
    emb = HashingTextEmbedder(n_features=64)
    qids = ["q1"]
    queries = ["weather in berlin tomorrow"]
    rewrites = {"q1": ["tomorrow berlin weather forecast", "forecast for berlin tomorrow"]}
    original = emb.encode_queries(queries)[0]
    mean, kappa, rbar = query_embeddings_and_kappa(
        qids, queries, emb, rewrites=rewrites, mean_mode="original"
    )
    np.testing.assert_allclose(mean[0], original, atol=1e-12)
    assert np.isfinite(kappa[0,0]) and kappa[0,0] > 0
    assert 0 <= rbar[0] <= 1


def test_rewrite_mean_mode_is_explicit():
    emb = HashingTextEmbedder(n_features=64)
    with np.testing.assert_raises(ValueError):
        query_embeddings_and_kappa(["q"], ["x"], emb, mean_mode="not-a-mode")
