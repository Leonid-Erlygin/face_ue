import numpy as np

from evaluation.modern_ai.methods import ModernUncertaintyModel, PosteriorModelConfig
from evaluation.modern_ai.scalable import streaming_deterministic_score


def _assert_streaming_matches_dense(gallery_prior):
    rng = np.random.default_rng(19)
    q = rng.normal(size=(6, 12)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    g = rng.normal(size=(17, 12)); g /= np.linalg.norm(g, axis=1, keepdims=True)
    k = rng.uniform(10, 300, size=(len(q), 1))
    cfg = PosteriorModelConfig(
        beta=0.37, gallery_prior=gallery_prior, predict_T=7.0,
        gallery_kappa=33.0, streaming_threshold_elements=10**9,
    )
    model = ModernUncertaintyModel(cfg)
    ids = [str(i) for i in range(len(q))]
    cids = [str(i) for i in range(len(g))]
    dense = model.score(ids, cids, q, k, g)
    streamed = streaming_deterministic_score(
        query_ids=ids, corpus_ids=cids, query_embeddings=q, query_kappa=k,
        gallery_embeddings=g, gallery_kappa=33.0, beta=0.37,
        gallery_prior=gallery_prior, predict_T=7.0,
        query_batch_size=2, gallery_chunk_size=4,
    )
    np.testing.assert_allclose(dense.unknown_prob, streamed.unknown_prob, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(dense.unknown_nonspecificity, streamed.unknown_nonspecificity, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(dense.kl_1, streamed.kl_1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(dense.kl_2, streamed.kl_2, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(dense.scores["galue_entropy"], streamed.scores["galue_entropy"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(dense.scores["mprisk"], streamed.scores["mprisk"], rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(dense.predicted_indices, streamed.predicted_indices)
    np.testing.assert_array_equal(dense.was_rejected, streamed.was_rejected)


def test_streaming_matches_dense_power_m0():
    _assert_streaming_matches_dense("power")


def test_streaming_matches_dense_vmf_m0():
    _assert_streaming_matches_dense("vMF")
