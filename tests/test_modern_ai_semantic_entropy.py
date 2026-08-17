import numpy as np

from evaluation.modern_ai.rag import cluster_semantic_samples, semantic_entropy_from_clusters


def test_semantic_entropy_collapses_paraphrase_clusters():
    samples = ["Paris", "The answer is Paris", "Berlin", "Berlin"]
    equivalent_pairs = {
        frozenset(("Paris", "The answer is Paris")),
    }

    def equivalent(a, b):
        return frozenset((a, b)) in equivalent_pairs

    clusters = cluster_semantic_samples(samples, equivalent)
    assert len(np.unique(clusters)) == 2
    features = semantic_entropy_from_clusters(clusters)
    assert np.isclose(features["generator_semantic_entropy_discrete"], np.log(2.0))
    assert np.isclose(features["generator_semantic_entropy_normalized"], 0.5)
    assert np.isclose(features["generator_self_consistency"], 0.5)
    assert features["generator_semantic_cluster_count"] == 2.0


def test_semantic_entropy_zero_for_unanimous_meaning():
    features = semantic_entropy_from_clusters([0, 0, 0, 0])
    assert features["generator_semantic_entropy_discrete"] == 0.0
    assert features["generator_self_consistency"] == 0.0
    assert features["generator_semantic_cluster_count"] == 1.0
