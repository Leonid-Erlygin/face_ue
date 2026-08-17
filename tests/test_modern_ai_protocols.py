import numpy as np

from evaluation.modern_ai.data import make_evidence_deletion_protocol
from evaluation.modern_ai.metrics import retrieval_error_masks
from evaluation.modern_ai.synthetic import synthetic_retrieval_dataset
from evaluation.modern_ai.types import RetrievalProtocol, RetrievalScores


def test_evidence_deletion_recomputes_known_state():
    ds = synthetic_retrieval_dataset(12)
    p = make_evidence_deletion_protocol(ds, unknown_fraction=0.5, seed=3)
    assert len(p.query_ids) == 12
    assert np.any(p.known_mask)
    assert np.any(~p.known_mask)
    for i, known in enumerate(p.known_mask):
        assert known == bool(p.relevant_doc_ids[i])
        assert all(d in p.corpus for d in p.relevant_doc_ids[i])


def test_multiple_relevant_docs_are_all_acceptable():
    p = RetrievalProtocol(
        name="multi", corpus={"d0":"a", "d1":"b", "d2":"c"},
        query_ids=("q",), queries=("query",), relevant_doc_ids=(("d0","d1"),),
        known_mask=np.array([True]), designated_unknown_mask=np.array([False]),
    )
    s = RetrievalScores(
        query_ids=("q",), corpus_ids=("d0","d1","d2"),
        predicted_indices=np.array([1]), was_rejected=np.array([False]),
        mean_known_probs=np.array([[.2,.7,.05]]), unknown_prob=np.array([.05]),
        unknown_nonspecificity=np.array([.1]), kl_1=np.array([0.]), kl_2=np.array([0.]),
        scores={"mprisk":np.array([.3])},
    )
    m = retrieval_error_masks(p, s)
    assert m["correct_retrieval"][0]
    assert m["true_accept"][0]
    assert not m["misretrieval"][0]
