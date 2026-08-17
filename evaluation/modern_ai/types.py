from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class RetrievalDataset:
    """In-memory IR dataset.

    qrels maps query_id -> {document_id: relevance}. A relevance value > 0 is
    considered relevant. The representation mirrors the BEIR data model while
    remaining independent of the optional ``beir`` dependency.
    """

    corpus: Mapping[str, str]
    queries: Mapping[str, str]
    qrels: Mapping[str, Mapping[str, float]]
    name: str = "dataset"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def relevant_docs(self, query_id: str) -> Tuple[str, ...]:
        rels = self.qrels.get(str(query_id), {})
        return tuple(str(doc_id) for doc_id, rel in rels.items() if float(rel) > 0)

    def validate(self) -> None:
        corpus_ids = set(map(str, self.corpus.keys()))
        query_ids = set(map(str, self.queries.keys()))
        missing_qrels = [qid for qid in self.qrels if str(qid) not in query_ids]
        if missing_qrels:
            raise ValueError(f"qrels reference missing queries: {missing_qrels[:5]}")
        bad_docs = []
        for qid in query_ids:
            for did in self.relevant_docs(qid):
                if did not in corpus_ids:
                    bad_docs.append((qid, did))
        if bad_docs:
            raise ValueError(f"qrels reference missing corpus documents: {bad_docs[:5]}")


@dataclass(frozen=True)
class RetrievalProtocol:
    """Open-set evidence retrieval protocol.

    ``known_mask[i]`` states whether query ``query_ids[i]`` has at least one
    relevant document in the *active* corpus. ``relevant_doc_ids`` contains the
    post-deletion qrels used to judge retrieval correctness.
    """

    name: str
    corpus: Mapping[str, str]
    query_ids: Tuple[str, ...]
    queries: Tuple[str, ...]
    relevant_doc_ids: Tuple[Tuple[str, ...], ...]
    known_mask: np.ndarray
    designated_unknown_mask: np.ndarray
    deleted_doc_ids: Tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.query_ids)
        if len(self.queries) != n or len(self.relevant_doc_ids) != n:
            raise ValueError("Protocol query arrays have inconsistent lengths.")
        if np.asarray(self.known_mask).shape != (n,):
            raise ValueError("known_mask must have shape [num_queries].")
        if np.asarray(self.designated_unknown_mask).shape != (n,):
            raise ValueError("designated_unknown_mask must have shape [num_queries].")


@dataclass
class RetrievalScores:
    query_ids: Tuple[str, ...]
    corpus_ids: Tuple[str, ...]
    predicted_indices: np.ndarray
    was_rejected: np.ndarray
    mean_known_probs: np.ndarray
    unknown_prob: np.ndarray
    unknown_nonspecificity: np.ndarray
    kl_1: np.ndarray
    kl_2: np.ndarray
    scores: Dict[str, np.ndarray]
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def predicted_doc_ids(self) -> Tuple[str, ...]:
        ids = np.asarray(self.corpus_ids, dtype=object)
        return tuple(ids[np.asarray(self.predicted_indices, dtype=int)].tolist())


@dataclass(frozen=True)
class RAGRecord:
    record_id: str
    group_id: str
    query: str
    contexts: Tuple[str, ...]
    response: str
    is_error: bool
    is_hallucination: bool
    reference_answers: Tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolRoutingExample:
    example_id: str
    query: str
    tools: Tuple[str, ...]
    known: bool
    relevant_tool_indices: Tuple[int, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
