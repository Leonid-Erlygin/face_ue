from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .data import make_evidence_deletion_protocol
from .types import RAGRecord, RetrievalDataset, RetrievalProtocol, ToolRoutingExample


def synthetic_retrieval_dataset(n: int = 20) -> RetrievalDataset:
    topics = [
        "cats", "dogs", "whales", "eagles", "volcanoes", "rivers", "planets", "comets",
        "batteries", "motors", "bridges", "tunnels", "viruses", "proteins", "neurons", "kidneys",
        "algebra", "geometry", "poetry", "painting", "forests", "deserts", "oceans", "glaciers",
    ][: int(n)]
    corpus = {}
    queries = {}
    qrels = {}
    for i, topic in enumerate(topics):
        did = f"d{i}"
        qid = f"q{i}"
        corpus[did] = f"Technical evidence about {topic}. This document specifically explains {topic}."
        queries[qid] = f"What does the evidence say about {topic}?"
        qrels[qid] = {did: 1}
    return RetrievalDataset(corpus=corpus, queries=queries, qrels=qrels, name="synthetic_retrieval")


def synthetic_open_set_protocol(n: int = 20, unknown_fraction: float = 0.4, seed: int = 777) -> RetrievalProtocol:
    return make_evidence_deletion_protocol(
        synthetic_retrieval_dataset(n), unknown_fraction=unknown_fraction, seed=seed
    )


def synthetic_rewrites(protocol: RetrievalProtocol) -> Dict[str, Tuple[str, ...]]:
    out = {}
    for qid, query in zip(protocol.query_ids, protocol.queries):
        out[qid] = (
            query.replace("What does the evidence say about", "Find information concerning"),
            query.replace("What does the evidence say about", "Retrieve evidence on"),
            query.replace("What does", "What exactly does"),
        )
    return out


def synthetic_tool_examples() -> List[ToolRoutingExample]:
    tools = (
        "weather_forecast\nGet weather for a city\nParameters: city",
        "currency_convert\nConvert money between currencies\nParameters: amount, from, to",
        "calendar_create\nCreate a calendar event\nParameters: title, time",
    )
    rows = []
    known_queries = [
        ("weather tomorrow in Berlin", 0),
        ("convert 30 euros to yen", 1),
        ("schedule a meeting at noon", 2),
        ("forecast rain in Paris", 0),
        ("exchange dollars for pounds", 1),
        ("put dentist appointment on calendar", 2),
        ("temperature in Rome", 0),
        ("convert 100 yen to euro", 1),
        ("create an event for Friday", 2),
    ]
    for i, (q, idx) in enumerate(known_queries):
        rows.append(ToolRoutingExample(f"known-{i}", q, tools, True, (idx,)))
    unknown_queries = [
        "write a poem about summer", "delete all my photos", "play a jazz song",
        "book a flight to Madrid", "summarize this research paper", "turn the kitchen lights off",
        "diagnose my headache", "order a pizza", "translate this sentence",
    ]
    for i, q in enumerate(unknown_queries):
        rows.append(ToolRoutingExample(f"unknown-{i}", q, tools, False, ()))
    return rows


def synthetic_ragtruth_records() -> List[RAGRecord]:
    """Repeated-response records with source-level good/bad evidence for CI tests."""
    out = []
    for i in range(12):
        topic = f"topic{i}"
        bad = i >= 6
        contexts = (
            (f"Unrelated material with no information about {topic}.",)
            if bad else
            (f"Authoritative evidence: the key fact about {topic} is value-{i}.",)
        )
        for j in range(2):
            hall = bool(bad)
            response = f"The answer is invented-{i}." if hall else f"The answer is value-{i}."
            out.append(RAGRecord(
                record_id=f"r{i}-{j}", group_id=f"source-{i}",
                query=f"What is the key fact about {topic}?", contexts=contexts,
                response=response, is_error=hall, is_hallucination=hall,
                reference_answers=(f"value-{i}",), metadata={"synthetic": True},
            ))
    return out


def synthetic_generator_features(records: List[RAGRecord]) -> Dict[str, np.ndarray]:
    # Deliberately imperfect generator signal so the hybrid path is exercised.
    rng = np.random.default_rng(11)
    y = np.asarray([r.is_error for r in records], dtype=float)
    return {
        "generator_mean_nll": np.clip(0.4 + 0.4 * y + rng.normal(0, 0.2, len(y)), 0, None),
        "generator_mean_entropy": np.clip(0.5 + 0.25 * y + rng.normal(0, 0.2, len(y)), 0, None),
    }
