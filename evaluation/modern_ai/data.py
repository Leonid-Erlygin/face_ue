from __future__ import annotations

import bz2
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .types import RAGRecord, RetrievalDataset, RetrievalProtocol, ToolRoutingExample


def _read_jsonl(path: str | Path) -> List[dict]:
    path = Path(path)
    opener = bz2.open if path.suffix == ".bz2" else open
    mode = "rt"
    with opener(path, mode, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_beir_local(data_dir: str | Path, split: str = "test", name: Optional[str] = None) -> RetrievalDataset:
    """Load BEIR's documented local corpus/queries/qrels format.

    Expected files are ``corpus.jsonl``, ``queries.jsonl`` and
    ``qrels/{split}.tsv``. This avoids pinning the experiments to a particular
    BEIR package release.
    """

    root = Path(data_dir)
    corpus_rows = _read_jsonl(root / "corpus.jsonl")
    query_rows = _read_jsonl(root / "queries.jsonl")

    corpus: Dict[str, str] = {}
    for row in corpus_rows:
        did = str(row["_id"])
        title = str(row.get("title") or "").strip()
        text = str(row.get("text") or "").strip()
        corpus[did] = (title + "\n" + text).strip() if title else text

    queries = {str(row["_id"]): str(row["text"]) for row in query_rows}

    qrels_path = root / "qrels" / f"{split}.tsv"
    qrels: Dict[str, Dict[str, float]] = {}
    with qrels_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().lower().split("\t")
        # BEIR uses query-id, corpus-id, score. Accept headerless variants too.
        has_header = any(x in {"query-id", "query_id", "score"} for x in header)
        if not has_header:
            f.seek(0)
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qid, did, score = str(parts[0]), str(parts[1]), float(parts[2])
            qrels.setdefault(qid, {})[did] = score

    ds = RetrievalDataset(corpus=corpus, queries=queries, qrels=qrels, name=name or root.name)
    ds.validate()
    return ds



def load_bright_hf(
    domain: str,
    *,
    use_long_documents: bool = False,
    dataset_name: str = "xlangai/BRIGHT",
) -> RetrievalDataset:
    """Load an official BRIGHT domain through Hugging Face Datasets.

    BRIGHT stores examples and documents as separate dataset configurations.
    This adapter maps ``gold_ids``/``gold_ids_long`` to qrels.  Some domains
    define query-specific ``excluded_ids``; a single global-gallery posterior
    cannot reproduce that evaluation exactly, so this loader refuses such data
    rather than silently changing the benchmark.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("load_bright_hf requires the optional 'datasets' package") from e

    doc_config = "long_documents" if use_long_documents else "documents"
    gold_field = "gold_ids_long" if use_long_documents else "gold_ids"
    examples = load_dataset(dataset_name, "examples", split=str(domain))
    documents = load_dataset(dataset_name, doc_config, split=str(domain))
    corpus = {str(row["id"]): str(row["content"]) for row in documents}
    queries: Dict[str, str] = {}
    qrels: Dict[str, Dict[str, float]] = {}
    nonempty_exclusions = []
    for row in examples:
        qid = str(row["id"])
        queries[qid] = str(row["query"])
        gold = row.get(gold_field) or []
        qrels[qid] = {str(d): 1.0 for d in gold}
        excluded = tuple(str(x) for x in (row.get("excluded_ids") or []))
        if excluded:
            nonempty_exclusions.append((qid, excluded))
    if nonempty_exclusions:
        raise ValueError(
            f"BRIGHT domain {domain!r} has query-specific excluded_ids for "
            f"{len(nonempty_exclusions)} queries. The global-gallery OSER protocol "
            "cannot apply those exclusions without changing K per query; choose a "
            "domain without exclusions or use a variable-gallery experiment."
        )
    ds = RetrievalDataset(
        corpus=corpus, queries=queries, qrels=qrels,
        name=f"BRIGHT-{domain}{'-long' if use_long_documents else ''}",
        metadata={"source": dataset_name, "domain": str(domain), "document_config": doc_config},
    )
    ds.validate()
    return ds


def split_query_ids(
    query_ids: Sequence[str],
    calibration_fraction: float = 0.3,
    seed: int = 777,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    ids = np.asarray([str(x) for x in query_ids], dtype=object)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    n_cal = int(round(float(calibration_fraction) * len(ids)))
    n_cal = min(max(n_cal, 1), max(len(ids) - 1, 1)) if len(ids) > 1 else len(ids)
    return tuple(ids[perm[:n_cal]].tolist()), tuple(ids[perm[n_cal:]].tolist())


def make_evidence_deletion_protocol(
    dataset: RetrievalDataset,
    query_ids: Optional[Sequence[str]] = None,
    unknown_fraction: float = 0.5,
    seed: int = 777,
    drop_collateral_unknowns: bool = False,
) -> RetrievalProtocol:
    """Create a global-corpus open-set protocol by deleting relevant evidence.

    A subset of queries is designated unknown. The union of all documents
    relevant to those queries is removed from the shared corpus. Since a deleted
    document can also support another query, the *actual* known/unknown state is
    recomputed after deletion. Collateral unknowns are retained by default and
    explicitly reported in metadata rather than being silently mislabeled.
    """

    dataset.validate()
    qids = [str(q) for q in (query_ids if query_ids is not None else dataset.queries.keys())]
    qids = [q for q in qids if q in dataset.queries and len(dataset.relevant_docs(q)) > 0]
    if not qids:
        raise ValueError("No answerable queries with qrels are available.")

    rng = np.random.default_rng(seed)
    n_unknown = int(round(float(unknown_fraction) * len(qids)))
    n_unknown = min(max(n_unknown, 1), len(qids) - 1) if len(qids) > 1 else 1
    designated = set(rng.choice(np.asarray(qids, dtype=object), size=n_unknown, replace=False).tolist())

    deleted = set()
    for qid in designated:
        deleted.update(dataset.relevant_docs(qid))

    active_corpus = {str(d): t for d, t in dataset.corpus.items() if str(d) not in deleted}
    rel_sets: List[Tuple[str, ...]] = []
    actual_known: List[bool] = []
    designated_mask: List[bool] = []
    keep_qids: List[str] = []
    keep_queries: List[str] = []

    collateral = 0
    for qid in qids:
        remaining = tuple(d for d in dataset.relevant_docs(qid) if d in active_corpus)
        is_known = len(remaining) > 0
        is_designated = qid in designated
        if (not is_designated) and (not is_known):
            collateral += 1
            if drop_collateral_unknowns:
                continue
        keep_qids.append(qid)
        keep_queries.append(str(dataset.queries[qid]))
        rel_sets.append(remaining)
        actual_known.append(is_known)
        designated_mask.append(is_designated)

    return RetrievalProtocol(
        name=f"{dataset.name}_evidence_deletion",
        corpus=active_corpus,
        query_ids=tuple(keep_qids),
        queries=tuple(keep_queries),
        relevant_doc_ids=tuple(rel_sets),
        known_mask=np.asarray(actual_known, dtype=bool),
        designated_unknown_mask=np.asarray(designated_mask, dtype=bool),
        deleted_doc_ids=tuple(sorted(deleted)),
        metadata={
            "unknown_fraction_requested": float(unknown_fraction),
            "num_designated_unknown": int(n_unknown),
            "num_actual_unknown": int(np.sum(~np.asarray(actual_known, dtype=bool))),
            "num_collateral_unknown": int(collateral),
            "num_deleted_documents": int(len(deleted)),
            "seed": int(seed),
        },
    )


def mix_cross_domain_unknown_queries(
    known_dataset: RetrievalDataset,
    unknown_dataset: RetrievalDataset,
    num_known: Optional[int] = None,
    num_unknown: Optional[int] = None,
    seed: int = 777,
) -> RetrievalProtocol:
    """Natural OOD protocol: known queries use one corpus, OOD queries another domain."""

    rng = np.random.default_rng(seed)
    known_ids = [q for q in known_dataset.queries if len(known_dataset.relevant_docs(q)) > 0]
    unknown_ids = list(unknown_dataset.queries.keys())
    if num_known is not None and len(known_ids) > num_known:
        known_ids = rng.choice(known_ids, size=num_known, replace=False).tolist()
    if num_unknown is not None and len(unknown_ids) > num_unknown:
        unknown_ids = rng.choice(unknown_ids, size=num_unknown, replace=False).tolist()

    qids: List[str] = []
    texts: List[str] = []
    rels: List[Tuple[str, ...]] = []
    known_mask: List[bool] = []
    for qid in known_ids:
        qids.append(f"known::{qid}")
        texts.append(str(known_dataset.queries[qid]))
        rels.append(tuple(d for d in known_dataset.relevant_docs(qid) if d in known_dataset.corpus))
        known_mask.append(True)
    for qid in unknown_ids:
        qids.append(f"ood::{qid}")
        texts.append(str(unknown_dataset.queries[qid]))
        rels.append(())
        known_mask.append(False)

    return RetrievalProtocol(
        name=f"{known_dataset.name}_with_{unknown_dataset.name}_ood",
        corpus=known_dataset.corpus,
        query_ids=tuple(qids),
        queries=tuple(texts),
        relevant_doc_ids=tuple(rels),
        known_mask=np.asarray(known_mask, dtype=bool),
        designated_unknown_mask=~np.asarray(known_mask, dtype=bool),
        metadata={"seed": int(seed), "unknown_source": unknown_dataset.name},
    )


def subset_protocol(protocol: RetrievalProtocol, query_ids: Sequence[str]) -> RetrievalProtocol:
    wanted = set(map(str, query_ids))
    idx = [i for i, qid in enumerate(protocol.query_ids) if qid in wanted]
    return RetrievalProtocol(
        name=protocol.name,
        corpus=protocol.corpus,
        query_ids=tuple(protocol.query_ids[i] for i in idx),
        queries=tuple(protocol.queries[i] for i in idx),
        relevant_doc_ids=tuple(protocol.relevant_doc_ids[i] for i in idx),
        known_mask=np.asarray(protocol.known_mask)[idx],
        designated_unknown_mask=np.asarray(protocol.designated_unknown_mask)[idx],
        deleted_doc_ids=protocol.deleted_doc_ids,
        metadata=dict(protocol.metadata),
    )


def sample_corpus_preserving_relevance(
    protocol: RetrievalProtocol,
    target_size: int,
    seed: int = 777,
) -> RetrievalProtocol:
    """Downsample negatives while retaining every relevant document still active."""

    relevant = set()
    for docs in protocol.relevant_doc_ids:
        relevant.update(docs)
    corpus_ids = list(map(str, protocol.corpus.keys()))
    if target_size < len(relevant):
        raise ValueError(
            f"target_size={target_size} is smaller than {len(relevant)} required relevant docs."
        )
    if target_size >= len(corpus_ids):
        return protocol
    negatives = [d for d in corpus_ids if d not in relevant]
    rng = np.random.default_rng(seed)
    n_neg = target_size - len(relevant)
    chosen_neg = rng.choice(np.asarray(negatives, dtype=object), size=n_neg, replace=False).tolist()
    chosen = set(relevant).union(chosen_neg)
    corpus = {d: protocol.corpus[d] for d in corpus_ids if d in chosen}
    return RetrievalProtocol(
        name=f"{protocol.name}_K{len(corpus)}",
        corpus=corpus,
        query_ids=protocol.query_ids,
        queries=protocol.queries,
        relevant_doc_ids=protocol.relevant_doc_ids,
        known_mask=protocol.known_mask.copy(),
        designated_unknown_mask=protocol.designated_unknown_mask.copy(),
        deleted_doc_ids=protocol.deleted_doc_ids,
        metadata={**dict(protocol.metadata), "corpus_target_size": int(target_size)},
    )



def stratified_protocol_split(
    protocol: RetrievalProtocol,
    calibration_fraction: float = 0.3,
    seed: int = 777,
) -> Tuple[RetrievalProtocol, RetrievalProtocol]:
    """Stratified query split with a shared active corpus.

    Gallery calibration needs both known and unknown validation queries.  This
    helper therefore splits each state independently whenever possible.
    """
    rng = np.random.default_rng(seed)
    known = np.asarray(protocol.known_mask, dtype=bool)
    cal_idx: List[int] = []
    test_idx: List[int] = []
    for state in (False, True):
        idx = np.flatnonzero(known == state)
        if not len(idx):
            continue
        idx = rng.permutation(idx)
        if len(idx) == 1:
            # Keep singleton in calibration only when otherwise calibration would
            # miss that state; the caller will detect an empty test if necessary.
            n_cal = 1
        else:
            n_cal = int(round(len(idx) * float(calibration_fraction)))
            n_cal = min(max(n_cal, 1), len(idx) - 1)
        cal_idx.extend(idx[:n_cal].tolist())
        test_idx.extend(idx[n_cal:].tolist())
    cal_idx = sorted(cal_idx)
    test_idx = sorted(test_idx)
    if not test_idx:
        raise ValueError("Protocol is too small for a calibration/test split.")
    cal = subset_protocol(protocol, [protocol.query_ids[i] for i in cal_idx])
    test = subset_protocol(protocol, [protocol.query_ids[i] for i in test_idx])
    return cal, test


def nested_corpus_protocols(
    protocol: RetrievalProtocol,
    target_sizes: Sequence[int],
    seed: int = 777,
) -> List[RetrievalProtocol]:
    """Create nested corpus-size stress tests while preserving all active qrels."""
    required = set()
    for rel in protocol.relevant_doc_ids:
        required.update(rel)
    all_ids = list(map(str, protocol.corpus.keys()))
    negatives = np.asarray([d for d in all_ids if d not in required], dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(negatives)
    out = []
    seen_actual_sizes = set()
    for target in sorted(set(int(x) for x in target_sizes)):
        if target < len(required):
            raise ValueError(
                f"target_size={target} is smaller than {len(required)} required relevant docs"
            )
        target = min(target, len(all_ids))
        if target in seen_actual_sizes:
            continue
        seen_actual_sizes.add(target)
        n_neg = target - len(required)
        keep = required.union(negatives[:n_neg].tolist())
        corpus = {d: protocol.corpus[d] for d in all_ids if d in keep}
        out.append(RetrievalProtocol(
            name=f"{protocol.name}_K{len(corpus)}",
            corpus=corpus, query_ids=protocol.query_ids, queries=protocol.queries,
            relevant_doc_ids=protocol.relevant_doc_ids, known_mask=protocol.known_mask.copy(),
            designated_unknown_mask=protocol.designated_unknown_mask.copy(),
            deleted_doc_ids=protocol.deleted_doc_ids,
            metadata={**dict(protocol.metadata), "corpus_target_size": len(corpus), "nested_seed": int(seed)},
        ))
    return out


def _ragtruth_passages(source_info: Mapping[str, object]) -> Tuple[str, ...]:
    passages = source_info.get("passages", "") if isinstance(source_info, Mapping) else ""
    if isinstance(passages, list):
        return tuple(str(x) for x in passages if str(x).strip())
    text = str(passages)
    # Official QA data stores passages as "passage 1:...\n\n passage 2:...".
    pieces = re.split(r"(?i)(?:^|\n\s*)passage\s+\d+\s*:\s*", text)
    pieces = tuple(p.strip() for p in pieces if p.strip())
    return pieces if pieces else ((text.strip(),) if text.strip() else ())


def load_ragtruth(
    response_path: str | Path,
    source_info_path: str | Path,
    split: Optional[str] = "test",
    task_type: str = "QA",
    count_implicit_true_as_hallucination: bool = False,
) -> List[RAGRecord]:
    sources = {str(x["source_id"]): x for x in _read_jsonl(source_info_path)}
    records: List[RAGRecord] = []
    for row in _read_jsonl(response_path):
        if split is not None and str(row.get("split")) != str(split):
            continue
        sid = str(row["source_id"])
        source = sources.get(sid)
        if source is None or str(source.get("task_type")) != str(task_type):
            continue
        si = source.get("source_info", {})
        if task_type == "QA":
            query = str(si.get("question", "")) if isinstance(si, Mapping) else ""
            contexts = _ragtruth_passages(si if isinstance(si, Mapping) else {})
        else:
            query = str(source.get("prompt", ""))
            contexts = (json.dumps(si, ensure_ascii=False) if isinstance(si, Mapping) else str(si),)

        labels = row.get("labels") or []
        hall_labels = []
        for lab in labels:
            implicit_true = bool(lab.get("implicit_true", False)) if isinstance(lab, Mapping) else False
            if count_implicit_true_as_hallucination or not implicit_true:
                hall_labels.append(lab)
        hallucination = len(hall_labels) > 0
        quality = str(row.get("quality", "good"))
        quality_error = quality not in {"", "good", "None", "none"}
        records.append(
            RAGRecord(
                record_id=str(row["id"]),
                group_id=sid,
                query=query,
                contexts=tuple(contexts),
                response=str(row.get("response", "")),
                is_error=bool(hallucination or quality_error),
                is_hallucination=bool(hallucination),
                metadata={
                    "model": row.get("model"),
                    "temperature": row.get("temperature"),
                    "quality": quality,
                    "task_type": task_type,
                    "source": source.get("source"),
                },
            )
        )
    return records


def load_crag_records(path: str | Path, split: Optional[int] = None) -> List[RAGRecord]:
    """Load CRAG QA/search-result records as RAG inputs.

    The official benchmark's search results are not relevance-labelled, so these
    records are for end-to-end answer-risk experiments rather than OSER qrels.
    """

    rows = _read_jsonl(path)
    out: List[RAGRecord] = []
    for row in rows:
        if split is not None and int(row.get("split", -1)) != int(split):
            continue
        contexts = []
        for r in row.get("search_results", []) or []:
            text = str(r.get("page_snippet") or r.get("page_result") or "").strip()
            if text:
                contexts.append(text)
        refs = [str(row.get("answer", ""))]
        alt = row.get("alt_ans") or []
        if isinstance(alt, str):
            alt = [alt]
        refs.extend(str(x) for x in alt)
        out.append(
            RAGRecord(
                record_id=str(row["interaction_id"]),
                group_id=str(row["interaction_id"]),
                query=str(row.get("query", "")),
                contexts=tuple(contexts),
                response="",
                is_error=False,
                is_hallucination=False,
                reference_answers=tuple(x for x in refs if x),
                metadata={
                    "domain": row.get("domain"),
                    "question_type": row.get("question_type"),
                    "dynamic": row.get("static_or_dynamic"),
                    "query_time": row.get("query_time"),
                    "popularity": row.get("popularity"),
                },
            )
        )
    return out


def _tool_to_text(tool: Mapping[str, object]) -> str:
    name = str(tool.get("name") or "")
    description = str(tool.get("description") or "")
    params = tool.get("parameters") or {}
    return f"{name}\n{description}\nParameters: {json.dumps(params, sort_keys=True, ensure_ascii=False)}".strip()


def load_bfcl_relevance_files(
    relevance_files: Sequence[str | Path],
    irrelevance_files: Sequence[str | Path],
) -> List[ToolRoutingExample]:
    """Load BFCL relevance/irrelevance JSONL categories.

    BFCL relevance examples intentionally do not have a unique callable ground
    truth, so this adapter evaluates the *call vs reject* decision. For tool-ID
    evaluation use ``load_generic_tool_routing`` with explicit relevant indices.
    """

    examples: List[ToolRoutingExample] = []
    for known, files in [(True, relevance_files), (False, irrelevance_files)]:
        for path in files:
            for row in _read_jsonl(path):
                question = row.get("question", "")
                if isinstance(question, list):
                    # BFCL often stores messages/turns. Preserve text only.
                    qtxt = "\n".join(
                        str(x.get("content", x)) if isinstance(x, Mapping) else str(x)
                        for x in question
                    )
                else:
                    qtxt = str(question)
                tools = tuple(_tool_to_text(t) for t in (row.get("function") or []))
                if not tools:
                    continue
                examples.append(
                    ToolRoutingExample(
                        example_id=str(row.get("id", f"{Path(path).stem}:{len(examples)}")),
                        query=qtxt,
                        tools=tools,
                        known=known,
                        relevant_tool_indices=(),
                        metadata={"source_file": str(path)},
                    )
                )
    return examples


def load_generic_tool_routing(path: str | Path) -> List[ToolRoutingExample]:
    """Load JSONL with fields id/query/tools/known/relevant_tool_indices."""
    out = []
    for row in _read_jsonl(path):
        tools = []
        for t in row["tools"]:
            tools.append(_tool_to_text(t) if isinstance(t, Mapping) else str(t))
        out.append(
            ToolRoutingExample(
                example_id=str(row.get("id", len(out))),
                query=str(row["query"]),
                tools=tuple(tools),
                known=bool(row["known"]),
                relevant_tool_indices=tuple(int(x) for x in row.get("relevant_tool_indices", [])),
                metadata=row.get("metadata", {}),
            )
        )
    return out
