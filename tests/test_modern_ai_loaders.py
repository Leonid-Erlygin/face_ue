import json

from evaluation.modern_ai.data import load_bfcl_relevance_files, load_crag_records, load_ragtruth


def _jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_ragtruth_loader_joins_source_and_preserves_group(tmp_path):
    sources = tmp_path / "source_info.jsonl"
    responses = tmp_path / "response.jsonl"
    _jsonl(sources, [{
        "source_id": "s1", "task_type": "QA", "source": "x", "prompt": "p",
        "source_info": {"question": "Who?", "passages": "passage 1: Alpha\n\npassage 2: Beta"},
    }])
    _jsonl(responses, [{
        "id": "r1", "source_id": "s1", "split": "test", "model": "m", "temperature": 0,
        "labels": [{"text": "bad", "implicit_true": False}], "quality": "good", "response": "wrong",
    }])
    rows = load_ragtruth(responses, sources)
    assert len(rows) == 1
    assert rows[0].group_id == "s1"
    assert rows[0].is_hallucination
    assert rows[0].contexts == ("Alpha", "Beta")


def test_crag_loader_accepts_official_search_snippet_shape(tmp_path):
    path = tmp_path / "crag.jsonl"
    _jsonl(path, [{
        "interaction_id":"i1", "query":"q", "answer":"a", "alt_ans":"alt",
        "search_results":[{"page_name":"n", "page_snippet":"evidence"}],
        "domain":"x", "question_type":"simple", "static_or_dynamic":"static",
    }])
    rows = load_crag_records(path)
    assert rows[0].contexts == ("evidence",)
    assert rows[0].reference_answers == ("a", "alt")


def test_bfcl_loader_handles_message_question_and_functions(tmp_path):
    rel = tmp_path / "rel.jsonl"
    irr = tmp_path / "irr.jsonl"
    base = {
        "question": [{"role":"user", "content":"weather berlin"}],
        "function": [{"name":"weather", "description":"forecast", "parameters":{"type":"object"}}],
    }
    _jsonl(rel, [{"id":"r", **base}])
    _jsonl(irr, [{"id":"u", **{**base, "question":"write a poem"}}])
    rows = load_bfcl_relevance_files([rel], [irr])
    assert [x.known for x in rows] == [True, False]
    assert "weather" in rows[0].tools[0]
