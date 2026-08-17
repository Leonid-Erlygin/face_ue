import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from evaluation.modern_ai.beir_data import ensure_beir_dataset, validate_beir_layout


def _make_fake_beir_zip(tmp_path: Path, dataset: str = "toy") -> tuple[Path, str]:
    src = tmp_path / dataset
    (src / "qrels").mkdir(parents=True)
    (src / "corpus.jsonl").write_text(json.dumps({"_id": "d1", "text": "doc"}) + "\n")
    (src / "queries.jsonl").write_text(json.dumps({"_id": "q1", "text": "query"}) + "\n")
    (src / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n")

    archive = tmp_path / f"{dataset}.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in src.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(tmp_path))
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    return archive, digest


def test_validate_beir_layout_has_actionable_error(tmp_path):
    root = tmp_path / "scifact"
    with pytest.raises(FileNotFoundError, match="experiments/prepare_beir.py"):
        validate_beir_layout(root, split="test")


def test_ensure_beir_dataset_downloads_and_validates_file_url(tmp_path):
    archive, digest = _make_fake_beir_zip(tmp_path / "source")
    target = tmp_path / "data" / "toy"
    out = ensure_beir_dataset(
        "toy",
        target,
        split="test",
        url=archive.resolve().as_uri(),
        md5=digest,
    )
    assert out == target
    assert (target / "corpus.jsonl").is_file()
    assert (target / "queries.jsonl").is_file()
    assert (target / "qrels" / "test.tsv").is_file()
