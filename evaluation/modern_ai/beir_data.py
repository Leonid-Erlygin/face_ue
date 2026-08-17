from __future__ import annotations

import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


@dataclass(frozen=True)
class BEIRDatasetSpec:
    name: str
    md5: Optional[str] = None

    @property
    def url(self) -> str:
        return f"{BEIR_BASE_URL}/{self.name}.zip"


# Checksums published by the official BEIR dataset table for datasets used by
# this extension. Unknown datasets can still be downloaded, but callers should
# provide --md5 if they want integrity verification.
BEIR_DATASET_SPECS: Dict[str, BEIRDatasetSpec] = {
    "scifact": BEIRDatasetSpec("scifact", "5f7d1de60b170fc8027bb7898e2efca1"),
    "fiqa": BEIRDatasetSpec("fiqa", "17918ed23cd04fb15047f73e6c3bd9d9"),
    "nfcorpus": BEIRDatasetSpec("nfcorpus", "a89dba18a62ef92f7d323ec890a0d38d"),
    "trec-covid": BEIRDatasetSpec("trec-covid", "ce62140cb23feb9becf6270d0d1fe6d1"),
    "nq": BEIRDatasetSpec("nq", "d4d3d2e48787a744b6f6e691ff534307"),
    "hotpotqa": BEIRDatasetSpec("hotpotqa", "f412724f78b0d91183a0e86805e16114"),
}


def beir_required_files(split: str = "test") -> Tuple[Path, ...]:
    return (
        Path("corpus.jsonl"),
        Path("queries.jsonl"),
        Path("qrels") / f"{split}.tsv",
    )


def missing_beir_files(data_dir: str | Path, split: str = "test") -> Tuple[Path, ...]:
    root = Path(data_dir)
    return tuple(rel for rel in beir_required_files(split) if not (root / rel).is_file())


def validate_beir_layout(data_dir: str | Path, split: str = "test") -> Path:
    root = Path(data_dir)
    missing = missing_beir_files(root, split=split)
    if missing:
        formatted = ", ".join(str(root / rel) for rel in missing)
        dataset = root.name
        raise FileNotFoundError(
            f"BEIR dataset {dataset!r} is not prepared at {root}. "
            f"Missing: {formatted}.\n"
            "Prepare it with:\n"
            f"  python experiments/prepare_beir.py --dataset {dataset} "
            f"--path {root} --split {split}"
        )
    return root


def _md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out, length=chunk_size)
        tmp.replace(destination)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            candidate = (destination / member.filename).resolve()
            try:
                candidate.relative_to(destination)
            except ValueError as exc:
                raise ValueError(f"Unsafe path in BEIR archive: {member.filename!r}") from exc
        zf.extractall(destination)


def _find_extracted_dataset_root(extraction_dir: Path, dataset: str, split: str) -> Path:
    candidates = [extraction_dir / dataset, extraction_dir]
    candidates.extend(p.parent for p in extraction_dir.rglob("corpus.jsonl"))

    seen = set()
    valid = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if not missing_beir_files(candidate, split=split):
            valid.append(candidate)

    if len(valid) == 1:
        return valid[0]
    if not valid:
        raise RuntimeError(
            f"Downloaded BEIR archive for {dataset!r} did not contain the expected "
            f"corpus.jsonl, queries.jsonl, qrels/{split}.tsv layout."
        )
    preferred = (extraction_dir / dataset).resolve()
    if preferred in valid:
        return preferred
    raise RuntimeError(
        f"Downloaded BEIR archive for {dataset!r} contains multiple candidate dataset roots: "
        + ", ".join(str(p) for p in valid)
    )


def ensure_beir_dataset(
    dataset: str,
    path: str | Path,
    *,
    split: str = "test",
    url: Optional[str] = None,
    md5: Optional[str] = None,
    download_cache: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Ensure an official-format BEIR dataset exists locally.

    Existing valid data is never downloaded again. A partial/invalid target is
    rejected unless ``force=True`` so an experiment runner cannot silently erase
    user data. Downloads are checksum-verified when a checksum is known.
    """

    dataset = str(dataset)
    target = Path(path)
    if not missing_beir_files(target, split=split):
        return target

    if target.exists():
        is_nonempty_dir = target.is_dir() and any(target.iterdir())
        is_wrong_type = not target.is_dir()
        if is_nonempty_dir or is_wrong_type:
            if not force:
                missing = ", ".join(str(x) for x in missing_beir_files(target, split=split))
                raise RuntimeError(
                    f"BEIR target {target} exists but is incomplete/invalid (missing {missing}). "
                    "Refusing to overwrite it automatically. Remove/fix the path or rerun "
                    "experiments/prepare_beir.py with --force."
                )
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()

    spec = BEIR_DATASET_SPECS.get(dataset, BEIRDatasetSpec(dataset))
    source_url = url or spec.url
    expected_md5 = (md5 or spec.md5)

    cache_root = Path(download_cache) if download_cache is not None else target.parent / ".downloads"
    archive = cache_root / f"{dataset}.zip"
    cache_root.mkdir(parents=True, exist_ok=True)

    needs_download = not archive.is_file()
    if archive.is_file() and expected_md5:
        actual = _md5(archive)
        if actual.lower() != expected_md5.lower():
            archive.unlink()
            needs_download = True

    if needs_download:
        print(f"Downloading BEIR/{dataset} from {source_url}")
        try:
            _download(source_url, archive)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download BEIR/{dataset} from {source_url}. "
                "Check network/DNS/proxy access. You can also download the archive manually "
                f"to {archive} and rerun the command."
            ) from exc

    if expected_md5:
        actual = _md5(archive)
        if actual.lower() != expected_md5.lower():
            archive.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {dataset}: expected {expected_md5}, got {actual}. "
                "The downloaded archive was removed."
            )
        print(f"Verified MD5: {actual}")

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"beir-{dataset}-", dir=target.parent) as tmp_name:
        extraction_dir = Path(tmp_name)
        _safe_extract_zip(archive, extraction_dir)
        source_root = _find_extracted_dataset_root(extraction_dir, dataset=dataset, split=split)
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(source_root), str(target))

    validate_beir_layout(target, split=split)
    print(f"BEIR/{dataset} ready at {target}")
    return target
