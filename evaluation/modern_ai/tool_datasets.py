from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import hashlib
import json
import os
import random
import shutil
import subprocess
import zipfile
from .types import ToolRoutingExample

TOOLBENCH_GDRIVE_ID = "1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk"
TOOLBENCH_REPO = "https://github.com/OpenBMB/ToolBench.git"
BFCL_REPO = "https://github.com/ShishirPatil/gorilla.git"


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _flatten_query(query: Any) -> str:
    if isinstance(query, str):
        return query.strip()
    if isinstance(query, Mapping):
        return str(query.get("content", query)).strip()
    if isinstance(query, Sequence) and not isinstance(query, (str, bytes)):
        parts: List[str] = []
        for item in query:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes, Mapping)):
                parts.extend(_flatten_query(x) for x in item)
            else:
                parts.append(_flatten_query(item))
        return "\n".join(x for x in parts if x).strip()
    return str(query).strip()


def toolbench_api_identity(api: Mapping[str, Any]) -> str:
    tool = str(api.get("tool_name") or api.get("tool") or "").strip()
    name = str(api.get("api_name") or api.get("name") or "").strip()
    category = str(api.get("category_name") or api.get("category") or "").strip()
    if not tool or not name:
        raise ValueError(f"ToolBench API is missing tool/api identity: keys={sorted(api.keys())}")
    return "::".join(x for x in (category, tool, name) if x)


def _parameter_text(values: Any) -> str:
    if not values:
        return "none"
    if isinstance(values, Mapping):
        values = [values]
    out = []
    for p in values:
        if isinstance(p, Mapping):
            name = p.get("name") or p.get("parameter_name") or p.get("key") or "parameter"
            typ = p.get("type") or p.get("data_type") or ""
            desc = p.get("description") or p.get("desc") or ""
            out.append(" ".join(str(x).strip() for x in (name, typ, desc) if str(x).strip()))
        else:
            out.append(str(p))
    return "; ".join(out) if out else "none"


def serialize_toolbench_api(api: Mapping[str, Any]) -> str:
    """Stable natural-language representation of one ToolBench API.

    The representation deliberately excludes runtime answers/results.  It contains
    only information available to a router at decision time: API/tool identity,
    description, method and schema.  This prevents answer leakage into the gallery.
    """
    return "\n".join(
        [
            f"Tool: {api.get('tool_name') or api.get('tool') or ''}",
            f"API: {api.get('api_name') or api.get('name') or ''}",
            f"Category: {api.get('category_name') or api.get('category') or ''}",
            f"Description: {api.get('api_description') or api.get('description') or ''}",
            f"Method: {api.get('method') or ''}",
            f"Required parameters: {_parameter_text(api.get('required_parameters'))}",
            f"Optional parameters: {_parameter_text(api.get('optional_parameters'))}",
        ]
    ).strip()


def _relevant_identity_set(row: Mapping[str, Any]) -> set[Tuple[str, str]]:
    raw = row.get("relevant APIs") or row.get("relevant_APIs") or row.get("relevant_apis") or []
    out: set[Tuple[str, str]] = set()
    for item in raw:
        if isinstance(item, Mapping):
            tool = str(item.get("tool_name") or item.get("tool") or "")
            api = str(item.get("api_name") or item.get("name") or "")
            if tool and api:
                out.add((tool, api))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 2:
            out.add((str(item[0]), str(item[1])))
    return out


def _iter_json_array(path: Path):
    """Stream a large top-level JSON array when ijson is available.

    ToolBench G1 is hundreds of MB; reading it via ``read_text`` can multiply
    memory use several times.  ``ijson`` keeps preparation bounded-memory.
    """
    try:
        import ijson
    except ImportError:
        # Compatibility fallback for small local/synthetic files.  Production
        # requirements-modern-ai.txt installs ijson.
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {path}")
        yield from data
        return
    with path.open("rb") as f:
        yield from ijson.items(f, "item")


def parse_toolbench_g1(path: str | Path) -> Tuple[Dict[str, dict], List[dict]]:
    """Parse official ``data/instruction/G1_query.json`` into unambiguous examples.

    ToolBench's official retriever preprocessing treats ``[tool_name, api_name]`` as
    the API identity and emits a positive qrel when that pair is in ``relevant APIs``.
    We retain only rows with exactly one *distinct* relevant API for ArcFace/SCF
    classification; multi-positive rows are better treated as later routing stress tests.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ToolBench G1 file not found: {p}")
    tools: Dict[str, dict] = {}
    examples: List[dict] = []
    for idx, row in enumerate(_iter_json_array(p)):
        query = _flatten_query(row.get("query", ""))
        if not query:
            continue
        relevant = _relevant_identity_set(row)
        matches: Dict[str, Mapping[str, Any]] = {}
        for api in row.get("api_list") or []:
            tool = str(api.get("tool_name") or api.get("tool") or "")
            name = str(api.get("api_name") or api.get("name") or "")
            if (tool, name) in relevant:
                tid = toolbench_api_identity(api)
                matches[tid] = api
        if len(matches) != 1:
            continue
        tool_id, api = next(iter(matches.items()))
        tool_text = serialize_toolbench_api(api)
        tools.setdefault(
            tool_id,
            {
                "tool_id": tool_id,
                "text": tool_text,
                "tool_name": str(api.get("tool_name") or api.get("tool") or ""),
                "api_name": str(api.get("api_name") or api.get("name") or ""),
                "category": str(api.get("category_name") or api.get("category") or ""),
                "raw": dict(api),
            },
        )
        examples.append(
            {
                "id": str(row.get("query_id", idx)),
                "query": query,
                "tool_id": tool_id,
            }
        )
    if not examples:
        raise ValueError(f"No unambiguous single-API examples could be parsed from {p}")
    return tools, examples


def toolbench_query_count_statistics(
    g1_query_path: str | Path,
    *,
    thresholds: Sequence[int] = (3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 50),
) -> dict:
    """Summarize the usable single-API query density in ToolBench G1.

    This is intentionally a *data-only* diagnostic: it does not inspect any model
    outputs or prepared validation/test predictions.  It is safe to run while the
    final OSR test remains untouched.
    """
    tools, examples = parse_toolbench_g1(g1_query_path)
    counts = Counter(str(row["tool_id"]) for row in examples)
    values = sorted(int(v) for v in counts.values())
    if not values:
        raise ValueError("No usable ToolBench API classes found")

    import numpy as np

    arr = np.asarray(values, dtype=float)
    threshold_counts = {
        str(int(t)): int(np.sum(arr >= int(t)))
        for t in thresholds
    }
    quantiles = {
        name: float(np.quantile(arr, q))
        for name, q in (
            ("q00", 0.00), ("q10", 0.10), ("q25", 0.25),
            ("q50", 0.50), ("q75", 0.75), ("q90", 0.90),
            ("q95", 0.95), ("q99", 0.99), ("q100", 1.00),
        )
    }
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
    return {
        "source": str(Path(g1_query_path)),
        "num_unique_single_api_classes": int(len(counts)),
        "num_usable_single_api_queries": int(len(examples)),
        "query_count_per_api": {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "quantiles": quantiles,
        },
        "num_api_classes_with_at_least_n_queries": threshold_counts,
        "top_20_api_classes_by_query_count": [
            {"tool_id": tool_id, "num_queries": int(n)}
            for tool_id, n in top
        ],
        "protocol_guidance": {
            "why": (
                "ArcFace and SCF should use disjoint query samples. A four-way "
                "known-class split therefore needs materially more than the previous "
                "minimum of three queries per API."
            ),
            "candidate_split": {
                "arcface_train_fraction": 0.50,
                "scf_train_fraction": 0.25,
                "validation_fraction": 0.125,
                "test_fraction": 0.125,
            },
            "recommended_minimum_queries_per_known_api": 8,
            "preferred_minimum_queries_per_known_api": 12,
        },
    }


def _split_items(items: Sequence[dict], rng: random.Random, train_fraction: float, val_fraction: float):
    xs = list(items)
    rng.shuffle(xs)
    n = len(xs)
    if n < 3:
        raise ValueError("Known classes need at least three query examples for train/val/test")
    n_val = max(1, int(round(n * val_fraction)))
    n_test = max(1, int(round(n * (1.0 - train_fraction - val_fraction))))
    if n_val + n_test >= n:
        n_val, n_test = 1, 1
    n_train = n - n_val - n_test
    return xs[:n_train], xs[n_train:n_train+n_val], xs[n_train+n_val:]


def build_toolbench_class_disjoint_protocol(
    g1_query_path: str | Path,
    output_dir: str | Path,
    *,
    num_known_tools: int = 1024,
    num_unknown_tools: int = 256,
    min_queries_per_tool: int = 3,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    add_tool_documents_to_train: bool = True,
    unknown_calibration_fraction: float = 0.5,
    seed: int = 777,
) -> dict:
    """Create class-disjoint ArcFace/SCF + OSR splits from ToolBench G1.

    Known tool/API classes appear in ArcFace/SCF training and in the fixed gallery.
    Unknown classes are never exposed to either training stage; only their queries
    appear in OSR validation/test.  This makes open-set claims class-disjoint.
    """
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 1:
        raise ValueError("Need 0 < train_fraction, val_fraction and train+val < 1")
    if not 0.0 < unknown_calibration_fraction < 1.0:
        raise ValueError("unknown_calibration_fraction must be in (0, 1)")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tools, examples = parse_toolbench_g1(g1_query_path)
    by_tool: Dict[str, List[dict]] = defaultdict(list)
    for row in examples:
        by_tool[row["tool_id"]].append(row)

    eligible = sorted(t for t, rows in by_tool.items() if len(rows) >= int(min_queries_per_tool))
    if int(num_known_tools) < 2:
        raise ValueError("num_known_tools must be >= 2")
    if int(num_unknown_tools) < 2:
        raise ValueError("num_unknown_tools must be >= 2 so validation/test unknown classes can be disjoint")
    need = int(num_known_tools) + int(num_unknown_tools)
    if len(eligible) < need:
        raise ValueError(
            f"Only {len(eligible)} ToolBench APIs have >= {min_queries_per_tool} queries, "
            f"but {need} classes were requested. Reduce num_known_tools/num_unknown_tools."
        )
    rng = random.Random(int(seed))
    rng.shuffle(eligible)
    known_tools = sorted(eligible[: int(num_known_tools)])
    unknown_tools = eligible[int(num_known_tools):need]
    rng.shuffle(unknown_tools)
    cut_unknown = max(1, int(round(len(unknown_tools) * float(unknown_calibration_fraction))))
    cut_unknown = min(cut_unknown, len(unknown_tools) - 1)
    unknown_calibration_tools = sorted(unknown_tools[:cut_unknown])
    unknown_test_tools = sorted(unknown_tools[cut_unknown:])
    labels = {tid: i for i, tid in enumerate(known_tools)}

    train_query_rows: List[dict] = []
    train_arcface_rows: List[dict] = []
    val_rows: List[dict] = []
    test_rows: List[dict] = []
    for tid in known_tools:
        tr, va, te = _split_items(by_tool[tid], rng, train_fraction, val_fraction)
        label = labels[tid]
        for x in tr:
            row = {"id": x["id"], "text": x["query"], "label": label, "tool_id": tid, "sample_type": "query"}
            train_query_rows.append(row)
            train_arcface_rows.append(row)
        if add_tool_documents_to_train:
            train_arcface_rows.append({
                "id": f"tool-doc::{tid}", "text": tools[tid]["text"], "label": label,
                "tool_id": tid, "sample_type": "tool_document",
            })
        for split_rows, xs in ((val_rows, va), (test_rows, te)):
            split_rows.extend({
                "id": x["id"], "query": x["query"], "known": True,
                "tool_id": tid, "label": label,
            } for x in xs)

    # Unknown API classes are disjoint not only from training but also between
    # calibration and final test.  This prevents tuning kappa_g/threshold behavior
    # on the same unknown semantic classes later used to claim open-set transfer.
    for tid in unknown_calibration_tools:
        val_rows.extend({
            "id": x["id"], "query": x["query"], "known": False,
            "tool_id": tid, "label": -1,
        } for x in by_tool[tid])
    for tid in unknown_test_tools:
        test_rows.extend({
            "id": x["id"], "query": x["query"], "known": False,
            "tool_id": tid, "label": -1,
        } for x in by_tool[tid])

    rng.shuffle(train_query_rows); rng.shuffle(train_arcface_rows); rng.shuffle(val_rows); rng.shuffle(test_rows)
    gallery = [
        {"index": labels[tid], "tool_id": tid, "text": tools[tid]["text"],
         "tool_name": tools[tid]["tool_name"], "api_name": tools[tid]["api_name"],
         "category": tools[tid]["category"]}
        for tid in known_tools
    ]
    # ArcFace gets canonical API descriptions in addition to query text so the
    # learned class geometry is aligned with the API-description gallery used at
    # OSR evaluation.  SCF gets query samples only: kappa(x) is a *query*
    # reliability signal, while gallery uncertainty remains the globally
    # calibrated kappa_g from the dissertation model.
    _write_jsonl(out / "train_arcface.jsonl", train_arcface_rows)
    _write_jsonl(out / "train_scf.jsonl", train_query_rows)
    # Backwards-compatible alias for tools/scripts created before the split.
    _write_jsonl(out / "train.jsonl", train_arcface_rows)
    _write_jsonl(out / "val.jsonl", val_rows)
    _write_jsonl(out / "test.jsonl", test_rows)
    _write_jsonl(out / "gallery.jsonl", gallery)
    _json_dump(out / "labels.json", labels)

    manifest = {
        "protocol": "toolbench_g1_class_disjoint_tool_routing",
        "source": str(Path(g1_query_path)),
        "seed": int(seed),
        "num_known_tools": len(known_tools),
        "num_unknown_tools": len(unknown_tools),
        "num_unknown_calibration_tools": len(unknown_calibration_tools),
        "num_unknown_test_tools": len(unknown_test_tools),
        "unknown_calibration_fraction": float(unknown_calibration_fraction),
        "min_queries_per_tool": int(min_queries_per_tool),
        "num_train_arcface_samples": len(train_arcface_rows),
        "num_train_scf_samples": len(train_query_rows),
        "num_val_queries": len(val_rows),
        "num_test_queries": len(test_rows),
        "num_val_known": sum(bool(x["known"]) for x in val_rows),
        "num_val_unknown": sum(not bool(x["known"]) for x in val_rows),
        "num_test_known": sum(bool(x["known"]) for x in test_rows),
        "num_test_unknown": sum(not bool(x["known"]) for x in test_rows),
        "add_tool_documents_to_train": bool(add_tool_documents_to_train),
        "known_tool_ids_sha256": hashlib.sha256("\n".join(known_tools).encode()).hexdigest(),
        "unknown_tool_ids_sha256": hashlib.sha256("\n".join(sorted(unknown_tools)).encode()).hexdigest(),
        "unknown_calibration_tool_ids_sha256": hashlib.sha256("\n".join(unknown_calibration_tools).encode()).hexdigest(),
        "unknown_test_tool_ids_sha256": hashlib.sha256("\n".join(unknown_test_tools).encode()).hexdigest(),
        "source_g1_sha256": _sha256_file(Path(g1_query_path)),
    }
    _json_dump(out / "manifest.json", manifest)
    return manifest


def load_toolbench_oser_examples(prepared_dir: str | Path, split: str) -> List[ToolRoutingExample]:
    root = Path(prepared_dir)
    gallery = _read_jsonl(root / "gallery.jsonl")
    tools = tuple(str(x["text"]) for x in sorted(gallery, key=lambda z: int(z["index"])))
    rows = _read_jsonl(root / f"{split}.jsonl")
    out: List[ToolRoutingExample] = []
    for row in rows:
        known = bool(row["known"])
        relevant = (int(row["label"]),) if known else ()
        out.append(ToolRoutingExample(
            example_id=str(row["id"]), query=str(row["query"]), tools=tools,
            known=known, relevant_tool_indices=relevant,
            metadata={"tool_id": str(row.get("tool_id", "")), "split": split, "source": "ToolBench-G1"},
        ))
    return out


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            try:
                target.relative_to(destination)
            except ValueError as e:
                raise ValueError(f"Unsafe path in ZIP archive: {member.filename}") from e
        zf.extractall(destination)


def _maybe_extract_reproduction_data(
    source_dir: Path,
    root: Path,
    *,
    force: bool = False,
) -> Optional[Path]:
    """Extract local ToolBench reproduction_data.zip once, if available."""

    archive = source_dir / "reproduction_data.zip"
    if not archive.is_file():
        print(
            f"[ToolBench] Optional archive not found: {archive}. "
            "Continuing because tool routing only requires data.zip."
        )
        return None

    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Not a valid ZIP archive: {archive}")

    # Keep a cheap extraction marker so we do not unpack this large archive
    # every time the preparation script is rerun.
    marker = root / ".reproduction_data_extracted"
    stat = archive.stat()
    signature = f"{stat.st_size}:{stat.st_mtime_ns}"

    already_extracted = (
        marker.is_file()
        and marker.read_text(encoding="utf-8").strip() == signature
    )

    if force or not already_extracted:
        print(f"[ToolBench] Extracting {archive} -> {root}")
        _safe_extract_zip(archive, root)

        marker.write_text(signature + "\n", encoding="utf-8")
    else:
        print(f"[ToolBench] reproduction_data.zip already extracted; skipping")

    return archive


def download_toolbench_data(
    output_root: str | Path,
    *,
    force: bool = False,
) -> Path:
    """
    Prepare ToolBench from locally supplied archives.

    Preferred archive location:

        /app/datasets/ToolBench_data/data.zip
        /app/datasets/ToolBench_data/reproduction_data.zip

    Override it with:

        TOOLBENCH_ARCHIVE_DIR=/some/other/path

    No network access is performed here.
    """

    root = Path(output_root)
    data_dir = root / "data"
    target = data_dir / "instruction" / "G1_query.json"

    source_dir = Path(
        os.environ.get(
            "TOOLBENCH_ARCHIVE_DIR",
            "/app/datasets/ToolBench_data",
        )
    ).expanduser()

    data_archive = source_dir / "data.zip"

    # Fast path: the dataset has already been extracted.
    if target.is_file() and not force:
        print(f"[ToolBench] Dataset already prepared: {target}")

        # reproduction_data is not needed by the routing benchmark, but if the
        # archive was supplied, make sure it is extracted as well.
        root.mkdir(parents=True, exist_ok=True)
        _maybe_extract_reproduction_data(
            source_dir,
            root,
            force=False,
        )

        return target

    # Dataset is not extracted, so a local data.zip is required.
    if not data_archive.is_file():
        raise FileNotFoundError(
            "ToolBench is not prepared and the local archive was not found.\n"
            f"Expected extracted dataset:\n  {target}\n"
            f"or local archive:\n  {data_archive}\n\n"
            "Place data.zip in /app/datasets/ToolBench_data or set "
            "TOOLBENCH_ARCHIVE_DIR."
        )

    if not zipfile.is_zipfile(data_archive):
        raise ValueError(
            f"ToolBench archive is not a valid ZIP file: {data_archive}"
        )

    root.mkdir(parents=True, exist_ok=True)

    if force and data_dir.exists():
        print(f"[ToolBench] Removing existing data directory: {data_dir}")
        shutil.rmtree(data_dir)

    print(f"[ToolBench] Extracting {data_archive} -> {root}")
    _safe_extract_zip(data_archive, root)

    # Normal expected archive layout:
    #
    #   data/
    #     instruction/
    #       G1_query.json
    #
    # Some releases/mirrors may add an outer directory. Normalize that case.
    if not target.is_file():
        matches = list(
            root.glob("**/data/instruction/G1_query.json")
        )

        if len(matches) == 1:
            discovered_target = matches[0]
            src_data = discovered_target.parents[1]

            if src_data.resolve() != data_dir.resolve():
                print(
                    "[ToolBench] Normalizing extracted data directory:\n"
                    f"  {src_data}\n"
                    f"  -> {data_dir}"
                )

                if data_dir.exists():
                    shutil.rmtree(data_dir)

                shutil.move(
                    str(src_data),
                    str(data_dir),
                )

        elif len(matches) > 1:
            raise RuntimeError(
                "Multiple ToolBench G1 files were found after extraction:\n"
                + "\n".join(f"  {p}" for p in matches)
            )

    if not target.is_file():
        raise FileNotFoundError(
            f"Extracted {data_archive}, but the required ToolBench file "
            f"was not found:\n  {target}"
        )

    print(f"[ToolBench] Found G1 dataset: {target}")

    reproduction_archive = _maybe_extract_reproduction_data(
        source_dir,
        root,
        force=force,
    )

    manifest = {
        "source": "local_toolbench_archives",
        "archive_dir": str(source_dir.resolve()),
        "data_archive": str(data_archive.resolve()),
        "data_archive_sha256": _sha256_file(data_archive),
        "reproduction_archive": (
            str(reproduction_archive.resolve())
            if reproduction_archive is not None
            else None
        ),
        "reproduction_archive_sha256": (
            _sha256_file(reproduction_archive)
            if reproduction_archive is not None
            else None
        ),
        "g1_query_path": str(target),
        "g1_query_sha256": _sha256_file(target),
    }

    _json_dump(
        root / "download_manifest.json",
        manifest,
    )

    print(
        "[ToolBench] Local dataset preparation complete:\n"
        f"  G1: {target}"
    )

    return target

def clone_bfcl(output_dir: str | Path, *, ref: str = "main", force: bool = False) -> dict:
    """Clone official Gorilla/BFCL and record the exact commit for reproducibility."""
    root = Path(output_dir)
    if root.exists() and force:
        shutil.rmtree(root)
    if not root.exists():
        subprocess.run(["git", "clone", "--filter=blob:none", BFCL_REPO, str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "fetch", "origin", ref], check=True)
    subprocess.run(["git", "-C", str(root), "checkout", "--detach", "FETCH_HEAD"], check=True)
    sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    return {"repository": BFCL_REPO, "requested_ref": ref, "commit": sha, "root": str(root)}


def _looks_like_bfcl_question(path: Path) -> bool:
    try:
        rows = _read_jsonl(path)
    except Exception:
        return False
    if not rows:
        return False
    row = rows[0]
    return isinstance(row, Mapping) and "id" in row and "question" in row and "function" in row


def _bfcl_possible_answer_for_question(question_path: Path, data_root: Path) -> Optional[Path]:
    candidate = data_root / "possible_answer" / question_path.name
    return candidate if candidate.exists() else None


def discover_bfcl_routing_files(gorilla_root: str | Path) -> dict:
    """Discover BFCL single-turn routing categories and matching answer files.

    The current Gorilla checkout keeps versioned question files under ``data/``
    and exact AST targets under ``data/possible_answer/`` with the same basename.
    We discover rather than hard-code the version prefix, but prefer the newest
    non-live version if several historical versions are present.
    """
    root = Path(gorilla_root)
    data_root = root / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"
    if not data_root.exists():
        raise FileNotFoundError(f"BFCL data directory not found: {data_root}")

    candidates: Dict[str, List[Path]] = {
        "known_multiple": [], "known_relevance": [], "unknown_irrelevance": []
    }
    for path in sorted(data_root.glob("*.json")):
        low = path.stem.lower()
        # Use static single-turn routing only.  Live data is valuable for LLM
        # function-calling evaluation but changes the distribution and can mix
        # temporal effects with our open-set routing question.
        if any(x in low for x in ("multi_turn", "parallel_multiple", "live_")):
            continue
        bucket = None
        if low.endswith("irrelevance"):
            bucket = "unknown_irrelevance"
        elif low.endswith("relevance"):
            bucket = "known_relevance"
        elif low.endswith("multiple"):
            bucket = "known_multiple"
        if bucket and _looks_like_bfcl_question(path):
            candidates[bucket].append(path)

    def version_key(path: Path) -> Tuple[int, str]:
        import re
        m = re.search(r"(?:^|_)v(\d+)(?:_|$)", path.stem.lower())
        return (int(m.group(1)) if m else -1, path.name)

    selected: Dict[str, List[str]] = {}
    answer_files: Dict[str, Dict[str, str]] = {}
    for bucket, paths in candidates.items():
        if paths:
            max_version = max(version_key(x)[0] for x in paths)
            paths = [x for x in paths if version_key(x)[0] == max_version]
        selected[bucket] = [str(x) for x in sorted(paths)]
        answer_files[bucket] = {}
        for qpath in paths:
            ans = _bfcl_possible_answer_for_question(qpath, data_root)
            if ans is not None:
                answer_files[bucket][str(qpath)] = str(ans)

    if not selected["unknown_irrelevance"]:
        raise FileNotFoundError(f"Could not discover a BFCL single-turn irrelevance dataset in {data_root}")
    if not (selected["known_multiple"] or selected["known_relevance"]):
        raise FileNotFoundError(f"Could not discover BFCL multiple/relevance datasets in {data_root}")
    return {"data_root": str(data_root), **selected, "answer_files": answer_files}


def serialize_bfcl_function(function: Mapping[str, Any]) -> str:
    """Natural, deterministic text representation of one BFCL function schema."""
    params = function.get("parameters") or {}
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except Exception:
            params = {"raw": params}
    properties = params.get("properties", {}) if isinstance(params, Mapping) else {}
    required = set(params.get("required", [])) if isinstance(params, Mapping) else set()
    pieces: List[str] = []
    if isinstance(properties, Mapping):
        for name, spec in properties.items():
            if isinstance(spec, Mapping):
                typ = spec.get("type", "")
                desc = spec.get("description", "")
            else:
                typ, desc = "", str(spec)
            status = "required" if name in required else "optional"
            pieces.append(f"{name} ({typ}, {status}): {desc}".strip())
    return "\n".join([
        f"Function: {function.get('name', '')}",
        f"Description: {function.get('description', '')}",
        "Parameters: " + ("; ".join(pieces) if pieces else "none"),
    ]).strip()


def _load_bfcl_targets(answer_path: Path) -> Dict[str, set[str]]:
    targets: Dict[str, set[str]] = {}
    for row in _read_jsonl(answer_path):
        names: set[str] = set()
        for call in row.get("ground_truth") or []:
            if isinstance(call, Mapping):
                names.update(str(k) for k in call.keys())
        targets[str(row.get("id", ""))] = names
    return targets


def bfcl_routing_examples_from_manifest(manifest: Mapping[str, Any]) -> List[ToolRoutingExample]:
    """Load BFCL as external variable-gallery open-set routing evaluation.

    * ``multiple``/``relevance`` are known/call examples.
    * ``irrelevance`` is a true none-of-the-provided-functions/reject condition.
    * When an official ``possible_answer`` file exists, the exact called function
      name is mapped back to the candidate list so tool-selection accuracy is
      reported in addition to call-vs-reject performance.
    """
    answer_manifest = manifest.get("answer_files") or {}
    out: List[ToolRoutingExample] = []
    for known, key in (
        (True, "known_multiple"),
        (True, "known_relevance"),
        (False, "unknown_irrelevance"),
    ):
        for p in manifest.get(key, []):
            p = str(p)
            answer_path = (answer_manifest.get(key) or {}).get(p)
            targets = _load_bfcl_targets(Path(answer_path)) if answer_path else {}
            for row in _read_jsonl(Path(p)):
                functions = row.get("function") or []
                function_names = [
                    str(x.get("name", "")) if isinstance(x, Mapping) else ""
                    for x in functions
                ]
                tools = tuple(
                    serialize_bfcl_function(x) if isinstance(x, Mapping) else str(x)
                    for x in functions
                )
                query = _flatten_query(row.get("question", ""))
                if not tools or not query:
                    continue
                example_id = str(row.get("id", f"{Path(p).stem}:{len(out)}"))
                target_names = targets.get(example_id, set())
                relevant = tuple(
                    i for i, name in enumerate(function_names) if name in target_names
                ) if known else ()
                # If an official answer exists but cannot be mapped to one of the
                # candidate functions, fail loudly rather than silently downgrading
                # a tool-ID benchmark into a call/reject-only benchmark.
                if known and target_names and not relevant:
                    raise ValueError(
                        f"BFCL target(s) {sorted(target_names)} for {example_id} do not match "
                        f"candidate functions {function_names} in {p}"
                    )
                out.append(ToolRoutingExample(
                    example_id=example_id,
                    query=query,
                    tools=tools,
                    known=known,
                    relevant_tool_indices=relevant,
                    metadata={
                        "source": "BFCL",
                        "source_file": p,
                        "answer_file": str(answer_path or ""),
                        "target_function_names": sorted(target_names),
                    },
                ))
    return out
