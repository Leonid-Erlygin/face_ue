#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.modern_ai.embedders import RepoArcFaceTextEmbedder, l2_normalize


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def audit(
    prepared_dir: str,
    backbone_path: str,
    softmax_weights_path: str,
    output_dir: str,
    *,
    model_name: str = "bert-base-uncased",
    num_features: int = 768,
    bottleneck_dim: int = 768,
    proj_depth: int = 2,
    batch_size: int = 64,
    max_length: int = 192,
    device: str | None = None,
) -> dict:
    root = Path(prepared_dir)
    gallery = sorted(_read_jsonl(root / "gallery.jsonl"), key=lambda x: int(x["index"]))
    val = [x for x in _read_jsonl(root / "val.jsonl") if bool(x["known"])]
    if not gallery or not val:
        raise ValueError("ArcFace audit needs a non-empty gallery and known validation queries")

    embedder = RepoArcFaceTextEmbedder(
        backbone_path=backbone_path, model_name=model_name,
        num_features=num_features, bottleneck_dim=bottleneck_dim,
        proj_depth=proj_depth, batch_size=batch_size, max_length=max_length,
        device=device,
    )
    tool_emb = embedder.encode_documents([x["text"] for x in gallery])
    query_emb = embedder.encode_queries([x["query"] for x in val])
    centers = torch.load(softmax_weights_path, map_location="cpu", weights_only=False)
    centers = l2_normalize(np.asarray(centers.detach().cpu(), dtype=np.float64))
    if len(centers) != len(gallery):
        raise ValueError(
            f"ArcFace centers ({len(centers)}) != prepared gallery ({len(gallery)}); "
            "checkpoint and protocol are not from the same training run"
        )

    labels = np.asarray([int(x["label"]) for x in val], dtype=int)
    doc_center_sim = tool_emb @ centers.T
    query_center_sim = query_emb @ centers.T
    query_doc_sim = query_emb @ tool_emb.T
    diag = doc_center_sim[np.arange(len(gallery)), np.arange(len(gallery))]
    doc_center_pred = np.argmax(doc_center_sim, axis=1)
    query_center_pred = np.argmax(query_center_sim, axis=1)
    query_doc_pred = np.argmax(query_doc_sim, axis=1)

    metrics = {
        "num_classes": int(len(gallery)),
        "num_known_validation_queries": int(len(val)),
        "api_description_to_own_center_cosine_mean": float(np.mean(diag)),
        "api_description_to_own_center_cosine_median": float(np.median(diag)),
        "api_description_nearest_center_accuracy": float(np.mean(doc_center_pred == np.arange(len(gallery)))),
        "validation_query_arcface_center_accuracy": float(np.mean(query_center_pred == labels)),
        "validation_query_api_description_gallery_accuracy": float(np.mean(query_doc_pred == labels)),
        "validation_query_true_center_cosine_mean": float(np.mean(query_center_sim[np.arange(len(val)), labels])),
        "validation_query_true_api_description_cosine_mean": float(np.mean(query_doc_sim[np.arange(len(val)), labels])),
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (out / "per_class.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["index", "tool_id", "own_center_cosine", "nearest_center_index", "center_match"])
        w.writeheader()
        for i, row in enumerate(gallery):
            w.writerow({
                "index": i, "tool_id": row.get("tool_id", ""),
                "own_center_cosine": float(diag[i]),
                "nearest_center_index": int(doc_center_pred[i]),
                "center_match": bool(doc_center_pred[i] == i),
            })
    return metrics


def main():
    p = argparse.ArgumentParser(description="Audit ArcFace center/API-description geometry for tool routing")
    p.add_argument("--prepared-dir", default="datasets/tool_routing/toolbench_g1")
    p.add_argument("--backbone-path", default="model_weights/backbone/bert_toolbench_arcface/backbone.pth")
    p.add_argument("--softmax-weights-path", default="model_weights/backbone/bert_toolbench_arcface/softmax_weight.pt")
    p.add_argument("--output-dir", default="outputs/tool_routing/arcface_geometry_audit")
    p.add_argument("--model-name", default="bert-base-uncased")
    p.add_argument("--num-features", type=int, default=768)
    p.add_argument("--bottleneck-dim", type=int, default=768)
    p.add_argument("--proj-depth", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    result = audit(**vars(args))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
