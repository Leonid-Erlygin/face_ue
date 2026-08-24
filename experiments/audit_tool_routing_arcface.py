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


def _routing_metrics(rows, gallery, embedder, *, text_key: str):
    gallery = sorted(gallery, key=lambda x: int(x["index"]))
    labels = np.asarray([int(x["label"]) for x in rows], dtype=int)
    q = embedder.encode_queries([str(x[text_key]) for x in rows])
    g = embedder.encode_documents([str(x["text"]) for x in gallery])
    sim = q @ g.T
    pred = np.argmax(sim, axis=1)
    true = sim[np.arange(len(rows)), labels]
    return {
        "num_classes": int(len(gallery)),
        "num_queries": int(len(rows)),
        "accuracy": float(np.mean(pred == labels)),
        "true_cosine_mean": float(np.mean(true)),
        "true_cosine_median": float(np.median(true)),
    }


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
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    stage_v2 = str(manifest.get("protocol", "")).endswith("stage_class_disjoint_v2")

    embedder = RepoArcFaceTextEmbedder(
        backbone_path=backbone_path, model_name=model_name,
        num_features=num_features, bottleneck_dim=bottleneck_dim,
        proj_depth=proj_depth, batch_size=batch_size, max_length=max_length,
        device=device,
    )
    centers = torch.load(softmax_weights_path, map_location="cpu", weights_only=False)
    centers = l2_normalize(np.asarray(centers.detach().cpu(), dtype=np.float64))

    if not stage_v2:
        gallery = sorted(_read_jsonl(root / "gallery.jsonl"), key=lambda x: int(x["index"]))
        val = [x for x in _read_jsonl(root / "val.jsonl") if bool(x["known"])]
        if not gallery or not val:
            raise ValueError("ArcFace audit needs a non-empty gallery and known validation queries")
        tool_emb = embedder.encode_documents([x["text"] for x in gallery])
        query_emb = embedder.encode_queries([x["query"] for x in val])
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
            "protocol": str(manifest.get("protocol", "legacy")),
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
        per_gallery = gallery
        diag_rows = diag
        pred_rows = doc_center_pred
    else:
        # ArcFace classifier geometry is audited only on the classes used to fit
        # the classifier.  Generalization is then audited on two *unseen* API
        # class sets without touching final-test APIs.
        arc_gallery = sorted(_read_jsonl(root / "arcface_gallery.jsonl"), key=lambda x: int(x["index"]))
        scf_gallery = sorted(_read_jsonl(root / "scf_gallery.jsonl"), key=lambda x: int(x["index"]))
        cal_gallery = sorted(_read_jsonl(root / "calibration_gallery.jsonl"), key=lambda x: int(x["index"]))
        arc_queries = [x for x in _read_jsonl(root / "train_arcface.jsonl") if x.get("sample_type") == "query"]
        scf_queries = _read_jsonl(root / "train_scf.jsonl")
        cal_queries = [x for x in _read_jsonl(root / "val.jsonl") if bool(x.get("known"))]
        if len(centers) != len(arc_gallery):
            raise ValueError(
                f"ArcFace centers ({len(centers)}) != ArcFace training gallery ({len(arc_gallery)})"
            )

        arc_doc_emb = embedder.encode_documents([x["text"] for x in arc_gallery])
        doc_center_sim = arc_doc_emb @ centers.T
        diag = doc_center_sim[np.arange(len(arc_gallery)), np.arange(len(arc_gallery))]
        doc_center_pred = np.argmax(doc_center_sim, axis=1)
        arc_q = embedder.encode_queries([x["text"] for x in arc_queries])
        arc_labels = np.asarray([int(x["label"]) for x in arc_queries], dtype=int)
        arc_center_sim = arc_q @ centers.T
        arc_center_pred = np.argmax(arc_center_sim, axis=1)

        arc_route = _routing_metrics(arc_queries, arc_gallery, embedder, text_key="text")
        scf_route = _routing_metrics(scf_queries, scf_gallery, embedder, text_key="text")
        cal_route = _routing_metrics(cal_queries, cal_gallery, embedder, text_key="query")
        metrics = {
            "protocol": str(manifest.get("protocol")),
            "uses_final_test_split": False,
            "num_arcface_classes": int(len(arc_gallery)),
            "api_description_to_own_center_cosine_mean": float(np.mean(diag)),
            "api_description_to_own_center_cosine_median": float(np.median(diag)),
            "api_description_nearest_center_accuracy": float(np.mean(doc_center_pred == np.arange(len(arc_gallery)))),
            "arcface_training_query_center_accuracy": float(np.mean(arc_center_pred == arc_labels)),
            "arcface_training_query_true_center_cosine_mean": float(np.mean(arc_center_sim[np.arange(len(arc_queries)), arc_labels])),
            "arcface_training_query_api_gallery": arc_route,
            "scf_stage_unseen_class_api_gallery": scf_route,
            "calibration_unseen_class_api_gallery": cal_route,
            # Compatibility key used by the fail-fast shell threshold.  In v2 it
            # intentionally means transfer to calibration API identities that
            # were never used by ArcFace training.
            "validation_query_api_description_gallery_accuracy": float(cal_route["accuracy"]),
        }
        per_gallery = arc_gallery
        diag_rows = diag
        pred_rows = doc_center_pred

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (out / "per_class.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["index", "tool_id", "own_center_cosine", "nearest_center_index", "center_match"])
        w.writeheader()
        for i, row in enumerate(per_gallery):
            w.writerow({
                "index": i, "tool_id": row.get("tool_id", ""),
                "own_center_cosine": float(diag_rows[i]),
                "nearest_center_index": int(pred_rows[i]),
                "center_match": bool(pred_rows[i] == i),
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
    p.add_argument("--min-query-gallery-accuracy", type=float, default=None)
    p.add_argument("--min-api-center-accuracy", type=float, default=None)
    args = p.parse_args()
    q_thr = args.min_query_gallery_accuracy
    c_thr = args.min_api_center_accuracy
    audit_args = vars(args).copy()
    audit_args.pop("min_query_gallery_accuracy")
    audit_args.pop("min_api_center_accuracy")
    result = audit(**audit_args)
    print(json.dumps(result, indent=2))

    failures = []
    if q_thr is not None and result["validation_query_api_description_gallery_accuracy"] < q_thr:
        failures.append(
            "ArcFace transfer query->API gallery accuracy "
            f"{result['validation_query_api_description_gallery_accuracy']:.4f} < {q_thr:.4f}"
        )
    if c_thr is not None and result["api_description_nearest_center_accuracy"] < c_thr:
        failures.append(
            "ArcFace-training API->center accuracy "
            f"{result['api_description_nearest_center_accuracy']:.4f} < {c_thr:.4f}"
        )
    if failures:
        print(
            "ArcFace geometry audit FAILED; SCF training would not be scientifically meaningful:\n  - "
            + "\n  - ".join(failures),
            file=sys.stderr,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
