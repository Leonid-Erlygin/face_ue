#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from evaluation.modern_ai.embedders import RepoSCFTextEmbedder, l2_normalize
from evaluation.modern_ai.scf_diagnostics import summarize_scf_split


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _prototype_summary(kappa, true_cos, correct, *, embedding_dim):
    raw = summarize_scf_split(
        kappa, true_cos, correct,
        embedding_dim=embedding_dim,
    )
    return {
        "count": raw["count"],
        "kappa": raw["kappa"],
        "prototype": raw["center"],
        "scf_stationarity": raw["scf_stationarity"],
        "kappa_by_prototype_outcome": raw["kappa_by_center_outcome"],
    }


def _evaluate_prototype(rows, *, text_key, embedder, gallery, split_name):
    gallery = sorted(gallery, key=lambda x: int(x["index"]))
    texts = [str(x[text_key]) for x in rows]
    labels = np.asarray([int(x["label"]) for x in rows], dtype=int)
    ids = [str(x.get("id", i)) for i, x in enumerate(rows)]
    mu, kappa = embedder.encode_queries_with_kappa(texts)
    mu = l2_normalize(mu)
    gallery_emb = embedder.encode_documents([str(x["text"]) for x in gallery])
    sim = mu @ gallery_emb.T
    true = sim[np.arange(len(rows)), labels]
    pred = np.argmax(sim, axis=1)
    correct = pred == labels
    summary = _prototype_summary(kappa, true, correct, embedding_dim=mu.shape[1])
    per = []
    for i in range(len(rows)):
        kval = float(np.asarray(kappa[i]).reshape(-1)[0])
        per.append({
            "split": split_name,
            "id": ids[i],
            "label": int(labels[i]),
            "kappa": kval,
            "log_kappa": float(np.log(kval)),
            "true_api_description_cosine": float(true[i]),
            "api_description_correct": bool(correct[i]),
        })
    return summary, per


def _evaluate_legacy(rows, *, text_key, embedder, centers, gallery_emb, split_name):
    texts = [str(x[text_key]) for x in rows]
    labels = np.asarray([int(x["label"]) for x in rows], dtype=int)
    ids = [str(x.get("id", i)) for i, x in enumerate(rows)]
    mu, kappa = embedder.encode_queries_with_kappa(texts)
    mu = l2_normalize(mu)
    c_sim = mu @ centers.T
    g_sim = mu @ gallery_emb.T
    true_c = c_sim[np.arange(len(rows)), labels]
    true_g = g_sim[np.arange(len(rows)), labels]
    c_pred = np.argmax(c_sim, axis=1)
    g_pred = np.argmax(g_sim, axis=1)
    summary = summarize_scf_split(
        kappa, true_c, c_pred == labels, true_g, g_pred == labels,
        embedding_dim=mu.shape[1],
    )
    per = []
    for i in range(len(rows)):
        kval = float(np.asarray(kappa[i]).reshape(-1)[0])
        per.append({
            "split": split_name, "id": ids[i], "label": int(labels[i]),
            "kappa": kval, "log_kappa": float(np.log(kval)),
            "true_center_cosine": float(true_c[i]),
            "center_correct": bool(c_pred[i] == labels[i]),
            "true_api_description_cosine": float(true_g[i]),
            "api_description_correct": bool(g_pred[i] == labels[i]),
        })
    return summary, per


def main():
    ap = argparse.ArgumentParser(description="Validation-only audit of ToolBench SCF concentration")
    ap.add_argument("--prepared-dir", default="datasets/tool_routing/toolbench_g1")
    ap.add_argument("--backbone-path", default="model_weights/backbone/bert_toolbench_arcface/backbone.pth")
    ap.add_argument("--softmax-weights-path", default="model_weights/backbone/bert_toolbench_arcface/softmax_weight.pt")
    ap.add_argument("--scf-checkpoint-path", default="outputs/tool_routing/scf/last.ckpt")
    ap.add_argument("--output-dir", default="outputs/tool_routing/scf_concentration_audit")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    root = Path(args.prepared_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    stage_v2 = str(manifest.get("protocol", "")).endswith("stage_class_disjoint_v2")
    train = _read_jsonl(root / "train_scf.jsonl")
    val = [x for x in _read_jsonl(root / "val.jsonl") if bool(x.get("known"))]
    if not train or not val:
        raise RuntimeError("Prepared ToolBench SCF-train/calibration-known split is empty")

    embedder = RepoSCFTextEmbedder(
        backbone_path=args.backbone_path,
        scf_checkpoint_path=args.scf_checkpoint_path,
        batch_size=args.batch_size,
    )

    if stage_v2:
        scf_gallery = _read_jsonl(root / "scf_gallery.jsonl")
        cal_gallery = _read_jsonl(root / "calibration_gallery.jsonl")
        tr_summary, tr_rows = _evaluate_prototype(
            train, text_key="text", embedder=embedder,
            gallery=scf_gallery, split_name="scf_train_classes",
        )
        va_summary, va_rows = _evaluate_prototype(
            val, text_key="query", embedder=embedder,
            gallery=cal_gallery, split_name="calibration_unseen_classes",
        )
        generalization = {
            "stationarity_mae_gap_calibration_minus_train": float(
                va_summary["scf_stationarity"]["mean_absolute_residual"]
                - tr_summary["scf_stationarity"]["mean_absolute_residual"]
            ),
            "routing_error_auroc_gap_calibration_minus_train": float(
                va_summary["prototype"]["error_auroc_negative_log_kappa"]
                - tr_summary["prototype"]["error_auroc_negative_log_kappa"]
            ),
        }
    else:
        gallery = sorted(_read_jsonl(root / "gallery.jsonl"), key=lambda x: int(x["index"]))
        centers = torch.load(args.softmax_weights_path, map_location="cpu", weights_only=False)
        centers = l2_normalize(np.asarray(centers.detach().cpu().numpy(), dtype=np.float64))
        if centers.shape[0] != len(gallery):
            raise RuntimeError(f"Center/gallery class mismatch: {centers.shape[0]} vs {len(gallery)}")
        gallery_emb = embedder.encode_documents([x["text"] for x in gallery])
        tr_summary, tr_rows = _evaluate_legacy(
            train, text_key="text", embedder=embedder, centers=centers,
            gallery_emb=gallery_emb, split_name="train",
        )
        va_summary, va_rows = _evaluate_legacy(
            val, text_key="query", embedder=embedder, centers=centers,
            gallery_emb=gallery_emb, split_name="known_validation",
        )
        generalization = {
            "stationarity_mae_gap_val_minus_train": float(
                va_summary["scf_stationarity"]["mean_absolute_residual"]
                - tr_summary["scf_stationarity"]["mean_absolute_residual"]
            ),
            "center_error_auroc_gap_val_minus_train": float(
                va_summary["center"]["error_auroc_negative_log_kappa"]
                - tr_summary["center"]["error_auroc_negative_log_kappa"]
            ),
        }

    summary = {
        "experiment": "tool_routing_scf_concentration_audit",
        "protocol": str(manifest.get("protocol", "legacy")),
        "uses_final_test_split": False,
        "train": tr_summary,
        "known_validation": va_summary,
        "generalization": generalization,
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    rows = tr_rows + va_rows
    with (out / "per_query.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}))
        w.writeheader(); w.writerows(rows)
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
