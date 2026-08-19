#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def export_arcface(checkpoint_path: str, output_dir: str) -> dict:
    ckpt_path = Path(checkpoint_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    backbone = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}
    if not backbone:
        raise KeyError(f"No backbone.* parameters found in {ckpt_path}")
    if "softmax_weights" not in state:
        raise KeyError(f"No softmax_weights parameter found in {ckpt_path}")
    weights = state["softmax_weights"].detach().cpu()
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save(backbone, out / "backbone.pth")
    torch.save(weights, out / "softmax_weight.pt")
    manifest = {
        "source_checkpoint": str(ckpt_path),
        "num_classes": int(weights.shape[0]),
        "embedding_dim": int(weights.shape[1]),
        "num_backbone_tensors": len(backbone),
        "backbone_path": str(out / "backbone.pth"),
        "softmax_weights_path": str(out / "softmax_weight.pt"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    p = argparse.ArgumentParser(description="Export BERT backbone + ArcFace centers for SCF training")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", default="model_weights/backbone/bert_toolbench_arcface")
    args = p.parse_args()
    print(json.dumps(export_arcface(args.checkpoint, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
