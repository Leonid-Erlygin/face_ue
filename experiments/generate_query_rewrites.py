#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf

from evaluation.modern_ai.query_uncertainty import TransformersRewriteGenerator
from experiments.modern_ai_experiments import make_protocol


def main():
    p = argparse.ArgumentParser(description="Generate paraphrase ensembles for HolUE query concentration")
    p.add_argument("--config", required=True, help="Retrieval config defining data/protocol")
    p.add_argument("--model", required=True, help="Local/Hugging Face causal LM")
    p.add_argument("--output", required=True)
    p.add_argument("--num-rewrites", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    protocol = make_protocol(cfg)
    generator = TransformersRewriteGenerator(
        args.model, num_rewrites=args.num_rewrites,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, device=args.device,
    )
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if out.exists():
        with out.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): existing.add(str(json.loads(line)["id"]))
    with out.open("a", encoding="utf-8") as f:
        for qid, query in zip(protocol.query_ids, protocol.queries):
            if qid in existing: continue
            rewrites = generator(query)
            f.write(json.dumps({"id": qid, "query": query, "rewrites": list(rewrites)}, ensure_ascii=False) + "\n")
            f.flush()
            print(qid, len(rewrites))


if __name__ == "__main__":
    main()
