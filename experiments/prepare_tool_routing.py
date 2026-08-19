#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.modern_ai.tool_datasets import (
    build_toolbench_class_disjoint_protocol,
    clone_bfcl,
    discover_bfcl_routing_files,
    download_toolbench_data,
    toolbench_query_count_statistics,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Download and prepare tool-routing research datasets")
    sub = p.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download-toolbench", help="Download the official ToolBench release")
    dl.add_argument("--output-root", default="external/ToolBench")
    dl.add_argument("--force", action="store_true")

    stats = sub.add_parser(
        "stats-toolbench",
        help="Report per-API query density before choosing a leakage-safe ArcFace/SCF split",
    )
    stats.add_argument("--g1-query-path", default="external/ToolBench/data/instruction/G1_query.json")
    stats.add_argument("--output", default="outputs/tool_routing/toolbench_g1_query_density.json")
    stats.add_argument(
        "--thresholds",
        default="3,4,5,6,8,10,12,16,20,25,32,50",
        help="Comma-separated minimum-query thresholds to count",
    )

    build = sub.add_parser("build-toolbench", help="Build class-disjoint ArcFace/SCF + OSR splits from G1")
    build.add_argument("--g1-query-path", default="external/ToolBench/data/instruction/G1_query.json")
    build.add_argument("--output-dir", default="datasets/tool_routing/toolbench_g1")
    build.add_argument("--num-known-tools", type=int, default=1024)
    build.add_argument("--num-unknown-tools", type=int, default=256)
    build.add_argument("--min-queries-per-tool", type=int, default=3)
    build.add_argument("--train-fraction", type=float, default=0.70)
    build.add_argument("--val-fraction", type=float, default=0.15)
    build.add_argument("--unknown-calibration-fraction", type=float, default=0.50)
    build.add_argument("--seed", type=int, default=777)
    build.add_argument("--no-tool-documents-in-train", action="store_true")

    bf = sub.add_parser("download-bfcl", help="Clone official Gorilla/BFCL and discover routing categories")
    bf.add_argument("--output-root", default="external/gorilla")
    bf.add_argument("--manifest", default="datasets/tool_routing/bfcl_manifest.json")
    bf.add_argument("--ref", default="main", help="Git ref. Exact resolved commit is stored in the manifest.")
    bf.add_argument("--force", action="store_true")

    allp = sub.add_parser("all", help="Download ToolBench/BFCL and build ToolBench OSR protocol")
    allp.add_argument("--toolbench-root", default="external/ToolBench")
    allp.add_argument("--prepared-dir", default="datasets/tool_routing/toolbench_g1")
    allp.add_argument("--bfcl-root", default="external/gorilla")
    allp.add_argument("--bfcl-manifest", default="datasets/tool_routing/bfcl_manifest.json")
    allp.add_argument("--bfcl-ref", default="main")
    allp.add_argument("--num-known-tools", type=int, default=1024)
    allp.add_argument("--num-unknown-tools", type=int, default=256)
    allp.add_argument("--min-queries-per-tool", type=int, default=3)
    allp.add_argument("--unknown-calibration-fraction", type=float, default=0.50)
    allp.add_argument("--seed", type=int, default=777)

    args = p.parse_args()
    if args.command == "download-toolbench":
        path = download_toolbench_data(args.output_root, force=args.force)
        print(json.dumps({"g1_query_path": str(path)}, indent=2))
    elif args.command == "stats-toolbench":
        thresholds = tuple(int(x.strip()) for x in str(args.thresholds).split(",") if x.strip())
        result = toolbench_query_count_statistics(args.g1_query_path, thresholds=thresholds)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    elif args.command == "build-toolbench":
        result = build_toolbench_class_disjoint_protocol(
            args.g1_query_path, args.output_dir,
            num_known_tools=args.num_known_tools,
            num_unknown_tools=args.num_unknown_tools,
            min_queries_per_tool=args.min_queries_per_tool,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            add_tool_documents_to_train=not args.no_tool_documents_in_train,
            unknown_calibration_fraction=args.unknown_calibration_fraction,
            seed=args.seed,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "download-bfcl":
        source = clone_bfcl(args.output_root, ref=args.ref, force=args.force)
        discovered = discover_bfcl_routing_files(args.output_root)
        manifest = {"source": source, **discovered}
        path = Path(args.manifest); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))
    else:
        g1 = download_toolbench_data(args.toolbench_root)
        protocol = build_toolbench_class_disjoint_protocol(
            g1, args.prepared_dir, num_known_tools=args.num_known_tools,
            num_unknown_tools=args.num_unknown_tools,
            min_queries_per_tool=args.min_queries_per_tool,
            unknown_calibration_fraction=args.unknown_calibration_fraction, seed=args.seed,
        )
        source = clone_bfcl(args.bfcl_root, ref=args.bfcl_ref)
        discovered = discover_bfcl_routing_files(args.bfcl_root)
        manifest = {"source": source, **discovered}
        mp = Path(args.bfcl_manifest); mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps({"toolbench": protocol, "bfcl": manifest}, indent=2))


if __name__ == "__main__":
    main()
