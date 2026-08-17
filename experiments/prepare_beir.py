from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf

from evaluation.modern_ai.beir_data import ensure_beir_dataset


def _iter_beir_sections(cfg: Mapping) -> Iterable[tuple[str, Mapping]]:
    for key in ("data", "unknown_data"):
        section = cfg.get(key)
        if section is not None and str(section.get("loader", "")).lower() == "beir":
            yield key, section


def _prepare_from_config(config_path: str | Path, force: bool = False) -> None:
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    found = False
    for section_name, section in _iter_beir_sections(cfg):
        found = True
        path = Path(str(section["path"]))
        dataset = str(section.get("dataset") or section.get("name") or path.name)
        split = str(section.get("split", "test"))
        print(f"Preparing {section_name}: BEIR/{dataset} -> {path}")
        ensure_beir_dataset(dataset, path, split=split, force=force)
    if not found:
        print(f"No BEIR data sections found in {config_path}; nothing to prepare.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/validate BEIR datasets used by modern-AI configs.")
    parser.add_argument("--config", type=str, help="YAML config; prepares data and unknown_data BEIR sections.")
    parser.add_argument("--dataset", type=str, help="BEIR dataset name, e.g. scifact or fiqa.")
    parser.add_argument("--path", type=str, help="Destination directory for --dataset mode.")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--url", type=str, default=None, help="Override official BEIR download URL.")
    parser.add_argument("--md5", type=str, default=None, help="Override/define expected MD5 checksum.")
    parser.add_argument("--force", action="store_true", help="Replace a partial target directory.")
    args = parser.parse_args()

    if args.config:
        if args.dataset or args.path:
            parser.error("Use either --config or --dataset/--path, not both.")
        _prepare_from_config(args.config, force=args.force)
        return

    if not args.dataset or not args.path:
        parser.error("Provide --config, or both --dataset and --path.")
    ensure_beir_dataset(
        args.dataset,
        args.path,
        split=args.split,
        url=args.url,
        md5=args.md5,
        force=args.force,
    )


if __name__ == "__main__":
    main()
