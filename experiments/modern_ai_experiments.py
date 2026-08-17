#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from importlib import metadata as importlib_metadata
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from typing import Any, Mapping, Optional

import numpy as np
from omegaconf import OmegaConf

from evaluation.reproducibility import seed_everything
from evaluation.modern_ai.data import (
    load_beir_local,
    load_bfcl_relevance_files,
    load_bright_hf,
    load_crag_records,
    load_generic_tool_routing,
    load_ragtruth,
    make_evidence_deletion_protocol,
    mix_cross_domain_unknown_queries,
)
from evaluation.modern_ai.embedders import CachedTextEmbedder, HashingTextEmbedder, SentenceTransformerEmbedder
from evaluation.modern_ai.methods import PosteriorModelConfig
from evaluation.modern_ai.query_uncertainty import load_rewrites_jsonl
from evaluation.modern_ai.rag import (
    TransformersConditionalSequenceScorer,
    TransformersRAGGenerator,
    TransformersSemanticEntropyScorer,
    load_binary_judgments_jsonl,
    load_feature_cache,
    load_generation_cache,
    run_end_to_end_rag_experiment,
    run_ragtruth_experiment,
    save_feature_cache,
    save_generation_cache,
)
from evaluation.modern_ai.retrieval_experiments import (
    run_corpus_scaling_experiment,
    run_retrieval_experiment,
    run_rewrite_ablation,
    run_kappa_root_sensitivity,
    run_mc_sensitivity_experiment,
    run_posterior_hyperparameter_sensitivity,
)
from evaluation.modern_ai.synthetic import (
    synthetic_generator_features,
    synthetic_open_set_protocol,
    synthetic_ragtruth_records,
    synthetic_rewrites,
    synthetic_tool_examples,
)
from evaluation.modern_ai.tool_routing import run_tool_routing_experiment


def _plain(node: Any):
    return OmegaConf.to_container(node, resolve=True) if OmegaConf.is_config(node) else node


def _cfg_dict(cfg: Any, key: str, default=None):
    value = cfg.get(key, default)
    return dict(_plain(value) or {})



def write_environment_manifest(out: Path) -> None:
    packages = {}
    for name in ["numpy", "scipy", "scikit-learn", "torch", "omegaconf", "sentence-transformers", "transformers", "datasets"]:
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = None
    commit = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        pass
    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": commit,
    }
    (out / "environment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def make_embedder(cfg) -> Any:
    ecfg = _cfg_dict(cfg, "embedder")
    kind = str(ecfg.pop("kind", "sentence_transformer"))
    cache_dir = ecfg.pop("cache_dir", None)
    if kind == "sentence_transformer":
        base = SentenceTransformerEmbedder(**ecfg)
    elif kind == "hashing":
        base = HashingTextEmbedder(**ecfg)
    else:
        raise ValueError(f"Unknown embedder.kind={kind}")
    return CachedTextEmbedder(base, cache_dir) if cache_dir else base


def make_posterior(cfg) -> PosteriorModelConfig:
    pcfg = _cfg_dict(cfg, "posterior")
    transfer = pcfg.pop("gallery_kappa_from_summary", None)
    if transfer and pcfg.get("gallery_kappa") is None:
        source = json.loads(Path(transfer).read_text(encoding="utf-8"))
        if "fitted_gallery_kappa" not in source:
            raise KeyError(f"{transfer} does not contain fitted_gallery_kappa")
        pcfg["gallery_kappa"] = float(source["fitted_gallery_kappa"])
    return PosteriorModelConfig(**pcfg)


def load_retrieval_dataset(cfg, prefix: str = "data"):
    dc = _cfg_dict(cfg, prefix)
    loader = str(dc.get("loader", "beir"))
    if loader == "beir":
        return load_beir_local(dc["path"], split=dc.get("split", "test"), name=dc.get("name"))
    if loader == "bright":
        return load_bright_hf(
            dc["domain"], use_long_documents=bool(dc.get("use_long_documents", False)),
            dataset_name=dc.get("dataset_name", "xlangai/BRIGHT"),
        )
    raise ValueError(f"Unknown data.loader={loader}")


def make_protocol(cfg):
    ds = load_retrieval_dataset(cfg)
    pc = _cfg_dict(cfg, "protocol")
    kind = str(pc.pop("kind", "evidence_deletion"))
    if kind == "evidence_deletion":
        return make_evidence_deletion_protocol(ds, **pc)
    if kind == "cross_domain":
        unknown = load_retrieval_dataset(cfg, prefix="unknown_data")
        return mix_cross_domain_unknown_queries(ds, unknown, **pc)
    raise ValueError(f"Unknown protocol.kind={kind}")


def common_kwargs(cfg):
    qc = _cfg_dict(cfg, "query_uncertainty")
    rewrites = None
    if qc.get("rewrites_path"):
        rewrites = load_rewrites_jsonl(qc.pop("rewrites_path"))
    return {
        "rewrites": rewrites,
        "default_query_kappa": float(qc.get("default_kappa", 300.0)),
        "query_mean_mode": str(qc.get("mean_mode", "original")),
        "calibration_fraction": float(cfg.get("calibration_fraction", 0.3)),
        "seed": int(cfg.get("seed", 777)),
    }


def run_smoke(cfg, out: Path):
    embedder = make_embedder(cfg)
    posterior = make_posterior(cfg)
    protocol = synthetic_open_set_protocol(seed=int(cfg.get("seed", 777)))
    rewrites = synthetic_rewrites(protocol)
    r = run_retrieval_experiment(
        protocol, embedder, posterior, rewrites=rewrites,
        query_mean_mode="original", calibration_fraction=0.4,
        seed=int(cfg.get("seed", 777)), output_dir=out / "retrieval",
    )
    run_rewrite_ablation(
        protocol, embedder, posterior, rewrites,
        calibration_fraction=0.4, seed=int(cfg.get("seed", 777)),
        output_dir=out / "rewrite_ablation",
    )
    active = len(protocol.corpus)
    sizes = sorted(set([max(4, active // 2), max(6, int(active * 0.75)), active]))
    # Ensure sizes cannot be below the number of relevant documents.
    required = len({d for x in protocol.relevant_doc_ids for d in x})
    sizes = [max(required, s) for s in sizes]
    run_corpus_scaling_experiment(
        protocol, embedder, posterior, sizes,
        calibration_fraction=0.4, seed=int(cfg.get("seed", 777)),
        output_dir=out / "scaling",
    )
    tool = run_tool_routing_experiment(
        synthetic_tool_examples(), embedder, posterior,
        calibration_fraction=0.4, kappa_grid_size=8,
        seed=int(cfg.get("seed", 777)), output_dir=out / "tool_routing",
    )
    rag_cfg = PosteriorModelConfig(**{**posterior.__dict__, "gallery_kappa": tool["summary"]["fitted_gallery_kappa"]})
    records = synthetic_ragtruth_records()
    rag = run_ragtruth_experiment(
        records, embedder, rag_cfg,
        generator_features=synthetic_generator_features(records),
        validation_fraction=0.33, n_boot=100,
        seed=int(cfg.get("seed", 777)), output_dir=out / "ragtruth",
    )
    summary = {
        "smoke_passed": True,
        "retrieval_oser_accuracy": r["summary"]["open_set"]["oser_accuracy"],
        # Smoke data use a hashing embedder and are a software check, not a scientific benchmark.
        "tool_pipeline_ran": True,
        "tool_examples_scored": int(tool["summary"]["split"]["num_test"]),
        "rag_methods": list(rag["summary"]["metrics"].keys()),
    }
    (out / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Modern-AI experiments for HolUE/GalUE/MPRisk")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help="OmegaConf dot-list override; repeat for sweeps (e.g. embedder.model_name=...)",
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))
    mode = str(cfg.get("mode", "retrieval"))
    out = Path(str(cfg.get("output_dir", f"outputs/modern_ai/{mode}")))
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(int(cfg.get("seed", 777)))
    write_environment_manifest(out)
    (out / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    if mode == "smoke":
        result = run_smoke(cfg, out)
    elif mode in {"retrieval", "retrieval_scaling", "rewrite_ablation", "kappa_root_sensitivity", "mc_sensitivity", "posterior_sensitivity"}:
        protocol = make_protocol(cfg)
        embedder = make_embedder(cfg)
        posterior = make_posterior(cfg)
        kw = common_kwargs(cfg)
        if mode == "retrieval":
            result = run_retrieval_experiment(protocol, embedder, posterior, output_dir=out, **kw)["summary"]
        elif mode == "retrieval_scaling":
            result = run_corpus_scaling_experiment(
                protocol, embedder, posterior,
                target_sizes=list(map(int, cfg.get("target_sizes"))),
                calibration_mode=str(cfg.get("scaling_calibration_mode", "refit")),
                output_dir=out, **kw,
            )
        elif mode == "rewrite_ablation":
            if kw["rewrites"] is None:
                raise ValueError("rewrite_ablation requires query_uncertainty.rewrites_path")
            rewrites = kw.pop("rewrites")
            result = run_rewrite_ablation(
                protocol, embedder, posterior, rewrites, output_dir=out, **kw
            )
        elif mode == "kappa_root_sensitivity":
            result = run_kappa_root_sensitivity(
                protocol, embedder, posterior, output_dir=out, **kw
            )
        elif mode == "posterior_sensitivity":
            sweeps = _plain(cfg.get("sweeps", {})) or {}
            result = run_posterior_hyperparameter_sensitivity(
                protocol, embedder, posterior, sweeps=sweeps, output_dir=out, **kw
            )
        else:
            result = run_mc_sensitivity_experiment(
                protocol, embedder, posterior,
                mc_samples=list(map(int, cfg.get("mc_samples", [0, 8, 32, 128]))),
                kappa_mode=str(cfg.get("mc_kappa_mode", "fixed_m0")),
                repeats=int(cfg.get("mc_repeats", 1)),
                output_dir=out, **kw,
            )
    elif mode == "ragtruth":
        dc = _cfg_dict(cfg, "data")
        records = load_ragtruth(
            dc["response_path"], dc["source_info_path"],
            split=dc.get("split", "test"), task_type=dc.get("task_type", "QA"),
            count_implicit_true_as_hallucination=bool(dc.get("count_implicit_true_as_hallucination", False)),
        )
        embedder = make_embedder(cfg); posterior = make_posterior(cfg)
        generator_features = None
        gc = _cfg_dict(cfg, "generator")
        if gc.get("sequence_scorer_model"):
            scorer = TransformersConditionalSequenceScorer(
                gc["sequence_scorer_model"], device=gc.get("device"),
                max_length=int(gc.get("max_length", 4096)),
            )
            generator_features = scorer.score(records)
        qc = _cfg_dict(cfg, "query_uncertainty")
        rewrites = load_rewrites_jsonl(qc["rewrites_path"]) if qc.get("rewrites_path") else None
        result = run_ragtruth_experiment(
            records, embedder, posterior, generator_features=generator_features,
            target=str(cfg.get("target", "hallucination")), rewrites=rewrites,
            default_query_kappa=float(qc.get("default_kappa", 300)),
            query_mean_mode=str(qc.get("mean_mode", "original")),
            validation_fraction=float(cfg.get("calibration_fraction", .3)),
            seed=int(cfg.get("seed",777)), n_boot=int(cfg.get("n_boot",1000)),
            output_dir=out,
        )["summary"]
    elif mode == "crag":
        dc = _cfg_dict(cfg, "data")
        records = load_crag_records(dc["path"], split=dc.get("split"))
        embedder = make_embedder(cfg); posterior = make_posterior(cfg)
        gc = _cfg_dict(cfg, "generator")
        cache = gc.get("cache_path")
        gen = None
        if cache and Path(cache).exists():
            responses, generator_features = load_generation_cache(cache, records)
        else:
            if not gc.get("model_name"):
                raise ValueError("CRAG needs generator.cache_path or generator.model_name")
            gen = TransformersRAGGenerator(
                gc["model_name"], device=gc.get("device"),
                max_input_tokens=int(gc.get("max_input_tokens",4096)),
                max_new_tokens=int(gc.get("max_new_tokens",128)),
                temperature=float(gc.get("temperature",0.0)),
            )
            responses, generator_features = gen.generate(records)
            if cache: save_generation_cache(cache, records, responses, generator_features)

        # Strong free-form generation baseline: semantic entropy/self-consistency
        # from repeated samples clustered by bidirectional NLI entailment.  This is
        # intentionally optional because it is much more expensive than token UE.
        sc = dict(gc.get("semantic_entropy") or {})
        if bool(sc.get("enabled", False)):
            feature_cache = sc.get("cache_path")
            if feature_cache and Path(feature_cache).exists():
                semantic_features = load_feature_cache(feature_cache, records)
            else:
                if not gc.get("model_name"):
                    raise ValueError(
                        "Semantic entropy needs generator.model_name even when the main generation cache exists"
                    )
                if gen is None:
                    gen = TransformersRAGGenerator(
                        gc["model_name"], device=gc.get("device"),
                        max_input_tokens=int(gc.get("max_input_tokens",4096)),
                        max_new_tokens=int(gc.get("max_new_tokens",128)),
                        temperature=float(gc.get("temperature",0.0)),
                    )
                nli_model = sc.get("nli_model_name")
                if not nli_model:
                    raise ValueError("semantic_entropy.enabled requires nli_model_name")
                sem = TransformersSemanticEntropyScorer(
                    gen, nli_model, nli_device=sc.get("nli_device"),
                    num_samples=int(sc.get("num_samples",10)),
                    temperature=float(sc.get("temperature",1.0)),
                    top_p=float(sc.get("top_p",1.0)),
                    entailment_threshold=float(sc.get("entailment_threshold",0.5)),
                    entailment_label_id=(
                        int(sc["entailment_label_id"])
                        if sc.get("entailment_label_id") is not None else None
                    ),
                    max_nli_tokens=int(sc.get("max_nli_tokens",512)),
                )
                semantic_features = sem.score(records)
                if feature_cache:
                    save_feature_cache(feature_cache, records, semantic_features)
            generator_features = {**generator_features, **semantic_features}
        judgments = None
        jc = _cfg_dict(cfg, "judgments")
        if jc.get("path"):
            judgments = load_binary_judgments_jsonl(
                jc["path"], id_field=jc.get("id_field","id"), error_field=jc.get("error_field","is_error")
            )
        result = run_end_to_end_rag_experiment(
            records, embedder, posterior, responses=responses,
            generator_features=generator_features, judgments=judgments,
            allow_reference_heuristic=bool(jc.get("allow_reference_heuristic",False)),
            validation_fraction=float(cfg.get("calibration_fraction",.3)),
            seed=int(cfg.get("seed",777)), n_boot=int(cfg.get("n_boot",1000)),
            output_dir=out,
        )["summary"]
    elif mode in {"bfcl", "tool_routing"}:
        dc = _cfg_dict(cfg, "data")
        if mode == "bfcl":
            examples = load_bfcl_relevance_files(dc.get("relevance_files",[]), dc.get("irrelevance_files",[]))
        else:
            examples = load_generic_tool_routing(dc["path"])
        embedder=make_embedder(cfg); posterior=make_posterior(cfg)
        qc=_cfg_dict(cfg,"query_uncertainty")
        rewrites=load_rewrites_jsonl(qc["rewrites_path"]) if qc.get("rewrites_path") else None
        result=run_tool_routing_experiment(
            examples,embedder,posterior,rewrites=rewrites,
            default_query_kappa=float(qc.get("default_kappa",300)),
            query_mean_mode=str(qc.get("mean_mode","original")),
            calibration_fraction=float(cfg.get("calibration_fraction",.3)),
            kappa_grid_size=int(cfg.get("kappa_grid_size",32)), seed=int(cfg.get("seed",777)),
            output_dir=out,
        )["summary"]
    else:
        raise ValueError(f"Unknown mode={mode}")

    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
