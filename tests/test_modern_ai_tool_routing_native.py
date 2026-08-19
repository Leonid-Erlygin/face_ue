import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from evaluation.modern_ai.embedders import BaseTextEmbedder, l2_normalize
from evaluation.modern_ai.methods import PosteriorModelConfig
from evaluation.modern_ai.tool_datasets import (
    bfcl_routing_examples_from_manifest,
    build_toolbench_class_disjoint_protocol,
    discover_bfcl_routing_files,
    toolbench_query_count_statistics,
)
from evaluation.modern_ai.tool_routing import run_tool_routing_experiment
from evaluation.modern_ai.types import ToolRoutingExample
from experiments.export_tool_routing_arcface import export_arcface
from training.dataset_classes.tool_routing import ToolRoutingClassificationDataset


def _write_g1(path: Path, num_tools=8, queries_per_tool=5):
    rows = []
    qid = 0
    for t in range(num_tools):
        api = {
            "category_name": "math",
            "tool_name": f"tool_{t}",
            "api_name": f"api_{t}",
            "api_description": f"Does operation number {t}",
            "required_parameters": [{"name": "value", "type": "integer"}],
            "optional_parameters": [],
            "method": "GET",
        }
        distractor = {
            "category_name": "math",
            "tool_name": f"distractor_{t}",
            "api_name": f"other_{t}",
            "api_description": "irrelevant",
        }
        for j in range(queries_per_tool):
            rows.append({
                "query_id": qid,
                "query": f"please run operation {t} example {j}",
                "api_list": [api, distractor],
                "relevant APIs": [[api["tool_name"], api["api_name"]]],
            })
            qid += 1
    path.write_text(json.dumps(rows), encoding="utf-8")


def _read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def test_toolbench_protocol_is_class_disjoint_and_separates_arcface_scf(tmp_path):
    g1 = tmp_path / "G1_query.json"
    out = tmp_path / "prepared"
    _write_g1(g1)
    manifest = build_toolbench_class_disjoint_protocol(
        g1, out, num_known_tools=4, num_unknown_tools=4,
        min_queries_per_tool=3, seed=11,
    )

    arc = _read_jsonl(out / "train_arcface.jsonl")
    scf = _read_jsonl(out / "train_scf.jsonl")
    val = _read_jsonl(out / "val.jsonl")
    test = _read_jsonl(out / "test.jsonl")
    gallery = _read_jsonl(out / "gallery.jsonl")

    assert len(gallery) == 4
    assert any(x["sample_type"] == "tool_document" for x in arc)
    assert all(x["sample_type"] == "query" for x in scf)
    assert len(arc) == len(scf) + 4
    assert (out / "train.jsonl").read_bytes() == (out / "train_arcface.jsonl").read_bytes()

    train_tools = {x["tool_id"] for x in arc}
    val_unknown = {x["tool_id"] for x in val if not x["known"]}
    test_unknown = {x["tool_id"] for x in test if not x["known"]}
    assert train_tools.isdisjoint(val_unknown)
    assert train_tools.isdisjoint(test_unknown)
    assert val_unknown.isdisjoint(test_unknown)
    assert manifest["num_unknown_calibration_tools"] == 2
    assert manifest["num_unknown_test_tools"] == 2
    assert len(manifest["source_g1_sha256"]) == 64

    ds_arc = ToolRoutingClassificationDataset(out / "train_arcface.jsonl")
    ds_scf = ToolRoutingClassificationDataset(out / "train_scf.jsonl")
    assert ds_arc.num_classes == 4 == ds_scf.num_classes


def test_toolbench_query_density_statistics_are_data_only_and_correct(tmp_path):
    g1 = tmp_path / "G1_query.json"
    rows = []
    qid = 0
    for t, n in enumerate([3, 5, 8, 12]):
        api = {
            "category_name": "math",
            "tool_name": f"tool_{t}",
            "api_name": f"api_{t}",
            "api_description": f"Does operation {t}",
        }
        for j in range(n):
            rows.append({
                "query_id": qid,
                "query": f"operation {t} query {j}",
                "api_list": [api],
                "relevant APIs": [[api["tool_name"], api["api_name"]]],
            })
            qid += 1
    g1.write_text(json.dumps(rows), encoding="utf-8")
    stats = toolbench_query_count_statistics(g1, thresholds=(3, 5, 8, 10, 12))
    assert stats["num_unique_single_api_classes"] == 4
    assert stats["num_usable_single_api_queries"] == 28
    assert stats["num_api_classes_with_at_least_n_queries"] == {
        "3": 4, "5": 3, "8": 2, "10": 1, "12": 1,
    }
    assert stats["protocol_guidance"]["recommended_minimum_queries_per_known_api"] == 8


def _jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")


def test_bfcl_manifest_maps_official_ground_truth_to_candidate_index(tmp_path):
    data = tmp_path / "gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"
    multiple = data / "BFCL_v4_multiple.json"
    irrelevance = data / "BFCL_v4_irrelevance.json"
    answer = data / "possible_answer/BFCL_v4_multiple.json"
    _jsonl(multiple, [{
        "id": "multiple_0",
        "question": [[{"role": "user", "content": "calculate a triangle"}]],
        "function": [
            {"name": "circle.get", "description": "circle", "parameters": {"type": "dict", "properties": {}}},
            {"name": "triangle.get", "description": "triangle", "parameters": {"type": "dict", "properties": {"a": {"type": "number"}}, "required": ["a"]}},
        ],
    }])
    _jsonl(answer, [{"id": "multiple_0", "ground_truth": [{"triangle.get": {"a": [3]}}]}])
    _jsonl(irrelevance, [{
        "id": "irrelevance_0",
        "question": [[{"role": "user", "content": "send an email"}]],
        "function": [{"name": "weather.get", "description": "weather", "parameters": {"type": "dict", "properties": {}}}],
    }])

    manifest = discover_bfcl_routing_files(tmp_path / "gorilla")
    examples = bfcl_routing_examples_from_manifest(manifest)
    known = next(x for x in examples if x.example_id == "multiple_0")
    unknown = next(x for x in examples if x.example_id == "irrelevance_0")
    assert known.known and known.relevant_tool_indices == (1,)
    assert "Function: triangle.get" in known.tools[1]
    assert not unknown.known and unknown.relevant_tool_indices == ()
    assert manifest["answer_files"]["known_multiple"][str(multiple)] == str(answer)


def test_arcface_exporter_exports_native_backbone_and_centers(tmp_path):
    ckpt = tmp_path / "model.ckpt"
    torch.save({"state_dict": {
        "backbone.proj.weight": torch.ones(3, 4),
        "backbone.final_proj.bias": torch.zeros(3),
        "softmax_weights": torch.arange(15, dtype=torch.float32).reshape(5, 3),
    }}, ckpt)
    out = tmp_path / "export"
    manifest = export_arcface(str(ckpt), str(out))
    state = torch.load(out / "backbone.pth", map_location="cpu", weights_only=False)
    centers = torch.load(out / "softmax_weight.pt", map_location="cpu", weights_only=False)
    assert set(state) == {"proj.weight", "final_proj.bias"}
    assert centers.shape == (5, 3)
    assert manifest["num_classes"] == 5


class _LearnedKappaEmbedder(BaseTextEmbedder):
    supports_query_kappa = True
    model_name = "unit-test-scf"

    @staticmethod
    def _vec(text):
        text = str(text).lower()
        if "alpha" in text:
            return [1.0, 0.0, 0.0, 0.0]
        if "beta" in text:
            return [0.0, 1.0, 0.0, 0.0]
        if "unknown-a" in text:
            return [-1.0, 0.0, 0.0, 0.0]
        if "unknown-b" in text:
            return [0.0, -1.0, 0.0, 0.0]
        return [0.0, 0.0, 1.0, 0.0]

    def encode_queries(self, texts):
        return l2_normalize(np.asarray([self._vec(x) for x in texts], dtype=float))

    def encode_documents(self, texts):
        return self.encode_queries(texts)

    def encode_queries_with_kappa(self, texts):
        emb = self.encode_queries(texts)
        # Intentionally non-constant to prove the learned-kappa path is carried
        # through as an uncertainty baseline.
        k = np.asarray([[100.0 + i] for i in range(len(texts))])
        return emb, k


def test_tool_routing_uses_prescribed_calibration_and_learned_kappa(tmp_path):
    gallery = ("alpha tool", "beta tool")
    cal = [
        ToolRoutingExample("ca1", "alpha request", gallery, True, (0,), {"tool_id":"a"}),
        ToolRoutingExample("ca2", "beta request", gallery, True, (1,), {"tool_id":"b"}),
        ToolRoutingExample("cu1", "unknown-a request", gallery, False, (), {"tool_id":"u1"}),
        ToolRoutingExample("cu2", "unknown-b request", gallery, False, (), {"tool_id":"u2"}),
    ]
    test = [
        ToolRoutingExample("ta1", "alpha another", gallery, True, (0,), {"tool_id":"a"}),
        ToolRoutingExample("tb1", "beta another", gallery, True, (1,), {"tool_id":"b"}),
        ToolRoutingExample("tu1", "unknown-a final", gallery, False, (), {"tool_id":"u3"}),
        ToolRoutingExample("tu2", "unknown-b final", gallery, False, (), {"tool_id":"u4"}),
    ]
    cfg = PosteriorModelConfig(
        gallery_kappa=30.0, beta=0.5, predict_T=20.0,
        gallery_prior="power", mc_samples=0,
    )
    result = run_tool_routing_experiment(
        test, _LearnedKappaEmbedder(), cfg,
        calibration_examples=cal, kappa_source="embedder",
        output_dir=tmp_path / "run",
    )
    summary = result["summary"]
    assert summary["split"]["strategy"] == "prescribed_validation_test"
    assert summary["split"]["num_calibration"] == 4
    assert summary["split"]["num_test"] == 4
    assert "negative_query_kappa" in result["test_scores"]
    assert summary["decision_metrics"]["num_tool_id_evaluable"] == 2
    assert (tmp_path / "run/per_example.csv").exists()


def test_tool_routing_configs_reference_separate_training_protocols():
    arc = OmegaConf.load("configs/uncertainty_models/text_model_toolbench_arcface.yaml")
    scf = OmegaConf.load("configs/uncertainty_models/text_model_toolbench_scf.yaml")
    osr = OmegaConf.load("configs/modern_ai/toolbench_tool_routing_osr.yaml")
    assert str(arc.data.train_path).endswith("train_arcface.jsonl")
    assert str(scf.data.train_path).endswith("train_scf.jsonl")
    assert str(scf.model.scheduler_params.scheduler) == "OneCycleLR"
    assert str(scf.model.scheduler_params.params.total_steps) == "auto"
    assert int(scf.trainer.max_epochs) >= 20
    assert int(scf.data.batch_size) <= 32
    assert str(osr.query_uncertainty.source) == "embedder"
    assert str(osr.posterior.gallery_kappa_strategy) == "boundary_roots_calibrated"


def test_internal_bfcl_style_split_keeps_candidate_galleries_disjoint(tmp_path):
    # Two distinct galleries in each known/unknown stratum are the minimum for a
    # leakage-safe internal calibration/test split.
    examples = [
        ToolRoutingExample("k1", "alpha request", ("alpha tool", "beta tool"), True, (0,), {}),
        ToolRoutingExample("k2", "alpha request again", ("alpha tool", "beta tool"), True, (0,), {}),
        ToolRoutingExample("k3", "beta request", ("beta tool", "gamma tool"), True, (0,), {}),
        ToolRoutingExample("k4", "beta request again", ("beta tool", "gamma tool"), True, (0,), {}),
        ToolRoutingExample("u1", "unknown-a request", ("alpha tool",), False, (), {}),
        ToolRoutingExample("u2", "unknown-a another", ("alpha tool",), False, (), {}),
        ToolRoutingExample("u3", "unknown-b request", ("beta tool",), False, (), {}),
        ToolRoutingExample("u4", "unknown-b another", ("beta tool",), False, (), {}),
    ]
    cfg = PosteriorModelConfig(
        gallery_kappa=30.0, beta=0.5, predict_T=20.0,
        gallery_prior="power", mc_samples=0,
    )
    result = run_tool_routing_experiment(
        examples, _LearnedKappaEmbedder(), cfg,
        kappa_source="embedder", calibration_fraction=0.5,
        output_dir=tmp_path / "bfcl_style",
    )
    split = result["summary"]["split"]
    assert split["strategy"] == "internal_gallery_group_stratified"
    assert set(split["calibration_gallery_hashes"]).isdisjoint(split["test_gallery_hashes"])
    for stats in split["gallery_group_counts"].values():
        assert stats["calibration"] >= 1
        assert stats["test"] >= 1


def test_arcface_class_balancing_equalizes_total_class_sampling_mass(tmp_path):
    rows = []
    for i in range(2):
        rows.append({"text": f"minority {i}", "label": 0})
    for i in range(8):
        rows.append({"text": f"majority {i}", "label": 1})
    path = tmp_path / "train.jsonl"
    _jsonl(path, rows)
    ds = ToolRoutingClassificationDataset(path)
    labels = np.asarray([int(row["label"]) for row in ds.rows])
    weights = ds.sample_weights.detach().cpu().numpy()
    totals = {
        label: float(np.sum(weights[labels == label]))
        for label in sorted(set(labels.tolist()))
    }
    assert np.isclose(totals[0], totals[1])


def test_bfcl_config_transfers_toolbench_gallery_concentration():
    cfg = OmegaConf.load("configs/modern_ai/bfcl.yaml")
    assert str(cfg.posterior.gallery_kappa_from_summary).endswith(
        "outputs/modern_ai/toolbench_g1_open_set_routing/summary.json"
    )
