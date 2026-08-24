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
    build_toolbench_stage_disjoint_protocol,
    discover_bfcl_routing_files,
    load_toolbench_oser_examples,
    toolbench_query_count_statistics,
)
from evaluation.modern_ai.tool_routing import fit_query_kappa_scale, run_tool_routing_experiment
from evaluation.modern_ai.types import ToolRoutingExample
from experiments.export_tool_routing_arcface import export_arcface
from evaluation.modern_ai.scf_diagnostics import vmf_mean_resultant
from training.dataset_classes.tool_routing import (
    ToolRoutingClassificationDataset, ToolRoutingPrototypeDataset,
)


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
    assert str(scf.data._target_).endswith("ToolRoutingPrototypeDataModule")
    assert bool(scf.model.use_batch_targets) is True
    assert "softmax_weights" not in scf.model
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



def test_stage_disjoint_toolbench_protocol_uses_disjoint_api_classes(tmp_path):
    g1 = tmp_path / "G1_query.json"
    out = tmp_path / "prepared_v2"
    _write_g1(g1, num_tools=24, queries_per_tool=3)
    manifest = build_toolbench_stage_disjoint_protocol(
        g1, out,
        num_arcface_tools=6,
        num_scf_tools=5,
        num_calibration_known_tools=4,
        num_test_known_tools=4,
        num_calibration_unknown_tools=2,
        num_test_unknown_tools=2,
        min_queries_per_tool=3,
        seed=19,
    )
    assert manifest["protocol"] == "toolbench_g1_stage_class_disjoint_v2"
    assert all(v == 0 for v in manifest["pairwise_stage_overlap_counts"].values())
    assert manifest["uses_final_test_for_training_or_calibration"] is False

    arc = _read_jsonl(out / "train_arcface.jsonl")
    scf = _read_jsonl(out / "train_scf.jsonl")
    val = _read_jsonl(out / "val.jsonl")
    test = _read_jsonl(out / "test.jsonl")
    arc_tools = {x["tool_id"] for x in arc}
    scf_tools = {x["tool_id"] for x in scf}
    cal_known = {x["tool_id"] for x in val if x["known"]}
    cal_unknown = {x["tool_id"] for x in val if not x["known"]}
    test_known = {x["tool_id"] for x in test if x["known"]}
    test_unknown = {x["tool_id"] for x in test if not x["known"]}
    sets = [arc_tools, scf_tools, cal_known, cal_unknown, test_known, test_unknown]
    for i, a in enumerate(sets):
        for b in sets[i+1:]:
            assert a.isdisjoint(b)

    assert all(x.get("target_text") for x in scf)
    ds = ToolRoutingPrototypeDataset(out / "train_scf.jsonl")
    assert len(ds) == 5 * 3
    assert len(_read_jsonl(out / "calibration_gallery.jsonl")) == 4
    assert len(_read_jsonl(out / "test_gallery.jsonl")) == 4

    cal_examples = load_toolbench_oser_examples(out, "val")
    test_examples = load_toolbench_oser_examples(out, "test")
    assert len(cal_examples[0].tools) == 4
    assert len(test_examples[0].tools) == 4
    assert cal_examples[0].tools != test_examples[0].tools


def test_fixed_k_tool_routing_can_transfer_between_disjoint_galleries(tmp_path):
    cal_gallery = ("alpha tool", "beta tool")
    test_gallery = ("gamma tool", "delta tool")
    cal = [
        ToolRoutingExample("ca", "alpha request", cal_gallery, True, (0,), {"tool_id": "ca"}),
        ToolRoutingExample("cb", "beta request", cal_gallery, True, (1,), {"tool_id": "cb"}),
        ToolRoutingExample("cu1", "unknown-a", cal_gallery, False, (), {"tool_id": "u1"}),
        ToolRoutingExample("cu2", "unknown-b", cal_gallery, False, (), {"tool_id": "u2"}),
    ]
    test = [
        ToolRoutingExample("tg", "gamma request", test_gallery, True, (0,), {"tool_id": "tg"}),
        ToolRoutingExample("td", "delta request", test_gallery, True, (1,), {"tool_id": "td"}),
        ToolRoutingExample("tu1", "unknown-a", test_gallery, False, (), {"tool_id": "u3"}),
        ToolRoutingExample("tu2", "unknown-b", test_gallery, False, (), {"tool_id": "u4"}),
    ]

    class E(_LearnedKappaEmbedder):
        @staticmethod
        def _vec(text):
            text = str(text).lower()
            if "alpha" in text or "gamma" in text:
                return [1.0, 0.0, 0.0, 0.0]
            if "beta" in text or "delta" in text:
                return [0.0, 1.0, 0.0, 0.0]
            if "unknown-a" in text:
                return [-1.0, 0.0, 0.0, 0.0]
            if "unknown-b" in text:
                return [0.0, -1.0, 0.0, 0.0]
            return [0.0, 0.0, 1.0, 0.0]

    cfg = PosteriorModelConfig(
        gallery_kappa=30.0, beta=0.5, predict_T=20.0,
        gallery_prior="power", mc_samples=0,
    )
    result = run_tool_routing_experiment(
        test, E(), cfg, calibration_examples=cal,
        kappa_source="embedder", output_dir=tmp_path / "fixed_transfer",
    )
    summary = result["summary"]
    assert summary["fixed_gallery"] is True
    assert summary["same_gallery_identities_in_calibration_and_test"] is False
    assert summary["kappa_calibration"]["same_gallery_identities_in_calibration_and_test"] is False


def test_query_kappa_scale_calibration_recovers_global_vmf_scale():
    # Construct known examples whose target cosine is exactly the vMF mean
    # resultant at 2.5x the raw kappa. The calibration should recover that
    # multiplicative scale without any unknown or test examples.
    raw_k = np.asarray([[1.5], [3.0], [6.0], [12.0]], dtype=float)
    true_scale = 2.5
    target_cos = vmf_mean_resultant(raw_k.reshape(-1) * true_scale, d=4)
    gallery = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    queries = np.asarray([
        [c, np.sqrt(max(1.0 - c*c, 0.0)), 0.0, 0.0] for c in target_cos
    ])
    examples = [
        ToolRoutingExample(f"k{i}", f"q{i}", ("target", "other"), True, (0,), {})
        for i in range(len(raw_k))
    ]
    scale, meta = fit_query_kappa_scale(
        examples, queries, raw_k, [gallery] * len(examples),
        min_scale=0.1, max_scale=10.0,
    )
    assert np.isclose(scale, true_scale, rtol=2e-3)
    assert meta["uses_unknown_queries"] is False
    assert meta["calibrated_stationarity_mae"] < meta["raw_stationarity_mae"]
    assert meta["objective_nll_calibrated"] < meta["objective_nll_raw"]


def test_toolbench_osr_config_enables_validation_only_query_kappa_calibration():
    cfg = OmegaConf.load("configs/modern_ai/toolbench_tool_routing_osr.yaml")
    assert str(cfg.query_uncertainty.concentration_calibration.strategy) == "vmf_nll_scale"
    assert float(cfg.query_uncertainty.concentration_calibration.min_scale) > 0
    assert float(cfg.query_uncertainty.concentration_calibration.max_scale) > 1
