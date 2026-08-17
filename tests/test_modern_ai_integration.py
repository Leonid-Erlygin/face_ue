from evaluation.modern_ai.embedders import HashingTextEmbedder
from evaluation.modern_ai.methods import PosteriorModelConfig
from evaluation.modern_ai.rag import run_ragtruth_experiment
from evaluation.modern_ai.retrieval_experiments import run_retrieval_experiment
from evaluation.modern_ai.synthetic import (
    synthetic_generator_features,
    synthetic_open_set_protocol,
    synthetic_ragtruth_records,
    synthetic_tool_examples,
)
from evaluation.modern_ai.tool_routing import run_tool_routing_experiment


def test_offline_modern_ai_stack_runs():
    emb = HashingTextEmbedder(64)
    cfg = PosteriorModelConfig(
        beta=.5, target_fpir=.2, predict_T=10.0, gallery_kappa=40.0,
        streaming_threshold_elements=100000,
    )
    p = synthetic_open_set_protocol(14, unknown_fraction=.35, seed=4)
    retrieval = run_retrieval_experiment(p, emb, cfg, calibration_fraction=.4, seed=2)
    assert "open_set" in retrieval["summary"]
    assert "mprisk" in retrieval["summary"]["uncertainty_detection"]

    tools = run_tool_routing_experiment(
        synthetic_tool_examples(), emb, cfg, calibration_fraction=.4, seed=2
    )
    assert "decision_metrics" in tools["summary"]

    records = synthetic_ragtruth_records()
    rag = run_ragtruth_experiment(
        records, emb, cfg, generator_features=synthetic_generator_features(records),
        validation_fraction=.33, n_boot=10, seed=2,
    )
    assert "hybrid" in rag["summary"]["metrics"]
