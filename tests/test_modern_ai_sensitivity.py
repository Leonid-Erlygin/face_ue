from dataclasses import replace

from evaluation.modern_ai.embedders import HashingTextEmbedder
from evaluation.modern_ai.methods import PosteriorModelConfig
from evaluation.modern_ai.retrieval_experiments import run_posterior_hyperparameter_sensitivity
from evaluation.modern_ai.synthetic import synthetic_open_set_protocol


def test_posterior_sensitivity_runs_main_effects_without_split_changes(tmp_path):
    protocol = synthetic_open_set_protocol(seed=19)
    embedder = HashingTextEmbedder(n_features=64)
    cfg = PosteriorModelConfig(
        beta=0.5,
        target_fpir=0.25,
        predict_T=20.0,
        mc_samples=0,
        streaming_threshold_elements=100,
        streaming_gallery_chunk_size=16,
        streaming_query_batch_size=8,
    )
    out = run_posterior_hyperparameter_sensitivity(
        protocol,
        embedder,
        cfg,
        sweeps={"beta": [0.25], "lambda_ns": [0.0, 2.0]},
        calibration_fraction=0.4,
        seed=19,
        output_dir=tmp_path,
    )
    assert out["experiment"] == "posterior_hyperparameter_sensitivity"
    assert len(out["results"]) == 4  # baseline + one beta + two lambda_ns
    assert (tmp_path / "posterior_sensitivity.json").exists()
