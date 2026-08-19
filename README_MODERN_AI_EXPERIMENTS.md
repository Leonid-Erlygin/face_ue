# Modern-AI Experimental Extension for GalUE, HolUE, and MPRisk

This extension applies the repository's original mixed-prior open-set posterior to modern embedding systems without replacing the dissertation method with a generic confidence heuristic.

The primary new task is **Open-Set Evidence Retrieval (OSER)**. RAG hallucination prediction and LLM tool routing are downstream/generalization tests built on top of that controlled task.

## 1. What is implemented

### Controlled open-set retrieval

`evaluation/modern_ai/retrieval_experiments.py`

- BEIR-compatible local loader (`corpus.jsonl`, `queries.jsonl`, `qrels/<split>.tsv`).
- BRIGHT loader through Hugging Face Datasets for domains with a global gallery.
- Controlled **evidence deletion**: remove every relevant passage for a designated subset of queries and recompute the actual known/unknown state after global deletion.
- Cross-domain unknown-query protocol.
- Query-disjoint calibration/test split stratified by known/unknown state.
- Standard retrieval metrics plus open-set FPIR/FNIR, OSER accuracy/F1, risk-coverage, AUROC/AUPRC, Brier/NLL/ECE.
- Calibration is held out. Scalar uncertainty scores use monotone calibration; HolUE uses its two native KL components.
- Multiple-qrel evaluation treats *any* relevant passage as correct.
- A separate qrel-aware diagnostic measures how much the original single-identity `r_ID` term over-counts posterior mass on other valid passages. Qrels are never used to change inference.

### Exact large-gallery posterior

`evaluation/modern_ai/scalable.py`

The original implementation materializes an `N_queries x K_gallery` posterior. For `M=0`, the new streaming implementation computes the **same posterior exactly** in gallery chunks:

- exact unknown posterior `p0`;
- exact top known posterior and Bayesian accept/reject decision;
- exact KL1 and deterministic KL2;
- exact GalUE posterior entropy/MSP;
- exact analytic-vMF reject non-specificity;
- exact MPRisk;
- cosine max/margin and softmax entropy/MSP baselines.

It lowers posterior working-memory storage from `O(NK)` to `O(NB)` for gallery chunk size `B`. Compute is still `O(NK)`. This is deliberate: the code does **not** renormalize over ANN top-k candidates and pretend that the omitted posterior tail does not exist.

A regression test numerically compares the streaming implementation with the original dense code.

### Query uncertainty from rewrites

`evaluation/modern_ai/query_uncertainty.py`

Paraphrase/query-rewrite embeddings define an empirical directional sample. A vMF concentration is estimated from resultant length.

Two distinct conditions are implemented:

1. `mean_mode=original`: keep the original query embedding fixed and use rewrites only for kappa. This is the clean HolUE uncertainty experiment.
2. `mean_mode=rewrite_mean`: use the rewrite ensemble mean and its kappa. This is a deployment variant, but it confounds representation changes and uncertainty changes.

`run_rewrite_ablation` evaluates deterministic original, rewrite-kappa/original-mean, and rewrite-kappa/rewrite-mean under the same split.

### Corpus-size stress test

`run_corpus_scaling_experiment` creates **nested** negative-document corpora while retaining all active relevant passages.

Two scientifically different settings are supported:

- `refit`: recalibrate gallery concentration at every K. This asks how well the method can operate when correctly recalibrated.
- `fixed_reference`: calibrate on the smallest/reference corpus and freeze kappa while the corpus grows. This tests deployment robustness to gallery growth.

### RAGTruth

`evaluation/modern_ai/rag.py`

- Loads QA RAGTruth responses and their source passages.
- Computes evidence uncertainty **once per source_id**, then broadcasts it to the multiple generated responses for that source.
- Splits calibration/evaluation by whole `source_id`; no response from one source can cross the split.
- Uses cluster bootstrap confidence intervals by `source_id`.
- Tests MPRisk, GalUE, HolUE, retrieval baselines, optional generator uncertainty, and a hybrid retrieval+generator model.
- Primary downstream hypothesis: retrieval-side uncertainty adds held-out predictive information beyond generation-side uncertainty.
- Gallery kappa must be **transferred from an OSER experiment** or explicitly fixed. Hallucination labels cannot be used to identify the gallery posterior.

`TransformersConditionalSequenceScorer` can score existing RAGTruth responses with a local causal LM. If that LM is not the model that generated the response, its likelihood is correctly described as a **surrogate** generator-confidence feature.

### CRAG / end-to-end RAG

- Loads CRAG questions, reference answers, and supplied web-search contexts.
- `TransformersRAGGenerator` runs a local Hugging Face causal LM and saves a generation cache.
- Generator uncertainty features include mean token NLL, mean token entropy, and worst-token NLL.
- An optional **semantic entropy / self-consistency** baseline samples multiple answers, clusters meanings using conservative bidirectional NLI entailment, and reports discrete semantic entropy, normalized semantic entropy, cluster disagreement, and cluster count. This is intentionally expensive and cached separately.
- Every generator-side feature is reported individually before the learned `generator_only` combination, so the hybrid comparison cannot hide a weak baseline inside an aggregate.
- Compares retrieval-only, generator-only, and hybrid error predictors on held-out examples.
- Uses paired group bootstrap for the key hybrid-vs-generator comparison.

The code requires external/official binary answer-error judgments by default. A simple reference-substring evaluator exists only for smoke/debugging and cannot be activated accidentally (`allow_reference_heuristic: false` in the production config).

### LLM agent / tool routing

`evaluation/modern_ai/tool_routing.py`

- Supports BFCL relevance/irrelevance JSONL files for **call vs reject** evaluation.
- Supports a generic JSONL format with explicit `relevant_tool_indices` for tool-ID evaluation.
- Handles a different function gallery size K for every example.
- Since the fixed-K boundary equation is not valid for variable galleries, kappa is fit empirically on validation data by minimizing target-FPIR error first and unknown-posterior NLL second.
- Reports call/reject accuracy, FPIR/FNIR, tool-ID accuracy where labels exist, uncertainty/error detection, and probability calibration.

## 2. Recommended experimental order

Run the experiments in this order. The order is part of the scientific design, not just an engineering convenience.

1. **OSER / BEIR**: establish that the posterior transfers from identity selection to evidence selection under a controlled known/unknown definition.
2. **OSER / BRIGHT**: test reasoning-intensive retrieval rather than only lexical/semantic near-neighbor retrieval.
3. **Corpus scaling**: test the K-dependence of the mixed prior and gallery concentration.
4. **Rewrite-kappa ablation**: test whether HolUE's query concentration predicts unstable/failing retrieval independently of changing the query representation.
5. **Cross-domain unknown queries**: complement artificial evidence deletion with natural OOD queries.
6. **M=0 vs Monte Carlo sensitivity**: test whether the mean-embedding posterior approximation materially changes decisions/risk.
7. **Kappa-root sensitivity**: expose all concentrations satisfying the same boundary and test whether conclusions depend on root selection.
8. **Prior/cost sensitivity**: vary beta, target FPIR, and each MPRisk loss weight one factor at a time.
9. **BFCL tool routing**: test a second modern open-set decision system with a particularly clean “none of the tools” interpretation.
10. **RAGTruth**: test whether source-level retrieval/evidence risk predicts downstream hallucination across repeated generator responses.
11. **CRAG end-to-end**: test whether retrieval risk adds information beyond generation uncertainty.

The strongest dissertation claim should depend on 1-5. Experiments 6-9 test robustness/external validity; they should not be allowed to redefine/tune the core posterior using downstream hallucination labels.

## 3. Installation

The original repository dependencies are still required. Additional modern-AI dependencies are listed in:

```bash
pip install -r requirements-modern-ai.txt
```

The offline smoke test needs only the numerical stack (`numpy`, `scipy`, `scikit-learn`, `torch`, `omegaconf`). It deliberately uses a hashing embedder and no downloaded model.

## 4. Smoke test

```bash
bash scripts/modern_ai/run_smoke.sh
```

or

```bash
python experiments/modern_ai_experiments.py --config configs/modern_ai/smoke.yaml
```

The smoke test exercises:

- controlled open-set retrieval;
- rewrite-kappa estimation;
- corpus scaling;
- tool routing;
- RAG source-risk and hybrid calibration;
- JSON/CSV artifact generation.

It is a software test, **not a scientific benchmark**.

### Automatic rejection curves and LaTeX tables

Every modern-AI runner now mirrors the paper-facing behavior of the original
repository.  After a numerical experiment finishes, it automatically generates
held-out rejection curves, Random/Oracle references, PRR values, and LaTeX
tables.  Retrieval/scaling/sensitivity runs are discovered recursively, so each
leaf experiment gets its own paper artifacts.

Typical output:

```text
outputs/modern_ai/<experiment>/
  summary.json
  per_query.csv              # or per_example.csv / per_response.csv
  rejection_curves/
    all_rejection_curves.csv
    prr_values.csv
    *_rejection_curve.png
    *_rejection_curve.pdf
  tables/
    uncertainty_metrics.csv
    uncertainty_metrics.tex
    operating_point.csv      # retrieval / tool routing
    operating_point.tex
  report_manifest.json       # at the runner output root
```

The convention is identical to the existing MPRisk plotting code: **larger
uncertainty is filtered first**, the x-axis is the filtered-out fraction, and PRR
is normalized between a seeded Random ranking (0) and an error-oracle ranking
(1).  The primary PRR metric is OSER F1 for evidence retrieval, task accuracy for
tool routing, and retained answer accuracy for RAG.

The defaults can be overridden in any modern-AI YAML:

```yaml
reporting:
  enabled: true
  rejection_fractions: [0.0, 0.5, 20]
  display_random_curve: true
  display_oracle_curve: true
  prr_in_legend: true
  figsize: [6.4, 4.8]
  legend_fontsize: 8
  round_num: 3
  highlight_best: true
  # Optional: restrict and order plotted/table methods.
  # methods: [max_similarity, galue_entropy, holue, mprisk_no_ns, mprisk]
```

For RAG outputs, raw retrieval/generator features and validation-fitted error
models are namespaced in CSVs (`raw__*`, `generator_raw__*`, `model__*`).  This
avoids ambiguous duplicate headers such as `mprisk` and ensures plots/tables use
the same held-out model outputs reported in `summary.json`.

## 5. BEIR OSER

Place a BEIR dataset in the documented local layout:

```text
data/beir/scifact/
  corpus.jsonl
  queries.jsonl
  qrels/
    test.tsv
```

Then:

```bash
bash scripts/modern_ai/run_beir_oser.sh
```

Edit `configs/modern_ai/beir_oser.yaml` to change the dataset, embedder, beta, target FPIR, or uncertainty settings.

The key output is:

```text
outputs/modern_ai/beir_scifact_oser/
  summary.json
  per_query.csv
  resolved_config.yaml
  environment.json
```

`summary.json` contains `fitted_gallery_kappa`. Downstream RAG configs can load it with `posterior.gallery_kappa_from_summary`.

## 6. BRIGHT OSER

```bash
bash scripts/modern_ai/run_bright_oser.sh
```

The adapter uses the official `xlangai/BRIGHT` dataset layout (`examples` and `documents`). Some BRIGHT domains define **query-specific excluded IDs**. A global-gallery posterior has a single K and cannot reproduce query-dependent gallery exclusions exactly, so the loader raises an error for such domains instead of silently reporting a non-comparable benchmark result.

## 7. Rewrite experiment

Generate paraphrases once and cache them:

```bash
python experiments/generate_query_rewrites.py \
  --config configs/modern_ai/beir_oser.yaml \
  --model <LOCAL_CAUSAL_LM> \
  --output data/rewrites/scifact.jsonl \
  --num-rewrites 5
```

Then:

```bash
python experiments/modern_ai_experiments.py \
  --config configs/modern_ai/rewrite_ablation.yaml
```

Never interpret `rewrite_mean + rewrite_kappa` alone as evidence that HolUE uncertainty improved. The `original_mean + rewrite_kappa` contrast is the relevant isolation test.

## 8. Scaling

```bash
bash scripts/modern_ai/run_beir_scaling.sh
```

Repeat after changing:

```yaml
scaling_calibration_mode: fixed_reference
```

This gives the important pair:

- performance when recalibration is allowed as K changes;
- degradation when the deployment gallery grows without recalibration.

If K is large enough, deterministic scoring automatically switches to the exact streaming path.

## 9. Approximation and kappa-identifiability stress tests

Run the deterministic-vs-Monte-Carlo experiment:

```bash
bash scripts/modern_ai/run_mc_sensitivity.sh
```

The default `fixed_m0` mode calibrates gallery kappa once with M=0 and freezes it for M=8/32/128, isolating posterior-integration effects. Repeat with `mc_kappa_mode: refit` to ask the separate operational question in which each MC method is allowed to recalibrate. `mc_repeats` quantifies sampling variability.

Run all gallery-kappa boundary roots:

```bash
bash scripts/modern_ai/run_kappa_root_sensitivity.sh
```

This reports every numerical root found for the same FPIR-matched cosine boundary and identifies the root selected by validation NLL. A large spread in downstream conclusions is evidence of parameter non-identifiability and should be reported.

Audit the prior and MPRisk loss assumptions:

```bash
bash scripts/modern_ai/run_posterior_sensitivity.sh
```

The default sweep varies `beta`, `target_fpir`, and each of `lambda_fa`, `lambda_id`, `lambda_fr`, `lambda_ns` one factor at a time on the identical query split. Posterior/operating-point changes re-identify gallery kappa; pure cost sweeps freeze the baseline kappa so the effect is attributable to the loss rather than a second calibration change. Report the whole sensitivity surface, not only the most favorable operating point.

## 10. RAGTruth

Put the official files at the paths configured in `configs/modern_ai/ragtruth.yaml`, then first run OSER so that:

```text
outputs/modern_ai/beir_scifact_oser/summary.json
```

exists.

Then:

```bash
bash scripts/modern_ai/run_ragtruth.sh
```

Optional generator-side confidence can be enabled by setting:

```yaml
generator:
  sequence_scorer_model: <LOCAL_CAUSAL_LM>
```

The main paper comparison is not “retrieval risk predicts every hallucination.” It is:

> Does retrieval/evidence risk predict a distinct subset of RAG failures, and does adding it improve a held-out hybrid predictor beyond generator uncertainty alone?

## 11. CRAG

Configure a local generation model or point `generator.cache_path` to an existing generation cache.

For a stronger free-form uncertainty baseline, enable semantic entropy in `configs/modern_ai/crag.yaml` and provide a local NLI model whose configuration exposes an unambiguous entailment label (or set `entailment_label_id` explicitly). Semantic equivalence is mutual entailment; exact duplicate samples bypass NLI. The primary discrete entropy does not depend on generation probabilities.

Provide independent/official judgments as JSONL:

```json
{"id": "interaction-id", "is_error": false}
{"id": "another-id", "is_error": true}
```

Then:

```bash
bash scripts/modern_ai/run_crag.sh
```

The run will fail rather than silently use the reference-substring heuristic unless you explicitly enable the debug-only option.

## 12. BFCL / tool routing

Populate the current BFCL relevance and irrelevance file paths in `configs/modern_ai/bfcl.yaml`, then:

```bash
bash scripts/modern_ai/run_bfcl.sh
```

For a benchmark that contains a unique or set-valued correct tool label, use generic JSONL:

```json
{
  "id": "1",
  "query": "What is the weather in Berlin?",
  "tools": [
    {"name": "weather", "description": "Get weather", "parameters": {}},
    {"name": "calendar", "description": "Create an event", "parameters": {}}
  ],
  "known": true,
  "relevant_tool_indices": [0]
}
```

and set runner mode to `tool_routing`.

## 13. Statistical rules encoded in the implementation

- Kappa selection uses calibration queries only.
- Error-probability calibration uses calibration labels only.
- RAGTruth split/bootstrap unit is `source_id`, not response row.
- Scalar calibration is monotone increasing in the pre-defined uncertainty score; it cannot reverse a bad ranking.
- `p(unknown)` is treated primarily as **unknown-state evidence**, not generic decision-error probability.
- HolUE uses `(KL1, KL2)` as two features rather than a hand-picked sum for its calibrated comparison.
- Multiple qrels are set-valued correct retrievals.
- Corpus scaling uses nested negative sets.
- Real benchmark output stores the resolved config and environment/package manifest.

## 14. Important limitations to report, not hide

1. **Document identity is not semantic answer identity.** MPRisk's original `r_ID` assumes exactly one correct gallery identity. The qrel-set audit quantifies this mismatch, but a fully set-valued Bayesian derivation would be a further theoretical contribution.
2. **Streaming fixes memory, not compute.** Exact full-gallery normalization remains O(NK). ANN/top-k approximations require a principled tail estimator before they can be claimed equivalent.
3. **Rewrite dispersion is an estimator choice.** It must be validated against retrieval instability/failure and compared with constant kappa; it is not automatically a calibrated posterior concentration.
4. **RAG failure has generator-side causes.** MPRisk is retrieval/evidence risk in the RAG experiments. The hybrid test is intentionally designed to find incremental rather than universal predictive value.
5. **Transferred kappa is a strong generalization test.** If a concentration calibrated on OSER fails badly on variable RAG context galleries, report the failure and test recalibration as an ablation instead of silently tuning on the test task.
6. **Benchmark semantics differ.** BFCL relevance/irrelevance evaluates whether any provided function should be called; tool-ID accuracy is only meaningful where target function labels exist.

## 15. Tests

```bash
pytest -q tests/test_modern_ai_*.py
```

The tests cover:

- exact dense/streaming posterior equivalence;
- evidence deletion state recomputation;
- multiple relevant passages;
- rewrite-kappa isolation;
- group-disjoint RAG splitting;
- end-to-end offline retrieval/tool/RAG integration.

## 16. Main code map

```text
evaluation/modern_ai/
  calibration.py             held-out monotone/logistic error calibration
  data.py                    BEIR, BRIGHT, RAGTruth, CRAG, BFCL loaders/protocols
  embedders.py               SentenceTransformers + cache + offline hashing
  methods.py                 adapter to original GalUE/HolUE/MPRisk posterior
  metrics.py                 OSER, retrieval, calibration, risk-coverage metrics
  query_uncertainty.py       rewrite-based vMF concentration
  rag.py                     RAGTruth + end-to-end RAG + local HF generation
  retrieval_experiments.py   OSER, scaling, rewrite ablations
  scalable.py                exact streaming deterministic posterior
  statistics.py              clustered/paired bootstrap and group splits
  synthetic.py               offline CI/smoke data
  tool_routing.py            BFCL/generic open-set function routing

experiments/
  modern_ai_experiments.py   config-driven runner
  generate_query_rewrites.py cached rewrite generation

configs/modern_ai/
  smoke.yaml
  beir_oser.yaml
  beir_scaling.yaml
  bright_oser.yaml
  beir_cross_domain.yaml
  rewrite_ablation.yaml
  mc_sensitivity.yaml
  kappa_root_sensitivity.yaml
  ragtruth.yaml
  crag.yaml
  bfcl.yaml
```
