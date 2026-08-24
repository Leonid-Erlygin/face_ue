# Native Tool-Routing Extension

This extension turns tool/function routing into a first-class open-set recognition task for the existing ArcFace → SCF → GalUE/HolUE/MPRisk stack.

## Scientific formulation

The routing class is a **function/API identity**, not a generated function call. A user request is the probe. Canonical API/function descriptions are the gallery. The system must either select the correct gallery item or reject the request when no suitable gallery item exists.

The primary ToolBench protocol is deliberately fixed-gallery and class-disjoint:

- known API classes are used for ArcFace/SCF training and form the deployment gallery;
- known validation/test prompts use held-out prompts for those classes;
- calibration-unknown API classes are absent from training and used only on validation;
- final-test-unknown API classes are absent from training **and disjoint from calibration unknown classes**.

This allows the original fixed-K posterior and boundary-root gallery-concentration calibration to be used without changing the dissertation model.

BFCL is used only as **external variable-gallery validation**. Its candidate-function gallery changes per prompt. The gallery concentration is transferred from ToolBench instead of being fit on BFCL labels. BFCL labels are used only for downstream held-out error calibration/reporting.

## Why the ArcFace and SCF training files differ

`train_arcface.jsonl` contains:

1. known user requests; and
2. one canonical API-description sample per known class.

The API-description samples make the ArcFace geometry compatible with the gallery used at deployment. After training, `audit_tool_routing_arcface.py` explicitly checks class-center ↔ API-description alignment and validation routing accuracy.

`train_scf.jsonl` contains **known user requests only**. The SCF head should estimate query-representation concentration, not confidence of deterministic gallery descriptions. Gallery concentration remains the global calibrated `kappa_g` of the dissertation model.

Both training stages use the repository's native classes:

- `training.models.lightning_wrappers.BERTEmbedder`
- `training.models.arcface.MetricLearningModel`
- `training.models.heads.SCFHead`
- `training.models.scf.SphereConfidenceFace`
- `training.models.losses.KLDiracVMF`

No parallel probabilistic text encoder is introduced.

## Datasets

### ToolBench G1 — training + primary OSR

The preparer downloads the official ToolBench release and consumes:

```text
external/ToolBench/data/instruction/G1_query.json
```

The official ToolBench retriever preprocessing identifies a relevant API by `(tool_name, api_name)`. The extension uses that same identity and keeps only prompts with exactly one distinct relevant API for ArcFace/SCF classification.

The original ToolBench split is not itself an open-set class-disjoint split, so the extension constructs and records a deterministic protocol from G1 using seed 777.

### BFCL — external routing validation

The preparer clones the official Gorilla repository, checks out an exact Git commit and records it in the manifest. It automatically discovers the newest static single-turn:

- Multiple/Relevance examples as `known` / should-call;
- Irrelevance examples as `unknown` / should-reject;
- corresponding `possible_answer` files when available, to recover the exact correct function index.

Live, parallel-multiple and multi-turn categories are intentionally excluded from this first routing experiment because they introduce additional temporal/compositional factors beyond one-step open-set selection.

## Install

```bash
pip install -r requirements-modern-ai.txt
```

New preparation dependencies are `gdown` and `ijson` (`ijson` avoids loading the full ToolBench G1 JSON array into memory).

## 1. Download and prepare

```bash
bash scripts/modern_ai/prepare_tool_routing.sh
```

Environment-variable overrides:

```bash
TOOL_ROUTING_KNOWN_TOOLS=1024 \
TOOL_ROUTING_UNKNOWN_TOOLS=256 \
TOOL_ROUTING_MIN_QUERIES=3 \
TOOL_ROUTING_UNKNOWN_CAL_FRACTION=0.5 \
TOOL_ROUTING_SEED=777 \
bash scripts/modern_ai/prepare_tool_routing.sh
```

If ToolBench has already been downloaded, skip the download and build directly:

```bash
python experiments/prepare_tool_routing.py build-toolbench \
  --g1-query-path /path/to/ToolBench/data/instruction/G1_query.json \
  --output-dir datasets/tool_routing/toolbench_g1 \
  --num-known-tools 1024 \
  --num-unknown-tools 256 \
  --seed 777
```

Prepared layout:

```text
datasets/tool_routing/toolbench_g1/
├── train_arcface.jsonl
├── train_scf.jsonl
├── train.jsonl              # compatibility alias of ArcFace train file
├── val.jsonl
├── test.jsonl
├── gallery.jsonl
├── labels.json
└── manifest.json

datasets/tool_routing/bfcl_manifest.json
```

`manifest.json` records source SHA-256, protocol counts and hashes of known, calibration-unknown and test-unknown class sets.

## 2. Train ArcFace and SCF

```bash
bash scripts/modern_ai/train_tool_routing.sh
```

This executes:

```text
ToolBench known classes
        |
        v
ArcFace BERT embedding training
        |
        +--> export native backbone.pth
        +--> export ArcFace softmax_weight.pt
        |
        v
ArcFace geometry audit
        |
        v
SCF head training on frozen ArcFace backbone
        |
        v
model_weights/text_models/trained_scf/toolbench.ckpt
```

Training configs:

```text
configs/uncertainty_models/text_model_toolbench_arcface.yaml
configs/uncertainty_models/text_model_toolbench_scf.yaml
```

The number of ArcFace classes is resolved automatically from the prepared dataset via `model.num_labels: auto`; existing fixed-class training configs are unchanged.

### Geometry audit

Before SCF training, the pipeline writes:

```text
outputs/tool_routing/arcface_geometry_audit/
├── summary.json
└── per_class.csv
```

Key checks are:

- API-description → own ArcFace-center cosine;
- nearest-center accuracy of API descriptions;
- validation-query accuracy against ArcFace centers;
- validation-query accuracy against encoded API descriptions.

A poor audit means the deterministic routing representation is mis-specified; uncertainty results should not be interpreted as evidence for SCF/HolUE until the mean embedding is fixed.

## 3. Primary class-disjoint ToolBench OSR

```bash
bash scripts/modern_ai/run_toolbench_osr.sh
```

Config:

```text
configs/modern_ai/toolbench_tool_routing_osr.yaml
```

The experiment uses:

```text
query -> native ArcFace mean direction mu_x
      -> native SCF concentration kappa_x

fixed known API gallery -> API-description ArcFace directions
                        -> globally calibrated kappa_g
```

ToolBench validation is the only dataset used to fit `kappa_g` for the primary tool-routing model. The final ToolBench test contains different unknown API classes.

Outputs include:

```text
outputs/modern_ai/toolbench_g1_open_set_routing/
├── summary.json
├── per_example.csv
├── rejection_curves/
└── tables/
```

The reporting layer produces rejection curves, PRR values and `booktabs` LaTeX tables using the same uncertainty convention as the existing dissertation experiments (large uncertainty rejected first).

## 4. External BFCL validation

Run ToolBench first, then:

```bash
bash scripts/modern_ai/run_bfcl.sh
```

`configs/modern_ai/bfcl.yaml` loads:

```text
outputs/modern_ai/toolbench_g1_open_set_routing/summary.json
```

and freezes the transferred `fitted_gallery_kappa`. It does **not** fit the posterior concentration on BFCL.

Because BFCL has a different candidate gallery per prompt, its downstream calibration/test split is grouped by the complete candidate-tool gallery; identical galleries cannot appear on both sides of that boundary.

## 5. Main SCF ablation

The most important controlled comparison is not learned SCF versus an arbitrary constant concentration. It is learned per-query SCF versus a constant concentration matched to the **median SCF kappa on ToolBench calibration**:

```bash
bash scripts/modern_ai/run_tool_routing_ablation.sh
```

It runs four conditions:

```text
ToolBench: ArcFace + learned SCF kappa_x
ToolBench: same ArcFace + constant matched kappa_x
BFCL:      ArcFace + learned SCF kappa_x
BFCL:      same ArcFace + transferred matched constant kappa_x
```

The ArcFace embedding space is identical. This isolates whether input-dependent concentration carries useful information beyond its average scale.

Expected output roots:

```text
outputs/modern_ai/toolbench_g1_open_set_routing/
outputs/modern_ai/toolbench_g1_open_set_routing_constant_kappa/
outputs/modern_ai/bfcl_open_set_routing/
outputs/modern_ai/bfcl_open_set_routing_constant_kappa/
```

## Metrics and baselines

The tool-routing runner evaluates decision quality and uncertainty separately.

Decision metrics include:

- call/reject accuracy;
- tool/function identity accuracy when an official target is available;
- FPIR: irrelevant/unknown request accepted;
- FNIR: relevant/known request rejected;
- AUROC/AUPRC of `p(unknown)` for irrelevance detection.

Uncertainty baselines include:

- maximum cosine similarity;
- top-1/top-2 margin;
- MSP / softmax entropy;
- `-kappa_x` (SCF itself);
- GalUE;
- HolUE;
- MPRisk;
- MPRisk without nonspecificity;
- `p(unknown)`.

For each uncertainty score the reporting layer computes error AUROC/AUPRC, AURC, rejection curves and PRR.

## Reproducibility and safety checks

- ToolBench source and protocol class sets are hashed.
- BFCL exact Git commit is recorded.
- Unknown classes used for ToolBench calibration and final testing are disjoint.
- BFCL internal downstream calibration is candidate-gallery-grouped.
- Embedding cache keys include ArcFace/SCF checkpoint file metadata, so retraining invalidates stale caches.
- The SCF inference adapter verifies that the backbone tensors embedded in the SCF checkpoint exactly match the supplied ArcFace backbone.
- BFCL fails loudly if an official ground-truth function name cannot be mapped to the provided candidates.
- A missing ToolBench `kappa_g` summary blocks BFCL rather than silently recalibrating on BFCL.

## Tests

Run all modern-AI regression tests:

```bash
pytest -q tests/test_modern_ai_*.py
```

The native tool-routing tests cover:

- class-disjoint ToolBench preparation;
- separate ArcFace/SCF training populations;
- disjoint calibration/test unknown API classes;
- class-balanced sampling;
- BFCL exact target-to-candidate mapping;
- BFCL candidate-gallery-grouped splitting;
- native ArcFace checkpoint export;
- native learned SCF-kappa evaluation;
- transferred ToolBench gallery concentration configuration.

## Recommended experiment order

Do not begin with BFCL. Use this order:

1. prepare ToolBench/BFCL;
2. train ArcFace;
3. inspect the ArcFace geometry audit;
4. train SCF;
5. run class-disjoint ToolBench OSR;
6. run the matched constant-kappa ablation;
7. only then run BFCL external transfer.

This sequence distinguishes failures of the deterministic embedding geometry from failures of SCF concentration or of the GalUE/HolUE/MPRisk posterior.

## Stage-class-disjoint protocol (v2)

The G1 density audit shows that almost every usable API has exactly three
single-API queries.  Therefore the primary protocol must **not** split queries of
the same API into ArcFace, SCF, calibration, and test partitions.  The current
protocol instead partitions API identities themselves:

| Stage | API classes | Typical real queries/API | Purpose |
|---|---:|---:|---|
| ArcFace | 512 | 3 | learn the deterministic spherical text geometry |
| SCF | 384 | 3 | learn query concentration on APIs unseen by ArcFace |
| calibration-known | 256 | 3 | fit gallery concentration / downstream calibration |
| final-test-known | 256 | 3 | final fixed-gallery known probes |
| calibration-unknown | 96 | 3 | FPIR/open-set calibration only |
| final-test-unknown | 96 | 3 | final open-set probes |

All six API sets are pairwise disjoint.  With the current ToolBench release this
uses 1600 of the 1602 API classes having at least three usable G1 queries.

### Prototype-target SCF

SCF no longer requires the ArcFace classifier center for the API identity.
For each SCF-stage query `x` and its API description `t`, the frozen ArcFace
encoder produces

`mu_x = f(x)` and `mu_t = f(t)`.

The native `KLDiracVMF` objective is then optimized with `mu_t` as `wc`.  This
preserves the SCF/vMF mathematics while making SCF API identities disjoint from
ArcFace identities and aligning the training target with the actual deployable
API-description gallery.

### Fixed-K gallery transfer

Calibration and final testing each use one fixed 256-API gallery, but the API
identities are different.  Gallery concentration is fitted only on the
calibration gallery and then frozen for the final gallery.  The evaluator keeps
this on the fixed-K posterior path rather than switching to the variable-gallery
BFCL fallback.
