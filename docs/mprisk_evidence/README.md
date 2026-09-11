# MPRisk: two-stage reference-prior evidence experiments

This package adds a separate experiment pipeline. It does not overwrite the existing
MPRisk, HolUE, published result files, or any `modern_ai` files. The old implementation
remains an explicitly named historical comparison, not a source of numbers attributed
to the evidence formulation.

## Install

Apply `mprisk_evidence_two_stage.patch` from the repository root, on top of the
previous HolUE temperature patch, hotfix, and fixed-decision patch:

```bash
git apply --check mprisk_evidence_two_stage.patch
git apply mprisk_evidence_two_stage.patch
python3 -m pytest -q tests/mprisk_evidence
```

The pipeline uses the scientific Python packages already needed by the repository,
plus `mpmath` for independent high-precision numerical checks. `pytest` is needed
only for tests. No package upgrade is required. The driver uses argparse and reads
the existing YAML files through OmegaConf; it does **not** change the working
directory or rely on a Hydra-decorated entry point. Simple `_target_` configuration
objects are instantiated recursively; unsupported advanced Hydra directives fail
explicitly.

The source configurations are:

- `configs/uncertainty_benchmark/mprisk_core_text_complete.yaml`
- `configs/uncertainty_benchmark/mprisk_core_bio_complete.yaml`

Dataset paths, original HolUE calibration architecture/epochs, and historical
MPRisk parameters are read from those files. A resolved copy is archived in each run.

## Stage A: run these two commands first

```bash
GPU_DEVICE=4 bash run_mprisk_evidence_sanity_text_docker.sh
GPU_DEVICE=5 bash run_mprisk_evidence_sanity_bio_docker.sh
```

Use available GPUs; running sequentially on GPU 4 is also valid. Defaults follow the
existing Docker setup: detached container, 16 GB shared memory, 160 GB RAM, 40 CPUs,
current UID/GID, repository mounted as `/app`. The image defaults to
`${USER}_$(basename "$HOST_APP_DIR")`, where `HOST_APP_DIR` is detected from the script
location. Override `DOCKER_IMAGE`, `HOST_APP_DIR`, `GPU_DEVICE`, `CPUS`, `THREADS`,
`MEMORY`, `SHM_SIZE`, `RUN_ID`, or `CONTAINER_NAME` as needed.

For example:

```bash
DOCKER_IMAGE=my_face_ue_image GPU_DEVICE=4 CPUS=24 \
  bash run_mprisk_evidence_sanity_text_docker.sh
```

The scripts print both the host result directory and the eventual ZIP filename,
plus `docker logs -f ...`. MLflow is not needed; an existing MLflow environment
variable/file is passed through when available but is not copied into results.

Without Docker:

```bash
python3 experiments/mprisk_evidence_experiments.py --stage sanity --domain text
python3 experiments/mprisk_evidence_experiments.py --stage sanity --domain bio
```

Sanity runs use **one target FPIR, 0.10**, on every dataset in the corresponding source
configuration: Yahoo, DBPedia, AG News, CLINC150, PAN; IJB-C, IJB-B, Whale, and the
VoxBlink protocol `large_12k-perspk5`. These are different datasets, not nine independent
modalities. At most 6,000 validation and 4,000 test probes are retained per dataset in
Stage A; the full gallery is kept. Saved original indices make subsampling explicit.
The nonlinear likelihood fit uses at most 3,000 fit-partition probes. These caps make
the sanity run smaller than the full run without declaring it a final benchmark.

Do **not** start full runs until these sanity archives have been reviewed. Completion
of a script is not evidence that the model outperforms anything. Optimizer convergence,
boundary solutions, concentration scales, observed prior/prevalence differences,
calibration, and low-quality false rejections must be inspected.

## Stage B: explicit approval required

After review, point each launcher to the manifest from its corresponding successful
sanity run:

```bash
APPROVE_FULL=1 \
SANITY_MANIFEST="$PWD/outputs/mprisk_evidence/<reviewed_text_run>/manifest.json" \
GPU_DEVICE=4 bash run_mprisk_evidence_full_text_docker.sh

APPROVE_FULL=1 \
SANITY_MANIFEST="$PWD/outputs/mprisk_evidence/<reviewed_bio_run>/manifest.json" \
GPU_DEVICE=5 bash run_mprisk_evidence_full_bio_docker.sh
```

The Python driver independently checks the stage, domain, successful status, model
and source-code fingerprint, source core-configuration fingerprint, and prior beta.
A synthetic smoke run cannot authorize a real full run. The `APPROVE_FULL` flag is a
human-review acknowledgement, not an automated statistical test.

Without Docker:

```bash
python3 experiments/mprisk_evidence_experiments.py --stage full --domain text \
  --confirm-full --approved-sanity /app/outputs/mprisk_evidence/<reviewed_text_run>/manifest.json
```

Full runs use all test probes and all source FPIR operating points:
text `[0.1, 0.2, 0.3, 0.4, 0.5]`; bio `[0.01, 0.05, 0.1, 0.2]`.
The probability-model fit remains capped at 3,000 validation fit probes by default;
`--max-fit 0` uses the complete fit partition. The cap and actual selected indices are
always saved. Full score tuning uses 1,024 random candidates plus deterministic
candidates, compared with 128 in sanity. The primary risk is not selected by test PRR.

## What is actually implemented

For the uniform-reference probe posterior `q_x`, uniform unknown background, and
vMF class densities, the implementation computes

```
B_i = S C_d(alpha*kappa_x) C_d(kappa_g)
      / C_d(||alpha*kappa_x*mu_x + kappa_g*g_i||)
eta_0 = beta / [beta + (1-beta)/K * sum_i B_i]
eta_i = ((1-beta)/K * B_i) / [beta + (1-beta)/K * sum_i B_i]
```

`alpha` rescales the probe concentration inside a **normalized distribution**.
`kappa_g` is one shared vMF gallery dispersion, fitted independently of the old
power-spherical recognizer. Neither is copied from the old FPIR-matching fit.
There is no posterior temperature and no `r_NS` in the primary score.

The primary row, `MPRisk`, is unit-cost conditional risk:
`1 - eta[action]`. The recognizer's action is an explicit input. The posterior's own
argmax is saved only as a counterfactual diagnostic and never replaces the reference
labels in the evaluation. Three components cover false acceptance, incorrect known
identity, and false rejection. Optional PRR-weighted variants are **scores**, not
probabilities or empirically measured error costs.

The theoretical interpretation requires compatible assumptions for the probe
reference posterior, class densities, unknown background, priors, and observation
channel. Correctly normalizing a model is not a proof that it describes real data.
See `MODEL.md` for the exact mathematical specification and limiting properties.

## Experimental controls

### Fixed recognition decisions

On each test dataset/FPIR, the historical `MPRisk raw` recognizer runs once, with
M=0 and without its internal score-weight tuning or calibration. Its decisions are
frozen for every uncertainty method. Decision hashes, achieved FPIR/FNIR/F1, and the
counterfactual posterior disagreement rate are saved.

**Important benchmark qualification:** the existing repository constructs a requested
FPIR operating point from labeled unknown probes in the evaluated benchmark. This
pipeline preserves that construction to compare uncertainty estimates at the same
benchmark decisions. It does not claim that those thresholds were learned without
benchmark labels, or that they guarantee prospective deployment FPIR. No test labels
are used to fit evidence parameters, score weights, or calibration transforms.

For validation decisions, the reference operating point is constructed using the
validation **fit** partition only, then frozen on selection and audit. Audit labels
therefore do not fit even the reference validation decision rule.

### Validation-only fitting

Validation is partitioned approximately 60/20/20 into fit, selection, and audit.
When sufficiently many identities are available, the partitions are identity-disjoint.
For datasets with few topic classes, the fallback is template-disjoint within class;
it is explicitly recorded and is not represented as unseen-identity validation.

Three optimization starts fit `alpha` and `kappa_g` by multiclass negative
log-likelihood on the fit partition. The start with smallest selection NLL is used.
A separately fitted point-vMF model controls for the change in gallery-density
family. The probability parameters are shared across FPIR operating points.

Score-weight searches, MSP temperature choice, supervised comparison heads, and
monotone risk calibration use only selection data. Audit is used only for reporting.
All normalizations use the relevant fitting subset; no test-set feature normalization
is used. The configured beta is kept separate from target FPIR and observed class
prevalence; both observed prevalence and the assumed prior are reported.

### Baselines and ablations

The study includes SCF, AccScr, MSP, entropy, margin, GalUE-style posterior score,
HolUE raw and its source calibration architecture refitted to the fixed decisions,
published-form MPRisk with/without nonspecificity, and a retuned published-form
MPRisk. Controlled supervision differs from the original publication: these rows
are recomputed comparisons, not copies of the published numbers.

Fair combinations include the original simple set (SCF/AccScr/MSP/Margin), the
explicitly listed complete baseline-score set, that set plus the three risk
components, linear HolUE KL features, and a KL+risk hybrid. The full study also
includes logistic and small-MLP comparison heads. Feature lists, search budgets,
normalization parameters and weights are retained. No row claims to combine scores
that were not actually available.

`MPRisk + NS diagnostic` uses the evidence model's own unknown probability and
scaled probe distribution; it is a learned diagnostic comparison, not part of the
primary definition. All seven nonempty subsets of the three genuine risk components
are evaluated in the full stage.

### Numerical and statistical reporting

Evidence is computed in float64 and log space. A convergent series handles Bessel
underflow; high-dimensional values are not repaired with an arbitrary asymptotic or
forced zero. A centered log-partition interpolation accelerates full-gallery
computation and is checked against exact evaluation. Independent high-precision and
spherical-quadrature tests run before data loading. Direct/interpolated scores are
also compared on actual probes.

The metrics retain the repository's F1 definition and PRR random/error-oracle
normalization. The latter is not claimed to be a globally F1-optimal oracle. Ties
use a shared label-independent ordering. AURC, error AUROC/AP, probability Brier/NLL,
calibration bins, and conditional false-rejection detection are also reported.

Paired bootstrap resamples identities when sufficiently many are available; the
few-class fallback is a within-class probe bootstrap and is labeled accordingly.
The optional numerical excess-risk bound is explicitly conditional on i.i.d. audit
observations and exact population coverage; the code does not certify these
assumptions, does not substitute ECE for total variation, and marks the calculation
`certified=false`.

Concentration sweeps are controlled **representation-distribution perturbations**,
not newly captured or re-encoded image/audio/text corruptions. A realistic paired
input-degradation experiment would need domain-specific corruption and re-encoding;
these tables must not be described as such an experiment.

## Outputs: send the two complete ZIPs

Each run creates one tree and one sibling ZIP, e.g.

```
outputs/mprisk_evidence/sanity_text_<run_id>/
  manifest.json
  file_inventory.json
  run.log
  tables/
  plots/
  reproducibility/
    arguments.json
    environment.json
    resolved_core_config.yaml
    source/...
  datasets/<dataset>/
    data_manifest.json
    validation_split.json
    probability_model_fit.json
    partition_numerics.json
    fpir_0.1/
      reference_decision_fit.json
      score_fits.json
      holue_calibrator/
      fusion_features.npz
      per_example/validation.npz
      per_example/test.npz
      status.json
  errors/                 # present when an individual dataset fails
  _cache/ or .../_cache/   # derived large cosine matrices; NOT in the ZIP
outputs/mprisk_evidence/sanity_text_<run_id>.zip
outputs/mprisk_evidence/sanity_text_<run_id>.zip.sha256
```

Per-example NPZ files preserve IDs, original decision, true class, partition,
concentrations, error components, probabilities, top alternatives, and every method's
score in aligned columns. Fitted transformations and selected original indices are
saved. The archive omits only derived scratch cosine matrices, bytecode, temporary
files, and raw input data/model checkpoints already held in the repository. It does
not omit the per-example information needed to recompute the reported metrics.
The included SHA-256 inventory and ZIP integrity check cover the shared files.

The primary sanity files are `main_mprisk_core_comparison.csv`,
`normalization_audit.csv`, `posterior_scoring.csv`, `quality_stratified.csv`,
`quality_error_groups.csv`, `error_type_detection_conditional.csv`, and
`model_parameters.csv`. Send the **whole ZIPs**, not just those tables: the fitted
parameter reports and per-example outputs are necessary for a reliable review.

Normal completion and handled Python failures both produce a ZIP; the manifest
separates `complete` from `failed`. A forced SIGKILL/container OOM/disk failure can
prevent finalization. Once no process is writing the directory:

```bash
bash collect_mprisk_evidence_results.sh /path/to/run --stopped
```

For an interrupted run, this records `interrupted`, never `complete`. To resume,
remove a stale `.running.lock` only after checking the process stopped, then rerun
the original Python command with the same `--run-dir` and `--resume`. Completed
datasets are reused; partial datasets are rerun rather than silently mixed. Resume
checks the code and experiment configuration.

## Small local smoke run

This exercises the driver and output format without the benchmark datasets:

```bash
python3 experiments/mprisk_evidence_experiments.py --synthetic --stage sanity \
  --domain text --device cpu --synthetic-n 200 --fit-iterations 10 --search-budget 16
```

It generates data from a specified observation model, not realistic benchmarks.
Its manifest is marked synthetic and it must never supply dissertation result rows.
No result in the delivered package is a completed real-domain experiment.

Binary error-NLL evaluation clips probabilities to `[1e-15, 1-1e-15]` only when
taking logarithms. Multiclass true-label log probabilities are retained in log space
without that probability floor. This numerical convention is distinct from the
posterior construction and must be kept when comparing exported metrics.
