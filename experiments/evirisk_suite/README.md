# EviRisk full dissertation rerun

This incremental patch applies **after `evirisk_whale_prr_selection.patch` and the
v12 verification patch**, from the original repository root `/app`. It adds a
full-suite driver; it does not replace the evidence formula or overwrite the
original paper experiments. It is based on the source snapshot returned with the
successful Whale PRR-selection run.

## Apply and run

```bash
cd /app
patch --dry-run -p1 < evirisk_full_suite.patch
patch -p1 < evirisk_full_suite.patch
python experiments/evirisk_full_suite.py --check-inputs
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python experiments/evirisk_full_suite.py \
  --out /app/outputs/evirisk_full_prr
```

The final archive is **`/app/outputs/evirisk_full_prr.zip`**. Return the entire ZIP,
not only the summary CSV. An adjacent SHA-256 checksum is also written. Failed
runtime jobs produce an explicitly incomplete archive with logs; a failing static
input check starts no experiments. Existing exports, source metadata, old
experiment outputs and native pooling caches are not overwritten.

Resume using **the same arguments**, plus `--resume`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python experiments/evirisk_full_suite.py \
  --out /app/outputs/evirisk_full_prr --resume
```

Resume verifies completed-job checksums, source code, settings, versions, and input
file signatures. It skips complete dataset/seed jobs and **restarts an incomplete
dataset/seed job from its beginning**. It does not append partial operating-point
tables or use results from another protocol. A `.suite.lock` prevents overlapping
writers. Remove a stale lock only after confirming no process is still using it.

## Scope and paths

Nine datasets, 41 operating points per seed, all gallery-G1 probes:

| Config dataset identifier | Display name | FPIR grid |
|---|---|---|
| `IJBC` | IJB-C | 0.01, 0.05, 0.1, 0.2 |
| `IJBB` | IJB-B | 0.01, 0.05, 0.1, 0.2 |
| `whale` | Whale | 0.01, 0.05, 0.1, 0.2 |
| `large_12k-perspk5` | VoxBlink | 0.01, 0.05, 0.1, 0.2 |
| `yahoo` | Yahoo Answers | 0.1, 0.2, 0.3, 0.4, 0.5 |
| `dbpedia` | DBPedia | 0.1, 0.2, 0.3, 0.4, 0.5 |
| `agnews` | AG News | 0.1, 0.2, 0.3, 0.4, 0.5 |
| `clinc150` | CLINC150 | 0.1, 0.2, 0.3, 0.4, 0.5 |
| `pan_test` | PAN-20-AV | 0.1, 0.2, 0.3, 0.4, 0.5 |

The original paths, calibration datasets, pooling conventions, and native baseline
settings come from:

- `configs/uncertainty_benchmark/mprisk_core_bio_complete.yaml`
- `configs/uncertainty_benchmark/mprisk_core_text_complete.yaml`

No image/audio encoder or SCF head is retrained. This is an uncertainty-estimator
rerun over the existing exported representations, using all available test probes.
The numerical FPIR grid is fixed above, not chosen after looking at scores.
A missing requested input is an error, not a reason to silently omit a dataset.

Optional path overrides need no source edit:

```json
{
  "whale": {
    "test_root": "/app/datasets/whale",
    "validation_root": "/app/datasets/whale_val"
  },
  "yahoo": {
    "test_root": "/app/datasets/text-ident/yahoo",
    "validation_root": "/your/existing/yahoo-validation"
  }
}
```

Pass `--paths-json /app/evirisk_paths.json` to input check, run, and resume. Names of
exports and metadata still follow the corresponding original `dataset_name`.
Use the actual existing validation root; the example does not create a new split.
Alternatively pass full source-compatible `--bio-config` and `--text-config`.
`--datasets IJBC whale` explicitly runs a subset and is labelled as partial scope.

## Frozen selection rule

1. Read fresh source metadata and reproduce source SCF pooling. Keep the base
   recognizer directions and decisions fixed across methods.
2. Split validation into fitting/selection/audit parts using the existing 60/20/20
   rule. Identity-disjoint splits are used when supported; small-class datasets use
   the original within-class template split and record that qualification.
3. Fit the finite-distribution likelihood model on the fitting part with the
   existing multistart optimizer. Retain every eligible converged candidate,
   including the original NLL incumbent.
4. For **each** retained candidate, tune three nonnegative FA/ID/FR weights on the
   selection part. Unit weights and the original weight alternatives are included.
5. Choose the probability candidate and weights jointly by selection PRR. NLL is
   an exact-PRR-tie breaker only. Save the choice before evaluating test predictions.
6. Report separate audit and test results. Audit/test outcomes are not used to
   select models, weights, feature standardizers or calibrators.

As in the source benchmark, **unknown test labels set the target-FPIR recognition
threshold**; validation uses unknown probes in its fitting part. This is an
operating-point benchmark, not a claim that a validation threshold transfers
unchanged to deployment. Parameter/weight fitting is otherwise validation-only.

The default fit limit of 3,000 concerns only likelihood-training rows. It does not
subsample test probes. Three likelihood starts, 250 iterations, probe-scale upper
bound 100, gallery-concentration upper bound 100,000, beta 0.5, seed 777 and 1,024
random weight candidates **per converged model** retain the corrected Whale rule.
The total EviRisk search budget thus depends on the number of eligible models.
It is explicitly recorded; the suite does not claim identical total search capacity
for every baseline. Native HolUE architecture/epochs follow the supplied configs.

Optional `--seeds 777 778 779` repeats splitting, estimator fitting and selection
for every listed seed (123 points); it does not retrain the encoders. All seeds
are retained, never the seed with the best test result. The default single-seed
bootstrap is conditional on the fitted estimators, not repeated-training inference.

## One deliberate evaluation-convention update

The default **`seeded-permutation`** tie policy leaves unequal scores unchanged,
but orders exact ties using a fixed label-independent permutation. It preserves
the original random and oracle RNG draw sequence, then draws the tie permutation.
This avoids platform-dependent default-sort changes such as those found in the
HolUE replay. The policy is identical in model selection, fusion/weight fitting,
audits, test metrics and bootstrap. It is frozen in `protocol.json`.

No rounded score bins or label-driven tie-breaking are introduced. For a strictly
historical replay, `--tie-policy legacy` is available, but it must be used for the
whole run. Do not combine its cells with the default-policy run. Existing tools
outside this new driver still default to the original legacy policy.

## Methods and coverage

**At all 41 points:**

- Main weighted **EviRisk** and its **EviRisk-1** unit-weight ablation, using the
  same selected finite-distribution model.
- The previous NLL-only model-selection control, including its retuned weights.
- Separate point comparators `GalUE-vMF (NLL; fixed action)` and its weighted
  variant. They are not relabelled as the native GalUE implementation.
- Native GalUE, HolUE and MPRisk three/four-component references, with source
  density/temperature settings. MPRisk references are retuned on the common
  selection objects; they are not copied publication cells. Text HolUE and MPRisk
  retain their **different** configured temperatures. Lin-All uses actual HolUE KL
  features, not MPRisk KL features under another name.
- SCF, AccScr, MSP, Entropy, Margin, Lin-4, Lin-KL, Lin-All, Lin-All+E, supervised
  logistic fusion and (full mode) a small supervised MLP fusion. Feature registries,
  weights, scaling, selected exact-score candidates and training indices are saved.
  Nested fusion families retain the exact smaller-family candidates and EviRisk.
- Positive-slope probability calibration `EviRisk-Cal`, trained on a transportable
  weighted-loss logit, **never on per-batch ordinal ranks**. Main EviRisk ranking is
  retained; calibrated probabilities are in separate files.
- Seven nonempty FA/ID/FR component ablations, with unit weights and with the
  selected costs held fixed. No claim that these fixed ablations are reoptimized.
- PRR, error AUROC/AUPRC, AURC, rejection/risk-coverage curves, conditional and
  all-object error-type metrics, posterior NLL/Brier, reliability bins, calibration
  summaries, paired bootstrap and sampled numerical checks.
- Constant/shuffled-concentration controls with frozen weights/parameters and
  unchanged recognizer. Diagnostic only, not test-selected replacement methods.

**At the predeclared FPIR 0.1 on every dataset (full mode):**

- Validation-size study of weights with three resamples. Posterior fixed: this is
  conditional weight-data efficiency, not end-to-end estimator data efficiency.
- Probability fit-size studies at 25% and 50%: regenerate multistart candidates
  and repeat the **PRR** selection procedure, never the discarded NLL-only rule.
- Fixed-parameter sensitivity to probe scale, class concentration and beta;
  unknown-background ablation; controlled distribution limits; concentration
  strata and diagnostic example records. These do not select the main method.

Operating-point and cross-dataset **weight-only transfer** are reported, with
posterior models locally fitted in the destination. They are not zero-shot
transfers of the whole probability model. Heavy fit-size/sensitivity experiments
are intentionally not repeated at every FPIR.

`--supplementary core` omits the FPIR-0.1 heavy studies and MLP comparison. It is a
reduced diagnostic scope, not the default full dissertation rerun. Earlier HolUE
MC/encoder-training studies are not rerun by this driver and must retain their
original provenance; no new raw-input corruption experiments are asserted.

## Outputs

```
evirisk_full_prr/
  manifest.json               # complete/incomplete/diagnostic failures
  protocol.json               # frozen settings, code hash, configs, input signatures
  coverage.json               # expected and present dataset/seed/FPIR rows
  numerical_preflight.json
  source/                     # executable source snapshot
  tables/                     # aggregated new results ONLY
    test_prr_wide.csv
    paired_point_estimates.csv
    bootstrap.csv
    probability_calibration.csv
    component_ablation.csv
    validation_size_weights.csv
    probability_fit_size.csv
    parameter_sensitivity.csv
    cross_dataset_weights.csv
  latex/prr_seed_777.tex
  runs/seed_777/DATASET/
    input_checks/{validation,test}.json
    data_manifest.json
    validation_split.json
    probability_model_fit.json
    fpir_0.1/
      selected_probability_model.json
      evidence_selection.json
      weights.json
      reference_decisions.json
      validation.npz
      test.npz
      validation_probabilities.npz
      test_probabilities.npz
      fusion_fit.json
      fusion_features.npz
      probability_calibrator.json
```

All curves and calibration bins needed to regenerate figures are saved. The
archive includes per-object **scores/probabilities/labels**, not raw embeddings or
scratch similarity matrices. Class labels/template/subject IDs in output records
may still be sensitive: return only through the intended research channel.

Bootstrap tables retain the source convention **comparator minus EviRisk**;
`paired_point_estimates.csv` gives the observed **EviRisk minus comparator**.
Intervals are conditional paired 95% percentile intervals (200 resamples default);
small-class fallback is stated per row. No multiple-comparison correction is
claimed. The audit squared-loss bound is recorded as a conditional diagnostic,
not certified under identity dependence.

A `complete_with_diagnostic_failures` manifest means all primary points finished
but at least one supplementary refit failed. Failed diagnostics are explicit and
are never replaced by old scores. All nine datasets, all primary methods, and the
relevant supplementary records should be reviewed before updating thesis claims.

Whale is a retrospective debugging case. Repeating it under this frozen rule does
not turn it into an untouched-holdout experiment. Ranking recovery does not imply
correct probability calibration or theoretical dominance over every comparator.

## Resource use and tests

Use the existing repository environment. Static check imports actual native
baseline targets; it does not install packages or download datasets. `--device`
controls cosine construction; evidence fitting is float64 CPU. Native modules
retain their source device behaviour (including CUDA auto-detection where used).
Jobs run sequentially in fresh processes. Disk-backed cosines are scratch files;
they are removed after successful jobs unless `--keep-cache` is specified, and
always excluded from the ZIP. Existing dataset caches are not used or modified.

No runtime or dataset-specific performance guarantee is made. The full run refits
probability models, HolUE and risk/fusion weights; it is materially more work than
replaying saved Whale predictions.

```bash
python -m pytest tests/mprisk_evidence -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python experiments/evirisk_full_suite.py --smoke \
  --out /app/outputs/evirisk_full_smoke
```

Smoke creates two generated datasets with reduced budgets, explicitly labelled
synthetic, and a stand-in for HolUE. It checks execution and packaging, not real
benchmark quality. The separate test report supplied with the patch records
additional native-baseline integration checks on generated IJB-format fixtures.
