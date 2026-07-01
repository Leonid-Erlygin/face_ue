
# MPRisk Experiments for the New Paper

This document describes the experimental pipeline for the follow-up paper on why raw KL divergence is not sufficient as an operational uncertainty score for open-set recognition (OSR), and why a calibrated/tuned posterior decision-risk score is preferable.

The new method is **MPRisk**: Mixed-Prior Bayes Risk for Open-Set Recognition.

---

## 1. Conceptual summary

The accepted HolUE paper uses KL divergence between a posterior class distribution and a prior class distribution as an uncertainty signal. The new paper investigates the limitation of this choice:

> KL divergence measures information gain, not the probability or cost of an OSR decision error.

For selective OSR, the operational target is closer to posterior decision risk. MPRisk therefore decomposes OSR risk into interpretable components:

```text
r_FA: risk of false acceptance
r_ID: risk of misidentification
r_FR: risk of false rejection
r_NS: mixed-prior reject non-specificity penalty
```

The final weighted risk score is

$$
u_\lambda(x)
=
\lambda_{\mathrm{FA}} r_{\mathrm{FA}}(x)
+
\lambda_{\mathrm{ID}} r_{\mathrm{ID}}(x)
+
\lambda_{\mathrm{FR}} r_{\mathrm{FR}}(x)
+
\lambda_{\mathrm{NS}} r_{\mathrm{NS}}(x).
$$

The weights $\lambda$ can be fixed or tuned on a validation set.

The mixed-prior term $r_{\mathrm{NS}}$ is essential. It exists because the unknown class is modeled not as a single discrete class, but as a continuous identity space. This allows the method to distinguish:

1. a confident true reject of a high-quality unknown probe;
2. a suspicious reject caused by a poor-quality in-gallery probe whose embedding drifted away from the gallery.

---

## 2. New code files

The new experimental code is organized into three groups.

### 2.1 Core experiments

```text
experiments/mprisk_core_experiments.py
configs/uncertainty_benchmark/mprisk_core_experiments.yaml
scripts/run_mprisk_core_experiments.sh
```

These experiments evaluate the main MPRisk variants, component ablations, error-type detection, and KL-vs-risk rank inversion.

### 2.2 Tuning experiments

```text
experiments/mprisk_tuning_experiments.py
experiments/mprisk_tuning_plots.py
configs/uncertainty_benchmark/mprisk_tuning_experiments.yaml
scripts/run_mprisk_tuning_experiments.sh
```

These experiments study validation-size dependence, operating-point transfer, cross-dataset transfer, and hyperparameter sensitivity.

### 2.3 Diagnostics and paper artifacts

```text
experiments/mprisk_diagnostics_experiments.py
experiments/mprisk_diagnostics_plots.py
configs/uncertainty_benchmark/mprisk_diagnostics_experiments.yaml
scripts/run_mprisk_diagnostics_experiments.sh
```

These experiments produce mixed-prior necessity diagnostics, reliability metrics, bootstrap confidence intervals, qualitative example tables, and runtime measurements.

---

## 3. Required method implementation

The experiments assume that the repository contains the new method class:

```python
evaluation.open_set_methods.class_prob_models.MPRiskPredictiveProb
```

The method should expose the following attributes after calling `setup()`:

```python
r_fa
r_id
r_fr
r_ns
risk_main
risk_ns
mprisk
oog_prob
oog_nonspecificity
mean_probs
kl_1
kl_2
```

The experiments also assume that the scalar monotone calibration class exists:

```python
evaluation.open_set_methods.calibration_methods.ScalarRiskCalibration
```

The calibrated MPRisk variant should return repository-style uncertainty:

```text
predicted_unc = -P(correct)
```

Therefore:

```text
P(error) = 1 + predicted_unc
```

---

## 4. Datasets and embeddings

The experiments reuse the same precomputed protocols and embeddings as the accepted HolUE paper.

Expected directory structure:

```text
project_dir/
  datasets/
    arcface_ijb/
      IJBC/
      IJBB/
    ms1m_ident/
    whale/
    whale_val/
    vb-clean_ident/
      Large_12k/
      Medium_1.2k/
  model_weights/
  configs/
  evaluation/
  experiments/
  scripts/
```

The default configs start with IJB-C only:

```yaml
test_datasets:
  - dataset_name: IJBC
```

After IJB-C works, uncomment IJB-B, Whale, and VoxBlink entries in the relevant config files.

---

## 5. Running all experiments

From the repository root or from `/app` inside the Docker container:

```bash
bash scripts/run_mprisk_core_experiments.sh
bash scripts/run_mprisk_tuning_experiments.sh
bash scripts/run_mprisk_diagnostics_experiments.sh
```

For debugging, run each script directly with full Hydra error output:

```bash
HYDRA_FULL_ERROR=1 python experiments/mprisk_core_experiments.py -cn=mprisk_core_experiments
HYDRA_FULL_ERROR=1 python experiments/mprisk_tuning_experiments.py -cn=mprisk_tuning_experiments
HYDRA_FULL_ERROR=1 python experiments/mprisk_diagnostics_experiments.py -cn=mprisk_diagnostics_experiments
```

---

# Part A. Core MPRisk Experiments

## A.1 Purpose

The core experiments answer:

1. Does MPRisk outperform KL-based HolUE on selective OSR?
2. Which MPRisk components are useful?
3. Which error types are detected by each method?
4. Does KL ranking disagree with posterior risk ranking?

## A.2 Config

```text
configs/uncertainty_benchmark/mprisk_core_experiments.yaml
```

Default:

```yaml
exp_dir: outputs/experiments/mprisk_core
far_list: [0.05]
beta_list: [0.5]
```

Recommended final run:

```yaml
far_list: [0.01, 0.05, 0.1, 0.2]
beta_list: [0.5]
```

Add all datasets once IJB-C works.

## A.3 Methods compared

The default core config includes:

```text
SCF
AccScr
GalUE
HolUE
MPRisk raw
MPRisk no NS
MPRisk
MPRisk cal
```

### SCF

Uses only sample-quality confidence.

### AccScr

Uses distance to acceptance threshold.

### GalUE

Gallery-aware posterior uncertainty without embedding uncertainty.

### HolUE

Original KL-based calibrated HolUE.

### MPRisk raw

Equal-cost posterior risk:

$$
r_{\mathrm{FA}} + r_{\mathrm{ID}} + r_{\mathrm{FR}} + r_{\mathrm{NS}}.
$$

### MPRisk no NS

Same as raw MPRisk but with

$$
\lambda_{\mathrm{NS}} = 0.
$$

This is used to test whether mixed-prior reject non-specificity helps.

### MPRisk

MPRisk with validation-tuned weights:

$$
\lambda^\star
=
\arg\max_{\lambda \geq 0}
\mathrm{PRR}_{\mathrm{val}}^{F_1}(u_\lambda).
$$

### MPRisk cal

MPRisk, followed by scalar monotone probability calibration.

## A.4 Outputs

Core outputs are written to:

```text
outputs/experiments/mprisk_core/
```

Main tables:

```text
outputs/experiments/mprisk_core/tables/main_mprisk_core_comparison.csv
outputs/experiments/mprisk_core/tables/error_type_detection.csv
outputs/experiments/mprisk_core/tables/mprisk_component_ablation.csv
outputs/experiments/mprisk_core/tables/kl_rank_inversion.csv
```

Per-example arrays:

```text
outputs/experiments/mprisk_core/per_example_npz/
```

KL-vs-risk plots:

```text
outputs/experiments/mprisk_core/plots/kl_vs_mprisk/
```

---

## A.5 Main comparison table

File:

```text
main_mprisk_core_comparison.csv
```

Important columns:

```text
dataset
method
far
beta
prr_f1
base_f1_class
base_fnir
base_fpir
error_rate
false_accept_count
false_reject_count
misidentification_count
error_auroc
error_auprc
lambda_fa
lambda_id
lambda_fr
lambda_ns
```

Use this table for the main new-paper result.

Recommended paper table:

| Dataset | FPIR | HolUE | MPRisk raw | MPRisk no NS | MPRisk | MPRisk cal |
|---|---:|---:|---:|---:|---:|---:|

Metric:

```text
PRR for F1 filtering
```

---

## A.6 Error-type detection table

File:

```text
error_type_detection.csv
```

This evaluates whether each method detects different OSR error types.

Targets:

```text
any_error
false_accept
false_reject
misidentification
```

Metrics:

```text
AUROC
AUPRC
```

Important columns:

```text
dataset
method
far
target
positive_count
auroc
auprc
```

Use this table to support the claim:

> MPRisk decomposes uncertainty into interpretable OSR error mechanisms.

---

## A.7 Component ablation

File:

```text
mprisk_component_ablation.csv
```

Variants include:

```text
FA only
ID only
FR only
NS only
ordinary equal
full equal
ordinary current
MPRisk current
NS current
```

Important columns:

```text
variant
prr_f1
error_auroc
error_auprc
```

This experiment shows which risk terms are useful.

Expected interpretation:

- `FA only` should help false-acceptance detection.
- `ID only` should help ambiguous identity detection.
- `FR only` should help false rejection detection.
- `NS only` should help poor-quality confident reject detection.
- `MPRisk current` should combine these effects.

---

## A.8 KL rank-inversion experiment

File:

```text
kl_rank_inversion.csv
```

This experiment measures whether KL uncertainty agrees with risk-based uncertainty.

KL uncertainty is computed as:

$$
u_{\mathrm{KL}}(x) = -(\mathrm{KL}_1(x) + \mathrm{KL}_2(x)).
$$

The pairwise inversion rate is:

$$
\mathrm{Inv}(u,r)
=
\Pr[
(u_i-u_j)(r_i-r_j) < 0
].
$$

Important columns:

```text
kl_reference
spearman
rank_inversion_rate
kl_error_auroc
kl_error_auprc
```

Possible references:

```text
empirical_error
method_uncertainty
mprisk
ordinary_risk
```

This experiment supports the theoretical claim:

> KL is an information-gain quantity and is not generally monotone with OSR error risk.

---

# Part B. Tuning Experiments

## B.1 Purpose

The tuning experiments answer:

1. How much validation data is needed to tune MPRisk weights?
2. Do tuned weights transfer across FPIR operating points?
3. Do tuned weights transfer across datasets?
4. Is MPRisk sensitive to `predict_T`, `kappa_input_scale`, Monte Carlo sampling, or non-specificity mode?

## B.2 Config

```text
configs/uncertainty_benchmark/mprisk_tuning_experiments.yaml
```

Default:

```yaml
exp_dir: outputs/experiments/mprisk_tuning
far_list: [0.05]
beta_list: [0.5]
```

## B.3 Main outputs

```text
outputs/experiments/mprisk_tuning/tables/cache_summary.csv
outputs/experiments/mprisk_tuning/tables/validation_size_ablation.csv
outputs/experiments/mprisk_tuning/tables/validation_size_ablation_summary.csv
outputs/experiments/mprisk_tuning/tables/operating_point_transfer.csv
outputs/experiments/mprisk_tuning/tables/hyperparameter_sensitivity.csv
```

If enabled:

```text
outputs/experiments/mprisk_tuning/tables/cross_dataset_transfer.csv
```

Plots:

```text
outputs/experiments/mprisk_tuning/plots/
```

---

## B.4 Validation-size ablation

Files:

```text
validation_size_ablation.csv
validation_size_ablation_summary.csv
```

Config section:

```yaml
validation_size_ablation:
  enabled: True
  validation_fractions: [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
  repeats: 5
  min_subset_size: 100
```

This experiment tunes $\lambda$ on random subsets of the validation set.

Important columns:

```text
validation_fraction
repeat
subset_size
val_prr
test_prr_f1
lambda_fa
lambda_id
lambda_fr
lambda_ns
```

Use this experiment to show:

> MPRisk tuning is stable and does not require an excessively large validation set.

Recommended paper plot:

```text
x-axis: validation fraction
y-axis: test PRR
error bars: std over repeats
```

---

## B.5 Operating-point transfer

File:

```text
operating_point_transfer.csv
```

Config section:

```yaml
operating_point_transfer:
  enabled: True
  train_fars: [0.05]
  eval_fars: [0.01, 0.05, 0.1, 0.2]
```

This experiment tunes lambdas at one FPIR and evaluates them at other FPIRs.

Important columns:

```text
train_far
eval_far
test_prr_f1
lambda_fa
lambda_id
lambda_fr
lambda_ns
```

Use this experiment to answer:

> Should MPRisk weights be tuned separately for each operating point?

Possible interpretations:

- If diagonal values are best, tune per operating point.
- If weights tuned at FPIR 0.05 transfer well, use a single default operating-point calibration.
- If a global model is later added, compare it here.

---

## B.6 Cross-dataset transfer

File:

```text
cross_dataset_transfer.csv
```

Disabled by default. Enable after adding multiple datasets:

```yaml
cross_dataset_transfer:
  enabled: True
  fars: [0.05]
```

This tunes lambdas using one dataset's validation protocol and evaluates them on another dataset's test protocol.

Important columns:

```text
source_dataset
target_dataset
test_prr_f1
lambda_fa
lambda_id
lambda_fr
lambda_ns
```

Use this experiment to support:

> MPRisk components capture transferable error mechanisms.

---

## B.7 Hyperparameter sensitivity

File:

```text
hyperparameter_sensitivity.csv
```

Config section:

```yaml
hyperparameter_sensitivity:
  enabled: True
  variants:
    - name: "T_7"
    - name: "T_10"
    - name: "T_20"
    - name: "T_30"
    - name: "T_50"
    - name: "kappa_input_0.5"
    - name: "kappa_input_1"
    - name: "kappa_input_2"
    - name: "weighted_mc_M10"
    - name: "weighted_mc_M50"
```

Important columns:

```text
variant
test_prr_f1
val_prr
lambda_fa
lambda_id
lambda_fr
lambda_ns
```

Use this experiment to show robustness to:

- posterior temperature `predict_T`;
- embedding uncertainty scaling `kappa_input_scale`;
- analytic vs Monte Carlo non-specificity estimation.

Recommended default for final method:

```yaml
nonspecificity_mode: analytic_vmf
M: 0
```

because it is deterministic and fast.

---

# Part C. Diagnostics and Paper Artifacts

## C.1 Purpose

Diagnostics experiments produce evidence for:

1. necessity of the mixed prior;
2. reliability of calibrated error probabilities;
3. statistical significance of PRR improvements;
4. qualitative interpretability;
5. runtime overhead.

## C.2 Config

```text
configs/uncertainty_benchmark/mprisk_diagnostics_experiments.yaml
```

Default:

```yaml
exp_dir: outputs/experiments/mprisk_diagnostics
far_list: [0.05]
beta_list: [0.5]
```

## C.3 Outputs

Tables:

```text
outputs/experiments/mprisk_diagnostics/tables/runtime_overhead.csv
outputs/experiments/mprisk_diagnostics/tables/mixed_prior_necessity.csv
outputs/experiments/mprisk_diagnostics/tables/reliability_metrics.csv
outputs/experiments/mprisk_diagnostics/tables/bootstrap_prr_differences.csv
outputs/experiments/mprisk_diagnostics/tables/qualitative_examples.csv
```

Optional:

```text
outputs/experiments/mprisk_diagnostics/tables/quality_stress.csv
```

Plots:

```text
outputs/experiments/mprisk_diagnostics/plots/
```

---

## C.4 Mixed-prior necessity

File:

```text
mixed_prior_necessity.csv
```

Variants:

```text
collapsed_unknown_risk
ordinary_risk_no_NS
equal_full_risk
MPRisk_current
r_FA
r_ID
r_FR
r_NS
unknown_nonspecificity
```

The key comparison is:

```text
collapsed_unknown_risk vs ordinary_risk_no_NS vs equal_full_risk vs MPRisk_current
```

### Collapsed unknown risk

This treats the unknown class as a single reject action:

$$
R_{\mathrm{collapsed}}(x) = 1 - p(\hat a(x) \mid x).
$$

If the system rejects and $p(\text{unknown} \mid x)$ is high, this risk becomes low. Therefore, a poor-quality known sample that drifts away from the gallery can look confidently rejected.

### Mixed-prior non-specificity

The mixed-prior penalty is:

$$
r_{\mathrm{NS}}(x) = \pi_0(x)\mathcal N_0(x).
$$

It is high when:

1. aggregate unknown probability is high;
2. posterior over unknown identities is diffuse.

This helps detect poor-quality confident false rejects.

Important columns:

```text
variant
prr_f1
false_reject_auroc
any_error_auroc
```

Use this experiment to justify the mixed prior.

---

## C.5 Reject non-specificity group summary

The same CSV includes rows with:

```text
variant = r_NS_group_summary
```

Groups:

```text
false_reject
true_reject
false_accept
misidentification
correct
```

Important columns:

```text
group
count
mean
std
median
q90
q99
```

Expected result:

> $r_{\mathrm{NS}}$ should be larger for suspicious false rejects than for confident true rejects.

---

## C.6 Reliability metrics

File:

```text
reliability_metrics.csv
```

Reliability diagrams:

```text
outputs/experiments/mprisk_diagnostics/tables/reliability_plots/
```

Methods evaluated by default:

```yaml
methods:
  - "HolUE"
  - "MPRisk cal"
```

Metrics:

```text
ECE
Brier
NLL
error AUROC
error AUPRC
```

Important columns:

```text
method
ece
brier
nll
error_auroc
error_auprc
```

For probability-like methods, the repository convention is:

```text
predicted_unc = -P(correct)
```

Therefore the diagnostics convert:

$$
P(\mathrm{error}) = 1 + \mathrm{predicted\_unc}.
$$

Use this experiment to separate two ideas:

1. lambda tuning improves ranking for selective OSR;
2. scalar calibration improves probability reliability.

---

## C.7 Paired bootstrap significance

File:

```text
bootstrap_prr_differences.csv
```

Default comparisons:

```text
MPRisk      vs HolUE
MPRisk cal  vs HolUE
MPRisk      vs MPRisk raw
MPRisk      vs MPRisk no NS
```

Important columns:

```text
method_a
method_b
prr_a_full
prr_b_full
delta_full
ci95_low
ci95_high
p_delta_le_0
```

Use in the paper as:

```text
Delta PRR = +0.04, 95% CI [0.02, 0.06].
```

If the confidence interval excludes zero, the improvement is statistically stable under paired bootstrap resampling of probe templates.

---

## C.8 Qualitative examples

File:

```text
qualitative_examples.csv
```

This file identifies probe templates worth visualizing.

Important columns:

```text
probe_template_id
true_subject_id
predicted_subject_id
was_rejected
error_kind
score_name
score_value
r_fa
r_id
r_fr
r_ns
mprisk
oog_prob
oog_nonspecificity
top1_subject_id
top1_posterior_prob
top2_subject_id
top2_posterior_prob
...
```

Recommended use:

1. Select rows where `score_name = r_ns` and `group = false_reject`.
2. Copy corresponding images using `probe_template_id`.
3. Show examples of poor-quality known probes rejected as unknown.
4. Compare `collapsed_unknown_risk` and MPRisk.

This creates a strong qualitative figure for the mixed-prior argument.

---

## C.9 Runtime overhead

File:

```text
runtime_overhead.csv
```

Important columns:

```text
dataset
method
num_probes
elapsed_sec_total
elapsed_ms_per_probe
```

Use this to show that MPRisk has similar cost to HolUE because it reuses the same posterior computation.

Recommended table:

| Method | ms/probe | Notes |
|---|---:|---|
| HolUE | ... | KL components + calibration |
| MPRisk raw | ... | risk components |
| MPRisk | ... | risk components + tuned weights |
| MPRisk cal | ... | scalar calibration |

---

## C.10 Optional quality stress test

Disabled by default:

```yaml
quality_stress:
  enabled: False
```

When enabled, it reruns MPRisk with different values of:

```yaml
kappa_input_scale
```

This is an embedding-space stress test, not image corruption. Lower values make $p(z \mid x)$ more diffuse.

Output:

```text
quality_stress.csv
```

Important columns:

```text
kappa_input_scale
score
prr_f1
false_reject_auroc
true_reject_mean_score
false_reject_mean_score
```

Use this to show how mixed-prior non-specificity responds to degraded embedding quality.

---

# 6. Recommended paper experiment set

For a concise but strong paper, include the following.

## Main table

Use:

```text
outputs/experiments/mprisk_core/tables/main_mprisk_core_comparison.csv
```

Report PRR for F1 filtering across datasets and FPIR values.

Methods:

```text
SCF
AccScr
GalUE
HolUE
MPRisk raw
MPRisk no NS
MPRisk
MPRisk cal
```

## Component ablation

Use:

```text
outputs/experiments/mprisk_core/tables/mprisk_component_ablation.csv
```

Show:

```text
r_FA
r_ID
r_FR
r_NS
ordinary risk
full risk
tuned risk
```

## Error-type detection

Use:

```text
outputs/experiments/mprisk_core/tables/error_type_detection.csv
```

Report AUROC/AUPRC for:

```text
false acceptance
false rejection
misidentification
any error
```

## KL rank inversion

Use:

```text
outputs/experiments/mprisk_core/tables/kl_rank_inversion.csv
```

Report rank-inversion rate and Spearman correlation between KL uncertainty and MPRisk.

## Mixed-prior necessity

Use:

```text
outputs/experiments/mprisk_diagnostics/tables/mixed_prior_necessity.csv
```

Compare:

```text
collapsed_unknown_risk
ordinary_risk_no_NS
equal_full_risk
MPRisk_current
```

## Validation-size ablation

Use:

```text
outputs/experiments/mprisk_tuning/tables/validation_size_ablation_summary.csv
```

Show that tuned MPRisk is stable under reduced validation data.

## Reliability

Use:

```text
outputs/experiments/mprisk_diagnostics/tables/reliability_metrics.csv
```

Compare HolUE calibrated vs MPRisk cal.

## Bootstrap significance

Use:

```text
outputs/experiments/mprisk_diagnostics/tables/bootstrap_prr_differences.csv
```

Report confidence intervals for MPRisk vs HolUE.

## Runtime

Use:

```text
outputs/experiments/mprisk_diagnostics/tables/runtime_overhead.csv
```

Show that MPRisk is lightweight.

---

# 7. Suggested final commands

For a full final run:

```bash
# 1. Main comparison, component ablation, error detection, KL inversion
bash scripts/run_mprisk_core_experiments.sh

# 2. Validation/tuning robustness
bash scripts/run_mprisk_tuning_experiments.sh

# 3. Mixed-prior diagnostics, reliability, bootstrap, runtime
bash scripts/run_mprisk_diagnostics_experiments.sh
```

For debugging:

```bash
HYDRA_FULL_ERROR=1 python experiments/mprisk_core_experiments.py -cn=mprisk_core_experiments
HYDRA_FULL_ERROR=1 python experiments/mprisk_tuning_experiments.py -cn=mprisk_tuning_experiments
HYDRA_FULL_ERROR=1 python experiments/mprisk_diagnostics_experiments.py -cn=mprisk_diagnostics_experiments
```

---

# 8. Common issues and fixes

## 8.1 `nonspecificity_mode` error

If you see:

```text
ValueError: nonspecificity_mode must be 'analytic_vmf' or 'weighted_mc'
```

Check spelling. Correct:

```yaml
nonspecificity_mode: analytic_vmf
```

Not:

```yaml
analytic_vMF
analytic-vmf
```

## 8.2 Calibrated MPRisk underperforms tuned MPRisk

This is possible if calibration is not monotone or if the wrong calibrator is used.

Use:

```yaml
calibration_transform:
  _target_: evaluation.open_set_methods.calibration_methods.ScalarRiskCalibration
```

Do not use `NNcalibration` for the scalar tuned MPRisk score unless intentionally running an ablation.

## 8.3 Bootstrap is slow

Reduce:

```yaml
bootstrap:
  num_bootstrap: 100
```

for debugging, then use `1000` for the final run.

## 8.4 Tuning is slow

Reduce:

```yaml
lambda_search:
  num_random: 1024
```

or for quick debugging:

```yaml
lambda_search:
  num_random: 128
```

## 8.5 Weighted MC is slow

Use:

```yaml
nonspecificity_mode: analytic_vmf
M: 0
```

for the main method. Use `weighted_mc` only as an ablation.

---

# 9. Interpretation guide

The new paper should emphasize the following findings.

## 9.1 KL is not operational risk

If KL rank inversion is high or KL AUROC is worse than MPRisk, this supports:

> KL is an information-gain score, not an error-risk score.

## 9.2 MPRisk is interpretable

If different MPRisk components detect different error types, this supports:

> OSR uncertainty is multi-mechanism and should be decomposed by error type.

## 9.3 Mixed prior is necessary

If `r_NS` improves false-rejection detection and full MPRisk improves over collapsed unknown risk, this supports:

> The continuous unknown prior is not a cosmetic modeling choice. It is needed to detect poor-quality confident false rejects.

## 9.4 Tuning is justified

If tuned MPRisk improves over raw MPRisk and is stable with validation-size ablation, this supports:

> Validation tuning corresponds to operational cost selection for the deployment metric and FPIR point.

## 9.5 Calibration is separate from tuning

If MPRisk and MPRisk cal have similar PRR but tuned+cal has better ECE/Brier/NLL, this supports:

> Tuning determines ranking; scalar calibration improves probability interpretability.

---

# 10. Recommended figures for the paper

1. **Concept figure:** diagram of MPRisk components and mixed-prior unknown continuum.
2. **Main PRR table:** MPRisk vs HolUE and baselines.
3. **Component ablation bar plot:** PRR or AUROC by component.
4. **KL-vs-risk scatter:** KL uncertainty vs MPRisk, colored by error type.
5. **Mixed-prior necessity plot:** collapsed unknown risk vs no-NS vs full MPRisk.
6. **Validation-size plot:** validation fraction vs test PRR.
7. **Reliability diagrams:** HolUE vs MPRisk cal.
8. **Qualitative examples:** top false rejects with high `r_NS`.

---

# 11. Minimal final paper claim supported by these experiments

A conservative claim:

> MPRisk replaces the KL information-gain score with a decision-theoretic posterior risk decomposition. Across OSR benchmarks, validation-tuned MPRisk provides a more operationally aligned uncertainty ranking than raw KL-based uncertainty. The mixed-prior non-specificity term is especially important for detecting poor-quality probes that are confidently but incorrectly rejected as unknown.

A stronger claim, if results support it:

> MPRisk improves risk-controlled OSR filtering over HolUE while preserving interpretability and providing calibrated error probabilities through scalar post-hoc calibration.


