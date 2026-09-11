# Full-stage experiment correspondence

The full driver consolidates the original core, fair-tuning, tuning, and diagnostics
families. It does not invoke `modern_ai`. It does not generate misleading four-term
ablations by setting a nonexistent primary nonspecificity component to zero.

| Previous purpose | Output in this pipeline |
|---|---|
| Main method comparison | `main_mprisk_core_comparison.csv` (audit and test explicitly separated) |
| Comparison to simple and complete linear score sets | `fair_tuning_comparison.csv`, `fair_tuning_weights.csv`, per-case `score_fits.json` |
| Error-type detection | `error_type_detection.csv` and `error_type_detection_conditional.csv` |
| Component ablation | `mprisk_component_ablation.csv`: seven nonempty subsets of FA/ID/FR |
| KL versus risk rankings | `ranking_comparison.csv`, saved per-example KL features |
| Linear KL, KL+risk hybrid, supervised heads | Named rows in the comparison tables, explicit feature lists and fitted parameters |
| Validation-size/weight stability | `validation_size_ablation.csv`, `lambda_stability.csv` |
| Validation-size effect on probability fitting | `probability_fit_sample_size.csv`, per-case refit reports |
| Operating-point transfer | `operating_point_transfer.csv` |
| Cross-dataset transfer | `cross_dataset_transfer.csv`: weights/source normalization, locally fitted destination posterior |
| Hyperparameter sensitivity | `hyperparameter_sensitivity.csv`: probe scale, gallery dispersion, unknown prior; diagnostic only |
| Need for unknown-event modeling | `background_model_ablation.csv`: not a claim that a continuum cannot be marginalized |
| Approximation and normalization | `numerical_checks.csv`, `normalization_audit.csv`, `partition_numerics.json` |
| Prior recovery/high-quality limits | `probe_distribution_sanity.csv` |
| Calibration | `reliability_metrics.csv`, `reliability_bins.csv`, `posterior_scoring.csv` |
| Paired uncertainty/significance | `bootstrap_prr_differences.csv` with the bootstrap unit stated |
| Numerical risk-bound illustration | `audit_risk_bound.csv`, explicitly conditional and not certified under dependence |
| Quality and validation-to-test shift | `quality_shift.csv`, `quality_stratified.csv`, `quality_error_groups.csv` |
| Qualitative numerical examples | `qualitative_examples.csv` with sample IDs and model outputs |
| Runtime | `runtime_overhead.csv`, explicit inclusion/exclusion of FAR matching, encoder and similarities |
| Figures | `plots/`: F1, empirical risk, FPIR, FNIR and reliability |

All model comparisons use the same fixed decisions in each dataset/FPIR case.
The original raw inputs and old checkpoint files are not bundled into the output;
their file metadata and pooled-array fingerprints are retained instead.

Two previous studies do not translate literally:

1. MC convergence of HolUE is not an approximation choice of the evidence posterior:
   its vMF evidence has an explicit expression. Exact/interpolated evidence checks
   replace a mandatory M=0/M>0 MPRisk sweep. Published HolUE is kept as a baseline.
2. An r_NS-specific geometric interpretation is not a primary component of this
   estimator. Its optional addition remains a diagnostic comparison, with the exact
   feature defined in the score-fit record.

The concentration sensitivity table is not a raw-image/audio/text corruption study.
Re-encoding corrupted inputs would require a separate, domain-specific protocol.
