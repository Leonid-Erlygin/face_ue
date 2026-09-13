# Incremental sanity correction: evidence MPRisk 1.1

## Target code — important

This patch targets the **separate-domain runner** that generated:

- `sanity_text_20260911T063537Z_3034090.zip`
- `sanity_bio_20260911T063622Z_3036102.zip`

Its base model version is `reference-prior-evidence-1.0`. Its Docker wrappers are
`run_mprisk_evidence_sanity_text_docker.sh` and
`run_mprisk_evidence_sanity_bio_docker.sh` in the repository root.

It is NOT an overlay for the subsequently described combined-domain kit with
`scripts/mprisk_evidence/run_sanity_docker.sh`. Both older deliveries had similarly
named patch files; the actual source in your result ZIPs is the authority here.
Do not reapply either old full patch. Use `git apply --check` first. A separate
preflight script in the delivery checks the exact expected runtime source hashes.
Do not use `--reject` or force a failed patch check.

## Install

From the repository root, with the incremental patch in that directory:

```bash
python3 check_mprisk_incremental_base.py .
git apply --check mprisk_evidence_sanity_incremental_v1_1.patch
git apply mprisk_evidence_sanity_incremental_v1_1.patch
python3 -m pytest -q tests/mprisk_evidence
```

The preflight script is shipped alongside the patch, not a new runtime dependency.
The patch adds no external dependency. Runtime continues to use NumPy, SciPy,
scikit-learn, pandas, PyTorch, OmegaConf, Matplotlib and mpmath. Pytest is for tests.
The original probabilistic model and beta=0.5 default are unchanged. The model
version becomes `reference-prior-evidence-1.1-logodds` to distinguish its output schema.

## Recommended second sanity run

Retain the first-run ZIPs. Set PREVIOUS_RUN to the ZIP or its original unpacked
run directory. It can be outside the repository; the launcher mounts it read-only
when an additional mount is necessary.

```bash
PREVIOUS_RUN="$PWD/outputs/mprisk_evidence/sanity_text_20260911T063537Z_3034090.zip" \
GPU_DEVICE=4 \
bash run_mprisk_evidence_sanity_text_docker.sh

PREVIOUS_RUN="$PWD/outputs/mprisk_evidence/sanity_bio_20260911T063622Z_3036102.zip" \
GPU_DEVICE=5 \
bash run_mprisk_evidence_sanity_bio_docker.sh
```

Use available GPUs (or run sequentially on one GPU). `GPU_DEVICE=cpu` removes the
GPU request. `DOCKER_IMAGE`, `HOST_APP_DIR`, `CPUS`, `MEMORY`, `SHM_SIZE`, and
`THREADS` overrides remain supported. `DRY_RUN=1` prints the Docker command.
The default resources are still 40 CPUs, 160 GB RAM, 16 GB shared memory, and
UID:GID from `id`, with the repository mounted at `/app`.

Do not use `--resume` on the old run: source and schema have changed. A fresh
run directory and adjacent ZIP are created automatically. A same-code interrupted
1.1 run can still be resumed with its original settings.

The equivalent local/container Python command is:

```bash
python3 experiments/mprisk_evidence_experiments.py \
  --stage sanity --domain text --device auto \
  --previous-run /path/to/old_text_sanity.zip
```

### Two phases inside each run

1. **Numerical-only replay.** The old evidence and point-model parameters, old
   decisions, original selected probes, and original validation partitions are
   reused. Primary scores are recalculated in log space. No parameter or calibrator
   is refitted in this phase. Results are explicitly labeled replay. The old
   DBPedia achieved FPIR (approximately 0.159) remains unchanged here.
2. **Corrected sanity.** The same data subsets and partitions are used. The
   parameter fits, threshold matching, score fitting and calibration use the
   corrected implementation. The benchmark recognition threshold may change;
   it is frozen across all compared methods. This is NOT labeled numerical-only.

Checks compare data fingerprints, IDs, targets, index arrays, beta and seed with
the old run. The replay also compares recomputed known-class log evidence against
the archived values. A mismatch fails rather than silently comparing other data.
The same raw SCF/protocol files and the same core configuration are required.
Archives contain data, not executable input: archived Python and pickled models
are not imported or loaded. NPZ reads use `allow_pickle=False`.

For a cheaper replay alone (still requires the original representation files):

```bash
PREVIOUS_RUN=/path/to/old_sanity.zip REPLAY_ONLY=1 GPU_DEVICE=4 \
  bash run_mprisk_evidence_sanity_text_docker.sh
```

A replay-only result cannot authorize the full stage.

## Numerical correction

With unnormalized class log weights `ell` and the fixed action `a`:

```python
log_wrong = logsumexp(ell_except_a)
score_log_odds = log_wrong - ell_a
risk_probability = expit(score_log_odds)
```

Ranking metrics use `score_log_odds`. Brier score, mean risk and reliability bins
use `risk_probability`. Binary NLL uses softplus of the original log odds, not a
clipped probability. Per-class NLL likewise uses the true class vs. its complement
in log space. Multiclass Brier is evaluated without subtracting nearly equal terms.
Probabilities can inevitably round to 0/1; that is no longer allowed to erase rankings.

For positive weighted components, the code keeps logarithms of the disjoint
`correct, FA, ID, FR` masses. It computes log odds of weighted loss divided by the
maximum cost, forming both numerator and complement explicitly. Thus unit costs
recover the primary score, and very small probabilities are not exponentiated
before weights are fitted. Zero-cost component ablations can have exact infinite
scores; the metrics handle their ties. General linear-fusion inputs must be finite,
so tuning anchors are strictly positive (minimum relative anchor cost 1e-12);
exact zero costs remain available in the explicit component ablations.

Validation score calibration is fitted to the original unsaturated log odds after
selection-only standardization. Calibrated probability and ranking score are saved
separately. Ranking is retained from the raw score, avoiding sigmoid saturation.
If selection contains only one error label, calibration is marked non-identifiable;
a Jeffreys-smoothed constant is reported for probability diagnostics, not passed off
as a successful fitted calibrator. The same happens on calibration optimization failure.

References for the numerical tools (not experimental claims):
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log_expit.html
- https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html

## FPIR construction

The historical golden-section search was applied to a non-monotone/discrete
objective and had a 1e6 upper concentration limit. It can miss a valid operating
point. Corrected runs first select an empirical cosine threshold directly.

For N unknown probes, the requested acceptance budget is floor(target_FPiR*N).
All ties receive the same decision. The policy is the largest attainable number
of acceptances no larger than that budget. Target, achieved value, accepted counts,
and any unavoidable tie shortfall are reported. We do not randomize using labels.

To construct the historical probability baselines, all sign-changing concentration
roots matching this threshold are searched on a fixed [1, 1e8] logarithmic bracket;
the largest root is selected deterministically (the high-concentration branch),
not using accuracy or PRR. A stable beta-function identity is used for the
power-spherical log normalizer. If no exact matching root exists, the approximation
and residual are explicitly flagged; direct threshold decisions remain authoritative.
The evidence-model fitting bounds are NOT expanded.

Validation reference thresholds use the validation-fit partition only. Test
thresholds retain the original **test-unknown-label benchmark convention**. This
is not a prospective, validation-only guarantee of test FPIR. It is explicit in
each manifest and `reference_operating_point.csv`.

## Parameter fitting and scientific checks

- Default fitting budget: 250 iterations per start. A nonconverged optimizer is
  retried once from its terminal point with 4x the budget, for every dataset.
- Only finite, converged candidates can be selected by selection NLL. If none
  converges, the attempts are saved and the dataset fails explicitly.
- Bounds remain as before. Boundary solutions are not automatically discarded
  or silently expanded. One-coordinate validation slices (0.1, 0.3, 1, 3, 10)
  and matched-row limiting-model comparisons are saved for review. They are not
  hyperparameters selected on test. Slices are not reoptimized profile likelihoods.
- Limiting comparisons include the selected finite model, uniform probe,
  point probe and point gallery, on the same fit/select/audit rows.
- Fit/select/audit/test error counts are exported for FA, FR and ID. Counts below
  ten in selection/audit generate triage warnings, not statistical certificates.
- Conditional probability calibration for accepted and rejected decisions is
  reported separately. The quality-by-similarity grid uses validation-fit
  covariates for its bin edges and preserves empty/sparse cells.

**No new low-quality validation examples are created.** The MS1M-to-IJB support
and distribution-shift problem is not fixed by a numerical patch. Zero false
rejections in validation are reported as such. No test labels are used to fit
an evidence model, calibrator, concentration scale, prior, or score weights.
Inspected test subsets should not be described as untouched confirmation of
subsequent model-development choices.

## Fusion and NS comparisons

`score_fits.json` contains the exact feature registry. The larger baseline now
includes the four historical risk components, the two HolUE KL features, and all
implemented standalone/pre-fitted baseline scores. It does not recursively include
itself or infinitely many other possible learned combinations.

Larger linear families evaluate the exact previously fitted smaller-model
predictions as candidates. If a nested model wins, it is reapplied directly to its
original columns. The selection objective therefore cannot be below that candidate;
this is NOT a guarantee of better test performance.

The NS increment keeps the fitted three-component weights fixed and varies only
the NS coefficient on a declared grid containing zero. Both original concentration
and fitted concentration scales are reported under distinct diagnostic names. The
zero-NS candidate uses the exact original core score. This is a controlled extra-
feature test, not a redefinition of the primary method or an assertion that NS is
an additional error probability.

Historical reconstructed risk scores are also evaluated in log space. Historical
paper-form rows and their retuning are therefore recomputed controls, not exact
reproductions of old numerical tables. The untouched old outputs remain in replay.

## Results to send

Send the entire two new ZIPs printed by the launchers. Each contains:

```
manifest.json
review_gates.json
file_inventory.json
run.log
reproducibility/
  source/
  protocol/
  environment.json
  resolved_core_config.yaml
numerical_replay/
  datasets/<dataset>/
    previous_probability_model_fit.json
    previous_reference_decision_fit.json
    previous_data_manifest.json
    previous_validation_split.json
    validation.npz
    test.npz
    status.json
tables/
  numerical_replay_comparison.csv
  numerical_replay_conditional.csv
  main_mprisk_core_comparison.csv
  error_type_detection_conditional.csv
  conditional_calibration.csv
  reference_operating_point.csv
  normalization_audit.csv
  validation_support.csv
  validation_parameter_slices.csv
  validation_model_limits.csv
  calibration_fit.csv
  quality_distance_grid.csv
  quality_distance_fr_detection.csv
  fair_tuning_comparison.csv
  fair_tuning_weights.csv
  ...
datasets/<dataset>/
  validation_split.json
  probability_model_fit.json
  evidence_fit_attempts.json
  point_fit_attempts.json
  fpir_0.1/
    reference_decision_fit.json
    score_fits.json
    fusion_features.npz
    per_example/validation.npz
    per_example/test.npz
```

In per-probe NPZ files, `scores` contains ranking scores. `probabilities` is a
separate matrix aligned with `probability_names`, with available calibrator log
odds in `probability_log_odds`. Never compute Brier from `scores` merely because a
row is named MPRisk. Structural inactive event log probabilities are -inf.

Ordinary failures are packaged as failed results. Recovery after a hard kill
continues to use `collect_mprisk_evidence_results.sh RUN_DIRECTORY --stopped`.
Original data and regenerable cosine caches are not copied into the ZIPs.

## Full-stage guard

A completed run is not automatically a scientifically approved run. The full
stage requires the matching reviewed source hash, domain, prior, configuration,
reference protocol and dataset coverage. A replay-only run is insufficient.
Technical blockers prevent promotion. Review warnings additionally require an
explicit JSON resolution with the matching `sanity_code_sha256`, exactly the
`acknowledged_warnings` listed in `review_gates.json`, and a nonempty scientific
`rationale`. This is a recorded review decision, not automatic validation of a
model. Do not fill it with a placeholder merely to bypass review. Keep full runs
on hold until these next sanity ZIPs have been examined.
