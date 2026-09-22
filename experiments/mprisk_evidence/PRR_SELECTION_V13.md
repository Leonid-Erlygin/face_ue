# EviRisk: retain converged probability fits and select by validation PRR

## What changed

The old runner used the smallest multiclass validation NLL to discard all but one
converged probability model before it fitted the three risk coefficients by PRR.
This is appropriate for an NLL-selected probability estimator, but is a regression
in the requested PRR-optimized ranking procedure. On the supplied Whale run the
other converged finite-q fit already provides a strong ranking.

The patch keeps the NLL optimizer and all original converged starts unchanged.
It selects the finite probability model and its three nonnegative risk weights
jointly by PRR on the existing validation selection subset. Unit weights and the
old NLL incumbent are explicitly included. NLL is used only to resolve exact PRR
ties. There is no change to the evidence integral, priors, class-density family,
concentration decoding, pooling, fixed recognizer, or temperature. No point-model
fallback, NS term, test-selected hyperparameter, or forced sign reversal is added.

**This is a change to the final model-selection objective, not a mathematical
correction to maximum likelihood.** The PRR-selected candidate may have worse
multiclass NLL. Absolute probability calibration must still be reported separately.

Candidate parameters use only the fit subset. Model choice and weights use only
the selection subset. The audit subset is unused for selection. Final model and
weights are written to disk before their test predictions are evaluated. The
existing reference-benchmark FPIR convention (using unknown test labels to set the
operating point) is retained and separately identified, not advertised as a
prospective deployment threshold.

The selection search has `candidate_count * search_budget` randomized cost
proposals, plus its seeds, not `search_budget` in total. This extra model-selection
capacity must be disclosed in comparisons. Only selection PRR is guaranteed not to
drop below the included NLL incumbent; test improvement is not guaranteed.

## Apply from /app

This patch is based on the v10 Whale diagnostics **after**
`whale_v12_verification.patch`. It does not re-patch the already corrected metric.

```bash
cd /app
patch --dry-run -p1 < evirisk_whale_prr_selection.patch
patch -p1 < evirisk_whale_prr_selection.patch
```

## Rerun, reusing the existing likelihood optimization

```bash
cd /app
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python experiments/evirisk_whale_diagnostics.py \
  --whale-root /app/datasets/whale \
  --val-root /app/datasets/whale_val \
  --out /app/outputs/whale_evirisk_prr_selection \
  --reuse-probability-fit /app/outputs/whale_evirisk_v10 \
  --probability-selection validation-prr \
  --fpirs 0.01 0.05 0.1 0.2 \
  --seed 777 --beta 0.5 --device cpu \
  --search-budget 1024 --bootstrap 200 \
  --include-holue --include-mprisk
```

`--reuse-probability-fit` expects the extracted completed run directory, not its
ZIP. It validates input fingerprints and the validation partition before reusing
all converged candidates. It does not hard-code the successful Whale parameters.
The SCF/embedding model is never retrained. Cost weights and optional HolUE fusion
are refitted on validation. Omitting `--reuse-probability-fit` reruns the original
likelihood optimizer first. `--probability-selection multiclass-nll` retains the
historical model-selection rule as an explicit experimental mode.

`--include-mprisk` uses the original power-spherical M=0 posterior and separately
retunes its three-component and four-component scores on the same selection
objects. The fourth component is the original NS penalty, not a new probability.
It is never passed as a disjoint event to the EviRisk loss function. The
four-component candidate set includes the selected three-component vector with
zero NS weight. These references use the original repository dependencies, just
like `--include-holue`.

The MPRisk reference is **not claimed to reproduce the paper table exactly**:
its common selection subset and declared search budget are those of this rerun.
Temperature and class-density settings come from the original repository config
and are recorded. Original MPRisk and GalUE-vMF remain distinct comparators.

## Important outputs

- `tables/probability_selection.csv`: every converged candidate, selection NLL,
  unit PRR, weighted PRR, and selection indicator, for each FPIR.
- `fpir_*/evidence_selection.json`: objective, selection rows, candidate history,
  selected parameter vector, and weights.
- `fpir_*/selected_probability_model.json`: the actual per-FPIR main model.
- `fpir_*/weights.json`: the actual selected three-component weights.
- `tables/main_comparison.csv`: main EviRisk, its unit control, the old NLL-selected
  weighted/unit controls, GalUE-vMF, HolUE and optional original MPRisk references.
- `tables/conditional_errors.csv`: includes within-rejection AUROC.
- `tables/bootstrap.csv`: paired differences, with EviRisk as the reference.
- `<out>.zip`: complete portable record, excluding the full similarity cache.

The root `probability_model_fit.json` deliberately preserves the old optimizer
report and its NLL incumbent. It is a **candidate-generation record**, not the
final model specification; use the per-FPIR files above. The v12 input-preflight
script is updated to honor that distinction when replaying new results.

## Optional retrospective preview from saved predictions

No raw embeddings or fitting are needed for this diagnostic:

```bash
python experiments/whale_verification/preview_evirisk_candidate.py \
  --results /app/outputs/whale_evirisk_v10 \
  --candidate-index 0 \
  --out /app/outputs/whale_candidate0_preview
```

This script inverts the original Bayes factors to recover the saved top-five
cosines, checks their maximum against AccScr and the saved threshold, and
bounds every omitted class by the fifth-largest cosine. Connected score-interval
groups give conservative PRR bounds for **unit** costs. The bounds are conditional
on the numerical inversion; a log-odds padding of 1e-4 is a sensitivity allowance,
not formally certified floating-point interval arithmetic.

It is a retrospective, truncated-gallery diagnosis. Do not paste its outputs into
main EviRisk benchmark tables or treat them as a full weighted rerun. The production
runner above always evaluates all classes and selects among all converged fits,
not among manually chosen top-five previews.

## Tests

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest -q tests/mprisk_evidence
```

Tests cover ranking-vs-NLL selection, unit/incumbent inclusion, converged-only
candidates, prevention of point/prior substitution, fit-selection separation,
unread non-selection outcomes, original metric RNG ordering, the nested MPRisk
reference, and omitted-tail ordering bounds. Synthetic end-to-end runs validate
both fresh fitting and reuse. Real Whale full-matrix performance after this patch
must come from the user-side run; it is not pre-filled in the code.
