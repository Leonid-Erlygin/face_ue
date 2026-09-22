# Whale verification (v12)

This directory is installed by `whale_v12_verification.patch`, applied from the
repository root (`/app`). There is no custom Python patch installer.

## Included changes

- `experiments/mprisk_evidence/metrics.py`: restores the random/oracle draws and
  tied-score sorting used by `experiments/mprisk_core_experiments.py` for PRR.
- `whale_input_preflight.py`: compares raw inputs, metadata order, original and
  adapter pooling, cached arrays, class mapping, recognition decisions, and
  selected posterior predictions. Reports sample-level and template-level SCF
  correlations separately.
- `verify_saved_results.py`: compares saved scores against the original PRR
  implementation, with optional high-precision vMF checks.
- `source_loader.py`: loads selected definitions from your trusted local source.
- `test_verification.py`: regression tests using synthetic input fixtures and the
  saved v10 Whale predictions.

Applying the patch does not run training or change datasets, caches, fitted
parameters, or saved results. The metric change affects subsequent evaluations
and any future fitting procedure that uses this PRR implementation. Existing
results are not rewritten; do not mix the two PRR conventions in one table.

## Apply from /app

Save the patch as `/app/whale_v12_verification.patch`, then:

```bash
cd /app
git apply --check whale_v12_verification.patch
git apply whale_v12_verification.patch
```

The existing-file hunk targets the metric implementation in the supplied v10
source snapshot. If you already applied the previous optional PRR fix, omit that
already-applied hunk when installing the verifier:

```bash
git apply --check --exclude=experiments/mprisk_evidence/metrics.py whale_v12_verification.patch
git apply --exclude=experiments/mprisk_evidence/metrics.py whale_v12_verification.patch
```

Use the exclusion only if that metric fix is already installed. Do not force an
unexplained failed patch or use `--reject`.

## Input verification: no training

The results directory must be extracted and contain `manifest.json`, `source/`,
`data_manifest.json`, and `fpir_*/test.npz`. Use the existing project environment
(Python 3.9+, NumPy, SciPy, pandas, scikit-learn).

```bash
cd /app
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python experiments/whale_verification/whale_input_preflight.py \
  --repo /app \
  --results /app/outputs/whale_evirisk_v10 \
  --whale-root /app/datasets/whale \
  --val-root /app/datasets/whale_val \
  --out /app/outputs/whale_v12_input_preflight
```

The default cache root is `/app/cache/template_cache_new_v2`; pass
`--cache-root` if yours differs. Caches are inspected, not overwritten.

Send back `validation.json`, `test.json`, and `summary.json` from the output
folder. Failed checks return a nonzero exit status and must be investigated before
another experiment. Matching metadata lengths or matching name lists cannot prove
the original export order when the embedding file has no row IDs. The reports
retain this limitation even when the numerical comparisons pass.

## Verify saved results independently

```bash
python experiments/whale_verification/verify_saved_results.py \
  --repo /app \
  --results /app/outputs/whale_evirisk_v10 \
  --out /app/outputs/whale_v12_saved_verification
```

Add `--gold` for the 70-digit vMF checks (requires `mpmath`). Neither verification
script fits models or chooses parameters. Selected definitions are executed from
your trusted source tree; this is not a sandbox for untrusted code.

## Regression tests

```bash
WHALE_AUDIT_REPO=/app \
WHALE_AUDIT_RESULTS=/app/outputs/whale_evirisk_v10 \
python experiments/whale_verification/test_verification.py
```

The input-parity and deliberate misalignment checks use synthetic fixtures, not
real Whale inputs. The metric test uses the supplied saved predictions. Passing
these tests does not certify the full real-data input pipeline.
