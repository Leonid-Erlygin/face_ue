# Full benchmark authorization under the existing validation protocol

## What this package is

This is **not a model patch**. It supplies the explicit review-resolution JSON files required
by `reference-prior-evidence-1.1-logodds` and thin wrappers around your existing full-stage
Docker launchers. The originals of `manifest.json` and `review_gates.json` have been copied
verbatim from the two reviewed 12 September sanity archives. Their hashes and the source
archive hashes are recorded in `provenance.json`. The original `hold` statuses have NOT
been changed. The resolution records your decision to benchmark the current protocol
while deferring validation-data improvements. It does not claim those issues are fixed.

In particular, the IJB clean-validation / low-quality-test discrepancy remains, as do sparse
validation support elsewhere and the reviewed boundary solutions. The results must retain
those limitations, including unfavorable results. They are not proof of calibrated risk,
no-extra-information sufficiency, or a universally correct class posterior. The existing
benchmark uses test unknown labels to establish FPIR operating points; model fitting and
calibration do not use those labels. The inspected sanity subsets are not untouched test
confirmation for later model changes.

## Run

Unzip `mprisk_full_run_approval.zip` in the repository root. Keep the experiment Python
files and core configuration YAML files unchanged from the corrected sanity run.

```bash
python3 mprisk_full_run_approval/verify_review_files.py

GPU_DEVICE=4 bash mprisk_full_run_approval/run_full_text.sh
GPU_DEVICE=5 bash mprisk_full_run_approval/run_full_bio.sh
```

Run sequentially instead when GPUs, RAM, or CPU capacity do not permit both containers.
Each container defaults to a 160g memory cap, 40 CPUs, and 16g shared memory. Two concurrent
runs therefore have combined configured caps of 320g and 80 CPUs, not guaranteed reservations.
`DOCKER_IMAGE`, `CPUS`, `MEMORY`, `SHM_SIZE`, and `THREADS` pass through unchanged.

```bash
DRY_RUN=1 GPU_DEVICE=4 bash mprisk_full_run_approval/run_full_text.sh
DOCKER_IMAGE=your_existing_image GPU_DEVICE=5 bash mprisk_full_run_approval/run_full_bio.sh
```

The equivalent direct invocation for text is:

```bash
APPROVE_FULL=1 \
SANITY_MANIFEST="$PWD/mprisk_full_run_approval/reviewed_sanity/text/manifest.json" \
REVIEW_RESOLUTION="$PWD/mprisk_full_run_approval/text_review_resolution.json" \
GPU_DEVICE=4 \
bash run_mprisk_evidence_full_text_docker.sh \
  --seed 777 --max-fit 3000 --fit-iterations 250 --search-budget 1024 --bootstrap 200
```

The bio invocation substitutes `bio` for `text`. The Python driver still verifies source
hash, configuration hash, domain, model, prior, numerical checks, and review metadata.
Do not edit hashes to get around a failed compatibility check.

## Scope and budgets

A pair of launches runs the full suite **once**, at global seed 777. The full driver does
not automatically run three independent global seeds.

- Text datasets: Yahoo, DBPedia, AG News, CLINC150, PAN; FPIR 0.10/0.20/0.30/0.40/0.50.
- Image/audio datasets: IJB-C, IJB-B, Whale, VoxBlink protocol large_12k-perspk5;
  FPIR 0.01/0.05/0.10/0.20.
- All test probes and galleries are used, unlike the sanity subsampling.
- Full validation data are loaded and repartitioned; the probability-model fit remains
  capped at 3,000 fit rows. Exact rows and partition IDs are saved.
- 250 optimization iterations per start; the existing convergence/retry policy remains.
- 1,024 random search candidates plus fixed/nested candidates; bootstrap=200.
- beta=0.5; empirical reference protocol; no test-driven parameter changes.

The suite includes baseline refits (including the existing fixed-decision HolUE comparison),
main PRR/error-detection/reliability tables, all three-risk-component subsets, fair fusions,
conditional error/calibration and quality-distance diagnostics, validation-size and
probability-fitting sample-size studies, parameter sensitivity, analytic limits, background
ablations, bootstrap comparisons, audit-bound diagnostics, within-domain cross-dataset
and operating-point score-weight transfer, timings, and plots. Cross-dataset transfer is of
score weights; it is not a refit-free transfer of all class/probe model parameters.

This is the full **risk-estimator** suite, not encoder training or an automatic rerun of
historical stand-alone SCF training, SCF geometric analysis, or HolUE Monte Carlo studies.
Those independent investigations retain their separate results.

For independent global-seed repeats, use the same wrappers with `SEED=778` and then
`SEED=779`, after the previous container on that GPU finishes. Every launch is detached:
do NOT launch a shell loop of seeds expecting it to wait. Save all repeat ZIPs. The same
approved source/protocol is used; seed changes also change validation partitions and
stochastic fitting, so the variation is not just Monte Carlo integration noise.

```bash
# Run after the seed-777 jobs finish:
SEED=778 GPU_DEVICE=4 bash mprisk_full_run_approval/run_full_text.sh
SEED=778 GPU_DEVICE=5 bash mprisk_full_run_approval/run_full_bio.sh
# Run after the seed-778 jobs finish:
SEED=779 GPU_DEVICE=4 bash mprisk_full_run_approval/run_full_text.sh
SEED=779 GPU_DEVICE=5 bash mprisk_full_run_approval/run_full_bio.sh
```

## Results to return

The launchers print a unique host directory, a detached-container log command, and a ZIP
path. The existing driver creates each adjacent ZIP after completion, including handled
failures. Typical paths:

```
outputs/mprisk_evidence/full_text_<run_id>.zip
outputs/mprisk_evidence/full_bio_<run_id>.zip
```

Send the whole ZIPs and keep their SHA-256 sidecars. The ZIPs contain all available tables,
per-probe predictions/log odds/probabilities, fitting artifacts, validation indices,
provenance, and logs. Regenerable cosine caches and raw source inputs are excluded.
`manifest.json` must say `status: complete`, with no failed datasets. `review_status` may
remain `hold` because the acknowledged scientific caveats still exist; it is not a
synonym for runtime failure. Send failed archives too when troubleshooting.

Leave method labels and filenames unchanged during this batch. A chosen presentation
name can later be mapped consistently in the dissertation, with the original published
method retained as a separately named comparator. Do not rename the implementation now:
its reviewed source hash is part of the full-run gate.

## Checks performed for this package

- The copied review metadata match the source ZIP bytes and hashes.
- Warning acknowledgements match every listed warning exactly; blockers are empty.
- Source hash reconstructed from the archived code matches both manifests.
- Shell syntax and existing-launcher dry-run command construction were tested locally.
- The existing Python full-stage guard was exercised with these review files and the
  archived source, stopping immediately after guard validation before data/experiments.
- Actual Docker/GPU/full benchmark execution was not performed here.
