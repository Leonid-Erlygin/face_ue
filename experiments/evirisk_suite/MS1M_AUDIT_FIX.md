# MS1M placeholder-name audit correction

Apply `evirisk_suite_ms1m_audit_fix.patch` after the full-suite patch, from the
repository root (`/app` inside Docker, `~/face_ue` on the reported host).
Stop the active suite before editing a bind-mounted source tree.

## Cause

The supplied `MXFaceDataset.create_identification_meta` in
`training/dataset_classes/lightning_datasets.py` deliberately creates

```python
mids = np.arange(len(self.labels))
names = np.zeros_like(mids)
```

and writes `name template_id media_id subject_id` to
`ms1m_face_tid_mid.txt`. Therefore the name field contains `0` for all rows;
it is not an image identity. The IJB-C and IJB-B configurations both use the
MS1M identification set for validation.

The previous audit intersected the probe and gallery name fields, obtained
`['0']`, and reported a leak. This is a false positive on that metadata schema.
The diagnostic error is distinct from the NumPy underflow warnings.

## Correction and limits

Only dataset `ms1m` with an entirely zero name column, exactly four columns and
sequential media IDs `0 .. N-1` gets the documented-placeholder interpretation.
The audit still checks disjoint template IDs, metadata/export row indices and
MS1M media indices. It records the raw name-overlap count as **1**, and the real
source-name check as **unavailable**, not as zero. This is deliberately not an
end-to-end image-provenance certificate. Original RecordIO IDs are needed to
establish upstream image identities. All other datasets and real names retain
the strict shared-source-name check. Mixed/unsupported dummy layouts still fail.
No data, split, pooling output, probability formula, fitted parameter, score,
metric, candidate-selection rule or random seed is changed.

The evidence kernels now use a fresh local `np.errstate(under='ignore')` for
vanishing probability tails. Other NumPy error policies are unchanged and the
caller policy is restored afterwards. Explicit finite-value/input checks and
log-domain ranking scores remain. This does NOT set `np.seterr(all='ignore')`,
change the native sampler or suppress all warnings globally. Other modules can
still report their own warnings.

## Apply

Inside a stopped/restarted environment, at the repository root:

```bash
patch --dry-run -p1 < evirisk_suite_ms1m_audit_fix.patch
patch -p1 < evirisk_suite_ms1m_audit_fix.patch
```

Run tests in the project environment:

```bash
python3 -m pytest tests/mprisk_evidence/test_ms1m_audit_fix.py -q
```

## Resume the reported run without weakening source checks

```bash
python3 -u experiments/evirisk_full_suite.py \
  --out /app/outputs/evirisk_full_prr \
  --resume --resume-audit-fix
```

Keep the original Docker image, inputs, command settings and library versions.
`--resume-audit-fix` does NOT allow arbitrary code changes. It verifies the old
source snapshot against the archived digest, checks the exact old/new contents
of the four patched source files, and rejects any additional source changes.
Settings, input signatures, configs, validation partitions and library versions
must still match. Completed jobs must pass their original checksums. A narrowly
verified source transition is saved under `provenance_history/`, including the
old source snapshot, contracts and manifest; it is included in the final ZIP.
The root numerical protocol identifier is unchanged, because the accepted
patch changes checks, logging and warning handling, not computation.

Completed dataset jobs are retained. Failed or interrupted dataset jobs restart
from the beginning, as in the original resume behaviour; there is no promise
of resuming a half-finished optimizer/operating point. Old incomplete attempts
are retained under `_cache/previous_attempts` and excluded from combined tables.
After this one-time transition, ordinary `--resume` works again.

A stopped process must release `.suite.lock` normally. If Docker force-killed
it, confirm that no process is writing the output before removing a stale lock.
The new flag does not remove locks or bypass modified-result checks. Do not edit
`protocol.json` to make resume pass. When a non-supported code change is detected,
use a new output directory instead.

## Host command (reported setup)

Save the patch in `~/face_ue` before these commands. Stop the container only if
it is still running; the reported `--rm` removes the container on exit, not the
bind-mounted repository/output files.

```bash
docker stop -t 120 l.erlygin_face_ue_evi_risk
cd ~/face_ue
patch --dry-run -p1 < evirisk_suite_ms1m_audit_fix.patch
patch -p1 < evirisk_suite_ms1m_audit_fix.patch
cd scripts
```

Reuse the original `docker run` command with this Python invocation:

```bash
python3 -u experiments/evirisk_full_suite.py \
  --out /app/outputs/evirisk_full_prr --resume --resume-audit-fix
```

The next logs name the dataset, seed and report directory, including failed
jobs. The final result remains `/app/outputs/evirisk_full_prr.zip`.

## Tests executed for delivery

136 existing and added unit/regression tests passed locally. Added tests execute
the actual MS1M metadata writer method on synthetic labels, test real filename
leaks and unsupported sentinel layouts, check local numerical-error behaviour,
and exercise guarded migration/tamper rejection. A synthetic full-suite run was
continued across the actual old/new source snapshots. Completed jobs were skipped;
an interrupted job was restarted, and all 136 stored array comparisons were
exactly equal to the baseline. These tests are not new real IJB/Whale experiments.
The local test runtime was Python 3.13; the patch adds no dependencies or syntax
newer than Python 3.9. The user's actual Python 3.9 image was not available here.
