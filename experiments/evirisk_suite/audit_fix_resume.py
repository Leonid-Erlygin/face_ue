"""One narrowly verified resume transition; never bypass the frozen contract.

Only the shipped MS1M-placeholder audit, local probability-underflow handling and
contextual logging changes are accepted. Stored and live code are both checked.
Model settings, split/grid, input signatures, library versions and all other
source files must be identical. Successful result inventories remain mandatory.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil

from experiments.evirisk_suite.protocol import code_files, code_digest
from experiments.mprisk_evidence.artifacts import sha256, write_json

# Filled from the exact tested patch pair, not from an arbitrary current checkout.
# Canonicalization tolerates trailing newline differences in the original export.
PATCH_PAIR = {'experiments/evirisk_suite/input_audit.py': {'old': 'dd5087c1ee7548f5efc4f349bc904b4485cffe8d4b8e31b73ca6b340752c32e6', 'new': '503a71dc23defb33bde83b55aff24929af1a2ff54686da563c64f7bbb9323098'}, 'experiments/evirisk_full_suite.py': {'old': '13f23db6de6b653bc41d1750f2df49c33227c7de9a37c1e629d7868e66752ade', 'new': 'e869bd10dc6053991fbc9976d6ed436169fdb725d253c46ed6917a69d8a80547'}, 'evaluation/open_set_methods/mprisk_evidence.py': {'old': 'd2fe03f014ca8cfcfc8f46fdc5e85cefe74b1842ca306562328c6baaa04af29c', 'new': '8f438e476b6dd22be71f58c35dce6342a4974f7f8f6ef9d4cbb561cc8437a4ce'}, 'experiments/mprisk_evidence/audits.py': {'old': '8428edc7400992c3cffd918cf463d9aff98ad621b931af3d1fef5717cc6ba9cd', 'new': '9f15b65efc2f5dcfcc4dc15ac4456c852d9cc507bcd2feaef8fe4d96204e33b7'}}
ADDED = 'experiments/evirisk_suite/audit_fix_resume.py'


def canonical_hash(path):
    return hashlib.sha256(Path(path).read_bytes().rstrip(b'\r\n') + b'\n').hexdigest()


def validate_migration(root, old, new, live_root):
    """Read-only check. Raises before changing protocol.json or completed results."""
    root, live_root = Path(root), Path(live_root)
    if {k:v for k,v in old.items() if k!='code_sha256'} != {
            k:v for k,v in new.items() if k!='code_sha256'}:
        raise ValueError('Audit-fix resume refused: settings, jobs/config, inputs or library versions changed.')
    source = root/'source'
    if code_digest(source) != old.get('code_sha256'):
        raise ValueError('Audit-fix resume refused: archived source does not match its original code digest.')
    if code_digest(live_root) != new.get('code_sha256'):
        raise ValueError('Audit-fix resume refused: live source changed after planning.')
    before = {str(p.relative_to(source)):p for p in code_files(source)}
    after = {str(p.relative_to(live_root)):p for p in code_files(live_root)}
    if set(before)-set(after) or set(after)-set(before) != {ADDED}:
        raise ValueError('Audit-fix resume refused: unexpected added/removed source files.')
    changed = {rel for rel in before if sha256(before[rel]) != sha256(after[rel])}
    if changed != set(PATCH_PAIR):
        raise ValueError('Audit-fix resume refused: changed source files do not match the audited patch: '+
                         ', '.join(sorted(changed)))
    for rel, pair in PATCH_PAIR.items():
        if canonical_hash(before[rel]) != pair['old'] or canonical_hash(after[rel]) != pair['new']:
            raise ValueError('Audit-fix resume refused: unsupported old/new contents of '+rel)
    # Reuse only completed, checksum-verified jobs. Failed jobs will restart.
    from experiments.evirisk_full_suite import verify_completed
    reusable = []
    for job in old['jobs']:
        key = f"seed_{job['seed']}/{job['dataset']}"
        dest = root/'runs'/key
        if (dest/'manifest.json').is_file():
            status = json.loads((dest/'manifest.json').read_text())
            if status.get('status') == 'complete':
                if not (dest/'completion_inventory.json').is_file() or not verify_completed(dest):
                    raise ValueError('Audit-fix resume refused: missing/invalid completed inventory: '+key)
                reusable.append(key)
    return dict(kind='ms1m-placeholder-audit-fix',
                reason='No changes to data, pooling, formulas, search, scores, metrics or random seeds.',
                changes={rel:dict(old_sha256=sha256(before[rel]),new_sha256=sha256(after[rel]))
                         for rel in sorted(changed)},
                added_source=ADDED,old_code_sha256=old['code_sha256'],
                new_code_sha256=new['code_sha256'],reusable_jobs=reusable,
                note='Incomplete jobs restart; completed jobs retain their original artifacts and code provenance.')


def record_migration(root, old, new, report):
    """Call after the suite lock is held and before replacing the root contract."""
    root = Path(root)
    if not (root/'.suite.lock').exists():
        raise ValueError('Audit-fix history requires the suite lock.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    dest = root/'provenance_history'/('ms1m-audit-fix-'+stamp)
    dest.mkdir(parents=True,exist_ok=False)
    # This history is deliberately outside _cache and is included in the result ZIP.
    shutil.copytree(root/'source', dest/'source_before')
    for name in ['manifest.json','protocol.json','RUN_PROTOCOL.md','coverage.json']:
        if (root/name).is_file(): shutil.copy2(root/name,dest/name)
    write_json(dest/'protocol_after.json',new)
    write_json(dest/'migration.json',dict(report,created_utc=stamp))
    return dest
