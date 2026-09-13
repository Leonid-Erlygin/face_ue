#!/usr/bin/env python3
"""Validate copied review metadata without editing it or importing experiment code."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    provenance = json.loads((root / 'provenance.json').read_text())
    for domain in ('text', 'bio'):
        d = root / 'reviewed_sanity' / domain
        manifest_bytes = (d / 'manifest.json').read_bytes()
        gates_bytes = (d / 'review_gates.json').read_bytes()
        meta = provenance[domain]
        for data, key in ((manifest_bytes, 'manifest_sha256'), (gates_bytes, 'review_gates_sha256')):
            if hashlib.sha256(data).hexdigest() != meta[key]:
                raise SystemExit(f'{domain}: original review metadata was modified ({key}).')
        m, g = json.loads(manifest_bytes), json.loads(gates_bytes)
        r = json.loads((root / f'{domain}_review_resolution.json').read_text())
        if m['status'] != 'complete' or m['stage'] != 'sanity' or m['domain'] != domain or m['synthetic']:
            raise SystemExit(f'{domain}: invalid sanity manifest.')
        if g.get('blockers'):
            raise SystemExit(f'{domain}: technical blockers remain: {g["blockers"]}')
        if set(r.get('acknowledged_warnings', [])) != set(g.get('review_warnings', [])):
            raise SystemExit(f'{domain}: warning acknowledgement mismatch.')
        if r.get('sanity_code_sha256') != m['code_sha256'] or not r.get('rationale', '').strip():
            raise SystemExit(f'{domain}: invalid review resolution.')
        if r.get('raw_probability_calibration_certified') is not False:
            raise SystemExit(f'{domain}: this authorization must not certify probability calibration.')
        print(f'{domain}: review files valid; {len(g["review_warnings"])} warnings acknowledged, not removed.')
    print('The experiment driver will independently check current code and core-configuration hashes.')

if __name__ == '__main__':
    main()
