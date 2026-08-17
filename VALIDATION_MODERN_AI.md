# Modern-AI extension validation

This file records software validation only. It does **not** claim benchmark results.

Validated locally on the reconstructed repository:

```bash
python -m compileall -q .
pytest -q tests/test_modern_ai_*.py
for f in scripts/modern_ai/*.sh; do bash -n "$f"; done
bash scripts/modern_ai/run_smoke.sh
```

Final status:

- 14 modern-AI regression/integration tests pass.
- Python compilation passes for the reconstructed repository.
- All `scripts/modern_ai/*.sh` pass shell syntax validation.
- Offline smoke execution reaches retrieval, rewrite uncertainty, corpus scaling, tool routing, and RAG hybrid evaluation.
- The dense and exact-streaming deterministic posteriors are regression-tested for both power-spherical and vMF gallery likelihoods.
- Synthetic smoke metrics are deliberately not scientific results; the hashing embedder exists only to test the software stack without downloads.

Real BEIR/BRIGHT/RAGTruth/CRAG/BFCL results require the corresponding datasets and production embedding/generation models. The supplied flattened dump did not contain those assets, so no real benchmark scores are fabricated in this package.
