# Applying this extension to the original repository

This v2 overlay fixes the patch-generation issue caused by separator blank lines in the flattened `combined.txt` dump.

## Safe application

From the root of your real repository:

```bash
git status --short
git apply --check /path/to/MODERN_AI_EXISTING_FILES.patch
git apply /path/to/MODERN_AI_EXISTING_FILES.patch
```

The patch changes only these existing files:

- `setup.py`
- `evaluation/samplers.py`
- `evaluation/open_set_methods/kappa_utils.py`

Then copy the **new/additive** files and directories from the overlay into the repository. If you copy the whole overlay, the three files above are byte-equivalent to the post-patch versions, so copying them again is harmless provided your checkout still matches the supplied dump.

Install `requirements-modern-ai.txt`, then follow `VALIDATION_MODERN_AI.md`.

Run `scripts/modern_ai/run_smoke.sh` first, followed by `scripts/modern_ai/run_beir_oser.sh`. The fitted gallery kappa from OSER is intentionally transferred into RAG experiments rather than fitted on hallucination labels.

## If `git apply --check` still fails

Do not force-apply. Run:

```bash
git diff -- setup.py evaluation/samplers.py evaluation/open_set_methods/kappa_utils.py
```

If those files contain local edits made after the supplied dump, merge the three small changes manually or regenerate a patch against the newer checkout.
