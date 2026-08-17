# Repository reconstruction note

The supplied artifact was `combined.txt.zip`, a flattened textual repository dump rather than an original Git checkout/archive.

The reconstruction script recovered every file body delimited by the dump's `# === ./path ===` markers (139 source/config files). The tree printed at the end of the dump listed additional paths whose file bodies were not present in the supplied artifact. Those absent bodies cannot be reconstructed faithfully and were not invented.

One malformed consequence of the flattened dump was an appended tree listing after `utils/reliability_diagrams.py`; that non-Python tail was removed during reconstruction.

The modern-AI extension was implemented on top of the recovered code and reuses `MPRiskPredictiveProb`, its HolUE KL implementation, `kappa_utils`, and the existing vMF sampler rather than reimplementing the dissertation equations independently. `setup.py` was changed from a fixed two-package list to `setuptools.find_packages()` so nested packages such as `evaluation.modern_ai` are installable.
