# SCF geometric calibration and HolUE ablations

This experiment tests whether SCF concentration has the geometric scale implied by
its vMF objective, and whether correcting that scale reduces HolUE's dependence on
error-supervised post-hoc fusion.

It intentionally uses only datasets already present in the dissertation:

- text: Yahoo Answers, DBPedia, AGNews, CLINC150, PAN;
- biometric/speaker: IJB-B, IJB-C, Whale, VoxBlink.

No tool-routing or other new benchmark is used.

## Geometric target

For a raw probe sample with normalized embedding direction `mu_x`, SCF
concentration `kappa_x`, and an independently enrolled true-class gallery proxy
`g_y`, the experiment evaluates

```text
A_d(kappa_x)  versus  mu_x^T g_y,
```

where `A_d(kappa)=I_{d/2}(kappa)/I_{d/2-1}(kappa)` is the vMF mean-resultant
length.

A single positive scale `s` is fitted on the validation protocol by minimizing
vMF negative log-likelihood:

```text
L(s) = mean_i[-log C_d(s*kappa_i) - s*kappa_i*cos_i].
```

The derivative is

```text
L'(s) proportional to sum_i kappa_i [A_d(s*kappa_i) - cos_i],
```

so an interior optimum is unique. The scale is fitted once per dataset/model and
then frozen for every FPIR operating point. Recognition-error labels are not used
in this fit.

The primary fit is sample-level because the SCF stationarity relation is a
sample-level statement. The runner also reports template-level diagnostics because
HolUE consumes pooled template concentration. Set
`geometric_calibration.fit_level=template` only as a sensitivity analysis.

## HolUE variants

At each existing dissertation FPIR, the same recognition protocol is evaluated
with four variants:

1. `HolUE raw supervised`: raw SCF kappa + the existing error-supervised fusion;
2. `HolUE direct`: raw SCF kappa + `-(KL1+KL2)`, no error-supervised fusion;
3. `HolUE-GC direct`: geometrically scaled kappa + `-(KL1+KL2)`, no error-supervised fusion;
4. `HolUE-GC supervised`: geometrically scaled kappa + the existing supervised fusion.

For the text config, the existing supervised transform is the HolUE MLP. For the
biometric config, the existing supervised ExpTransform is retained so the
comparison stays aligned with the current dissertation benchmark.

All four variants keep `M: 0`, matching the current dissertation HolUE setup. In
that deterministic approximation the probe kappa does not change the sampled
embedding or the mixed posterior/OSR decision; it enters the HolUE uncertainty
through `KL2`. Consequently the geometric-scale ablation changes the uncertainty
model while keeping the base recognition decision fixed.

The experiment deliberately does **not** compare SCF probe kappa numerically with
`gallery_kappa`. The dissertation configs use a vMF probe uncertainty model but a
Power-Spherical gallery prior, so the two raw concentration parameters belong to
different distribution families and are not directly commensurate.

## r_NS diagnostic

For rejected probes the runner also recomputes

```text
N_d(kappa) = C_d(2*kappa) / [S_{d-1} C_d(kappa)^2]
r_NS = p0 * N_d(kappa)
```

with raw and geometrically calibrated kappa while keeping the same M=0 posterior
and OSR decision. It reports false-reject vs true-reject AUROC overall and in the
high-`p0` strata where the collapsed unknown posterior already strongly supports
rejection. This directly tests whether non-specificity adds information beyond the
ordinary collapsed false-rejection term `r_FR = 1-p0`.

## Run

Text:

```bash
bash scripts/run_scf_geometry_holue_text.sh
```

Biometric/speaker:

```bash
bash scripts/run_scf_geometry_holue_bio.sh
```

The configs are:

```text
configs/uncertainty_benchmark/scf_geometry_holue_text.yaml
configs/uncertainty_benchmark/scf_geometry_holue_bio.yaml
```

## Main outputs

```text
outputs/experiments/scf_geometry_holue_*/
  tables/
    scf_geometric_calibration.csv
    geometric_scale_manifest.json
    holue_geometry_variants.csv
    holue_geometry_deltas.csv
    r_ns_geometry_diagnostics.csv
  geometry/<dataset>/
    calibration_sample_points.csv
    calibration_template_points.csv
    test_sample_points.csv
    test_template_points.csv
    *_reliability_raw.csv
    *_reliability_geometric.csv
    *_reliability.png
    *_reliability.pdf
  curves/<dataset>/<far>/<variant>/
    curve.csv
    random_curve.csv
    oracle_curve.csv
  per_example_npz/
  r_ns_per_example/<dataset>/
```

`geometric_scale_manifest.json` also contains an oracle test scale fitted only as a
diagnostic. It is never used to generate HolUE test results. Its ratio to the
validation-fitted scale measures how well the geometric concentration scale
transfers from validation to test.
