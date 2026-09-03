# HolUE temperature scaling: normalized construction and legacy formula

This note records the derivation that should be used when the dissertation is
updated after the HolUE experiments are rerun. It intentionally distinguishes
between the accepted-paper formula and the normalized implementation.

## Notation

Let

- `K` be the number of gallery identities;
- `rho = (1 - beta) / K` be the prior mass of each gallery identity;
- `u0 = beta / S_{d-1}` be the uniform prior density of the unknown continuum on
  the unit sphere;
- `p_i(z) = p(z | c=i)` be the gallery class-conditional density;
- `p_x(z) = p(z | x)` be the probabilistic embedding density of a probe.

Define unnormalized local evidence terms

\[
    w_i(z)=\rho p_i(z),\qquad w_0(z)=u_0.
\]

For `T>0`, set

\[
D_T(z)=\sum_{i=1}^K w_i(z)^{1/T}+w_0(z)^{1/T}.
\]

## Normalized temperature construction

Define the temperature-scaled conditional probabilities at a fixed embedding
`z` by

\[
a_{i,T}(z)=\frac{w_i(z)^{1/T}}{D_T(z)},\qquad
 a_{0,T}(z)=\frac{w_0(z)^{1/T}}{D_T(z)}.
\]

For the unknown continuum, conditional on choosing the unknown component at
embedding `z`, the unknown identity is the direction `z` itself. Marginalizing
with respect to the probabilistic embedding gives

\[
\Pi_{i,T}(x)=\int a_{i,T}(z)p_x(z)\,dz,
\]

and the continuous unknown posterior density

\[
q_{0,T}(u\mid x)=a_{0,T}(u)p_x(u).
\]

### Proposition: the mixed posterior is normalized

For every `T>0`,

\[
\sum_{i=1}^K \Pi_{i,T}(x)+
\int q_{0,T}(u\mid x)\,du=1.
\]

**Proof.** By the definition of `D_T`, for every `z`,

\[
\sum_{i=1}^K a_{i,T}(z)+a_{0,T}(z)=1.
\]

Therefore

\[
\begin{aligned}
\sum_i \Pi_{i,T}(x)+\int q_{0,T}(u\mid x)du
&=\int p_x(z)\left(\sum_i a_{i,T}(z)+a_{0,T}(z)\right)dz\\
&=\int p_x(z)dz=1.
\end{aligned}
\]

No power of a Dirac delta is needed in this construction.

## Why the accepted-paper formula is not normalized

There are two distinct normalization issues in the published derivation.

First, the appendix denotes

\[
p_T(z\mid c)=\frac{S_{d-1}}{S_{d-1}^{1/T}}\,\delta(z-u_c)
\]

for a continuous unknown identity.  If this object is interpreted literally as
an ordinary conditional probability measure, then

\[
\int p_T(z\mid c)\,dz=S_{d-1}^{1-1/T},
\]

which is not one for general `T`.  It is therefore safer to interpret the
quantity only as an *unnormalized class evidence* and to normalize the class
responsibilities explicitly.  The normalized construction above does exactly
that and never raises or rescales a Dirac probability measure.

Second, and more importantly for the final HolUE posterior, the accepted-paper
derivation uses `D_T(z)` for the gallery terms but the
*unscaled* evidence denominator

\[
D_1(z)=\sum_i w_i(z)+w_0(z)=p(z)
\]

for the continuous unknown term. At a fixed `z`, the gallery contribution is

\[
G_T(z)=\frac{\sum_i w_i(z)^{1/T}}{D_T(z)}
      =1-\frac{w_0(z)^{1/T}}{D_T(z)},
\]

whereas the paper's unknown factor is

\[
\widetilde U_T(z)=\frac{w_0(z)^{1/T}}{D_1(z)}.
\]

Thus

\[
G_T(z)+\widetilde U_T(z)
=1+w_0(z)^{1/T}\left(\frac{1}{D_1(z)}-\frac{1}{D_T(z)}\right).
\]

This equals one only when `D_1(z)=D_T(z)`, which holds for `T=1` and may occur
at isolated accidental points, but not in general for `T != 1`. Integrating the
pointwise expression against `p_x(z)` therefore does not restore normalization
in general.

The implementation mode `legacy_paper` reproduces this denominator mismatch
only for an ablation. The default `normalized` mode uses `D_T` consistently.

## KL decomposition under normalized scaling

With the normalized construction,

\[
\mathrm{KL}_1=
\sum_{i=1}^K \Pi_{i,T}(x)
\log\frac{\Pi_{i,T}(x)}{(1-\beta)/K},
\]

and

\[
\mathrm{KL}_2=
\int q_{0,T}(u\mid x)
\log\frac{q_{0,T}(u\mid x)}{\beta/S_{d-1}}\,du.
\]

Since

\[
q_{0,T}(u\mid x)=p_x(u)\frac{w_0^{1/T}}{D_T(u)},
\]

we obtain

\[
\log\frac{q_{0,T}(u\mid x)}{\beta/S_{d-1}}
=\log p_x(u)+\left(\frac1T-1\right)\log\frac{\beta}{S_{d-1}}
-\log D_T(u).
\]

The code estimates the integral by the same deterministic mean-embedding
approximation (`M=0`) used in the published HolUE experiments, or by Monte Carlo
samples (`M>0`) for the convergence study.

## Required rerun/ablation set

1. Rerun every dissertation HolUE table with `temperature_scaling_mode=normalized`.
2. Compare `normalized` and `legacy_paper` at the published `T=20`, retraining
   the HolUE calibrator separately for each formula.
3. Audit total posterior mass over a temperature grid. The normalized mass must
   be one up to numerical error; the legacy mass defect should be reported.
4. Run a temperature-sensitivity study on the validation/test protocol without
   retuning on test data.
5. Run `M=0,8,16,32` (optionally `64`) to quantify the mean-embedding
   approximation relative to Monte Carlo integration while keeping the OSR
   decisions fixed for the rejection-quality comparison.
6. Report rank correlation/inversion between legacy and normalized HolUE scores.
7. Stratify the comparison by SCF concentration quantiles, especially for
   IJB-B/IJB-C because their validation set (MS1MV2) is intentionally cleaner
   than the IJB test imagery.

## Reproduction commands

The patch targets the source tree represented by `combined.txt(3).zip`.  Apply
it from the repository root:

```bash
git apply holue_proper_temperature_scaling.patch
```

Run the focused tests first:

```bash
pytest -q tests/test_holue_temperature_scaling.py tests/test_scf_geometric_calibration.py
```

Regenerate the dissertation-facing biometric and text core experiments (the
patched active HolUE configurations explicitly select normalized scaling):

```bash
python experiments/mprisk_core_experiments.py --config-name mprisk_core_bio_complete
python experiments/mprisk_core_experiments.py --config-name mprisk_core_text_complete
```

Regenerate the HolUE/SCF geometry experiments used for interpretation:

```bash
python experiments/scf_geometry_holue_experiments.py --config-name scf_geometry_holue_bio
python experiments/scf_geometry_holue_experiments.py --config-name scf_geometry_holue_text
```

Run the dedicated temperature-scaling audit/ablation suite:

```bash
python experiments/holue_temperature_scaling_experiments.py --config-name holue_temperature_scaling_bio
python experiments/holue_temperature_scaling_experiments.py --config-name holue_temperature_scaling_text
```

The audit suite writes:

- `final_formula_comparison.csv`: final calibrated normalized versus legacy
  formula at the published operating point, with the calibrator retrained on
  the validation set separately for each formula;
- `normalization_audit.csv`: empirical posterior-mass check over the temperature
  grid;
- `temperature_sensitivity.csv`: raw KL sensitivity with recognition decisions
  and gallery concentration held fixed;
- `mc_convergence.csv`: mean-embedding (`M=0`) versus Monte Carlo estimates over
  several seeds;
- `ranking_stability.csv`: Spearman correlation and sampled pairwise rank
  inversion rates;
- `quality_stratified.csv`: uncertainty quality split by SCF concentration
  quantiles;
- `quality_shift.csv`: validation-to-test SCF concentration shift, including
  the MS1MV2-to-IJB-B/IJB-C comparison;
- per-example `.npz` files with KL components and uncertainty scores.

Do not select a temperature from the test-set sensitivity table.  If the new
results suggest that the published temperature is no longer adequate, choose
(or learn) a new temperature using validation data only and then rerun the test
set once with that frozen choice.
