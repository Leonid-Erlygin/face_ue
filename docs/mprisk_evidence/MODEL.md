# Mathematical specification implemented by the experiments

Let the ambient representation dimension be d and let u(z)=1/S be the uniform
density on the unit sphere. The encoder returns q_x(z)=p_ref(z|x) under reference
prior u. It is NOT assumed to be the posterior already conditioned on the deployment
gallery. Suppose p(x|z,Y)=p(x|z). The known-class latent densities are f_i(z), their
prior probabilities pi_i sum to 1-beta, and the unknown latent density is u.

Bayes' rule under the reference model implies

    p(x|z) = p_ref(x) q_x(z)/u(z).

Consequently p(x|Y=i)=p_ref(x) B_i(x), where

    B_i(x) = integral q_x(z) f_i(z)/u(z) d sigma(z),
    B_0(x) = 1.

The common p_ref(x) cancels. The posterior is

    eta_0 = beta / D,
    eta_i = pi_i B_i / D,
    D = beta + sum_i pi_i B_i.

All masses are nonnegative and sum to one. For vMF probe and class distributions,
multiplication of the exponential densities gives the explicit evidence integral

    B_i = S C_d(k_x) C_d(k_g) / C_d(h_i),
    h_i = ||k_x mu_x + k_g g_i||.

The experiment fits k_x=alpha*kappa_x and a shared k_g on validation. Equal known
class priors pi_i=(1-beta)/K are used by the runner. The numerical posterior helper
also supports arbitrary positive conditional known-class weights for tests.

For a fixed recognizer action a in {0,1,...,K}, with 0 denoting rejection:

    r_FA = 1[a != 0] eta_0
    r_ID = 1[a != 0] sum_{i>0,i != a} eta_i
    r_FR = 1[a == 0] (1-eta_0)

For unit costs their sum is 1-eta_a. For externally specified nonnegative costs the
corresponding weighted sum equals conditional expected loss under this probability
model. This identity becomes a statement about actual error probabilities only when
the model components are correctly specified. A normalized model is not automatically
calibrated. Validation-PRR-selected weights are treated as ranking weights, not
estimated application costs.

When q_x=u, every B_i=1 and the complete class posterior returns to the prior.
The unit-cost rejection risk then equals 1-beta; acceptance as i has risk 1-pi_i.
When q_x concentrates at mu_x, the posterior approaches point-based inference using
the same gallery likelihoods and priors. Neither limit implies that a relatively low
network concentration in high dimension is already uninformative.

The mixed-prior interpretation can retain discrete known labels and a uniform
continuum of unknown class directions. With delta-distributed latent direction
conditional on an unknown class u, its posterior density is eta_0 q_x(u). Marginalizing
this continuum gives eta_0. Ordinary recognition loss assigns the same cost to every
unknown identity, so no additional internal-spread term is needed after eta is known.
The full mixed posterior can still be used for information-theoretic analyses.

This is a different reference-prior construction from averaging local class
posteriors directly against an already gallery-conditioned probe distribution. The
old implementation is preserved for comparisons; its published results are not
reinterpreted as outcomes of these equations.

## Stable calculation

Write A_d(k)=log E_uniform exp(k mu^T Z) and F_d(k)=A_d(k)-k. Then

    log B_i = [h_i-k_x-k_g] + F_d(h_i)-F_d(k_x)-F_d(k_g).

The linear difference is evaluated without catastrophic cancellation as

    h_i-k_x-k_g = -2 k_x k_g (1-cos_i)/(h_i+k_x+k_g),

with the zero case handled exactly. F_d is evaluated using exponentially scaled
Bessel functions and a convergent log-domain hypergeometric series where the scaled
Bessel function underflows. The interpolation accelerates F_d, not unnormalized
probabilities. Independent numerical checks are in `audits.py` and the test suite.

References for the numerical and representation ingredients:
- SciPy reference, `scipy.special.ive` (exponentially scaled modified Bessel function).
- Li et al., *Spherical Confidence Learning for Face Recognition*, CVPR 2021.
- Franc, Prusa, Voracek, *Optimal Strategies for Reject Option Classifiers*, JMLR 2023.
The source-specific evidence derivation above is provided explicitly rather than
attributing it to any of those papers.
