# Coherent array, revision 1

The apparatus is a stationary classical harmonic array with a specified local Langevin bath and a coherent optical field readout. The completed source keeps the exact Gibbs distribution, all damped normal modes, static coherent scattering, self motion and mean-field plateau. It transfers the normalized connected self relaxation to the collective field. The physical error is that distinct-particle motion need not relax like self motion, even when these exact static and single-particle checks pass.

No model evaluation has been run. Scientific validation and local completed controls are finished; empirical difficulty is unknown.

## State, measurement and complete approximation

The potential is K times a positive quadratic form with onsite coefficient .6 and known nearest-neighbor coupling. Mass is one, damping .4, and independent force noise has covariance .8*T. Thus the stationary position covariance is T times the inverse Hessian, momentum covariance is T times the identity, and the mixed block vanishes. Positive pinning and damping give stable, mixing dynamics throughout the declared domain. The baths remain active; the array is not isolated during measurement.

The detector measures the full real two-time correlation of the complex scattered field, normalized by sqrt(5). Both quadratures and the field amplitude are operationally specified in the public README. The mean field is included. The probe is weak enough that it does not change the mechanical dynamics. This is a linear field measurement rather than a prescription to reconstruct a field correlation from an intensity measurement.

Let C_ij(t)=<u_i(t)u_j(0)> and v_i=C_ii(0). The exact pair contribution is

    exp[i*q*(i-j)] * exp[-q^2*(v_i+v_j-2*C_ij(t))/2].

Summing over i,j and dividing by five gives F(q,t). The modal implementation evaluates the exact damped covariance. The completed source instead constructs

    F_source = F_infinity + (S-F_infinity)*(Fs-Fs_infinity)/(1-Fs_infinity),

where S=F(q,0), F_infinity=|<A_q>|^2, and Fs is the exact mean self field correlation. It has the exact static value and elastic plateau, and it is exact for independent identical trapped scatterers and for one scatterer. The general source is a valid stationary correlation kernel: Fs-Fs_infinity is a positive-semidefinite average of connected self covariances; rescaling by the positive collective variance and adding the nonnegative mean-field constant preserves this property. This does not claim that the supplied microscopic coupled array realizes the alternative kernel.

The uncorrected Vineyard convolution F=S*Fs was considered first. It misses the elastic plateau in a trapped system, so the stronger source above was chosen before any task evaluation. Its original predictions remain in the prototype report. The exact-plateau modification is our stated closure, not a formula attributed to Vineyard's original paper. The earlier isolated finite-array outline was also rejected for plateau claims: finite undamped harmonic modes recur. Positive damping is part of the actual final apparatus, not just a preparation device.

## Reference and scientific checks

The independent reference constructs the full ten-dimensional physical drift and noise matrices, solves the stationary Lyapunov equation, and applies a full matrix exponential. For each pair it obtains the joint two-dimensional displacement covariance and integrates the actual field factors with Gauss-Hermite quadrature. It does not use the scalar modal time formula or the oracle's exponential-of-pair-variance formula. The specified physical Hessian is necessarily common, but its construction and subsequent state/readout calculations are separate.

The final report covers 720 structured controls, including coupling interiors .25,.5,.75, plus 64 random full-domain controls. The largest reference error is 8.84e-12, and changing Hermite order32 to48 changes values by at most 8.84e-12. Scored-reference errors are below2.3e-15. This is bounded numerical validation, not a rigorous uniform quadrature bound over every real input.

Equal-time and uncoupled source equivalence, signed-wavevector reflection, positive connected variance, and the long-time plateau pass. Both correlation kernels are positive on17-time grids for36 parameter/control settings, with minimum eigenvalue1.09e-5. The stationary Lyapunov residual, five-mode equipartition and bath injection/dissipation balance agree at roundoff. The baths exchange energy and momentum, so no isolated conservation claim is made. The one-scatterer limit and full broad wavevector/time screen are preserved in the bounded prototype.

## Calibration and grading

K is unknown in [.8,1.2]. The private value1.07 is absent from public source and constructors, which start with stiffness=None. Nine calibration settings use temperatures .04,.08,.12, each with a coupled equal-time reciprocal-wavevector measurement and two nonzero-delay uncoupled measurements. Thirty-two independent repeats give288 records. Every sigma is the fixed instrument uncertainty .003, independent of K and the response. Calibration seed171113 and separate noise seed171119 are private metadata.

At q=2*pi, all static pair phases are positive. Write the positive off-diagonal constants as a_ij; then S=1/5 sum exp[-a_ij/K], so dS/dK>0. The coupled static records identify K globally by themselves. The finite-delay uncoupled records also exercise the exact dynamics. The sampled derivative minimum .8178 is not asserted to be a proved numerical box bound. Forty-one noiseless full-range fits, including endpoints, recover K within6.80e-9. Both models agree on calibration within8.89e-16.

The actual fit is1.0700010616 with reduced chi square1.00428. Diagnostic source errors are .14904,.29000,.40714. The three groups use the same ordinary full-field observable at finite delay and known coupling, varying temperature; their smallest absolute reference signal is .3230. There is also a shared exact-limit anchor. Across the whole sampled K interval the respective group gaps exceed .13719,.26854,.38337. No near-zero individual signal is used as a normalizer, and the public domains retain all near-equivalent regimes.

All256 independent noise realizations pass calibration and parameter checks for both controls. Every oracle group and shortcut exact anchor passes, while all three shortcut diagnostic groups fail. The maximum oracle error is .00080044 and minimum shortcut error .148717 at the ordinary .04 group gate. Parameter error remains below .03 and reduced calibration chi square below1.5. Local fresh copies give oracle7/7 and shortcut4 passes plus3 intended failures, each below .44 seconds. Standard agent/verifier limits remain600/60 seconds. The full author validation completed once successfully in4.10 seconds; no data regeneration or failed scientific revision is hidden.

The public/private calibration files are identical. Docker copies only the public README, starter, test and data. The instruction is the unchanged neutral text allowing replacement of every helper. The source's forward functions are AST-identical to the completed shortcut, whose only addition is fitting. Grading imposes numeric API/output requirements without a solver or model-form restriction.

## History, sources and classification

The complete approved feasibility package is copied under `../../prototype/`, including its original author manifest, peer, wording followup, broad272-point screen and superseded variants. A376-path archive inventory and ten actual closest/alternative AUTHOR reads found no exact coherent/self dynamic-scattering task. Fixed-trap disorder was rejected as an exact archived mechanism. Phonon transport, quantum paired resonators, hard-core coherence and multiphoton phase share normal-mode or correlation themes but test different physical approximations. Single-file memory also has a shared calibration subspace; it concerns rank conservation and preparation-dependent nonstationary increments, rather than this equilibrium stationary field closure. Their outcomes remain untouched.

Primary background, with access scope retained in the prototype:

- [Vineyard, Scattering of Slow Neutrons by a Liquid](https://journals.aps.org/pr/abstract/10.1103/PhysRev.110.999): primary abstract read; supports the static/self convolution approximation. No claim of reading its restricted full text or attributing our plateau correction to it.
- [Singwi, Coherent Scattering of Slow Neutrons by a Liquid](https://journals.aps.org/pr/abstract/10.1103/PhysRev.136.A969): primary abstract read through the APS search result; supports the role of correlated neighboring motion. A subsequent direct open returned a cache miss.
- [Van Hove, Correlations in Space and Time and Born Approximation Scattering in Systems of Interacting Particles](https://journals.aps.org/pr/abstract/10.1103/PhysRev.95.249): primary abstract read for the space-time pair-correlation basis of weak scattering. The finite-array formulas and numbers are derived from the explicit apparatus here.

Retaining the complete self-relaxation closure can be an intended physical failure. Deriving the correct collective covariance but making a Gaussian, modal, normalization or indexing mistake is mathematical/implementation or mixed, and must be classified from the full trial trajectory. Scientific separation does not establish an agent failure rate.

Evidence is in `../../results/coherent-array-r1-validation.json`, `coherent-array-r1-local-controls.json`, the public-input audit and frozen provenance. Reproduction uses the staged `scripts/validate_coherent_array.py`; `--generate` deliberately creates a new absent calibration dataset. The author has launched no Docker or model-agent evaluation.
