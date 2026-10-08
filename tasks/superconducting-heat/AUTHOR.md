# Thermal phase oscillator and superconducting heat interference

## Revision 6 public-information correction

Revision 5 accidentally supplied an unused exact-parity branch in the public
starter and completed shortcut. Revision 6 removes that branch; only the
complete factorized approximation remains public. The correction was declared
before revision 5 outcomes were available. All revision 5 trials are retained.
The physical apparatus, exact oracle, independent reference, original calibration
bytes, tolerances and 600/60-second limits are unchanged. A fresh frozen
three-trial evaluation is required for this corrected public package.

The original 256 actual fitted-prediction noise checks remain applicable through
bit-identical reachable predictions: 192 off-grid parameter/preparation checks
and all 144 calibration predictions agree exactly. A fresh targeted validation
also checks all 96 domain preparations using the sole factorized implementation,
independent reference, refinements and thermodynamic identities, plus new local
oracle and shortcut verification. This targeted rerun includes one noisy draw;
it is not reported as a fresh 256-draw run. Its evidence and inheritance are
recorded in `results/superconducting-heat-r6-validation-inheritance.json`.
The older validator's direct domain calls used an unintended exact-branch
default; the fresh targeted run corrects that author validation issue. Its
256 noisy predictions did use the selected factorized implementation.

## Physics

This revision retains the BCS heat family, the pinned noninteracting proximity
construction, all circular magnetic drive coherence and pair channels, and
finite stationary spin feedback. The new physical bottleneck is the correlation
between quantum charge-displacement coherence and exchanged oscillator energy.

The complete scientific derivation, weak-cell reservoir construction and
independent normal-state limit are in the staged runnable proposal bundle
`superconducting-heat/phase-oscillator-prototype/README.md`. Primary mechanism
sources are Martinis (2009 quasiparticle P(E) supplement, Eq.11/21), Ingold and
Nazarov (cond-mat/0508728 Eq.99–101), and Ansari et al. (1211.4745 Eq.14). These
papers do not specify our rotating magnetic barrier or pinned islands.

Let ell>0 mean oscillator gain. The exact conditional anomalous covariance
is (-1)^ell P_ell, while its sum is DW=exp[-2rho(2nB+1)]. The completed shortcut
is the positive factorized covariance DW*P_ell. It retains the correct thermal
normal weights, their detailed balance, all static/opposite-spin and both
rotating/equal-spin pair channels, all transfer channels and full finite spin
balance. Each channel covariance matrix has eigenvalues P_ell*(1+/-DW)>=0;
both controls are passive detailed-balance rate models. The shortcut loses
energy/coherence correlation, rather than a normalization or known force.

With delta=n*w-ell*24, transfer conserves ER=EL+delta, creation EL+ER=delta.
Reverse probabilities are P_-ell; creation/annihilation is counted once for
positive delta. The number prefactor is 2G/e^2 for positive quasiparticle
energies including both normal-dispersion branches. Explicit independent
2x2 Fock/spin matrices confirm static per-spin weights1/2 and rotating a²/2.
The heat identity is QL+QR+Pgate-Qenv=0, where Qenv is oscillator gain and
Pgate=kB*w*(IL+IR)/2. Reservoir entropy after stationary spin relaxation is
-QL/TL-QR/TR+Qenv/Tph. Gate work can be negative under thermal bias; positivity
is required for equilibrium passivity, not arbitrarily for a heat engine.

Every transition spin Jacobian has negative diagonal entries and equal absolute
off-diagonal/diagonal contribution per column. Finite positive spin relaxation
makes the summed Jacobian strictly column dominant. The integrated Jacobian
between possible roots has the same property, proving injectivity. Large-mu
normal-metal slopes retain the positive definite r4 spin-balance matrix since
sum P_ell=1 and the oscillator has finite recoil moments; this gives existence
with inward large-box forcing. Both solvers expand their search windows and
check actual balance rather than assuming |mu|<=w.

The physical phase-contrast readout independently equilibrates the two spin
states at phi and phi+pi. It removes the large phase-independent photon heat
background. Its three predeclared groups have .04 relative RMSE gates,
unchanged from all prior versions. Original 144 calibration records, sigma
.0005pW, seeds33021/43021, conductance20..80uS, chi²1.5, parameter3%,600/60s
limits and both calibration file bytes are preserved. Default mean/rho0 is
the literal archived r4 implementation; calibration therefore identifies the
same unknown G exactly. Normal-state and rho0 anchors remain exact for the
completed shortcut. Phase contrast is explicitly public apparatus, not an
undisclosed metric change.

Proposal evidence shows scalar/Bessel and independent thermal Fock/singlet
agreement, convergence of photon sums and quadrature, first-law and entropy
identities, strict spin-balance Jacobians and an independent closed normal
phase-space solution. The full validator records256 noisy fits and local
oracle/shortcut tests. Neither the scientific package nor proposal runs paid
model trials. Root independently approved the prototype for packaging;
final actual-source review and model evaluation remain separately recorded.

R4 full task, runnable validator, original reports and every raw job/trajectory
are archived before this replacement at `archives/superconducting-heat-r4`.
R4 had one genuine pass, one algebraic implementation failure that also solved
finite spin feedback, and one service-capacity interruption. All are preserved
and excluded from any claim of zero physical passes.
