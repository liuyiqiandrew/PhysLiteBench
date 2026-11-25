# Deforming crystal current: bounded feasibility

The numerical prototype is complete and awaiting independent peer review. No
task harness, calibration file, model trial or Docker run exists. All source
and report variants produced for this candidate are retained here; there was
one scientific execution, which completed in .947 seconds.

The source is a complete zero-field internal-relaxation and polarization model.
It solves the strictly convex three-component free energy, computes the exact
cell volume and laboratory polarization, and differentiates that polarization
with the full internal Hessian. It then interprets the laboratory density
response through the instantaneous electrode face as the terminal current.
This is the legitimate improper polarization susceptibility, used as an
approximation to a different measured response. No constitutive input is
silently removed from the internal equilibrium calculation.

## Independent derivation and closed apparatus

Set J=det(F), x=s−s0 and gamma=4. The internal equation is
(K+gamma |x|²)x=h, with Hessian
(K+gamma |x|²)I+2 gamma xxᵀ. Its eigenvalues are at least K, so the equilibrium
is globally unique. The code uses a scalar radial root; the reference solves
all three stationarity equations independently at each deformed state.

In units q0/a0², the bulk polarization is P=−F s/J. For a unit reference top
face the current area vector is a=J F^(−T)e3. The supplied approximation is
−a·dP/dlambda. The grounded slab's weighting potential is the material third
coordinate divided by its thickness. Each complete +/− cell contributes a
charge displacement independent of transverse area, giving upper electrode
charge per reference cell area s3, up to a fixed termination constant. Thus the
physical normalized coefficient is ds3/dlambda. Writing M=F^(−1)G, the source
exceeds this by (M s)3−tr(M)s3. These formulas are author-only and absent from
the apparatus outline.

The reference does not use that response formula. It independently relaxes
s at four perturbed strains, deposits both charges of every cell with
charge- and first-moment-preserving interpolation, solves the grounded
one-dimensional Poisson equation, and differences the upper electrode charge.
The reduction to the planar average is exact for total electrode charge under
periodic lateral boundary conditions: every nonzero lateral Fourier component
integrates to zero. For an oblique affine slab the positive metric factor in
Poisson's equation cancels against the flux-to-total-charge conversion. The
reference is independent in equilibrium solution, response differentiation and
electrical readout, while necessarily using the same specified cell free energy.
It is not a microscopic rederivation of that effective constitutive energy.

The outline explicitly defines the free energy as the total zero-macroscopic-
field material free energy, including internal electrostatic effects. It does
not add charged particles to a nonelectrostatic spring while neglecting their
interactions. Ideal shorted electrodes select zero macroscopic field in the
bulk. Bounded microscopic surface backaction layers contribute vanishing
corrections to the bulk current coefficient in the stated thickness limit;
there is no undisclosed strain-dependent surface law. The physical readout and
reference are compatible with the specified complete neutral-cell termination.

## Calibration and separation

There are eighteen possible calibration settings: nine longitudinal stretches
from .88 to1.12, each with positive and negative longitudinal rate. Both models
coincide exactly for these drives. With F=diag(1,1,l), h3=.35(l²−1), x has only
a third component, and the positive response is .7l/(K+12x3²). Its denominator
increases strictly with K: its derivative is
(K−12x3²)/(K+12x3²). Throughout the calibration interval,
|x3|<=.08904/.8=.1113 and12x3²<.149<K. Each positive setting therefore decreases
strictly with K; any one identifies the parameter globally, and the combined
noiseless least-squares objective has only one zero. This does not assert a
single stationary point for every possible noisy objective.

All41 noiseless fits spanning both endpoints recover K within8.38e−9. Exact
calibration agreement is2.23e−16. A future harness must set sigma from a fixed
instrument uncertainty, independently of response and fitted K. No sigma or
private true value has been chosen in this feasibility stage.

The preserved sweep includes246 ordinary transverse settings over41 K values,
three stretches and two drive axes. The123 transverse-x cases alone have true
signals .349–.553, source signals .110–.290, and45.6–68.5% pointwise gaps. These
are viable positive, nonzero held-out preparations, well separated from4%.
Transverse-y cases are also retained, but some source predictions cross zero;
those are not proposed as the basis of scored groups. This is a transparent
restriction to healthy illustrative signals, not a removed failed validation.
The prototype's additional rigid-rotation limit has zero true current and a
nonzero source response. It checks the same physical assumption but is not
proposed as a scored zero-response task.

## Numerical scope and limitations

The145 checked states comprise81 structured strain/rate/K controls and64 random
interior controls. They are not every corner of the six-dimensional F box.
Maximum response disagreement with the Poisson reference is6.19e−12. Combined
step/grid refinement changes results by1.86e−11; changing slab thickness from
nine to fifteen cells changes them by1.07e−11. Induced-charge/offset agreement
is3.25e−14 and rigid frame covariance is6.67e−16. The source/report contain every
case and all fit rows, including weak or sign-changing variants.

Global convexity does not depend on sampling. On the stated F box, bounds on
|h| give |x|<.278; with the specified c and s0 both charges remain in their
selected material cells. F has positive determinant because its diagonal is
positive. The small perturbation limit preserves this branch. These facts
exclude a hidden coordinate-wrap or unstable-root failure. The
mechanical lattice control is prescribed, not solved as a free-boundary
elasticity problem.

A future package still needs complete public-domain and signed-current checks,
noise calibration, actual local verifier controls, input audit, frozen peer
review and model evaluation. Empirical difficulty is unknown. A trajectory that
recognizes transported electrode charge and then miscodes a determinant,
normal, derivative, sign or matrix index is an implementation failure, not the
intended retained-readout approximation.

## Archive and primary-source scope

The archive audit records the actual author/assessment inventory and six closest
AUTHOR files read. Piezoelectric-waves uses a different electrostatic constraint
in acoustic propagation; terminal-current-noise counts stochastic capacitor
redistribution; prestrained-solid concerns the incremental elastic Hessian;
actuator-frame-current concerns persistent-force channels. Nematic boundary
work and coherent-alloy compatibility share broader continuum themes. None is
the same moving-electrode polarization-density approximation. Broad moving-
frame and electrical-readout overlap is acknowledged.

Primary support is Vanderbilt, *Berry-phase theory of proper piezoelectric
response*, Journal of Physics and Chemistry of Solids61,147–151(2000),
[arXiv cond-mat/9903137](https://arxiv.org/pdf/cond-mat/9903137),
[DOI](https://doi.org/10.1016/S0022-3697(99)00273-5). The full five-page primary
PDF was read through the browser, especially sectionsIII.A–C and equations
15–16,25. It establishes that measured current differs from a laboratory
polarization derivative, gives longitudinal equality, and connects the measured
response to surface charge. The explicit classical effective-cell potential,
parameter domain and Poisson prototype here are our construction; they are not
a claimed quantitative model of a named crystal or a Berry-phase computation.

Shuttleworth surface stress was considered but not prototyped because it is
close to previously screened variational-stress tasks. A micropolar-flow idea
was not developed because a sufficiently strong exact calibration subspace was
not identified. A Berry-band implementation was rejected as unnecessary
complexity: the molecular-cell construction already isolates the same electrical
measurement distinction. These rejected branches are recorded in the audit.
