# Rotating-flow FENE stress, revision 2

This is a same-family physical revision of FENE r1, whose preserved batch scored
1/3: two agents retained Peterlin preaveraging and one derived the exact
potential-flow distribution. That evidence is in `archives/fene-stress-r1` and
`results/zero-three-fene-r1-trial-reviews.json`. It is not discarded or relabeled.
No model evaluation of r2 has yet been made.

The imposed laboratory flow now contains both extension and rotation. The source
uses both in its complete positive conformation closure. Rich Hookean calibration
includes nonzero signed rotation, so its cross-covariance and full flow response
are tested. Finite-spring hidden preparations have stationary configuration
currents. The r1 Boltzmann expression does not solve this mixed-flow problem.
Pure rotation alone still has the radial equilibrium distribution and a
circulating current; the non-potential extension-plus-rotation preparation is the
substantive revision.

For the specified independent per-bead drag, the relative coordinate satisfies
`dQ=(kappa Q-2 grad(U)/zeta)dt+sqrt(4T/zeta)dW`. The source preaverages the spring
factor, solves `kappa C+C kappa^T-(4/zeta)(f C-T I)=0`, and determines
`f=1/(1-tr(C)/L²)`. Its scalar root lies beyond the positive-Lyapunov stability
boundary. It predicts the stress difference `f(Cxx-Cyy)`. This is a valid
Peterlin constitutive approximation, not an unfinished fit or a wrong bead
factor. The physical readout is the actual connector force-dipole average.

The oracle solves the stationary Fokker–Planck equation in disk polynomials,
weighted by the zero-flow density `(1-r²/L²)^(L²/(2T))`. A weak form eliminates
singular spring-force quadrature. The exact stationary second-moment identity
then gives stress `zeta/2*(rate*tr(C)-2*rotation*Cxy)`. The independent verifier
solves positive, conservative polar probability balances with exponentially
fitted edge fluxes and integrates the actual nonlinear spring force. Its two
meshes, 64×96 and 128×192, are Richardson-extrapolated. It does not import the
oracle, its polynomial basis, or its moment readout.

The finite disk has natural zero normal probability flux. Throughout the public
range `b=L²/T` is 5–15; the first and second spring-force moments are integrable.
The Hookean relaxation margin is at least `2/4.8-.25=1/6` even with rotation.
No arbitrary branch, initial ensemble, or imposed configuration symmetry is
required in the public task. The numerical even polynomial basis uses the
inversion symmetry of this unique stationary problem.

Validation is in `results/fene-stress-r2-validation.json` relative to this stage.
There are 24 unique calibration settings, each repeated six times, with fixed
instrument sigma .0005. Seeds are 241241 (data), 241243 (256 noise draws), and
241247 (domain). Every fitted parameter and calibration passes; all 256 oracle
predictions pass, while all 256 completed shortcuts fail all three finite-spring
groups at the unchanged .04 relative group tolerance. Nominal oracle errors are
below 3.7e-5; shortcut errors are .579, .166 and .573. Across noise, maximum oracle
error is .000258 and minimum shortcut error is .165845.

Forty-eight full-domain corner/random checks compare independent force averages:
relative disagreement is at most 2.68e-5. Spectral degree 24→28 changes the stress
by at most 1.52e-12 relative. The unextrapolated finite-volume change can be 1.13%;
the independent prototype verifies second-order refinement and extrapolation.
The weak spectral density has at most 3.24e-6 negative tail mass at difficult
corners; its moments are converged and the verifier distribution is positive.
This small numerical tail is disclosed rather than treated as physical density.
Both controls have positive covariance, extension below L, positive flow power
and the required sign symmetries. Pure rotation and zero-flow equilibrium,
zero-rotation r1 limits, probability conservation, full-range parameter recovery
and nonzero mixed-flow current are checked. The full validator takes about 24 seconds
with one BLAS thread. Local controls pass 7/7 for the oracle and 4/7 for
the shortcut (the three intended finite-spring failures), in 1.02 and .92 seconds.

The public fit is unfinished and the drag starts as `None`. The prediction
helpers are a complete closure; the neutral instruction explicitly permits
replacement. Data contain only input/value/sigma, with no response-dependent
uncertainty. Private references, hidden cases, true drag, hint and this note stay
outside the agent image. Runtime remains 600s agent/60s verifier.

A failure that retains force preaveraging is the intended physical error. An
agent that derives the stationary current problem but miscodes it is an
implementation or numerical failure and must not be counted as a physical
failure. All measured outcomes must be preserved.

Primary background: [Ciuperca and Palade, stationary FENE distributions for
arbitrary velocity gradients](https://arxiv.org/abs/0902.4298) (2009), and
[Keunings, On the Peterlin approximation for finitely extensible dumbbells](https://doi.org/10.1016/S0377-0257(96)01497-8)
(1997). The task's derivation and independent numerical checks determine its
normalization. This revision shares the FENE mechanism with r1; it is not claimed
to be an independent polymer family.
