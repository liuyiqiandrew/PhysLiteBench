# Viscoelastic waving sheet: feasibility only

No numerical prototype, task package, calibration data or model evaluation has
been created. Waving-sheet r1 remains preserved with two passes and one physical
failure, plus its three unscored Docker interruptions.

The viable proposal uses the same anchored extensible sheet,
`x=X, y=a cos(kX-omega*t)`, in an incompressible Oldroyd-B liquid. The known solvent
fraction is `beta=eta_s/(eta_s+eta_p)` and the polymer relaxation time is known.
The unknown is the common zero-frequency kinematic viscosity `nu0`, with known
density. Readouts retain the signed first-harmonic laboratory velocity and the
stationary far-field pumping coefficient divided by `a^2`. The small-amplitude
limit keeps instantaneous polymer deformation below the Hookean instability.
There is no imposed pressure gradient, shear or counterflow. A full package
would state the constitutive material model and preparation explicitly.

The completed approximation would use linear laboratory-frame Jeffreys stress,
`tau_p + lambda*d_t(tau_p) = 2*eta_p*D`, together with full fluid momentum
convection and the exact moving-wall expansion. It retains every correction
established in r1. Its first-order complex viscosity is exactly
`nu_star=nu0*[beta+(1-beta)/(1-i*omega*lambda)]`. Thus all first-harmonic calibration
records are shared, including complex phases and nonzero polymer response. This
is a linear constitutive approximation extended to quadratic pumping; it is not
a generally objective nonlinear material law. That limitation must be stated
in private author documentation, rather than describing it as fully objective.

Finite inertia is useful for identifying `nu0`: with the same harmonic
conventions as r1, `U+i*V=omega*exp(-s*h)`, where
`s^2=k^2-i*omega/nu_star` and `Re(s)>0`. Two-height amplitude and phase measurements
recover `s`, provided their phase difference is unambiguous over the selected
parameter range. Then `nu0` is determined uniquely. For example, retain r1's
`nu0 in [.7,1.4] mm^2/s`, `k in [.7,1.3] /mm`, and `omega in [.5,8] /s`, with
`beta=.25` and known `lambda=1 s`. Since `|nu_star| >= beta*nu0`,
`|s| <= sqrt(1.3^2+8/(.25*.7)) < 6.89 /mm`. A height separation of `.1 mm`
therefore bounds the phase difference below `.689 rad`, avoiding any wrapping.
This proves injectivity, but does not establish a noisy fit's conditioning.
In a strictly creeping calculation,
first-order velocity alone is independent of viscosity and cannot calibrate it.

For the creeping benchmark, the physical pumping relative to its Newtonian
value is `(1+beta*De^2)/(1+De^2)`, whereas the linear Jeffreys approximation gives
the Newtonian value. At `beta=.25, De=2` the ratio is `.4`, a nonzero signal with
150% relative source error. This is an analytic limiting check, not a measured
finite-inertia domain margin. [Lauga, Propulsion in a viscoelastic fluid,
Phys. Fluids 19, 083104 (2007)](https://arxiv.org/html/cond-mat/0703548).

A seemingly stronger corotational Jeffreys alternative should be rejected for
this readout. Appendix B of the same paper gives identical quadratic pumping
for the Oldroyd-like objective derivatives. Independently, any planar
incompressible harmonic strain matrix is symmetric and traceless, so its
anticommutator with its complex conjugate is proportional to the identity.
The changed stretching term therefore contributes only pressure at this order;
changing the finite-inertia harmonic profile does not cure that degeneracy.

An independent reference would solve the first-harmonic velocity/polymer-stress
boundary problem and the forced mean momentum equation directly, retaining the
full upper-convected tensor transport. The alternative oracle can integrate its
mean stress and Reynolds momentum flux. Required future checks include the
Newtonian and creeping limits, signed wave reversal, stress work, decay at
infinity, domain convergence and the viscosity phase-bound argument. None has
been run for this revision.

Actual overlap review: archived dumbbell-stress already compares corotational
and upper-convected Hookean stress; its calibration is rigid rotation and its
observable is transient molecular stress. FENE r1/r2 instead retain affine
transport and preaverage a nonlinear spring force. This proposal shares polymer
constitutive physics with those tasks and the apparatus with waving r1. Its
specific approximation would be the full linear viscoelastic response used for
a quadratic pumping observable, rather than another omission of Newtonian mean
convection. It must be labeled a same-family revision, not a new theory family.

Recommendation: scientifically viable for a small finite-inertia prototype,
but the non-objective linear source makes the correction fairly recognizable.
The separately preserved fixed-number band-calorimetry outline remains the
simpler next candidate if a different apparatus is preferred. No full package
is warranted until the finite-inertia gap and independent reference are checked.
