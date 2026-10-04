# Electroviscous flow, revision 1

The completed shortcut makes a single streaming-potential estimate from pressure
flow, then solves the resulting electroosmotic flow. It omits that induced
flow's contribution to the current constraint. The approximation is valid for
weak hydraulic/electric feedback and solves its stated equations consistently.
Neutral-wall calibration cannot observe this feedback and identifies viscosity.

For the symmetric half-slit, transverse equilibrium and Gauss's law give

    c_plus=c0 exp(-phi/Vt), c_minus=c0 exp(phi/Vt), Vt=RT/F,
    rho=-2 F c0 sinh(phi/Vt), phi''=-rho/epsilon,
    phi'(0)=0, phi'(h)=surface_charge/epsilon.

The wall charge is fixed and immobile. It imposes a normal derivative, not an
absolute potential, so a small axial potential gradient is consistent with the
boundary. Diffuse-layer volume conductivity is retained; there is no separate
surface current or compact-layer physics.

The pressure and unit-electric-field velocity responses are

    u_p=g(h²-y²)/(2 eta), u_E=epsilon[phi(y)-phi(h)]/eta.

Let brackets denote the half-slit spatial average, G=<sigma>, I_p=<rho*u_p>,
and A=<rho*u_E>. The two controls differ only in axial field:

    oracle:   E=-I_p/(G+A), u=u_p+E*u_E;
    shortcut: E=-I_p/G,     u=u_p+E*u_E.

The additional conductance A=epsilon²<phi'²>/eta is nonnegative. The hydraulic
and electrical cross coefficients obey <u_E>=I_p/g. At open circuit the
pressure power g<u> equals <eta*u'²+sigma*E²>. The shortcut is not repaired by
fitting viscosity: its full resulting velocity must re-enter the current
constraint. Hidden cases keep the shortcut's mean flow positive; failure does
not depend on a conspicuous negative hydraulic conductance.

The oracle solves only the nonlinear Poisson–Boltzmann boundary problem, then
uses analytic creeping-flow responses and Gaussian quadrature. The independent
reference solves a joint five-state collocation problem for potential, its
derivative, velocity, its derivative, and integrated current, with the unknown
axial electric field as an additional boundary-value parameter. It imposes
zero integrated current at both ends of the half-domain and does not use the
oracle's analytic flow response or eliminated field denominator.

Inputs are half-gap in micrometres, reservoir concentration in mol/m³, immobile
wall charge in C/m², and signed pressure gradient in Pa/m. Outputs are velocity
in m/s. Over the public domain, even the pressure-only center speed bounds ionic
Peclet number by .057 and fluid Reynolds number below 8e-6. The stated leading
linear electrokinetic response neglects distortion of transverse equilibrium;
there is no unknown electrokinetic coefficient beyond the fitted viscosity.

Calibration seed 9314 uses 120 neutral-wall readings, eta=.0012 Pa s, and sigma
2.56e-8 m/s. The completed controls fit eta=.00119963271 and reduced chi-square
.95823. Their calibration predictions are exactly equal. Hidden groups vary
wall-charge sign, overlap, geometry, and mean/point readout. Each group's error
is RMSE divided by its reference RMS velocity; the limit is .025. Oracle errors
are at most .000240, and shortcut errors are .1689 to .1941.

Independent joint collocation agrees to 3.1e-13 relative; refinement changes
predictions by 3.3e-13. Transverse charge neutralization, reciprocal hydraulic
coupling, and the positive gradient-integral form of A agree to 9e-13. Current
closure error is below 4e-16 and pressure-power balance below 3e-13. Public
viscosity/geometry/charge corners agree with the reference below 7e-13. Hidden
feedback conductance is .427 to .559 of ohmic conductance; the shortcut leaves
at least .427 of the original streaming current uncancelled.

Noise seed 19314 generates 256 additional calibration realizations. All pass
calibration and parameter checks; maximum viscosity error is .264%. Hidden
checks at observed fitted-parameter extrema give oracle error below .00206
and shortcut error above .1684. This is a calibration Monte Carlo plus extrema
sensitivity, not 256 complete hidden runs or an exhaustive noise bound.

Run scripts/validate_electroviscous_flow.py to read frozen calibration and write
jobs/electroviscous-flow-validation/summary.json. Use --generate only to replace
both calibration copies intentionally before freezing a revision. The completed
shortcut is scripts/electroviscous_flow_baseline.py. A teammate reviewed the PB boundary, analytic flow response, positive feedback
conductance, independent joint BVP scaling, and public physical assumptions.
Local verifier controls give oracle 7/7 and shortcut four passing checks plus
three intended hidden failures. Harbor agent difficulty is not yet measured.
