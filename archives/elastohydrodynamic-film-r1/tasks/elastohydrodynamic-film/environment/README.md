# Relaxation of a liquid film over a bonded elastic layer

An incompressible Newtonian liquid of viscosity .15 Pa s fills a thin gap
between a fixed rigid plate and a homogeneous isotropic elastic coating.
The coating has thickness `thickness` and is perfectly bonded to a rigid
support below it. Its Poisson ratio is .48. Its shared unknown Young modulus
is between .7 and 1.6 MPa. There is no viscoelasticity, surface tension,
permeation, slip, body force or external flow after preparation.

The system is translation invariant in y and periodic in x. Use plane strain:
u_y=0 and all y derivatives vanish. The elastic displacement is zero at the
bottom of the coating. Its upper surface is loaded by the liquid. The
coating obeys isotropic infinitesimal-strain elasticity with the specified
Young modulus and Poisson ratio; its mechanical response is quasistatic.
The liquid and solid inertia are negligible.

Work to leading order in the thin liquid gap, `gap`, compared with both the
coating thickness and the lateral pressure wavelength. Use the usual
no-slip lubrication limit in the liquid. At this order the liquid exerts
normal pressure on the coating, the elastic surface has zero shear traction,
and lateral surface motion contributes only higher-order Couette flux.
Keep the complete elastic response through the coating thickness. The gap
variation is small compared with its mean: linearize liquid transport about
the given gap. These are the specified controlled approximations.

Each experiment starts with a prescribed excess liquid pressure

    p(x,0) = pressure * cos(wavenumber*x).

The coating has already reached mechanical equilibrium with this pressure.
Distributed preparation ports are closed at t=0 without changing the initial
pressure or displacement. There is subsequently no fluid exchange. For
positive wavenumber the periodic length is 2*pi/wavenumber. Wavenumber zero
means uniform loading and zero lateral strain at infinity. A large ambient
pressure prevents cavitation where the excess pressure is negative.

The detector measures the cosine amplitude of the coating's downward normal
surface displacement at `elapsed_time`. Downward displacement increases the
liquid gap and is positive. Return this amplitude in nanometres relative to
the uniformly unloaded reference surface. All trials use fresh preparations.
The detector measures the actual displacement, not a fraction of its initial
value. For the homogeneous calibration the fluid remains uniformly loaded.

Input dictionaries contain:

- `pressure`: initial excess-pressure amplitude, 0 to100 Pa.
- `wavenumber`: 0 to4 in inverse millimetres (angular wavenumber, not cycles/mm).
- `thickness`: .5 to1.5 mm.
- `gap`: 5 to15 micrometres.
- `elapsed_time`: 0 to10 s.

All specified controls, viscosity and Poisson ratio are exact. The task asks
for the leading linear model described above; higher-order effects of the
small gap, deformation and inertia are outside the requested approximation.
Calibration observes uniform loading at varied pressure and coating
thickness, with independent Gaussian errors of the listed standard deviation.

Implement `Model.fit(records)`, returning self, and `Model.predict(experiments)`,
returning a finite NumPy array with shape `(len(experiments),)`. Expose the
fitted Young modulus in MPa as `Model.young_modulus`. Records in
`data/calibration.json` have `input`, `value` and `sigma` fields. Run
`python -m pytest -q test_public.py`.
