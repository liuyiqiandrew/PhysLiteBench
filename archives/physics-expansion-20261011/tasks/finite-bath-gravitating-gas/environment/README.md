# A gravitating gas with a thermal reservoir

A classical gas of N identical particles of mass m has total mass M=Nm.
It is confined inside a rigid spherical reflecting wall of radius R. The
particles interact through Newtonian gravity with one unknown strength G
in [0.7,1.4] mm^3/(mass-unit second^2). The same G applies to every run.
There is no central body, rotation, external gravitational field, or
surface mass. The wall supports pressure and does not move.

Use the smooth, collisional Boltzmann mean-field description, with
N tending to infinity. Each stationary state has a local Maxwell velocity
distribution at uniform temperature T. Define theta=k_B*T/m, in
mm^2/second^2. The regular spherical density profile has finite central
density and obeys hydrostatic balance with P=rho*theta and its own
Newtonian potential. The total enclosed mass at the wall is M.

The gas can exchange heat with a spatially uniform reservoir of constant
heat capacity b*N*k_B. The reservoir exerts no force, contributes no
gravitating mass, exchanges no particles, and occupies no gas volume.
Its coupling energy is negligible. At thermal equilibrium its temperature
is T; its energy is b*M*theta plus an irrelevant constant, and its entropy
obeys dS_b/dE_b=1/T. During a finite-bath stability test the gas and
reservoir form an isolated composite with fixed total energy. At b=0 the
reservoir is absent and gas energy is fixed. The separate value
`bath_capacity: "infinite"` denotes an ideal imposed-temperature reservoir.

Two readouts are available:

- `temperature`: theta of the regular stationary profile whose central to
  wall density ratio is `contrast`. This contrast labels a preparation;
  it is not constrained during perturbations.
- `critical_temperature`: theta at the first loss of local thermodynamic
  stability on the regular spherical branch, followed from dilute states
  to increasing density contrast. Stability means a local maximum of
  total Boltzmann entropy under infinitesimal smooth spherical density
  and thermal perturbations, at fixed M, R, and composite energy. For the
  infinite reservoir it means local stability at fixed imposed T.
  Follow a sequence of separately prepared composites with decreasing
  total energy; for the infinite reservoir lower the imposed temperature.
  Report the temperature at the first instability, not the minimum
  temperature anywhere on the equilibrium branch.

The apparatus reports this mean-field local threshold, not a collapse
waiting time or a global entropy maximum. Finite-N activation, binaries,
short-distance encounters, and collisionless dynamical stability are
outside this description.

Each experiment contains `observable`, `mass` M in [0.8,1.6] mass-units,
`radius` R in [0.7,1.5] mm, and `bath_capacity` b in [0,2] or `"infinite"`.
For `temperature` it also contains `contrast` in [1.5,25]. For
`critical_temperature` no contrast is supplied. Both outputs have units
mm^2/second^2. Predictions must cover the full input domain.

`data/calibration.json` is a list of independent records. Each contains
an `input` experiment, a noisy `value`, and its known additive Gaussian
standard deviation `sigma`, in output units. Calibration includes ordinary
profile temperatures with finite reservoirs and critical temperatures
with an infinite reservoir.

Implement `Model.fit(records)`, returning self and setting `self.gravity`
to the fitted G. Implement `Model.predict(experiments)`, returning a finite
one-dimensional NumPy array with one temperature per experiment, in input
order. Prediction inputs contain no measured value or sigma. NumPy and
SciPy are available. The supplied response is a starting model and may
be changed; preserve these method signatures and the parameter attribute.
