# Counterions around a charged cylindrical rod

A rigid, impenetrable, infinitely long cylindrical rod of radius a carries
uniform positive charge +e*u*tau0 per unit length. Here e is the elementary
charge, u is a known dimensionless preparation multiplier, and tau0 is one
unknown line density in [0.7,1.4] nm^-1, shared by every preparation.

The rod is coaxial with an uncharged impenetrable cylindrical wall of
radius R>a. The annulus contains only mobile monovalent counterions of
charge -e. Their number per unit axial length is exactly u*tau0, so each
finite cell is neutral. No salt is added. The counterions cannot enter the
rod or cross the outer wall, and they do not exchange with a reservoir.
The rod is fixed and supplies no translational ideal-gas pressure.

The temperature is T. The dielectric permittivity epsilon is homogeneous
throughout the rod, solution, and exterior, with known Bjerrum length
lB=e^2/(4*pi*epsilon*k_B*T). Use classical Poisson-Boltzmann mean-field
equilibrium for point counterions: ideal local ion entropy plus the
self-consistent electrostatic field. Consider the translationally
invariant, cylindrically symmetric equilibrium at each finite R. There
are no end fields, dielectric images, specific adsorption, or other
interactions. Ion correlations beyond this mean-field model are excluded.

Let n_R(r) be the equilibrium number density in that finite cell. The
outer electric field vanishes by neutrality, so its osmotic pressure
above the pure-solvent pressure is P_R=k_B*T*n_R(R). The mean ion density
in accessible volume is nbar_R=u*tau0/[pi*(R^2-a^2)]. The readout is the
dimensionless limit

    phi = lim_(R/a -> infinity) P_R/(k_B*T*nbar_R).

First take the infinite axial-length limit, then take the displayed
limit of canonical finite-cell equilibria, holding a, u, tau0, and lB
fixed. Do not instead impose a normalized ion distribution directly on an
infinite transverse plane.

An experiment is a dictionary with `charge_multiplier` u in [0.2,1],
`bjerrum_length` lB in [0.6,1.25] nm, and `rod_radius` a in [0.5,2] nm.
Predictions must cover this full domain.

`data/calibration.json` is a list of independent records. Each contains
an `input` experiment, a noisy `value` of phi, and its known additive
Gaussian standard deviation `sigma`. Calibration uses u<=0.4.

Implement `Model.fit(records)`, returning self and setting
`self.line_density` to the fitted tau0 in nm^-1. Implement
`Model.predict(experiments)`, returning a finite one-dimensional NumPy
array with one phi per experiment, in input order. Prediction inputs
contain no measured value or sigma. NumPy and SciPy are available.
The supplied response is a starting model and may be changed; preserve
these methods and the parameter attribute.
