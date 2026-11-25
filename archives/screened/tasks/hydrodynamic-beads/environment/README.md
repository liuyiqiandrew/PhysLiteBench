# Two trapped beads in a fluid

Two identical spheres of radius a = 0.5 micrometers are held by separate harmonic
optical traps in an unbounded incompressible Newtonian fluid. The fluid viscosity
is eta = 0.001 Pa s and its temperature is 300 K. No other conservative force acts
between the spheres. Particle and fluid inertia, walls, and optical cross-talk
are negligible. Both beads move only along their line of centers.

Use the leading far-field Stokes approximation: each sphere has its isolated
Stokes drag and produces the velocity field of a point force in the surrounding
fluid. Neglect higher multipoles and evaluate this response at the nominal bead
separation. Position fluctuations and imposed shifts are small compared with
that separation; retain only this constant-mobility linear approximation.
Thermal fluctuations obey the fluctuation-dissipation relation for this same
fluid response.

The trap stiffnesses are k1 = stiffness*factors[0] and
k2 = stiffness*factors[1], in pN/micrometer. The single unknown `stiffness` is
between 0.1 and 0.8 pN/micrometer. The two positive factors are known.
Boltzmann's constant is k_B = 1.380649e-5 pN micrometer/K.

There are two measurements. `variance` measures the equilibrium positional
variance of the selected bead about its trap center, in micrometer^2. `mean`
starts from equilibrium with both trap centers fixed. At time zero the centers
are shifted by `shift[0]` and `shift[1]` and held there. It measures the ensemble
mean displacement of the selected bead from its original trap center, in
micrometers, at the given time. Initial mean displacements are zero. Positive
shifts and displacements for both beads point in the same direction along the
line of centers. The shifts do not change the nominal separation used in the
linear fluid response.

## Interface

Every experiment contains `measurement` (`variance` or `mean`), `bead` (0 or 1),
`separation` (micrometers, 5 to 15), and `factors` (a list of two numbers, each
0.6 to 1.6). A `mean` experiment also contains `shift` (a list of two
micrometer displacements, each -0.15 to 0.15) and `time` (seconds, 0 to 1).
A `variance` experiment has no shift or time fields and refers to equilibrium
before any center shift.

Implement `Model` in `model.py`. `fit(records)` must return `self` and set
`stiffness`. `predict(experiments)` must return a finite NumPy array of shape
`(len(experiments),)`. You may edit the entire implementation within this API.

`data/calibration.json` is a list of records. Each record has an `input`
experiment dictionary, measured `value`, and known Gaussian measurement standard
deviation `sigma`. Errors in different records are independent. Experimental
settings are exact. Use the fitted apparatus parameter for predictions over
the stated input ranges.

Run `python -m pytest -q test_public.py` to check the interface and calibration.
