# Conducting flow in an insulating rectangular duct

A long straight duct has cross section 0<x<W, 0<y<H with W=0.012 m and H=0.008 m.
It contains an incompressible Newtonian conducting liquid with density
rho=1000 kg/m³ and electrical conductivity sigma=1e5 S/m. Dynamic viscosity eta
is unknown, in [0.006,0.018] Pa s. All four walls are rigid, no-slip, and
perfectly electrically insulating. There are no electrodes or imposed electric
currents. The applied static magnetic field is uniform, B=(Bx,By,0).

Flow is fully developed far from the ends: velocity is u(x,y,t) times the +z unit vector and all fields are independent of z. There is no imposed axial
electric field. Neglect charge-relaxation and displacement-current transients;
use the local Ohm law of a moving conductor. The magnetic Reynolds number is
negligible, so the imposed field is unaffected by the liquid. Ignore Hall and
thermoelectric effects. Temperature is held uniform, material constants are
constant, and heating does not change the flow. The stated continuum model is
laminar; entrance effects and turbulence are absent.

Each experiment starts with the liquid at rest. At time zero a constant axial
pressure gradient is applied: pressure decreases by g Pa per meter along +z.
The gradient is then held fixed. The magnetic field is already present.

## Interface

Implement `Model.fit(records)`, returning `self` and setting `viscosity`, and
`Model.predict(experiments)`, returning a finite NumPy array of shape
`(len(experiments),)` in m/s. You may edit the entire implementation.

Each experiment contains `magnetic_x` and `magnetic_y` (Bx and By, each
-0.65 to 0.65 tesla),
`pressure_gradient` (g, -0.2 to 0.2 Pa/m), `time` (0 to 12 s), and
`observable`, either `mean` for cross-sectional mean velocity or `point`.
For `point`, `x` and `y` give fractions of W and H between 0.15 and 0.85.
Coordinates `x` and `y` may also be present for a mean reading and are ignored.
The same viscosity applies to all preparations.

Calibration records contain `input`, measured scalar `value`, and the known
independent Gaussian measurement standard deviation `sigma`. Experimental
settings are exact. Run `python -m pytest -q test_public.py` to check the API
and calibration.
