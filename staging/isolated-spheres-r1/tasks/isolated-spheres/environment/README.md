# Thermal force between conducting spheres

Two identical rigid metal spheres are held at a controlled center separation in vacuum. They are normal, nonmagnetic conductors with finite nonzero dc conductivity and negligible static electric screening length compared with their radius. The spheres and surrounding radiation are in thermal equilibrium at known temperature `T`. The reported interaction is the leading classical high-temperature thermal force, in the limit that the thermal wavelength is small compared with the surface gap. The material remains a normal conductor in this limit.

Each sphere is prepared with exactly zero total electric charge. The supports and thermal environment cannot exchange charge with either sphere, and there is no electrical connection between them or to any external conductor. The spheres remain individually isolated throughout equilibration and measurement. Radius and separation are fixed by the mechanical supports; the measured force is the mean force exerted on the right sphere along the line from the left center to the right center. An attractive force is negative. No external electric field is present during this force measurement. Background forces and the forces of the supports are subtracted.

Lengths are expressed in a known unit `ell`. The common radius `radius` is unknown in `[0.85, 1.15]`; the center separation is in `[3.0, 4.5]`. Force output is expressed in units `k_B*T/ell`. The same radius applies to every experiment.

A separate calibration protocol uses one sphere alone, under the same electrical isolation, in a weak uniform static field along a fixed axis. The reported quantity is its mean induced dipole component along that axis, with the zero-field mean subtracted. `field` is the signed field in a known unit `E_star`, with `field` in `[-1, 1]`. Dipole output is in units `4*pi*epsilon_0*E_star*ell^3`. The field is weak enough for linear electrostatic response, and the sphere is centered at the origin.

Each experiment is one of:

- `{"kind": "dipole", "field": ...}`: the single-sphere induced dipole.
- `{"kind": "force", "separation": ...}`: the equilibrium pair force.

`data/calibration.json` contains independent dipole readings with `input`, `value`, and known fixed instrument uncertainty `sigma = 0.0003` in the stated dipole units. Implement `Model.fit(records)` and `Model.predict(experiments)`. `fit` returns `self` and stores the inferred `self.radius`. `predict` returns a finite NumPy array of shape `(len(experiments),)` in each experiment's stated output units. You may use any numerical method consistent with the apparatus.
