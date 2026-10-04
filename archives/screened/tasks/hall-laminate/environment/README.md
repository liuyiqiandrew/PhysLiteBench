# Layered conductor

A periodic laminate consists of two isotropic conducting phases A and B. Each has one classical Drude carrier species with known charge-density magnitude nq=1e4 C/m³. Carriers in A are positive with mobility mu; those in B are negative with mobility 2mu. The common positive mu is unknown between 0.2 and 1.5 m²/(V s). Local carrier momentum relaxation is fast. There is no anomalous Hall effect, magnetic response, thermoelectric effect, or frequency dependence.

Flat parallel layers fill the material. `fraction` is A's volume fraction; `normal_angle` specifies their normal in the laboratory xy plane, measured from +x. The layer period is much smaller than the specimen and measurement distances. Interfaces are ideal, with no contact resistance or separate surface conduction. Interface charge may establish the stationary electric field; neglect finite interface thickness and additional material corrections. Use the macroscopic homogenization limit of this geometry.

Apply a uniform static magnetic field `field` along +z. In a region far from contacts and outer edges, impose average current density <jx>=1 A/m² and <jy>=0. Report either `longitudinal_resistivity`=<Ex>/<jx> or `transverse_resistivity`=<Ey>/<jx>, in ohm m. These are signed laboratory components; do not antisymmetrize the transverse signal. The detector measures volume-averaged electric fields.

Every experiment supplies `field` in [-2.5,2.5] T, `fraction` in [0,1], `normal_angle` in [0,pi], and `observable`. Calibration records longitudinal resistivity in pure-phase specimens (`fraction` 0 or 1) at varied fields, and in mixed laminates at zero magnetic field with varied fractions and orientations. Errors are independent Gaussian with known standard deviations; all settings and geometry are exact. One mu applies throughout.

Implement `Model.fit(records)`, returning `self`, and `Model.predict(experiments)`, returning a finite NumPy array of shape `(len(experiments),)`. Store mu in `Model.mobility` in m²/(V s). Records in `data/calibration.json` contain `input`, `value`, and `sigma`.
