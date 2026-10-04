# Elastic bulk-wave measurements

An infinite homogeneous compressible elastic solid has reference mass density `rho0 = 1000 kg/m^3`. Its stored elastic energy per reference volume is

    W(F) = 10^6 modulus [0.3 (I1 - 3) + 0.2 (I2 - 3)
                        - 1.4 log(J) + log(J)^2] J/m^3,
    C = F^T F,  I1 = tr(C),  I2 = [(tr C)^2 - tr(C^2)]/2,  J = det(F).

The unknown positive `modulus` is in MPa, lies in `[0.8, 1.6]`, and is shared by every preparation. This energy is the complete constitutive law. There is no dissipation, thermal expansion or material dispersion.

Each preparation has a spatially uniform static deformation `x = deformation @ X` relative to its unstressed reference. Remote actuators maintain that deformation. Infinitesimal bulk plane waves are measured far from the actuators in the static current configuration. Their wavelength is large compared with the material's microscopic scale. The material is at rest before the wave is launched. The total mass is conserved during preparation. The measurement is phase speed in meters per second, for a specified branch of the three bulk acoustic modes.

Each experiment supplies:

- `deformation`: the 3 by 3 deformation gradient from reference to current Cartesian coordinates, with positive determinant. Its singular values lie in `[0.85, 1.45]` and its determinant lies in `[0.95, 1.18]`.
- `direction`: a nonzero three-vector specifying the wavevector direction in current Cartesian coordinates; normalize it to unit length. Wavevector magnitude and phase speed are measured in current distance units.
- `branch`: `0`, `1`, or `2`, selecting the phase speeds in nondecreasing order, including multiplicity.

`data/calibration.json` contains independent readings with keys `input`, `value`, and `sigma`. Every speed has known instrument uncertainty `sigma = 0.02 m/s`.

Implement `Model.fit(records)` and `Model.predict(experiments)` in `model.py`. `fit` returns `self` and stores the inferred parameter in `self.modulus`. `predict` returns a finite NumPy array of shape `(len(experiments),)` in meters per second. You may use any numerical method consistent with the stated apparatus.
