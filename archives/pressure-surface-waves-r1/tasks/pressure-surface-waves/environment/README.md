# Surface waves in a loaded elastic solid

A homogeneous isotropic hyperelastic solid occupies a half-space. Its strain energy per undeformed volume is

    W(F) = mu/2 * (tr(F^T F) - 3 - 2*ln(J)) + lambda/2 * ln(J)^2,
    J = det(F),   mu = 10^6 Pa,   lambda = 2*10^6 Pa.

These elastic constants are known. The unknown density per undeformed volume, `density`, lies in `[900, 1300] kg/m^3` and is common to all experiments. Mass is conserved during deformation. The material has no dissipation or surface energy, and gravity is negligible.

The homogeneous static deformation has principal stretches `(stretch, 1, normal_stretch)`. The two tangential stretches are maintained remotely. The exposed plane is normal to the third principal axis. An ideal pressure actuator applies known compressive pressure `pressure` to this face: the force is normal to the instantaneous surface, and its magnitude per actual surface area remains equal to that pressure during motion. The actuator has negligible inertia and adds no mass, elasticity or tangential traction. The normal stretch is the positive value selected by static normal force balance. The flat face is at current coordinate `z=0`, with solid in `z>0`.

Measure a small-amplitude plane surface wave propagating along the current first principal axis `x`. The displacement lies in the `(x,z)` plane, is independent of the second coordinate, and decays into the solid. Its amplitude tends to zero before the phase speed is measured. There is no additional applied time-dependent force. The requested observable is the phase speed `omega/k` in m/s, where `k` is the wave number measured along the deformed surface. The specified continuum and loading regime supports a stable subsonic surface-wave branch. No change to the imposed tangential prestretches occurs far from the wave.

Each input contains `stretch` in `[0.95, 1.2]` and `pressure` in `[0, 800000] Pa`. Calibration records in `data/calibration.json` contain `input`, measured `value`, and independent Gaussian uncertainty `sigma = 0.002 m/s`.

Implement `Model.fit(records)`, returning `self` and storing the inferred `self.density`. `Model.predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` containing the phase speeds in input order. You may use any numerical method consistent with the apparatus.
