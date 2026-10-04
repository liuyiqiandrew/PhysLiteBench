# A sphere driven through a viscous liquid

A rigid sphere of radius a=0.001 m and density 2300 kg/m^3 translates without rotation along the z axis in an unbounded incompressible Newtonian liquid. The liquid density is 1000 kg/m^3. Its unknown dynamic viscosity `viscosity` is between 0.004 and 0.016 Pa s, constant across experiments. The particle mass is its density times 4*pi*a^3/3. There are no walls, trap, gravity, Brownian forcing, or compressibility effects.

The motion is in the linear hydrodynamic regime: neglect convective acceleration, but retain the local fluid acceleration. In the exterior fluid solve

    rho * partial_t u = -grad(p) + viscosity * Laplacian(u),
    div(u) = 0.

The no-slip fluid velocity at the sphere equals its translational velocity, and fluid velocity tends to zero at infinity. The mechanical force exerted by the fluid on the sphere is the integral of the pressure and viscous traction over its surface. The sphere obeys Newton's law with that force and the prescribed external force. For oscillatory motion, work to leading order in the displacement amplitude, imposing the surface condition on the mean sphere surface. The allowed force amplitudes and frequencies keep both convective Reynolds number and relative displacement small.

Each experiment applies an external force

    F_ext(t) = force * cos(frequency * t)

along z. `force` is a signed amplitude in [-1e-9, 1e-9] newtons. `frequency` is angular frequency in radians/second, either zero or between 5 and 120. Transients have died out before measurement. For nonzero frequency, write the velocity as

    V(t) = V_cos * cos(frequency*t) + V_sin * sin(frequency*t).

Return V_cos for `component: "cosine"` or V_sin for `component: "sine"`, in m/s. Equivalently, the complex amplitude is V_cos+i*V_sin when using exp(-i*frequency*t). At zero frequency, return the terminal steady velocity for cosine and zero for sine. Every input contains `force`, `frequency`, and `component`.

Calibration measures terminal steady velocities under constant external forces. These data identify the viscosity. Measurement errors are independent Gaussian errors with the stated standard deviations. Predictions at other frequencies use the same material parameters and fluid equations.

Implement `Model.fit(records)`, returning `self`, and `Model.predict(experiments)`, returning a finite NumPy array of shape `(len(experiments),)`. Expose `Model.viscosity`. Calibration records in `data/calibration.json` have keys `input`, `value`, and `sigma`. The starter provides a frequency-response calculation; check its hydrodynamic assumptions against the complete apparatus.
