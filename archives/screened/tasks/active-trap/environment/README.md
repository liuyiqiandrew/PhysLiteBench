# Active particles in a planar trap

Dilute, noninteracting particles move in an unbounded plane. Inertia is negligible. Each particle propels at constant speed `speed` along its own axis. That axis undergoes free rotational Brownian motion: its angle increment has zero mean and variance `2 * rotational_diffusion * dt`. There is no torque. Independent translational Brownian increments have variance `2 * diffusion * dt` along each Cartesian axis. An isotropic harmonic trap adds deterministic velocity `-trap_rate * (x, y)` relative to its center. All noises are independent, and all particles share the same unknown rotational diffusion coefficient.

Lengths are in micrometers and times in seconds. `rotational_diffusion` is in [0.4, 1.1] per second. Each input specifies a separate preparation. The readouts are:

- `free_msd`: release at the origin with uniformly distributed axes, no trap, and measure the ensemble mean squared two-dimensional displacement after `time` seconds.
- `trap_variance`: allow full stationary relaxation in the specified trap, then measure the variance of one Cartesian position coordinate.
- `trap_fourier`: after full stationary relaxation, measure the ensemble mean of `cos(wavenumber * x)`, where x is measured from the trap center. The detector returns a signed mean, not its magnitude or square.

Every input contains `readout`, `speed` in [0, 2.4] micrometers/second and `diffusion` in [0, 0.12] square micrometers/second. Free measurements have `time` in [0.1, 8]. Trapped measurements have `trap_rate` in [0.4, 2.5] per second. Fourier measurements additionally have `wavenumber` in [0, 12] inverse micrometers, with `wavenumber * speed / trap_rate <= 7`. The two position readouts have units of square micrometers; the Fourier readout is dimensionless.

Calibration records in `data/calibration.json` contain `input`, measured `value`, and independent Gaussian standard deviation `sigma`. Implement `Model.fit(records)`, return `self`, and store the fitted coefficient as `self.rotational_diffusion`. `Model.predict(experiments)` must return a finite NumPy array of shape `(len(experiments),)` in input order. Each prediction uses its own preparation. Use only the supplied calibration data to fit the common coefficient.
