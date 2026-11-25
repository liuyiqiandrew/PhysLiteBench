# Particle capture by a spherical coating

A stationary sphere of radius a is surrounded by an infinite three-dimensional dilute solution of noninteracting Brownian particles. The solvent is at rest. There are no forces outside the sphere and particle motion has a common unknown diffusivity D. Use fixed length and time units L0 and t0, with `0.7 <= diffusivity <= 1.3` in units L0^2/t0. The particle concentration far from the sphere is maintained at the known constant `c_infinity=1` in units L0^-3.

The entire spherical coating has one state G, either 0 or 1. This single state follows a continuous-time Markov process: the transition rate from 0 to 1 is k01, and the rate from 1 to 0 is k10. These known rates are independent of all particle positions, arrival events and reaction counts. A state transition changes the coating uniformly and instantaneously.

In state i the coating has the known finite intrinsic reaction velocity kappa_i. Its definition is the usual diffusion-reaction boundary rule: removal flux per surface area equals kappa_i times the particle concentration immediately outside the surface. Unreacted particles are reflected. This local rule applies with the current state of the coating, without a delayed surface reaction or stored surface population. Zero reaction velocity means reflection. Reactions are irreversible, products are removed immediately, and there is no saturation or site blocking. Absorption does not change the target or its switching rates.

The coupled particle distribution and coating process have reached their stationary state under the maintained reservoir before each reading. The observable is the long-time mean number of particles absorbed by this sphere per unit time. Every absorption contributes one count. Positive output denotes capture, in units t0^-1.

Each experiment supplies these exact known controls:

- `radius`: a, from 0.6 to 1.8, in units L0;
- `reactivity0` and `reactivity1`: kappa_0 and kappa_1, each from 0 to 15, in units L0/t0;
- `rate01` and `rate10`: k01 and k10, each from 0.03 to 1.5, in units t0^-1.

`data/calibration.json` contains independent records `{"input": {...}, "value": ..., "sigma": 0.01}`. Instrument errors are additive Gaussian with the fixed standard deviation 0.01 t0^-1. They are independent across records.

Implement `Model.fit(records)`, returning `self` and storing the fitted common diffusivity in `self.diffusivity`. Implement `Model.predict(experiments)`, returning a finite one-dimensional NumPy array of mean capture rates in input order. Preserve this interface and the stated units.
