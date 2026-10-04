# Tagged-particle covariance

Identical Brownian point particles move on an infinite straight line. Their common diffusion constant is `D`; an isolated particle has displacement variance `2*D*t`. Thermal forcing of different particles is independent. Particles cannot pass each other: the many-particle diffusion has perfectly reflecting collision boundaries at coincident neighboring positions. There are no other interactions, external forces, hydrodynamic couplings or particle exchanges.

In each independent experimental shot, particle `j` starts exactly at `j/density` for every integer `j`. The particle at the origin is marked without changing its dynamics. All position constraints are released at time zero, which is also the start of the measurement clock. This identical initial configuration is prepared afresh for every shot. The detector follows the marked particle, whose position is `X(t)`, without exerting a force.

For two fixed positive times `time_a` and `time_b`, the reported quantity is

    lim(lambda -> infinity) Cov[X(lambda*time_a), X(lambda*time_b)] / sqrt(lambda).

Here `lambda` is dimensionless and scales both times together. The covariance is over independent shots with the stated identical starting positions; both positions in a shot belong to the same marked particle trajectory. The infinite-line, infinite-file model is used before this time limit. The reported quantity is the limiting coefficient, rather than a measurement at a specified finite lambda.

Use fixed length and time units `L0` and `T0`. Numerical positions are in `L0`, times in `T0`, density in `1/L0`, and D in `L0**2/T0`. The output is in `L0**2`. The common unknown D lies in `[0.8, 1.2]`. Each input contains:

| Field | Meaning | Allowed values |
|---|---|---|
| `density` | Initial number per unit length | `[0.7, 1.5]` |
| `time_a` | First fixed time multiplier, in `T0` | `[0.25, 4.0]` |
| `time_b` | Second fixed time multiplier, in `T0` | `[0.25, 4.0]` |

`data/calibration.json` is a list of independent records with `input`, `value`, and `sigma`. `value` estimates the specified limiting covariance coefficient. Each Gaussian instrument error has the known fixed standard deviation `sigma=0.002`, in output units.

Implement `Model.fit(records)`, returning `self` and storing the inferred common diffusivity as `self.diffusivity`. Implement `Model.predict(experiments)`, returning a finite one-dimensional NumPy array in input order. Predictions use the fitted diffusivity and the known controls in each experiment.
