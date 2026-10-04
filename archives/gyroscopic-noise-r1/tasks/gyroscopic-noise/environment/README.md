# Gyroscopic thermal-motion apparatus

A classical particle moves in a two-dimensional harmonic trap on an unbounded plane. Its mass and Boltzmann constant are one in the units used below. The potential is x^T K x/2, where K=R(theta) diag(k1,k2) R(theta)^T and R(theta)=[[cos(theta),-sin(theta)],[sin(theta),cos(theta)]]. The coordinate vector is x=(x,y). The signed perpendicular magnetic control B produces the force B J v, with J(vx,vy)=(vy,-vx). The harmonic force and magnetic force act throughout preparation and measurement.

The particle is coupled to a single equilibrium thermal reservoir at the specified temperature T. The reservoir is an ideal classical Markov Langevin bath: its deterministic force is -gamma*v, its thermal force is Gaussian and white, and the particle equilibrates with this bath before any signal is recorded. The same positive isotropic drag gamma is used for every experiment. There is no external drive, feedback, additional noise source, confinement boundary, or measurement backaction. All parameters describe this ideal model exactly.

Two displacement detectors share a clock. The recorded scalar signal is

    Q(t) = x(t) + weight*y(t-delay).

The detector delay is nonnegative and is applied to the y channel only. The output is the stationary two-sided power spectral density

    S_Q(omega) = integral from -infinity to infinity
                 exp(i*omega*t) <Q(t) Q(0)> dt.

Frequencies are angular frequencies. The convention for a harmonic component is exp(-i*omega*t). The mean signal is zero. PSD units are displacement squared times time. A zero-frequency input means the continuous omega -> 0 limit. The quoted measurement uncertainty applies to the estimated PSD, not to an extra physical force or detector-noise spectrum.

## Inputs and calibration

Each experiment is a dictionary with these fields:

| Field | Meaning | Allowed range |
|---|---|---|
| `frequency` | omega | [0, 2.2] |
| `field` | signed B | [-1.6, 1.6] |
| `stiffness_1` | k1 | [.7, 1.3] |
| `stiffness_2` | k2 | [1.6, 2.5] |
| `trap_angle` | theta in radians | [-pi/2, pi/2] |
| `temperature` | T | [.5, 1.4] |
| `weight` | y-channel multiplier | [-1.3, 1.3] |
| `delay` | time shift of the y channel | [0, 3] |

The sole unknown is `drag` = gamma in [.4,1.1]. Calibration records in `data/calibration.json` contain `input`, measured `value`, and standard uncertainty `sigma`. The uncertainty is the fixed instrument value .0015 for every record. Records are independent measurements.

## Python interface

Implement `Model` in `/app/model.py`. `Model().fit(records)` must return that instance and set the fitted numeric attribute `drag`. `predict(experiments)` must return a finite one-dimensional NumPy array of PSD values in input order. Use the apparatus and calibration to predict any allowed input. Run `python -m pytest -q /app/test_public.py` for the visible checks.
