# Pulsed gas density

A dilute classical gas moves around a ring of circumference `2*pi` length units. Particles do not interact. Before each run, position is uniform and velocity is an independent zero-mean Gaussian with standard deviation `thermal_speed`. The particle mass and Boltzmann constant are one in the chosen units.

Between impulses the unwrapped coordinate and velocity obey

```
dx = v dt
dv = -friction*v dt + sqrt(2*friction)*thermal_speed dW
```

`W` is a standard Wiener process for each particle. The bath remains attached throughout the run. `friction`, in inverse time units, is the same unknown number in `[0.08, 0.4]` for every run. `thermal_speed` is set separately for each run. Coordinates used by the impulses and detector are modulo `2*pi`.

At time zero, a spatial impulse changes velocity by `a1*sin(k1*x + phase1)`. At time `delay`, a second impulse changes velocity by `a2*sin(k2*x + phase2)`, evaluated at the particle's position at that time. Positions are continuous across each impulse; the stated velocity change is added to the incoming velocity. The apparatus makes no other intervention. If `delay=0`, apply the first impulse and then the second at the same position. A zero amplitude means that impulse is absent.

At time `delay + after`, measure the ensemble average of `cos(mode*x)` when `component="cos"`, or `sin(mode*x)` when `component="sin"`. The ensemble includes the initial preparation and all bath noise. These are signed, dimensionless density Fourier components, with no conditioning or normalization by pulse amplitude.

## Inputs and outputs

Implement `Model` in `model.py`:

- `fit(records)` sets the scalar attribute `friction` and returns `self`.
- `predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` in input order.

Each experiment is a dictionary with all these fields:

| Field | Meaning and allowed values |
|---|---|
| `thermal_speed` | Equilibrium velocity standard deviation, `[0.5, 1.0]` length/time |
| `a1`, `a2` | Signed velocity impulse amplitudes, `[-1.0, 1.0]` length/time |
| `k1`, `k2` | Spatial wave numbers, integers 1 through 3 in inverse length units |
| `phase1`, `phase2` | Spatial phases, `[-pi, pi]` radians |
| `delay` | Time between impulses, `[0, 3.5]` |
| `after` | Time from second impulse to readout, `[0.2, 3.5]` |
| `mode` | Detector wave number, integer 1 through 3 |
| `component` | `"cos"` or `"sin"` |

`data/calibration.json` is a list of records with `input` (an experiment), `value` (a measured Fourier component), and `sigma` (its known independent Gaussian measurement standard deviation). Calibration uses single-impulse preparations. Predictions can use any preparation in the stated domain.

Run `python -m pytest -q test_public.py` to check the interface and calibration. You may replace any implementation code while preserving this API.
