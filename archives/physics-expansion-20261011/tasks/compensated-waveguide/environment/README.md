# Spectroscopy of a narrow channel

A single nonrelativistic quantum particle occupies a straight, centered planar
channel with hard walls. Length is measured in a fixed unit and energy in
units where hbar²/(2m)=1. For each positive width parameter `epsilon`, the
accessible region is exactly

```
0 < x < 1,   -epsilon*a(x)/2 < y < epsilon*a(x)/2,
a(x) = exp(amplitude*cos(2*pi*x) + harmonic*cos(4*pi*x + phase)).
```

The wavefunction vanishes on both side walls and on the end caps at x=0,1.
There is no centerline bending. The Hamiltonian in physical Cartesian
coordinates is

```
H_epsilon = -d²/dx² - d²/dy²
            + well*cos(2*pi*x) - pi²/(epsilon*a(x))².
```

The last two terms are an externally applied potential, constant across each
cross section. The negative term is tuned exactly as written at each width;
it cancels the local ground transverse energy. It is part of the apparatus,
not a subtraction performed on the measured gaps. There are no other fields,
interactions, or boundary potentials.

Let E_0(epsilon) <= E_1(epsilon) <= ... be the eigenvalues of this full
two-dimensional Hamiltonian, ordered with multiplicity. A spectrometer
measures

```
value = gain * lim_(epsilon -> 0) [E_level(epsilon) - E_0(epsilon)].
```

Hold every input below fixed while taking the width limit. The integer
`level` labels the full ordered spectrum, with ground level zero. The common
unknown dimensionless spectrometer gain is in `[0.8, 1.3]`. The readout has
energy units. 

## Interface

Implement `Model` in `model.py`:

- `fit(records)` sets scalar attribute `gain` and returns `self`.
- `predict(experiments)` returns a finite NumPy array of shape
  `(len(experiments),)` in input order.

Every experiment contains:

| Field | Allowed values |
|---|---|
| `amplitude` | `[0, 0.8]` |
| `harmonic` | `[0, 0.5]` |
| `phase` | `[0, 2*pi]` radians |
| `well` | `[-20, 20]` in energy units |
| `level` | Integer 1, 2, or 3 |

`data/calibration.json` contains records with `input`, measured `value`, and
known independent Gaussian standard deviation `sigma`. Calibration uses
constant-width channels (`amplitude=harmonic=0`) with varied well strengths
and spectral levels. Prediction experiments can use the full stated domain.
All preparations use the same gain.

Run `python -m pytest -q test_public.py` to check interface and calibration.
You may replace any implementation code while preserving the API.
