# Equilibrium magnetization with two- and four-spin interactions

A magnet consists of N classical spins s_i = +/-1. Write S=sum_i s_i.
The uniform infinite-range interaction has Hamiltonian

    H = -J*S^2/(2*N) - Q*S^4/(4*N^3) - h*S.

The positive two-spin coupling J is unknown, between 0.3 and 0.8.
The four-spin coupling is known: Q=1.2. Take the thermodynamic limit
N -> infinity. This Hamiltonian defines the idealized material exactly;
there are no additional interaction terms or unknown material constants.

The magnet is in canonical equilibrium with a bath at temperature T.
Each measurement starts a fresh experiment and allows full equilibration
at its specified field and temperature. No constraint fixes the initial
magnetization or retains a branch from an earlier experiment. Ignore finite
size, domains, and delays from nucleation. The observable is equilibrium
magnetization per spin m=<S>/N. Every requested field is nonzero, and all
requested states are away from phase-transition lines.

Use units in which kB=1. J, Q, h, and T share one energy unit; magnetization
per spin is dimensionless.

## Interface

Each experiment has `temperature` (T, 0.55 to 3.0) and `field`
(h, -0.9 to 0.9, nonzero). Return magnetization per spin between -1 and 1.

Implement `Model` in `model.py`. `fit(records)` must return `self` and set
`coupling` to the fitted J. `predict(experiments)` must return a finite NumPy
array of shape `(len(experiments),)`. You may edit the entire implementation
within this API.

`data/calibration.json` is a list of records. Each record has an `input`
experiment dictionary, measured `value`, and known independent Gaussian
measurement standard deviation `sigma`. Experimental settings are exact.
Use the same fitted coupling and known Q for all allowed preparations.

Run `python -m pytest -q test_public.py` to check the interface and calibration.
