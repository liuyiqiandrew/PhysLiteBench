# Magnetic equilibrium

A ferromagnet consists of N classical spins s_i = +/-1 with uniform
infinite-range interaction. Its Hamiltonian is
H = -J/(2N)*(sum_i s_i)^2 - h*sum_i s_i.
Take the thermodynamic limit N -> infinity. The positive coupling J is unknown.
The magnet is in canonical equilibrium with a bath at temperature T; equilibrate
fully at each field and temperature. The task concerns equilibrium, with no
metastable preparation or hysteresis protocol. Use units in which kB=1;
J, h, and T are given in the same energy unit. The observable is equilibrium
magnetization per spin m. Every requested field is nonzero.

## Interface

Each experiment has `temperature` (0.4 to 4.) and `field` (-1 to 1, nonzero).
Return magnetization per spin, between -1 and 1.
`coupling` is J, between 0.3 and 1.2, in the energy unit above.

Implement `Model` in `model.py`. `fit(records)` must return `self` and set
`coupling`. `predict(experiments)` must return a finite NumPy array of shape
`(len(experiments),)`. You may edit the entire implementation within this API.

`data/calibration.json` is a list of records. Each record has an `input`
experiment dictionary, measured `value`, and known Gaussian measurement standard
deviation `sigma`. Errors in different records are independent. Experimental
settings are exact. Use the fitted apparatus parameter for all predictions over
the stated input ranges.

Run `python -m pytest -q test_public.py` to check the interface and calibration.
