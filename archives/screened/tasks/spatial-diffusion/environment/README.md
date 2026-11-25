# Diffusion in a patterned medium

Dilute, noninteracting Brownian particles move along a periodic channel of length
L = 10 micrometers. Position x is identified modulo L. The accessible cross section
is constant, and there is no potential or imposed flow. The surrounding bath is
at a uniform temperature of 300 K. A patterned dissipative coating changes the
local mechanical drag to zeta(x) = zeta0 / g(x), where

    g(x) = 1 + contrast * cos(2*pi*x/L).

The coating changes drag only; it does not change accessible volume or exert a
conservative force. Thermal force fluctuations obey the local fluctuation-
dissipation relation with this same drag and temperature. Measurements are in
the overdamped limit: momentum relaxation is much faster than every observation
time. The unknown uniform-medium diffusivity D0 = k_B T / zeta0 is in
micrometer^2/s, between 0.1 and 1.2. The drag profile and its contrast are exact.

At time zero, prepare the probability density

    p(x, 0) = [1 + first*cos(2*pi*x/L) + second*cos(4*pi*x/L)] / L.

The measured value is the ensemble average of cos(2*pi*mode*x/L) at the requested
time. This is an average over particle positions, with uniform detection
sensitivity per particle. It is dimensionless and lies between -1 and 1.

## Interface

Each experiment dictionary has `contrast` (-0.8 to 0.8), `first` and `second`
(dimensionless, with abs(first)+abs(second) <= 0.8), `time` (seconds, 0 to 80),
and `mode` (integer 1 or 2). Initial preparation and time conventions are the
same for every contrast. The channel has periodic boundary conditions.

Implement `Model` in `model.py`. `fit(records)` must return `self` and set
`diffusivity`. `predict(experiments)` must return a finite NumPy array of shape
`(len(experiments),)`. You may edit the entire implementation within this API.

`data/calibration.json` is a list of records. Each record has an `input`
experiment dictionary, measured `value`, and known Gaussian measurement standard
deviation `sigma`. Errors in different records are independent. Experimental
settings are exact. Use the fitted apparatus parameter for predictions over
the stated input ranges.

Run `python -m pytest -q test_public.py` to check the interface and calibration.
