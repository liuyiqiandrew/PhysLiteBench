# Effusive beam

A dilute equilibrium argon gas supplies an effusive beam through an aperture
much smaller than the mean free path. The wall normal pointing into vacuum is +z.
The reservoir remains at one unknown temperature T. Atoms have mass
39.948*1.66053906660e-27 kg. The Boltzmann constant is 1.380649e-23 J/K.
There are no collisions beyond the aperture and no external forces.
The data represent individual outward aperture-crossing events. Every event is
counted once with equal detection efficiency, independent of speed and direction.
There is no downstream angular selection or residence-time weighting. The
instrument measures velocity components for each event. An experiment requests
the fraction of these events whose specified velocity component is at most v.

## Interface

Each experiment has `component` (`x`, `y`, or `z`) and `threshold`
(velocity v in m/s, -1200 to 1200). Return its cumulative probability.
`temperature` is in kelvin, between 150 and 1000.

Implement `Model` in `model.py`. `fit(records)` must return `self` and set
`temperature`. `predict(experiments)` must return a finite NumPy array of shape
`(len(experiments),)`. You may edit the entire implementation within this API.

`data/calibration.json` is a list of records. Each record has an `input`
experiment dictionary, measured `value`, and known Gaussian measurement standard
deviation `sigma`. Errors in different records are independent. Experimental
settings are exact. Use the fitted apparatus parameter for all predictions over
the stated input ranges.

Run `python -m pytest -q test_public.py` to check the interface and calibration.
