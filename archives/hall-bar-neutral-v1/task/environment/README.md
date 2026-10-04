# Hall bar

A long rectangular conducting bar contains one isotropic classical carrier
population of density n = 2e21 m^-3 and charge q = +1.602176634e-19 C.
Momentum relaxes with one field-independent relaxation time. The carrier mobility
is unknown. Ignore quantum effects, heating, and any field dependence of density.
A voltage source maintains the specified longitudinal electric field Ex in the
uniform central section. The two lateral faces are electrically insulating.
Measurements are made after charge redistribution has reached steady state,
far from the contacts. A uniform magnetic field B points along +z. The bar's
longitudinal direction is x and its transverse direction is y.
The observable is longitudinal current density jx in A/m^2.

## Interface

Each experiment has `electric_field` (Ex, V/m, -3 to 3) and `magnetic_field`
(B, tesla, -6 to 6). Return the signed longitudinal current density.
`mobility` is positive, in m^2/(V s), between 0.05 and 1.5.

Implement `Model` in `model.py`. `fit(records)` must return `self` and set
`mobility`. `predict(experiments)` must return a finite NumPy array of shape
`(len(experiments),)`. You may edit the entire implementation within this API.

`data/calibration.json` is a list of records. Each record has an `input`
experiment dictionary, measured `value`, and known Gaussian measurement standard
deviation `sigma`. Errors in different records are independent. Experimental
settings are exact. Use the fitted apparatus parameter for all predictions over
the stated input ranges.

Run `python -m pytest -q test_public.py` to check the interface and calibration.
