# Radiative plates

Two very large parallel, opaque plates face each other across a vacuum.
Their separation is much greater than thermal wavelengths and small compared
with their extent, so their mutual view
factor is one and edge leakage is negligible. Both surfaces are diffuse gray
and in local thermal equilibrium, with wavelength-independent emissivity over
the relevant spectrum. The gap neither absorbs nor emits. Supports conduct no
heat. Thermostats maintain the specified surface temperatures T1 and T2.
Plate 1 has constant unknown emissivity epsilon1. Plate 2 accepts a panel whose
known emissivity epsilon2 is supplied for each run. Panels can range from black
to reflective gray. The observable is net steady radiative heat flux leaving
plate 1 per unit facing area, positive from plate 1 to plate 2.
Use sigma=5.670374419e-8 W/(m^2 K^4).

## Interface

Each experiment has `temperature_1` and `temperature_2` (K, 250 to 700),
and `emissivity_2` (dimensionless, 0.1 to 1).
Return heat flux in W/m^2. `emissivity` is epsilon1, between 0.05 and 0.95.

Implement `Model` in `model.py`. `fit(records)` must return `self` and set
`emissivity`. `predict(experiments)` must return a finite NumPy array of shape
`(len(experiments),)`. You may edit the entire implementation within this API.

`data/calibration.json` is a list of records. Each record has an `input`
experiment dictionary, measured `value`, and known Gaussian measurement standard
deviation `sigma`. Errors in different records are independent. Experimental
settings are exact. Use the fitted apparatus parameter for all predictions over
the stated input ranges.

Run `python -m pytest -q test_public.py` to check the interface and calibration.
