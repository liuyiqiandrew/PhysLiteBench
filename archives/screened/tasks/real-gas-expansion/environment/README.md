# Real gas expansion

One mole of a gas obeys the van der Waals equation
P = n R T/(V - n b) - a n^2/V^2, with n=1 mol,
R=8.31446261815324 J/(mol K), a=0.36 Pa m^6/mol^2, and
b=4.3e-5 m^3/mol. Its molar constant-volume heat capacity Cv is constant over
the stated temperature and volume range, but unknown. Treat this as an equilibrium
thermodynamic material and neglect any surface energy. All allowed endpoint
states are in the single stable gas phase.
The apparatus supports two protocols. In `heating`, the gas stays at fixed volume
while a measured heat input changes its temperature from T0 to T1. Return the
heat supplied in joules. In `expansion`, an insulated rigid outer vessel has two
compartments. Initially the gas occupies V0 at T0 and the other compartment is
evacuated. A partition is removed, and the gas equilibrates in total volume V1.
The vessel exchanges no heat or work with its surroundings. Return final gas
temperature in kelvin.

## Interface

For `protocol="heating"`, inputs are `initial_temperature`, `final_temperature`
(K, 300 to 450), and `volume` (m^3, 0.0003 to 0.002).
For `protocol="expansion"`, inputs are `initial_temperature` (K, 400 to 550),
`initial_volume` and `final_volume` (m^3, 0.0003 to 0.003), with V1 >= V0.
`heat_capacity` is molar Cv in J/(mol K), between 15 and 40.

Implement `Model` in `model.py`. `fit(records)` must return `self` and set
`heat_capacity`. `predict(experiments)` must return a finite NumPy array of shape
`(len(experiments),)`. You may edit the entire implementation within this API.

`data/calibration.json` is a list of records. Each record has an `input`
experiment dictionary, measured `value`, and known Gaussian measurement standard
deviation `sigma`. Errors in different records are independent. Experimental
settings are exact. Use the fitted apparatus parameter for all predictions over
the stated input ranges.

Run `python -m pytest -q test_public.py` to check the interface and calibration.
