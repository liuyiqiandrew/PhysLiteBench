# Pressure of an equilibrated fluid

A single-component classical fluid has the following molar Helmholtz free energy
when constrained to be homogeneous:

    f(v, T) = -T*ln(v - 1) - attraction/v,  v > 1.

Reduced units set the known excluded molar volume b and gas constant R to one.
Volume v, temperature T, free energy per mole f, pressure P, and `attraction`
are expressed in these same reduced units. A term depending only on temperature
can be omitted from f because the requested measurements are isothermal.
The material constant `attraction` is positive and unknown, between 2.6 and 3.4.
The model is exact for this idealized fluid; no other material parameters need
fitting. There is no additional solid phase.

The apparatus contains a fixed amount of fluid in a vessel with prescribed total
volume and maintains a specified uniform temperature. The input `volume` is total
volume divided by the total amount of fluid. The fluid reaches global thermal,
mechanical, and material equilibrium before its pressure is read. Macroscopic
regions of different densities are allowed to form and exchange material; no
constraint enforces a homogeneous density. Work in the thermodynamic limit and
neglect interface energy, capillary pressure, gravity, and any delay due to
nucleation. The observable is the equilibrium pressure on the vessel wall.

## Interface

Each experiment has `temperature` (T, 0.6 to 2.2) and `volume` (v, 1.3 to 35).
All inputs are exact. Return pressure in the reduced units defined above.

Implement `Model` in `model.py`. `fit(records)` must return `self` and set
`attraction`. `predict(experiments)` must return a finite NumPy array of shape
`(len(experiments),)`. You may edit the entire implementation within this API.

`data/calibration.json` is a list of records. Each record has an `input`
experiment dictionary, measured `value`, and known Gaussian measurement standard
deviation `sigma`. Errors in different records are independent. Use the fitted
material parameter for predictions over the stated input ranges.

Run `python -m pytest -q test_public.py` to check the interface and calibration.
