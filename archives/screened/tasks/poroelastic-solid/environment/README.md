# Sealed porous solid

A homogeneous, isotropic porous specimen is fully saturated by a compressible
liquid. Its drained skeleton is a linear elastic solid with unknown Young's
modulus E and known drained Poisson ratio nu=0.25. It obeys small-strain Biot
poroelasticity with Biot coefficient alpha=0.8 and Biot modulus M=2400 MPa.
All constants are uniform and independent of strain.

The apparatus imposes a uniform strain tensor on the specimen and measures a
component of the total Cauchy stress. The reference state has zero strain, zero
stress and zero gauge pore pressure. The specimen is sealed throughout every
experiment: no liquid enters or leaves. Strain is applied quasistatically, and
the state remains uniform. Ignore inertia, gravity, thermal effects and leakage.
The liquid stays in a single phase throughout the allowed strains.

Stress and normal strain are positive in tension. Pore pressure p is positive
in compression. The Biot coefficient is the increase in fluid content per unit
volumetric strain at fixed p; it also gives the isotropic compressive stress
contribution per unit p. The fluid content is the increment of liquid mass
divided by reference liquid density and specimen reference volume. At fixed
strain its increase per unit p is 1/M. These are linear constitutive definitions
relative to the reference state. E and nu refer to the response with pore
pressure held fixed, rather than to the response of an impermeable specimen.

## Interface

Implement `Model` in `model.py`. `fit(records)` must return `self` and set
`young_modulus`, in MPa, between 800 and 2200. `predict(experiments)` must return
a finite NumPy array with shape `(len(experiments),)` containing measured total
stress components in MPa. You may edit the entire implementation within this API.

Each experiment is a dictionary with:

- `strain`: six dimensionless tensor components `[xx, yy, zz, xy, xz, yz]`, each
  between -0.002 and 0.002. Off-diagonal entries are tensor shear strains, not
  engineering shear strains; the tensor is symmetric.
- `component`: one of `"xx"`, `"yy"`, `"zz"`, `"xy"`, `"xz"`, `"yz"`, selecting
  the requested stress component in the same coordinate system.

Each strain is applied independently from the reference state. Predict all
allowed strains using the fitted E and the given material constants.

`data/calibration.json` is a list of records. Each has an `input` experiment,
measured `value` in MPa, and known independent Gaussian measurement standard
deviation `sigma` in MPa. Experimental settings are exact.

Run `python -m pytest -q test_public.py` to check the interface and calibration.
