# Induced dipole

An isotropic neutral particle has no permanent electric dipole. Its induced
dipole responds reversibly and linearly to the local externally imposed static
electric field, p = alpha E. The polarizability alpha is unknown. The field source
maintains its specified field while the particle moves. Neglect interactions
between particles, radiation, thermal fluctuations, and the particle's effect
on the source. The particle is small compared with the field variation scale.
Along the measurement axis the field is E(x)=E0+g*x, directed along that axis;
it is the on-axis profile of a three-dimensional electrostatic field.
The apparatus measures either axial dipole moment or axial mechanical force.
The laboratory reports fields in kV/m, positions in mm, gradients in
(kV/m)/mm, dipoles in 1e-21 C m, forces in 1e-15 N, and polarizability in
1e-24 C m^2/V. These unit choices give p=alpha*E numerically.

## Interface

Each experiment has `observable` (`dipole` or `force`), `field_offset`
(E0, -5 to 5), `gradient` (g, -3 to 3), and `position` (x, -1 to 1),
in the laboratory units above. `polarizability` is in those units, between 0.2 and 5.

Implement `Model` in `model.py`. `fit(records)` must return `self` and set
`polarizability`. `predict(experiments)` must return a finite NumPy array of shape
`(len(experiments),)`. You may edit the entire implementation within this API.

`data/calibration.json` is a list of records. Each record has an `input`
experiment dictionary, measured `value`, and known Gaussian measurement standard
deviation `sigma`. Errors in different records are independent. Experimental
settings are exact. Use the fitted apparatus parameter for all predictions over
the stated input ranges.

Run `python -m pytest -q test_public.py` to check the interface and calibration.
