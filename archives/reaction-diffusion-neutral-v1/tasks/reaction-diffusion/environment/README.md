# Electrolyte cell

A quiescent, isothermal ideal dilute solution occupies a uniform one-dimensional cell, 0 ≤ x ≤ 0.001 m. The fully dissociated monovalent salts AC and BC share an anion C⁻; A⁺ and B⁺ have no chemical reactions. The ionic diffusion coefficients are D, 4D, and 2D for A⁺, B⁺, and C⁻ respectively, with D unknown. Both ends are blocking and electrically insulating. No current is applied. The measurements resolve electroneutral bulk transport; Debye layers and charge-relaxation transients are outside the stated model.

`data/calibration.npz` contains `x` (m), `t` (s), `initial` (mol m⁻³; shape R×X×2), `concentration` (R×T×X×2), and `sigma` (same shape). The species axis is [A⁺, B⁺]. Measurements have independent Gaussian errors. Initial concentrations are known and linear between mesh points.

`TransportModel.fit(data)` accepts this dictionary, returns self, and stores positive D in `.diffusivity` (m² s⁻¹). `predict(t, x, initial)` returns a finite NumPy array with shape `(len(t), len(x), 2)`. `x` is an equally spaced mesh spanning the cell. `t` contains increasing nonnegative times measured from preparation.

Run `python -m pytest -q test_public.py`.
