# Temperature and voltage of a conducting rod

A uniform thermoelectric rod of length 0.01 m has insulated lateral faces and
negligible mechanical expansion. Its two end temperatures are held at the
specified values by ideal reservoirs. A current source maintains the specified
spatially uniform electrical current density J, positive along increasing x.
There are no other heat paths. Electrical relaxation is much faster than the
temperature measurements. Local equilibrium and linear thermoelectric transport
apply; the material is reciprocal and isotropic, with no magnetic field.

The volume heat capacity is 1.5e6 J/(m³ K), electrical resistivity is 2e-5 ohm m,
and thermal conductivity at zero current is the unknown positive constant k
between 0.6 and 2.5 W/(m K). These three properties are temperature independent.
The measured Seebeck coefficient is S(T)=2e-4+4e-6(T-300) V/K, with T in kelvin.
The electric-field convention is E=rho J+S(T) dT/dx. Report the terminal
voltage integral of E from the left end to the right end.

`data/calibration.npz` contains `x` (m), `t` (s), `initial` (R×X kelvin),
`current` (R values in A/m²), `boundary` (R×2 end temperatures),
`temperature` (R×T×X kelvin), `voltage` (R×T volts), and their independent
Gaussian uncertainties `sigma_temperature` and `sigma_voltage`. Initial
temperatures are linear between mesh nodes and match the two reservoirs.
All settings are exact. Currents can range from -2.5e5 to 2.5e5 A/m².

Complete `Model` in `model.py`. `fit(data)` returns self and stores k in
`conductivity`. `predict(t, x, initial, current, boundary)` returns a dictionary
with finite NumPy arrays `temperature` of shape `(len(t),len(x))` and `voltage`
of shape `(len(t),)`. The equally spaced mesh includes both ends. Times are
nonnegative and increasing from preparation and need not include zero. Use
one fitted conductivity for all preparations.

Run `python -m pytest -q test_public.py`.
