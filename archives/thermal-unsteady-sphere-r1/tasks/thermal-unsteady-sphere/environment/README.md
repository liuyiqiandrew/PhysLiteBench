# Driven and thermal sphere measurements

A rigid sphere of radius `radius` and mass density 2200 kg/m³ is immersed in an
unbounded Newtonian liquid of density 1000 kg/m³. The liquid and sphere are at
uniform equilibrium temperature `temperature`. The liquid has constant material
properties throughout the stated temperature range. Its dynamic viscosity is
the only unknown. The sphere has a no-slip impermeable surface. There are no
nearby walls, traps, imposed background flows, or other particles. Gravity is
compensated.

Use the classical linear incompressible continuum description. Retain local
fluid acceleration and the particle's inertia; neglect convective inertia.
At infinity the mean fluid velocity is zero. Thermal fluctuations arise from
the same equilibrated viscous liquid. There is no additional random force
source, particle heating, or measurement backaction. The stated frequencies
are well below acoustic and molecular scales; the incompressible model defines
the requested readings in this range.

For a driven reading an external force along a fixed Cartesian axis is
`F(t)=Re[F0 exp(-i omega t)]`, with real amplitude `F0=1e-13 N`.
The response has reached its steady or periodic state. Write the mean particle
velocity on this axis as `Re[V exp(-i omega t)]`. An `in_phase` reading is
`Re(V)`, and a `quadrature` reading is `Im(V)`, both in µm/s.

For a `velocity_psd` reading the external force is absent. The particle velocity
has its stationary equilibrium statistics. The measured spectrum of the same
Cartesian component is

```
Svv(omega) = integral from -infinity to infinity
            exp(i omega t) <v(t) v(0)> dt.
```

This two-sided angular-frequency spectrum is reported in µm²/s. The detector
resolves particle velocity directly without filtering or added detector noise.
Free diffusion of the particle's position is allowed.

Each experiment dictionary contains:

- `radius` in `[.5,1.5]`, in µm;
- `angular_frequency`, either zero or in `[.2,8]`, in units of `1e6 rad/s`;
- `temperature` in `[280,320]`, in K;
- `observable`, one of `"in_phase"`, `"quadrature"`, or `"velocity_psd"`.

Use Boltzmann's constant `kB=1.380649e-23 J/K`. The common unknown
`viscosity` lies in `[.7,1.4]` mPa s. Radius, density, temperature, frequency and
force amplitude are known controls.

## Interface

Implement `Model.fit(records)` and `Model.predict(experiments)` in `model.py`.
`fit` receives records from `data/calibration.json`, stores the fitted viscosity
in mPa s in `self.viscosity`, and returns `self`. Each record contains `input`,
measured `value`, and independent Gaussian instrument standard deviation
`sigma`, in the units of that reading. `predict` receives a list of experiment
dictionaries and returns a one-dimensional NumPy array of readings in the same
order.

Run `python -m pytest -q test_public.py` to check the interface and calibration.
