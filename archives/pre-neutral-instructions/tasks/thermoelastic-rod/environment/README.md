# Heat exchange in a clamped thermoelastic rod

A slender rod has reference length L=0.05 m and cross-sectional area
A=1e-4 m². Its ends are fixed at their reference separation. Transverse surfaces
are mechanically free. Axial mechanical equilibration is much faster than any
thermal time, and inertia, viscous stress, and mechanical dissipation are
negligible. The rod is initially stress free at the reference temperature
T0=300 K. Temperatures below are departures theta=T-T0 in kelvin.

Use the one-dimensional linear thermoelastic material with Young's modulus
E=2e9 Pa, position-dependent stress-free axial expansion coefficient
alpha(x)=0.002*[1+0.65*cos(2*pi*x/L)] K^-1, and volumetric heat capacity at fixed local axial strain c_e=1e6 J/(m³ K).
The modulus, heat capacity, and conductivity are uniform. Expansion is the
only spatially varying material property. These are effective one-dimensional
properties after transverse relaxation;
there is no additional Poisson-ratio parameter. Thermal conductivity k is
unknown, in [80,220] W/(m K). Heat conduction obeys Fourier's law. Retain terms
linear in temperature departures and axial strains about T0, including reversible
thermoelastic coupling; neglect changes of geometry in the thermal problem.
There are no other constitutive terms or energy sources.

The lateral surface and end x=L are thermally insulated. At time zero, end x=0
is connected to a well-mixed finite thermal body with heat capacity C_b=6 J/K
through a massless contact of known conductance H in W/K. Heat flowing into the
rod is H times the body's temperature minus the rod's surface temperature.
Neither body nor rod exchanges heat elsewhere. H=0 disconnects them. Each
experiment is a fresh preparation; the clamps have already equilibrated
mechanically with the initial temperature profile, and are then held fixed.

Initially, theta(x)=a+b*cos(pi*x/L)+c*cos(2*pi*x/L), and the body's departure is
bath_initial. All specified departures have magnitude at most 1 K, so the
linear model applies. The rod remains elastically stable.

## Interface

Implement `Model.fit(records)`, returning `self` and setting `conductivity`, and
`Model.predict(experiments)`, returning a finite NumPy array of shape
`(len(experiments),)`. You may edit the entire implementation within this API.

Each experiment supplies `mean` (a), `first` (b), `second` (c), `bath_initial`,
`contact` (H, 0 to 1.2 W/K), `time` (0 to 80 s), and `observable`:

- `mean`: spatially averaged rod temperature departure.
- `first`: (1/L) integral theta(x)*cos(pi*x/L) dx.
- `second`: (1/L) integral theta(x)*cos(2*pi*x/L) dx.
- `bath`: the well-mixed body's temperature departure.

Every output has units K. Calibration records have `input`, measured scalar
`value`, and independent Gaussian measurement standard deviation `sigma`.
Experimental settings are exact. Use the same conductivity for all preparations.
Run `python -m pytest -q test_public.py` for interface and calibration checks.
