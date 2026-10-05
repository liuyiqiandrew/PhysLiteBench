# Heat exchange in a clamped rod

A slender rod has reference length L=0.05 m and cross-sectional area A=1e-4 m².
Its ends are fixed at their reference separation. Transverse surfaces are
mechanically free. Axial mechanical equilibration is much faster than thermal
evolution, so neglect axial inertia. The effective one-dimensional material
properties below already include transverse relaxation. The reference state is
unstrained at T0=300 K. Write theta=T−T0 in K.

The material has axial strain epsilon and local internal viscous strain z. Its
Helmholtz free energy per reference volume, in J/m³, is

```
psi = E0*epsilon²/2 + E1*(epsilon-z)²/2
      - theta*[beta0(x)*epsilon + beta1(x)*(epsilon-z)]
      - ce*theta²/(2*T0).
```

Here E0=E1=2e9 Pa, ce=1e6 J/(m³ K),
beta0(x)=E0*(0.002 K^-1)*[1+0.65*cos(2*pi*x/L)], and
beta1(x)=chi*beta0(x). The known material setting chi is in [0,1]. The internal
strain obeys

```
eta(x)*dz/dt = E1*(epsilon-z) - beta1(x)*theta,
eta(x) = E1*(12 s)*[1+0.97*cos(2*pi*x/L)]  in Pa s.
```

There are no other constitutive terms or energy sources. Thermal conductivity
k is uniform and unknown, in [80,220] W/(m K). Heat conduction obeys Fourier's
law. Retain terms linear in temperature departures and strains about T0,
including reversible thermal coupling. Viscous heat production is quadratic
in the perturbations and is omitted at this order. Neglect geometry changes
in heat conduction.

The lateral surface and end x=L are thermally insulated. At time zero, end x=0
is connected to a well-mixed body of heat capacity C_b=6 J/K through a massless
contact of known conductance H in W/K. Heat flowing into the rod is
H*(T_body-T_surface). Neither the body nor the rod exchanges heat elsewhere.
H=0 disconnects the body.

Each experiment is a fresh preparation. With the clamps fixed, preparation
heaters maintain theta(x)=a+b*cos(pi*x/L)+c*cos(2*pi*x/L) until internal
relaxation is complete. At time zero the heaters are removed and the specified
contact is made. The clamps remain fixed. Initially max_x|theta(x)|<=1 K and
|bath_initial|<=1 K.

## Interface

Implement `Model.fit(records)`, returning `self` and setting `conductivity`, and
`Model.predict(experiments)`, returning a finite NumPy array of shape
`(len(experiments),)`. You may edit the entire implementation within this API.
Use the same conductivity for all experiments and material settings.

Each experiment supplies `mean` (a), `first` (b), `second` (c), `bath_initial`,
`chi` (0 to 1), `contact` (H, 0 to 1.2 W/K), `time` (0 to 80 s), and
`observable`:

- `mean`: the spatially averaged rod temperature departure.
- `first`: (1/L) integral theta(x)*cos(pi*x/L) dx.
- `second`: (1/L) integral theta(x)*cos(2*pi*x/L) dx.
- `bath`: the well-mixed body's temperature departure.

Every output is in K. Calibration records have `input`, measured scalar
`value`, and independent Gaussian measurement standard deviation `sigma`.
Experimental settings are exact. Run `python -m pytest -q test_public.py` for
interface and calibration checks.
