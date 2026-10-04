# Liquid-film relaxation over a patterned elastic coating

A thin incompressible Newtonian liquid of viscosity .15 Pa s occupies a gap
between a fixed rigid plate and a bonded isotropic elastic coating. The
coating thickness is `thickness`, and its Poisson ratio is .48 everywhere.
Its Young modulus is fixed in the material and varies laterally:

    E(x)=young_modulus*[1+contrast*cos(pattern_mode*k0*x+pattern_phase)],
    k0=fundamental_wavenumber.

The one unknown `young_modulus` is between .7 and1.6 MPa and is shared by all
experiments. The contrast, integer pattern mode and phase are known. The
pattern extends uniformly through the coating depth. The coating has no
viscoelasticity or material transport. Use isotropic infinitesimal-strain
elasticity at every point with the local modulus above.

The system is invariant in y, with plane strain u_y=0 and no y derivatives.
It is periodic in x with period2*pi/k0. Both elastic displacement components
vanish at the perfectly bonded rigid bottom. The top carries the liquid's
normal pressure and zero shear traction at the order retained here. The
solid responds quasistatically. The liquid and solid inertia are negligible.
There are no body forces, surface tension, permeation, slip or external flow
after preparation.

Use the leading no-slip lubrication model for the liquid gap, retaining terms
linear in its displacement from the uniform reference gap. The gap is small
compared with the coating thickness and every lateral wavelength. Liquid
shear traction and flow due to the elastic surface's lateral velocity are
higher-order corrections at this limit. The full coating thickness and the
stated modulus pattern remain resolved in elasticity. These are the
specified approximations, not a limit of a thin coating.

All tractions and displacements are incremental about a static preloaded
reference. The rigid plate follows that reference surface so the reference
gap is uniform. A large background fluid pressure prevents cavitation; its
balanced preload is already subtracted from the elastic traction. Only the
specified excess pressure acts in the incremental problem. Higher-order
changes of geometry or material constants due to preload are excluded.

Prepare the excess liquid pressure

    p(x,0)=pressure*cos(pressure_mode*k0*x+pressure_phase).

The coating has reached mechanical equilibrium with this profile. Distributed
preparation ports close at t=0 without changing the initial pressure or
displacement; there is no later exchange of liquid. Each experiment is a
fresh preparation. At `elapsed_time`, let a(x,t) be the coating's downward
normal surface displacement relative to the reference. Downward displacement
increases the gap. For detector_mode>0, report in nanometres

    2*mean_over_period[a(x,t)*cos(detector_mode*k0*x+detector_phase)].

For detector_mode=0, report mean_over_period[a(x,t)] instead; detector_phase is
then unused. These Fourier coefficients may have either sign. They are actual
displacements, not fractions of the initial displacement.

Every input contains the following exact controls:

- `pressure`: amplitude0 to60 Pa.
- `pressure_mode`: integer0,1,2 or3; `pressure_phase`: -pi to pi.
- `fundamental_wavenumber`: .5 to1.2 inverse millimetres (angular wavenumber).
- `contrast`: 0 to.7; `pattern_mode`: integer1 or2; `pattern_phase`: -pi to pi.
- `thickness`: .8 to1.2 mm; `gap`: 8 to12 micrometres.
- `elapsed_time`: 0 to2 s.
- `detector_mode`: integer0 to5; `detector_phase`: -pi to pi.

Calibration uses homogeneous coatings (contrast0) with spatially varying
pressure preparations at several wavelengths and times. Independent Gaussian
measurement errors have the listed sigma. The task requests the stated
linear, quasistatic, leading thin-gap model. Finite-deformation, inertia and
higher-order liquid-gap effects are outside its scope.

Implement `Model.fit(records)`, returning self and setting `Model.young_modulus`
in MPa. Implement `Model.predict(experiments)` returning a finite NumPy array
with shape `(len(experiments),)`. Records in `data/calibration.json` have
`input`, `value` and `sigma`. Run `python -m pytest -q test_public.py`.
