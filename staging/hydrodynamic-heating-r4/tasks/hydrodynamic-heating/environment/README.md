# Optical calorimetry at a thermostatted fluid contact

A uniform charged fluid fills a plane slab 0<z<d, with vacuum optical media on either side. Its stationary positive background makes it neutral at rest. Use reduced units epsilon_0=mu_0=c=1. The fluid has equilibrium number density n0, carrier charge q and mass m. The only unknown is the common plasma frequency p, with p²=n0*q²/m, in [.85,1.15]. All other parameters below are known.

Consider linear transverse response about rest. The fluid is incompressible and Newtonian, with mass density rho=m*n0, kinematic shear viscosity nu=.04, and momentum relaxation force per unit mass -gamma*v, where gamma=.06. Its viscous stress is eta*(grad(v)+grad(v)^T), eta=nu*rho. Electric force per unit mass is q*E/m, current is J=q*n0*v, and the fields obey Maxwell's equations. Magnetic force on the perturbation is second order and omitted. There is no additional polarization, surface conductivity or pressure-driven flow. These continuum equations apply at all listed settings.

Each stationary plane face carries a massless, optically inert tangential contact layer anchored to the substrate. The layer is a locally equilibrated elastic element and a Newtonian dashpot in series. If e is elastic extension and v_d is dashpot velocity, fluid slip velocity is de/dt+v_d. The common tangential traction is k(T)*e=kappa*v_d, with opposite traction acting on the fluid. The positive known ratios alpha=kappa/rho and tau=kappa/k0 are identical at the two faces. The contact adds no inertia or normal motion. The usual tangential electric and magnetic fields are continuous at each face; there is no other surface response.

Both complete contacts are held at the same known positive absolute temperature T0. Measured isothermal force data give k(T)=k0*[1+(T-T0)/(2*T0)] for .9*T0<=T<=1.1*T0. Each elastic element has a known positive constant heat capacity per area C0 at fixed extension throughout this neighborhood. The dashpot coefficient is temperature independent. There is no additional internal relaxation or work channel. C0 has units energy/(area*temperature), k0 has units traction/length, and e is a length.

A normally incident plane wave from z<0 has electric field along y. At the front plane its incoming field is Re[E0*exp(-i*frequency*t)], with E0 real. No wave is incident from z>d. Use this clock for all phase measurements. All measurements are in the periodic steady state after transients, followed by the limit E0→0 at leading quadratic order in the heat signal. Normal incidence makes the phase uniform over the wall.

A calibrated thermostat at the front contact records the signed thermal heat removed per area from the whole contact, including its elastic element and dashpot. Positive heat flows into the thermostat. It maintains T0 with negligible thermal response time at the drive and measurement frequencies. Equilibrium background heat is subtracted. Thermal feedback and spatial heat transport do not modify the leading electromagnetic or mechanical response. Heat removed from the bulk fluid, stationary background and rear contact is collected separately. The detector has unit efficiency.

Divide the front-contact heat signal by the mean incident electromagnetic power per area, E0²/2. The `readout` selector specifies:

- `mean`: its cycle average.
- `in_phase`: twice the cycle average of the product of that signal and cos(2*frequency*t).
- `quadrature`: twice the cycle average of the product of that signal and sin(2*frequency*t).

Every output is dimensionless. Signed harmonic quadratures use the incoming-field phase convention above.

Each input dictionary contains `frequency` in [.7,1.5], `thickness`=d in [.3,1.2], `friction`=alpha in [.08,.4], `relaxation`=tau in [.2,1.2], and `readout`. In these fixed units alpha has dimensions length/time and tau has dimensions time. All controls are exact.

`data/calibration.json` contains independent records with `input`, measured `value`, and fixed Gaussian standard deviation `sigma`. The uncertainty does not depend on p. Implement `Model.fit(records)`, returning self and storing p as `plasma_frequency`. `Model.predict(experiments)` must return a finite NumPy array of shape `(len(experiments),)` in input order. You may replace the entire implementation within this interface. Run `python -m pytest -q test_public.py`.
