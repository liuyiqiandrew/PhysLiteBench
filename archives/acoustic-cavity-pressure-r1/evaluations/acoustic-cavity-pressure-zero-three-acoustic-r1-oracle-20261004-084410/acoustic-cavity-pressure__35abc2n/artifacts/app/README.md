# Driven liquid in a closed cavity

A uniform liquid fills a one-dimensional cavity 0 <= x <= length of unit cross-sectional area. Its rigid endwalls remain fixed. The mean rest density is rho0=1, and the total mass per area is exactly rho0*length. There is no mass exchange. Transverse variation is absent. Use units l0, t0 and m0: density is in m0/l0^3, pressure in m0/(l0*t0^2), and dynamic viscosity in m0/(l0*t0). The rest pressure is p0=10.

A uniform thermostat fixes the temperature and removes dissipated heat without applying momentum. At this temperature the liquid has the exact barotropic equation of state

    p(rho) = p0 + B*((rho/rho0)^5 - 1)/5,  B=1.

The small-signal sound speed is c0=sqrt(B/rho0)=1. The only unknown is the positive constant longitudinal dynamic viscosity `viscosity` in [.08,.18]. With velocity u in the x direction, the normal stress is -p + viscosity*du/dx. The full one-dimensional continuum equations are

    d(rho)/dt + d(rho*u)/dx = 0,
    d(rho*u)/dt + d(rho*u*u)/dx
        = -d(p)/dx + viscosity*d^2(u)/dx^2 + f(x,t).

The walls impose u(0,t)=u(length,t)=0. No other momentum sources act on the liquid. An actuator prescribes force per current volume, not force per unit mass:

    f(x,t) = epsilon*drive*sin(mode*pi*x/length)*cos(omega*t),
    omega = frequency_ratio*c0*mode*pi/length.

Known controls are `length` in [2.5,4], integer `mode` in {1,2,3}, `frequency_ratio` in [.8,1.2], and positive `drive` in [.1,.3]. The drive has units of force per volume. All coefficients and controls remain constant in each run. For each positive small epsilon the system reaches its periodic steady state at the stated fixed mass. The reported responses take epsilon to zero only after that steady state has been reached. Thermal expansion and any temperature variation are excluded by the stated isothermal constitutive law.

Each input contains these four controls and a `readout`:

- `readout="density"`: A nonperturbing optical probe measures the density on the liquid side of the left wall. Report the magnitude of its Fourier component at omega, divided by epsilon in the small-epsilon limit. The harmonic convention is rho(0,t)=mean(rho(0,t))+Re[rho_hat*exp(-i*omega*t)]+higher harmonics, so this is the peak harmonic amplitude, not its RMS value.
- `readout="force"`: A support transducer measures the compressive force per area exerted by the liquid on the fixed left endwall. Report its period average minus p0, divided by epsilon^2 in the small-epsilon limit. A positive value means increased outward force on the wall. The transducer includes the full liquid normal traction.

Every observation uses a separately prepared cavity. Calibration records in `data/calibration.json` contain `input`, measured `value`, and independent Gaussian uncertainty `sigma=.0002`, fixed by the instrument in the readout units.

Implement `Model.fit(records)`, returning self and storing the inferred `self.viscosity`. `Model.predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` in input order. Run `python -m pytest -q test_public.py`.
