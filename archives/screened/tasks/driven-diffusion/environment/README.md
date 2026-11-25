# Driven tracer in a periodic landscape

Independent classical Brownian tracers move on an infinite one-dimensional line
in the static potential U(x)=amplitude*cos(x), of period 2*pi. A constant uniform
force force also acts along increasing x. Use reduced units with k_B*T=1. The
constant viscous drag coefficient gamma is unknown in [0.5,1.5]. Neglect inertia.
The fluid is at rest and supplies Gaussian white thermal force with covariance
2*gamma*k_B*T*delta(t-t'). There are no other forces, interactions, memory,
position-dependent drag, or changes of bath temperature.

The applied force and potential are held fixed for each run. The position x is
unwrapped: crossing a spatial period does not reset the tracked coordinate.
After stationary transport is established, a detector reports either

    observable="drift": lim[t->infinity] E[x(t)-x(0)]/t,
    observable="diffusion": lim[t->infinity] Var[x(t)-x(0)]/(2*t).

These are asymptotic transport coefficients, with no finite-time correction.
They have reduced units of length/time and length^2/time, respectively. The
phase x modulo 2*pi is initially drawn from its stationary distribution for
the stated force; the result is independent of the chosen spatial period.
Inputs provide amplitude in [0,3], force in [-3,3], and observable. The same
gamma applies in all experiments.

Implement Model.fit(records), returning self and storing gamma in friction,
and Model.predict(experiments), returning a finite NumPy array with shape
(len(experiments),). Calibration measures diffusion at zero applied force for
several known potential amplitudes. All settings are exact. Records in
data/calibration.json contain input, value, and sigma, the independent Gaussian
measurement standard deviation. Run python -m pytest -q test_public.py.
