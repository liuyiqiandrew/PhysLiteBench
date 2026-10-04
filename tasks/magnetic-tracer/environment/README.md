# A trapped charged tracer

A dilute classical tracer of mass m and positive charge q=1 moves in the xy
plane of a stationary fluid. Use reduced units with k_B=1. The plane is
unbounded. A harmonic trap has potential

    U(x,y) = (kx*x^2 + ky*y^2)/2.

The known stiffnesses kx and ky lie in [0.7,1.8]. A uniform static magnetic
field B points along +z; its signed value lies in [-2,2]. The fluid has known
uniform temperature T in [0.5,1.2] and constant isotropic drag gamma, unknown in
[0.5,1.5]. The tracer obeys

    dr/dt = v,
    m dv/dt = -grad U - gamma*v + v cross (B zhat) + eta(t),
    E[eta_i(t) eta_j(s)] = 2*gamma*T*delta_ij*delta(t-s).

The force is zero-mean Gaussian and Markovian. There is no electric field, fluid flow,
other force or interaction. The tracer does not alter the bath or fields.

At preparation, position is a centered Gaussian with specified covariance
`initial_covariance`, a symmetric positive matrix with eigenvalues in [0.2,2].
Velocities are independent of position and Maxwell distributed at the bath
temperature: E[v_i v_j]=(T/m)*delta_ij. Each experiment starts from this new
preparation and evolves in the stated fixed trap and bath.

Measurements are ensemble averages at a fixed time in [0,4]. The readout is
one of `xx`, `xy`, `yy`, or `area`. The first three mean E[x(t)^2], E[x(t)y(t)]
and E[y(t)^2]. The mechanical area readout is

    E[integral_0^time (x(s)*v_y(s) - y(s)*v_x(s)) ds].

This area has no factor of one half. For every readout, form the observable and
its ensemble average in the finite-mass dynamics above, then take m to zero
with the trap, gamma, B, T, preparation covariance and observation time fixed.
The data report this limit; mass is not a fitted parameter.

An input dictionary has `time`, `field`, `temperature`, `stiffness` (length two),
`initial_covariance` (2 by 2), and `readout`. Calibration records in
`data/calibration.json` contain `input`, measured `value`, and independent
Gaussian standard deviation `sigma`. Settings are exact; instrument uncertainty
is fixed independently of the unknown gamma.

Implement `Model.fit(records)`, returning self and storing gamma as `friction`.
`Model.predict(experiments)` returns a finite NumPy array with one value per
input, in the same order. Run `python -m pytest -q test_public.py`.
