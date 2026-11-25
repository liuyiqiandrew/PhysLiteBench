# A charged tracer with a retained thermal contact

A dilute classical tracer of positive charge q=1 and mass m moves in an unbounded
xy plane. Set k_B=1 and use fixed reference units. The harmonic trap is
U=(kx*x*x+ky*y*y)/2. Each stiffness is in [.7,1.8]. A uniform signed magnetic
field B in [-2,2] points along +z. The ordinary fluid has temperature T in
[.5,1.2] and a common unknown isotropic friction gamma in [.5,1.5].

The tracer is also coupled to a linear thermal contact with force f=(fx,fy).
Its known memory_drag zeta is in [0,1.3], memory_time tau in [.2,1.6], and
memory_temperature Tb in [.4,1.4]. The complete finite-mass dynamics are

    dr = v dt,
    m dv = [-grad U - gamma*v + v cross (B zhat) + f] dt
           + sqrt(2*gamma*T) dW,
    tau df = [-f-zeta*v] dt + sqrt(2*zeta*Tb) dZ.

W and Z are independent standard two-dimensional Wiener processes. All scalar
drag, memory and noise coefficients multiply the identity matrix in the plane.
The force f is retained continuously and is not reset during a record. These
equations specify the entire contact and both reservoirs; there are no further
forces, collisions, sampling steps, fluid flow, or feedback. The contact with
zeta=0 contributes zero force. All parameters and fields are constant in time.

At preparation, position is a centered Gaussian with the supplied symmetric
positive initial_covariance, whose eigenvalues lie in [.2,2]. Initial velocity
is centered Gaussian with covariance (T/m)*I. Initial contact force is centered
Gaussian with covariance (zeta*Tb/tau)*I. These three initial random vectors are
mutually independent and independent of all subsequent Wiener increments.
Every experiment starts from this preparation; it need not be stationary.

The observation time is in [0,4]. Readouts xx, xy and yy mean E[x(t)^2],
E[x(t)y(t)] and E[y(t)^2]. The mechanical area readout is

    E[integral_0^time (x(s)*v_y(s)-y(s)*v_x(s)) ds].

It has no factor of one half. First form each observable and its ensemble
average in the finite-mass system above, then take m to zero with gamma, B,
both temperatures, both contact coefficients, the trap, preparation and
observation time fixed. In particular tau is held fixed in this limit. Mass
is not an input or an inferred parameter.

An experiment dictionary has time, field, temperature, stiffness (length two),
initial_covariance (2 by 2), and readout, as well as memory_drag, memory_time,
and memory_temperature. For compatibility, omitted memory_drag is zero,
omitted memory_time is one, and omitted memory_temperature equals temperature.
These defaults apply independently to each dictionary.

Calibration in data/calibration.json uses field=0 and memory_drag=0. Each record
contains input, measured value and an independent Gaussian standard uncertainty
sigma. All settings are exact; uncertainty is fixed independently of gamma.

Implement Model.fit(records), returning self and storing gamma as friction.
Model.predict(experiments) returns a finite one-dimensional NumPy array in input
order, including an empty array for an empty list. Any implementation preserving
this interface is allowed. Run python -m pytest -q test_public.py.
