# Magnetic tracer revision 3

Revision 2 scored 2/3 passes in the independent 2026-10-11 rerun. Both successes
added the known fast-velocity area anomaly. Revision 3 introduces a retained
linear thermal force at a known finite memory time, preserving that physics
family and the original memory-free calibration. The new completed shortcut
correctly handles the area anomaly in its instantaneous-contact approximation;
it adds the two friction coefficients and uses their weighted temperature.
This is a coherent zero-memory limit, but the prediction experiments hold the
memory time fixed. The distinguishing failure is loss of bath memory and of
the shared noise created by eliminating velocity, not an unfinished fit.

Write M=(gamma I-B J)^(-1), J=[[0,1],[-1,0]], and K=diag(kx,ky). The slow state
is (r,f), with drift [[-MK,M],[zeta MK/tau,-I/tau-zeta M/tau]]. The white-fluid
noise enters it through [M;-zeta M/tau], so its two blocks are correlated.
Add independent contact noise to the f block. Solve the stationary covariance
Lyapunov equation and propagate the specified initial covariance. The mechanical
area equals the integrated quadratic slow drift readout plus T*tr(J M)*time.
Its transient integral is evaluated by a second Lyapunov equation.

The private reference instead constructs the complete six-dimensional
finite-mass OU process, measures the area from position-velocity covariances,
and extrapolates after measurement using three masses. It does not use the
four-dimensional reduction. The validator checks mass refinement, randomized
physical settings, covariance positivity, field/reflection symmetry, equilibrium
zero area, 256 calibration-noise realizations, and both completed controls.
The original calibration observations and 2.5% prediction tolerance are kept.
The new hidden groups include both memory-sensitive experiments and a
memory-free anchor. See the staged validation report for measured errors.
