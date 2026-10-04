# Slow deformation of a plasma tube

A straight circular flux tube contains classical collisionless ions with a
neutralizing cold electron population. Initially the ion density n0=1 and axial
magnetic field B0=1 along z are uniform. Use reduced units with ion mass,
positive ion charge, and magnetic permeability all equal to 1. Initial ion
velocities are isotropic Maxwellian with unknown pressure p0 in [0.03,0.08].
The same p0 applies to every fresh preparation.

External actuators slowly change the tube radius by factor a and its length
by factor b, each in [0.7,1.3]. The side boundary moves with the flux tube,
conserving its magnetic flux. Particles are retained; two moving end pistons
reflect ions specularly in each piston's instantaneous rest frame. No particles
or heat pass through any boundary. Actuator work need not vanish. The field
remains uniform and axial in the bulk, with ideal induction and negligible
resistivity. Electron pressure, finite ion gyroradius corrections, and boundary
layers are negligible in the measured bulk region.

Gyro periods and thermal ion bounce times are much shorter than the deformation
time, which is much shorter than every collision or pitch-angle-scattering
time. Take the strict quasistatic, strongly magnetized, collisionless limit.
The initial radius and length are large compared with ion gyroradii. No waves,
instabilities, or extra relaxation act; the stated parameter range lies below
both mirror and firehose thresholds. Parallel and perpendicular heat fluxes
are negligible. After a smooth deformation, the endpoint is at rest. Readings
average over gyro and bounce phases before collisional relaxation.

A detector reports the signed normal component of the total ion-plus-magnetic
momentum flux through a virtual surface in the bulk rest frame. Its unit normal
is (sin(theta),0,cos(theta)). Include magnetic pressure and magnetic tension;
positive momentum flux is compressive. There is no bulk ram contribution at
the endpoint. Inputs are radius_ratio a, length_ratio b, and theta in [0,pi/2].
Calibration uses a=b. All geometry and magnetic settings are exact.

Implement Model.fit(records), returning self and storing p0 in pressure, and
Model.predict(experiments), returning a finite NumPy array with shape
(len(experiments),) in the reduced pressure units. Records in
data/calibration.json contain input, value, and sigma, the independent Gaussian
measurement standard deviation. Run python -m pytest -q test_public.py.
