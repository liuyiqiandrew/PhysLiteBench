# Charged tracer in a periodic film

Dilute independent classical tracers of positive charge q=1 move in the xy plane
of a stationary fluid at k_B T=1. Use reduced units throughout. The plane is
periodic with period 2*pi along each axis. The constant isotropic viscous drag
coefficient gamma is unknown in [0.5,1.5]. The fluid provides a Gaussian thermal
force with covariance 2*gamma*k_B*T times the unit matrix times delta(t-t').
The tracer obeys Newton's equation with this thermal force, viscous force
-gamma*v, and Lorentz force q*v cross B. There is no electric field, potential,
interaction, flow, or other force. Tracers do not alter the applied fields.

External currents maintain the static field

    B_z(x) = field_offset + field_amplitude*cos(x + field_phase).

The other field components vanish. Offsets and amplitudes are between -1.5
and 1.5; field_phase is an angle in radians. The field is already present at
preparation. Initial velocities have the Maxwell distribution independently
of position. Measurements are in the strict small-mass limit m/gamma -> 0 at
fixed positive observation times and fixed gamma, temperature, field, and
spatial periods. Momentum relaxation transients are outside the measurement.

Each run starts with normalized position density

    rho(x,y,0) = [1 + amplitude*cos(kx*x + ky*y + initial_phase)]/(2*pi)^2.

The input initial_wave is [kx,ky], where kx is an integer in [-2,2] and ky is
1 or 2. The known density amplitude lies in [0.2,0.7]. For time between 0.05
and 6, the detector reports the expectation of

    cos(mx*x + ky*y + detector_phase),

with integer detector_mode mx in [-2,2]. Every particle has equal detection
weight. Each input contains field_offset, field_amplitude, field_phase,
initial_wave, initial_phase, amplitude, detector_mode, detector_phase, and
time. Phases lie between -pi and pi. The same gamma applies throughout.

Implement Model.fit(records), returning self and storing gamma in friction,
and Model.predict(experiments), returning a finite NumPy array with shape
(len(experiments),). Calibration uses spatially uniform fields, including zero
field. Records in data/calibration.json contain input, value, and sigma, the
independent Gaussian measurement standard deviation. Settings are exact.
Run python -m pytest -q test_public.py.
