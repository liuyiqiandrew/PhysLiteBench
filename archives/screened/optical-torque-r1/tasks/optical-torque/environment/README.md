# Mechanical torque on an electric dipole

A point particle in vacuum has a linear anisotropic electric-dipole response and no magnetic or higher-multipole response. Use epsilon0=mu0=c=omega=1 and the field convention Re[E*exp(-i*t)]. Its three orthonormal body axes are the columns of the known proper rotation matrix `orientation`. In that body frame its complete effective polarizability is diagonal, with entries

    alpha_j = response_strength*r_j / (1 - i*response_strength*r_j/(6*pi)),
    [r_0,r_1,r_2] = [1,1.7,2.6].

The real positive `response_strength` is the sole unknown, common to all preparations and in [0.6,1.6]. The specified response includes radiation reaction and has no material absorption. The exact point-dipole model is used throughout. The dipole equals the effective polarizability tensor times the incident electric field. Its scattered field is the outgoing vacuum electric-dipole field; the incident field excludes it. No other particles or interfaces are present.

Each preparation illuminates the particle with one coherent propagating plane wave. `direction` is its real unit wavevector k. `field_real` plus i*`field_imag` is the complex electric amplitude e at the particle center, with k dot e=0 and norm at most1. In coordinates measured from that center the incident fields are E(r)=e*exp(i*k dot r) and H(r)=(k cross e)*exp(i*k dot r). Illumination sources do not scatter. All controls are exact.

An ideal support holds the particle center and its body orientation fixed without altering the fields. For `observable="force"`, the readout is the time-averaged electromagnetic force on the particle, projected along the real unit vector `axis`. For `observable="torque"`, it is the time-averaged electromagnetic torque on the particle about its own center, projected along `axis`; positive means the right-hand sense about that axis. This torque excludes any moment of the translational force about a different origin. The support records the electromagnetic force or torque exerted on the particle, with its sign as defined above. Fields have reached monochromatic steady state, and every experiment is independently prepared.

All vectors have three entries and `orientation` is a3-by-3 real proper orthogonal matrix. Each experiment also contains the `observable` and `axis` fields. Calibration records measure force for linearly polarized waves along body principal axes at varied amplitudes and orientations. Each record in `data/calibration.json` has `input`, measured `value`, and independent Gaussian instrument uncertainty `sigma`, fixed at .00005 independently of the unknown parameter or response.

Implement `Model.fit(records)`, returning self and storing the fitted `response_strength`. `Model.predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` in input order. Any implementation satisfying this interface may be used.
