# Force on an electric dipole

An isotropic point particle in vacuum has only an electric dipole response. Use units epsilon_0=mu_0=c=omega=1 and the complex-field convention Re[E(r)*exp(-i*t)]. The particle has no magnetic polarizability or higher multipole response. Its effective electric polarizability is

    alpha = response_strength / (1 - i*response_strength/(6*pi)).

The real positive `response_strength` is the only unknown and lies in [0.6,1.6]. This response includes radiation reaction. The particle has no material absorption. The exact point-dipole model is used throughout; its dipole amplitude is alpha times the incident electric field at its position, and its scattered field is the outgoing vacuum electric-dipole field. The incident field excludes that scattered field. The external illumination sources have no scattering response. There are no material interfaces or other particles.

Illumination consists of one or two coherent propagating plane waves at the same frequency. For each entry in `waves`, `direction` is its real unit wavevector k, and `real` plus i*`imag` is its complex electric amplitude e at the coordinate origin. Its incident fields are E(r)=e*exp(i*k dot r) and H(r)=(k cross e)*exp(i*k dot r). Amplitudes are transverse, k dot e=0, with norm at most 1, and each direction has z component at least 0.3. All waves maintain their relative phases.

The particle is held at the known `position`, with each coordinate in [-2,2]. An ideal support measures the time-averaged electromagnetic force exerted on the particle, projected along the real unit vector `axis`. Positive output means force in the direction of that axis. This is the force on the particle alone. The fields have reached their monochromatic steady state. The support does not change the electromagnetic fields, and each experiment uses a fresh independent preparation. All inputs except response_strength are known exactly.

Calibration records in `data/calibration.json` contain `input`, measured force projection `value`, and known independent Gaussian standard deviation `sigma`. Implement `Model.fit(records)`, returning self and storing the fitted scalar `response_strength`. Implement `Model.predict(experiments)`, returning a finite NumPy array of shape `(len(experiments),)` in input order.
