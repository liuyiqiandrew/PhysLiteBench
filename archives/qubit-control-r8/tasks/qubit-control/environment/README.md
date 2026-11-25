# Two probes in a resonator bath

Two spin-1/2 probes couple to four independent quantum harmonic resonators. Use reduced units with hbar=k_B=1. During both waiting intervals the complete Hamiltonian is

    H = sum_k omega_k b_k^dagger b_k
        + sqrt(gamma) * (w1 sigma_z1 + w2 sigma_z2)
          * sum_k sqrt(a_k) (b_k + b_k^dagger).

Here [b_j,b_k^dagger]=delta_jk, sigma_z has eigenvalues +1 and -1, omega=[1,1.35,1.9,2.6], and a=[0.8,0.6,0.4,0.3]. These are exact apparatus constants. The unknown common gamma lies in [0.04,0.25], in squared inverse-time units. There are no other Hamiltonian terms or dissipative processes during the wait. The two real control weights [w1,w2] may each lie in [-1.2,1.2]; zero disconnects that probe from every resonator.

Before every experiment the probes are prepared parallel to +z, independently of the bath. Each resonator is freshly prepared in the thermal state of its uncoupled Hamiltonian omega_k b_k^dagger b_k, at the known temperature in [0,0.8]. Temperature zero means the vacuum. The coupling is off during preparation and readout, and is switched on with the specified weights for the entire sequence of two waits and the intermediate pulse. Switching adds no separate pulse or spin operation. The pulse acts only on the probes; the resonators evolve continuously throughout the two waits. Experiments are independent.

An experiment applies an ideal instantaneous preparation rotation to each probe, waits for `waits[0]`, applies an ideal instantaneous intermediate rotation to each probe, waits for `waits[1]`, switches the coupling off, and applies an ideal instantaneous readout rotation to each probe. Both waits are nonnegative and their sum lies in [0,10]. An instantaneous pulse has no bath evolution during its zero duration. A rotation is [azimuth, angle] in radians about the transverse axis (cos(azimuth),sin(azimuth),0), with unitary exp(-i*angle*axis.dot(sigma)/2). Both angles lie in [-pi,pi]. The detector returns the probability that both probes are parallel to +z after readout; the resonators are unobserved.

Each experiment dictionary contains `waits` (length two), `temperature`, `weights` (length two, held fixed throughout the experiment), and `preparation`, `intermediate`, and `readout` (each containing two rotations). Calibration records in `data/calibration.json` use one nonzero waiting interval and identity intermediate rotations, with both individual and paired coupling settings. Each record has `experiment`, measured `probability`, and known independent Gaussian uncertainty `sigma`. Settings are exact. The uncertainty is fixed by the instrument, independently of the unknown parameter.

Implement `QubitModel.fit(runs)`, returning self and storing the common fitted parameter in `gamma`. `predict(experiments)` returns a finite NumPy probability array of shape (len(experiments),), in input order. The same gamma applies to all preparations. Any implementation satisfying this interface may be used.
