# Equilibrated probes in a resonator bath

Two spin-1/2 probes couple to four quantum harmonic resonators. Use reduced units with hbar=k_B=1. The complete Hamiltonian is

    H = sum_k omega_k b_k^dagger b_k
        + sqrt(gamma) * (w1 sigma_z1 + w2 sigma_z2)
          * sum_k sqrt(a_k) (b_k + b_k^dagger).

Here [b_j,b_k^dagger]=delta_jk, sigma_z has eigenvalues +1 and -1, omega=[1,1.35,1.9,2.6], and a=[0.8,0.6,0.4,0.3]. These are exact apparatus constants. The unknown common gamma lies in [0.04,0.25], in squared inverse-time units. There are no additional spin fields, counterterms or interactions. The two real control weights [w1,w2] may each lie in [-1.2,1.2]; zero disconnects that probe from every resonator. The same weights apply throughout a given experiment.

Before each experiment, the entire interacting system of both probes and all four resonators is prepared in the canonical Gibbs state exp(-H/T)/Tr(exp(-H/T)) of this Hamiltonian, at known temperature T in [0.2,1.2]. Preparation includes equilibration of spin populations; it is not limited by the conserved sigma_z values of the subsequent isolated evolution. The equilibration apparatus is then removed. From that point the whole system evolves unitarily under H, except for the specified instantaneous local spin rotations. Neither the coupling nor the resonator state is reset or switched during the protocol. Every experiment begins with a fresh full canonical preparation at its own stated settings.

An experiment applies an ideal instantaneous preparation rotation to each probe, waits for `wait` in [0,10], and applies an ideal instantaneous readout rotation to each probe. Rotations act only on the spins, with no resonator evolution during their zero duration. A rotation is [azimuth, angle] in radians about the transverse axis (cos(azimuth),sin(azimuth),0), with unitary exp(-i*angle*axis.dot(sigma)/2). Both angles lie in [-pi,pi]. The detector returns the probability that both probes are parallel to +z after readout; the resonators are unobserved.

Each experiment dictionary contains `wait`, `temperature`, `weights` (length two), and `preparation` and `readout` (each containing two rotations). Calibration records in `data/calibration.json` use identity preparation rotations with paired nonzero coupling weights. They measure the same joint two-probe detector probability. Each record has `experiment`, measured `probability`, and known independent Gaussian uncertainty `sigma`. Settings are exact. The uncertainty is a fixed instrument standard deviation, independent of gamma.

Implement `QubitModel.fit(runs)`, returning self and storing the common fitted parameter in `gamma`. `predict(experiments)` returns a finite NumPy probability array of shape `(len(experiments),)`, in input order. The same gamma applies to all preparations. Any implementation satisfying this interface may be used.
