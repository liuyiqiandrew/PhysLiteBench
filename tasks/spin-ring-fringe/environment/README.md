# Spin ring exchange fringe

Prepare two independent rings, each with n spin-1/2 sites labeled 0 through
n−1. For a setting [beta, h, Jx, Jy], prepare the ordinary Gibbs ensemble
at inverse temperature beta with Hamiltonian

    H = − sum_j [h Z_j/2 + Jx X_j X_{j+1} + Jy Y_j Y_{j+1}].

X, Y and Z are Pauli matrices, the index wraps at n, and the boundary bond
is identical to the other bonds. All spin states participate. Planck's
reduced constant and Boltzmann's constant are one. Fields and couplings
are in fixed energy units, and beta is in the corresponding inverse units.

An ancillary probe controls an instantaneous operation. On arm zero nothing
happens. On arm one, at each addressed site j, the two spins at site j on
the two rings exchange states. This is the ordinary two-qubit SWAP in the
laboratory Z basis, with unit amplitude for every basis-state exchange. The
gate durations and all additional evolution are negligible. The readout is
the real part of the overlap of the two final arm states, averaged over the
initial ensemble and multiplied by a common visibility in [0.65,0.95].

An input dictionary has keys n (6 through 32), first and second (the two
four-entry settings), and sites (1 through 4 distinct site labels, in any
order). The addressed sites can be anywhere on the rings. Allowed beta is
in [0,4], and h, Jx, Jy are each in [−1.5,1.5]. The same site labels and
Pauli axes are used in both rings.

Each calibration record contains input, value and sigma. Values have
independent Gaussian errors of standard deviation sigma. Fit the common
visibility with Model.fit(records), store it in self.visibility, and return
self. Model.predict(inputs) returns the raw dimensionless fringe signals
as a finite NumPy array of shape (len(inputs),), including empty inputs.

For calibration, the sum of squared residuals in units of sigma, divided
by the number of records minus one, must be below 2.5. The fitted
visibility must be within 0.005 of the instrument visibility. The
root-mean-square signal error must be below 0.008 in each test family.

For example, take n=12, sites=[0,3], and first=second=[2,0,1,0].
The predicted signal divided by the fitted visibility is 0.49118946.
