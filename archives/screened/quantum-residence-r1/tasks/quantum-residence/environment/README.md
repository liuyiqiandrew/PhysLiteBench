# Population in a scattering region

A nonrelativistic particle of mass m=1 scatters coherently in one spatial dimension. Use reduced units with hbar=1. The stationary Hamiltonian is H=−(1/2)d²/dx²+V(x). The potential is the specified nonnegative constant `potential` inside 0<x<L and zero in both semi-infinite leads. The common unknown region width L lies in [0.6,1.4]. The particle mass and potential offset are unchanged at the two interfaces. Wavefunction and its spatial derivative are continuous. There are no interactions, additional channels, absorption, decoherence or time-dependent forces.

A dilute monoenergetic beam is incident from the left at kinetic energy `energy` in [0.5,3.5]. No particles enter from the right. The potential height is in [0,4.5], so propagating and evanescent interior states are allowed. Beam particles are independent, and all measurements are stationary.

An ideal non-invasive detector counts the mean number of particles present in 0<x<L, including the complete coherent state inside that region. It reports this population divided by the incoming particle flux. The output therefore has units of time. It does not condition on subsequent reflection or transmission. The detector excludes both exterior leads. Beam intensity cancels in the reported ratio.

For scattering amplitudes, the reference planes are at the two region faces: on the left an incident wave is exp(i*k*x) and a reflected wave is r*exp(−i*k*x); on the right the transmitted wave is t*exp(i*k*(x−L)), where k=sqrt(2*energy). A right-incident state uses the corresponding inward wave referenced to x=L. The lead speeds are equal. These phase-reference conventions also apply to the supplied helpers.

Each input has `energy` and `potential`. The same L applies to all preparations. Calibration switches the potential height to zero while retaining the physical detector region. Each record in `data/calibration.json` has `input`, the measured population-to-flux ratio `value`, and an independent Gaussian uncertainty `sigma` in time units. The instrument uncertainty is fixed independently of L and the noiseless response. Energies and potential heights are exact.

Implement `Model.fit(records)`, storing the fitted scalar `width` and returning self. `predict(experiments)` must return a finite NumPy array of shape `(len(experiments),)` in input order. Any implementation satisfying this interface may be used.
