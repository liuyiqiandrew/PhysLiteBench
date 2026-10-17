# Superconducting tunnel contact

Two large isotropic s-wave superconducting electrodes, L and R, are separated by a nonmagnetic tunnel barrier. Each electrode has a flat normal-state density of electronic states and a local equilibrium Fermi distribution at its prescribed temperature. The chemical potentials are equal. A stiff superconducting phase bias holds the order-parameter phase difference at the specified constant value; the voltage is zero. The phase controller supplies no electrical power. Only heat crossing the measured tunnel contact is counted; there is no parallel phonon or photon heat channel.

The mean-field electrode Hamiltonian is
`H_j = sum_(k,s) xi_jk c†_jks c_jks + sum_k [Delta_j exp(i phi_j) c†_jk_up c†_j,-k,down + h.c.]`,
and the barrier couples electrons by
`H_T = sum_(k,q,s) [t c†_Lks c_Rqs + h.c.]`.
The spin-independent tunneling amplitude is real and energy independent. Use the leading nonzero order in the individual channel transparencies. Charging effects, lifetime broadening, gap suppression by the barrier, and phase fluctuations are absent.

The known electrode gap magnitudes, expressed as `Delta/k_B` in kelvin, are
`d_j(T) = 1.764 Tc_j tanh(1.74 sqrt(Tc_j/T - 1))`,
with `Tc_L = 1.2 K` and `Tc_R = 1.9 K`. This supplied gap law defines the material model. All requested temperatures are between `0.25 K` and `1.05 K`; both electrodes remain superconducting and their gap edges are separated. Each measurement is stationary, with the electrode temperatures maintained by external thermal reservoirs.

The sole unknown is the normal-state electrical conductance `conductance`, in microsiemens, between 20 and 80. The readout is heat power from L to R in picowatts, defined as the stationary loss of L's electronic energy relative to its chemical potential. Negative values denote heat entering L. Phase is in radians and lies between `-pi` and `pi`.

Use `k_B = 1.380649e-23 J/K` and `e = 1.602176634e-19 C`.

## Interface

`Model.fit(records)` returns `self` and sets `conductance`.
Each JSON record has `input`, `value`, and `sigma`; the latter two are the measured heat power and its independent Gaussian standard uncertainty, in pW. Each input has `left_temperature`, `right_temperature`, and `phase`.

`Model.predict(experiments)` accepts a list of input dictionaries and returns a finite NumPy array with shape `(len(experiments),)`, in pW. The public calibration is `data/calibration.json`.
