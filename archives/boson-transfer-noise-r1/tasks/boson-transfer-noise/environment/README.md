# Bosonic energy transport

Two passive resonators connect two independent single-channel bosonic reservoirs. Use reduced units with hbar=kB=1 and zero chemical potential. The device is static, number conserving and linear, with no internal loss, gain, coherent input, frequency conversion or additional reservoir. Its resonator Hamiltonian is a† H a, with

    H = [[1.4 + shift, g], [g, 1.9 + shift]].

The left port couples only to the first resonator with rate .7, and the right port only to the second with rate 1.1. The ideal frequency-independent contact model is specified by the scattering relation b(omega)=S(omega) a(omega), where

    W = diag(sqrt(.7), sqrt(1.1))
    S(omega) = I - W [i(H - omega I) + W²/2]^-1 W.

Here a and b are incoming and outgoing annihilation fields, normalized by [a_i(omega), a_j†(omega')] = 2*pi*delta_ij*delta(omega-omega'). Frequencies are positive. The scattering matrix is unitary. The stated contact model is the ideal apparatus throughout the measured frequency range.

Incoming modes from each reservoir are in a thermal Gaussian state at its specified temperature. Different reservoirs and different frequency modes are independent; there is no squeezing or initial coherence. The reservoirs remain macroscopic and maintain their stated temperatures. The unknown positive resonator coupling g, stored as `coupling`, is common to all experiments and lies in [.3,.8]. The known common frequency shift lies in [-.2,.2], and each known temperature lies in [.15,3].

An ideal frequency-resolved reservoir calorimeter records the energy gained by the right reservoir in a band [omega_low,omega_high]. Both edges lie in [.4,3.4], with omega_high - omega_low at least .2. The band is a detector selection and does not modify the scattering device. Energy gain counts outgoing energy into that reservoir minus its incoming energy in the selected band. Use two projective measurements of the reservoir's band energy, before and after a long observation window. The apparatus has reached its stationary transport regime; endpoint device-energy contributions do not affect the long-time rates.

The available readouts are `current`, the limit of mean energy gain divided by observation time, and `noise`, the limit of its centered variance divided by observation time. Positive current means energy delivered to the right reservoir. Noise is a variance rate, with no extra one-sided-spectrum factor of two. Zero-point energy cancels in the energy difference. Temperatures, shifts, band edges and clock times are exact.

Each experiment dictionary has `shift`, `left_temperature`, `right_temperature`, `band` (a two-element list), and `readout`. `data/calibration.json` contains current records at unequal temperatures and noise records at equal temperatures. Each record has `input`, measured `value`, and independent Gaussian uncertainty `sigma`. The fixed instrument uncertainty is .0005 in the corresponding reduced readout units and is independent of g.

Implement `Model.fit(records)`, returning self and storing the fitted `coupling`. `Model.predict(experiments)` returns a finite NumPy array of shape `(len(experiments),)` in input order. Use the same fitted g for all controls and readouts.
