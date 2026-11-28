# A reversible seam in a critical magnet

A ferromagnetic ring contains N spin-½ sites in order j=0,…,N−1. All spin states are allowed. A slow seam at the bond from N−1 to 0 has classical polarity ξ=+1 or −1. At fixed polarity the critical transverse-field Ising Hamiltonian is

    Hξ = −Σⱼ Xⱼ − Σⱼ₌₀ᴺ⁻² Zⱼ Zⱼ₊₁ − ξ Z_(N−1) Z₀ .

Here the last product denotes the Z operators at sites N−1 and 0. X and Z are Pauli matrices, and the exchange, ℏ and Boltzmann's constant are one. The low-energy velocity is two lattice sites per unit time. Use inverse temperature β=N × aspect/2. Take N to infinity through multiples of 64 at each specified aspect.

With `seam="free"`, spins and seam equilibrate together. The unnormalized spin density for polarity ξ is exp(g h ξ) exp(−βHξ), with dial `bias`=h and common gain g in [0.75,1.15]. With `seam=1` or `seam=-1` clamp that bond at the indicated sign. The calibration disconnects the spins, frees the seam, and measures its mean sign at each supplied dial value. Calibration values are rounded to eight decimals.

Before the spin readout an ideal parity monitor can retain shots with a specified eigenvalue of P=∏ⱼ Xⱼ. The integer `parity` is +1 or −1 for this selection, or 0 to keep all shots. The seam, parity and spin measurement belong to the same shot. Pool all retained shots for the selected seam protocol. Every spin measurement uses an even number of Z operators, so their product commutes with P.

The input `sites` is an increasing list of two, four, six or eight fractions u. Measure O=∏ᵤ Z at site Nu. The ring coordinates start directly after the seam and increase along the positive direction. The reference instrument measures the same product at the same temperature and size with the seam clamped positive and the parity monitor off. Let C_N(h) be the mean O in the specified ensemble divided by this reference mean. Use the large-ring limit C(h) of the ratio. For `read="spin"` return C(h). For `read="response"` return its derivative with respect to the dial h at fixed gain and all other controls. A clamped seam is insensitive to the bias. A contrast can exceed one or change sign.

Aspect lies in [0.4,1.5], and h lies in [−1.4,1.4]. Each site lies strictly between 0 and 1 and is an integer multiple of 1/64. Consecutive sites around the ring are at least 2/64 apart. The calibration inputs contain only `bias`.

For a check at g=1, take sites [0.125,0.4375], aspect=0.85, bias=0.32 and parity=−1. A freely annealed seam gives spin contrast 1.11506357 and dial response 0.15321969. Clamping the seam to −1 gives spin contrast 0.49761299. Clamping it positive gives 1.20252516. Without parity selection the positive clamped case gives one by the definition of the reference.

Implement Model in /app/model.py. Its fit(records) sets `gain` to the fitted g and returns self. Its predict(inputs) returns a finite one-dimensional real NumPy array in input order, also for empty inputs. Maximum absolute errors must be below 0.00002 on calibration and below 0.004 on spin contrasts and responses. The starter approximates a negative seam by a single uniformly located domain wall and treats seam polarity and spin-flip parity as independent of the thermal spin correlations.
