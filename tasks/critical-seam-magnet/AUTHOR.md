# A reversible seam in a critical magnet

In the scaling limit the equal-time ring corresponds to a critical Ising torus of horizontal period one and imaginary-time period i times the aspect. Antiferromagnetic closure is a spatial twist. Inserting the global flip P in a thermal trace twists the imaginary-time direction. The parity gate therefore combines two physical boundary conditions. Relative partition weights also enter the free-seam measurement. The seam dial response is the gain times the conditional covariance of the seam polarity and spin product, divided by the unselected welded-ring reference.

## Continuum calculation

For n marked sites sum over neutral charges eᵢ=±1 with Σeᵢ=0. Define four nonnegative blocks by

    Bν² = Σₑ |θν(Σᵢ eᵢuᵢ/2)|² ∏ᵢ<ⱼ |θ₁(uᵢ−uⱼ)|^(eᵢeⱼ/2).

Use period-one theta functions with modulus i times the aspect. All factors common to the four blocks cancel in the calibrated contrast. The corresponding empty trace has amplitudes |θν(0)|. For neither twist the physical sign combination is ++++. A time twist gives −−++, a spatial twist gives −++−, and both twists give −+−+ in numerical order ν=1,2,3,4. These follow from the transform between spin structures and physical disorder lines. The spin-inserted trace is normalized to the physical welded trace before constructing the parity-selected and annealed ratio. For the torus correlation formula and physical disorder-line transform see [Bayraktaroğlu and Izyurov, equations 2.1, 4.3 and 4.10](https://arxiv.org/html/2506.11324v1). An earlier derivation is given by [Di Francesco, Saleur and Zuber](https://doi.org/10.1016/0550-3213(87)90202-1).

## Independent quantum-chain checks

The private reference works on the finite quantum ring. In a fixed physical parity p and seam ξ, the Jordan–Wigner fermions have boundary sign t=−ξp. Let a and b be the two Majoranas per site. The signed forward shift minus identity has symbol exp(ik)−1. The thermal contraction Cⱼₗ=〈i bⱼ aₗ〉 follows by Fourier transforming that symbol divided by its magnitude, times tanh(β|exp(ik)−1|). Its zero-momentum value is zero.

For an ordered even probe, let S be the union of the nonwrapping bond intervals joining consecutive marked pairs. The spin product has expectation det C[S,S+1]. Its trace with P is (−1)^(N+|S|) times the minor on the complementary rows and columns. This identity keeps the contribution of a zero mode to a trace with an insertion even though its parity partition trace vanishes. The reference combines the four trace weights at the microscopic level, and compares the result at 256 and 512 sites. The 512-site estimates agree with every target within 0.0000016. Removing the leading inverse-square correction reduces that bound to 2.1 × 10⁻¹⁰.

Literal diagonalization of the original eight-spin Hamiltonian agrees with the finite-chain determinant method within 4 × 10⁻¹⁵ for both seam signs, both selected parities, the unselected monitor and the bias derivative. Central differences of the independently stated bias response also reproduce every response target.

The completed approximation omits the zero-mode block from the spin-inserted traces on the grounds that it vanishes in the empty partition trace. All other continuum blocks and seam populations are retained. Its maximum errors on clamped seams, annealed seams and their responses are 1.003, 0.657 and 0.278 respectively, against tolerance 0.004. Run `OPENBLAS_NUM_THREADS=1 python scripts/validate_critical_seam_magnet.py` to reproduce the checks.
