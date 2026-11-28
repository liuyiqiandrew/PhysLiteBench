# A coherent Fermi edge measured by spatial parity

In the denominator, the reference arcs each contain a single sign-changing interval acting on a consecutive filled Fermi sea. For the numerator the parity operation multiplies a one-particle wavefunction by −1 on the union and by +1 outside. The many-particle matrix elements are Slater determinants of its plane-wave overlaps, including off-diagonal minors for a coherent source. [Abanov, Ivanov and Qian, J. Phys. A 44, 485001 (2011)](https://doi.org/10.1088/1751-8113/44/48/485001)

For a filled sea, apply generalized Fisher–Hartwig asymptotics to its sign symbol. With 2m jump endpoints x_j all neutral minimal representations have equal leading power N^(−m/2). Write e_j=±1, with sum e_j=0. Up to a common constant the sector weight is

    exp[iπ Σ e_j (delta_j + d x_j)] × ∏_{j<k} [2 sin π(x_k−x_j)]^(e_j e_k/2).

The reference contains m single arcs whose two leading terms add in phase. Its common prefactor cancels the numerator's, leaving the denominator ∏ 2[2 sin π(b−a)]^(−1/2). The choice of endpoint grid and subsequence fixes all microscopic reference phases. [Deift, Its and Krasovsky, Ann. Math. 174, 1243 (2011)](https://annals.math.princeton.edu/2011/174-2/p12)

Relative to the filled determinant, the three excited wavefunctions are the Schur polynomials p1, (p1²+p2)/2 and (p1²−p2)/2 in the particle coordinates, where p_k is the sum of exp(2πikx) over all particles. This follows by dividing the corresponding alternating determinant of plane waves by the Vandermonde determinant. Differentiate the smooth-symbol generating function of each jump sector to insert these polynomials. The positive traces p_k have means μ_k=Σ e_j exp(2πikx_j)/2. The negative ones have means −conjugate(μ_k). The only second cumulants are k between p_k and p_−k. Higher connected cumulants vanish in that sector. These are formal sector moments; summing conjugate representations gives a Hermitian observable. Keeping only the diagonal Slater entries retains exact populations and loses the measured packet interference.

The independent calculation integrates the finite plane-wave sign matrix exactly and takes microscopic minors, then extrapolates four ring sizes in 1/N. It checks both the coherent doublet and the full continuum model. Small determinants also check the complementary-minor acceleration against direct Slater determinants. Run `OPENBLAS_NUM_THREADS=1 python scripts/validate_fermi_arc_interferometer.py` from the repository root.
