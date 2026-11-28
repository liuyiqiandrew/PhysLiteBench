# Photon transfer fringe, revision 2

The apparatus measures the signed transfer of photons during a pulse. An entrance meter resolves the total in a bank, and an optional intermediate meter heralds on total parity. Both projections preserve degeneracies. A classical calculation that assigns each photon a path gives the correct calibration when one thermal source is mixed with vacuum. Several populated or squeezed inputs let the measurements distinguish indistinguishable bosons and reveal coherences that survive the first meter. The target is three unhinted GPT-5.6 Luna High attempts with completed physical failures. The 1,800-second solve allowance and 180-second verifier allowance follow the previous expansion to allow a submitted model to finish.

## Derivation

Write Γ(W) for the number-conserving Fock-space action of a passive single-particle matrix W. Define D(λ) with diagonal entries exp(iλ) on the bank and one elsewhere. Ignoring the first count while retaining its phase exp(−iλq) amounts to applying D(−λ) to the state averaged over a common bank rotation D(φ), with φ uniform on [0,2π]. This preserves every coherence between configurations with the same bank total.

For the herald, write Z for the diagonal sign change on its rails. Its retained projector is [1+sΓ(Z)]/2 for parity s. Let U and V be the before and after meshes. The two density-matrix arms contribute matrices

    Kᵢⱼ = U† Zⁱ V† D(λ) V Zʲ U D(−λ),    i,j ∈ {0,1}.

Sum their expectations with weights sⁱ⁺ʲ/4 and divide by the same expression at λ=0. Without a herald only i=j=0 enters with unit weight. The fringe is the rotated real part at the calibrated visibility. These expressions follow by expanding both projectors in the sequential Born rule.

The oracle purifies each thermal rail of occupation n with a second rail. Its pair amplitude before squeezing is t=√[n/(1+n)]. After the physical squeezer, the pair matrix for the physical and auxiliary rails has entries B₁₁=−exp(iχ)tanh r, B₁₂=B₂₁=t/cosh r and B₂₂=−B₁₁* t². Passive preparation transforms B into PBPᵀ, with P extended by identity on the auxiliary modes. The squared vacuum amplitude of the prepared pure state is c=∏[(1+n)cosh r]⁻¹. If C is its pair matrix after a bank phase rotation and C′ is obtained by applying K to it, the Gaussian overlap is

    c exp[−½ Σⱼ log(1−zⱼ)],    zⱼ = eigenvalues of C†C′.

The eigenvalues have modulus below one. Choosing each logarithm continuously from zero fixes the phase even when the determinant of the product crosses the negative real axis. The oracle averages 64 equally spaced bank phases.

## Independent checks

The private reference uses the physical modes' real quadrature covariance V and the normal-ordered form Γ(K) = :exp[a†(K−I)a]:. Let a=Lx with x=(q₁,…,qₘ,p₁,…,pₘ) and L=(I,iI)/√2. Gaussian normal moments have covariance V−I/2 in these coordinates. Setting H=L†(K−I)L gives the expectation det[I−(V−I/2)(H+Hᵀ)]⁻¹ᐟ². Continuation from the identity fixes the square root. The reference averages 112 midpoint phases. The validator compares 192-point results for the two largest selected pulse sizes in each family.

A third calculation constructs squeezed thermal density matrices in the number basis, propagates each conserved total-photon sector and applies the number and parity projectors explicitly. For its three-rail test, the probability beyond ten total photons is 1.63×10⁻⁷. The largest conditional-fringe difference for the absent, even and odd heralds is 3.7×10⁻⁷. A swapped squeezed vacuum provides the exact even-parity example in the public description. The Gaussian squeezed-state counting formalism is developed in [Hamilton et al., Gaussian Boson Sampling](https://doi.org/10.1103/PhysRevLett.119.170501). The use of Fock-space traces for transport statistics is described in [Klich, Full Counting Statistics](https://arxiv.org/abs/cond-mat/0209642).

## Calibration and controls

With occupation n in the only lit rail and transfer probability p=sin²t, the thermal geometric generating function gives [1+np(1−exp(−iλ))]⁻¹. The rotated real parts on twelve distinct preparations identify the common scale. The readings are rounded to eight decimal places. The verifier uses the public visibility range and maximum calibration error below 0.00002. On each test preparation the allowed absolute fringe error is 0.001, one part per thousand of full scale. On the 32 hidden readings, the two Gaussian calculations agree within 6×10⁻⁹ including rounding. The smallest retention probability is 0.227.

The starter assigns independently routed photons to each rail and replaces each pumped source by a thermal rail of the same mean. The revision 1 shortcut retained the exact individual squeezed-source photon statistics. Its per-rail generating function at z is the inverse square root of [1+(N+|M|)(1−z)][1+(N−|M|)(1−z)] with mean N and pair amplitude M. It resolved the rays individually at each mesh boundary. That control passed calibration and missed the mixing, squeezed, even-herald and odd-herald families by 0.07866, 0.07018, 0.08157 and 0.04475 respectively.

Run `OPENBLAS_NUM_THREADS=1 python scripts/validate_photon_transfer_fringe.py` from the repository root. The explicit `--generate` option rebuilds the stored readings using seed 911247.

## Revision 2 optical switch

A trigger photon in mode X applies −iεL_X to leading order. With one photon in each vacuum trigger, each order has amplitude proportional to −ε². The retained control bra gives a coherent sum T=(T₀+exp(i angle)T₁)/2, where T₀ applies A first and T₁ applies B first. The probability divided by ε⁴ and the transfer numerator both contain Tᵢ†…Tⱼ with i≠j. The revision 2 control preserves the exact herald response for each order separately and drops these interference terms, as would an environmental record that distinguished the orders. It passes every revision 1 group and misses the three switched groups by 0.04146, 0.01667 and 0.23823. The parent calibration and 32 revision 1 readings are preserved.

Move each passive mesh past the linear operators. Each numerator term is an ordered product of four linear fields followed by Γ(K). On purified pair states C and C′, define G=(I−C′C†)⁻¹. For fields with annihilation and creation coefficient rows (uᵢ,vᵢ), the normalized transition contraction of the i-th and j-th field is (uᵢ+vᵢC†)G(C′uⱼᵀ+vⱼᵀ). Wick contraction of the four fields gives the sum over three pairings, weighted by the passive overlap and the phases of the control arms. The source twirl is applied before summing, and the denominator uses the same sum at zero transfer angle.

The second implementation normal orders exponentials of the four fields before performing the real quadrature integral. Between coherent states its linear source is vᵢα*+uᵢKα. Commuting an annihilator past a later creator gives the additional ordered covariance uᵢvⱼᵀ. The remaining covariance is the Gaussian one tilted by the final passive action. The comparison on all 18 additional rows agrees within 5×10⁻⁹ including rounding. The smallest scaled success probability is 0.03056. The standalone validator also implements both weak clicks as matrices on occupation states with up to 18 total photons, independent of Gaussian contractions. Its conditional reading and scaled success agree within 1.3×10⁻⁸ on a two-rail coherently ordered preparation. The public subtraction-port example follows from the commutator [a,a†]=1 and the parity of a thermal rail.
