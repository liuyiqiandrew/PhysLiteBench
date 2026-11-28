# Majorana island fringe, revision 1

Parity-preserving gates on a charge-isolated superconducting island rotate its Majorana modes. A probe in coherent superposition between two pulse sequences measures their many-body relative phase. A second qubit controls the coupling during the pulses and selects which trials the detector keeps. The physical quantities here are the parity-constrained thermal average, the spinor lift of each rotation and the postselected two-arm coherence. A relative 2π pulse reverses the fringe even though the Majorana coordinates return to their original values. Joint parity measurements on superconducting islands and trace identities for paired fermions motivate the model. [Hyart et al., 2013](https://arxiv.org/abs/1303.4379) · [Klich, 2014](https://arxiv.org/abs/1403.7824)

## Measurement

For arm a let Uₐ₊ and Uₐ₋ be the ordered island unitaries at the two meter Z values. On meter outcome s the island operator for this arm is

    Kₐ = (Uₐ₊ + s exp(−iφ) Uₐ₋)/2.

The unnormalized probability of retaining the shot is p = (⟨K₀†K₀⟩+⟨K₁†K₁⟩)/2. The measured reading is V Re[exp(−iλ)⟨K₀†K₁⟩]/p. These expressions follow by projecting the complete two-qubit state once at the end. They include interference between the meter histories as well as between the arm histories. The identical-arm calibration gives V cos λ for every retained outcome.

In the full Gibbs state the mode pairs are independent with polarizations tⱼ=tanh(βEⱼ/2). The contractions within a pair are ⟨γ₂ⱼγ₂ⱼ₊₁⟩=itⱼ. For total parity P and required parity η, each charge-projected average is

    ⟨O⟩η = [⟨O⟩ + η⟨OP⟩]/[1+η∏tⱼ].

The oracle computes each ordered pulse product directly. For a rotation α, its two terms are c=cos(α/2) times the identity and s=−sin(α/2) times γ(u)γ(v). In the Wick matrix take the ordered forms sγ(u), γ(v) for every pulse, put their pair contractions above the diagonal and antisymmetrize. Add c to the upper-diagonal element of each pulse pair and −c to its transpose. In the Pfaffian expansion, selecting this c removes the adjacent two forms without changing the sign of the remaining pairing. All other terms contract the selected bilinears by Wick's theorem. This identity yields the signed ordered expectation even when a trace crosses zero. For ⟨OP⟩ append the n parity bilinears, each with c=0 and s=−i.

## Independent checks

The private reference multiplies Majorana monomials explicitly in the addressed contact basis and uses anticommutation to track the sign. It computes the charge insertion by replacing each tⱼ by 1/tⱼ in the contractions and multiplying by ∏tⱼ. This follows by taking the trace of Pρ separately on each thermal pair. A third implementation evolves the actual occupation-basis density amplitudes with Pauli-string Majoranas and projects the meter. The validator compares this calculation against the sparse reference for every fixture through six fermion modes. Larger preparations include both contact frames and noncommuting pulses on islands through 32 fermion modes.

Seed 718305 gives 12 calibration records and four families of eight held-out measurements. The winding family rotates one arm through 2π before a further pulse. The meter family changes coherent coupling to the same retained meter. The charge family probes a cold odd island where discarding the conserved parity changes the occupations appreciably. The extended family uses delocalized contacts on larger islands. The fixed prediction tolerance of 0.003 resolves the observable to three parts per thousand of full contrast. Independent calculations agree below 2 × 10⁻⁹. The visibility is identifiable from a linear fit to the calibration signals, whose largest rounding error is 5 × 10⁻⁹.

The completed shortcut evolves both qubits coherently but uses an island in thermal contact with a parity reservoir. It fits the visibility and misses the cold fixed-charge family by up to 0.594. The public starter instead uses the correctly charge-conditioned mean pair parity as a static classical field. It calibrates correctly and misses the coherent meter family by up to 1.415. Both models give finite predictions and leave different signatures in the verifier metrics.

Run `OPENBLAS_NUM_THREADS=1 python scripts/validate_majorana_island_fringe.py` to reproduce the fixtures and check the oracle, shortcut and starter. Add `--generate` to write the fixtures. The solve and verification limits match the previous physical-failure panels at 1,800 and 180 seconds. The oracle grades all held-out signals in under two seconds with one BLAS thread.
