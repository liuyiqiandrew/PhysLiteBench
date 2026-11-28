# Two quantum repeater arms with an erased source marker

Two remote Bell-pair measurements share a polarization-programmable memory source between stations L and R. A source qubit can preserve coherence between two preparations of the bank. Each arm swaps one stored pair to a remote user. Predict the acceptance of the source marker, the joint optical herald, and the correlation of the four output spins.

## Pair preparation

Every qubit uses the basis |0⟩,|1⟩. Angles are in radians. An ordered pair setting `[β,θ,φ]` prepares the normalized **ket with this phase convention**

    (I ⊗ Rᵧ(θ)) [cos(β)|00⟩ + exp(iφ) sin(β)|11⟩],

where Rᵧ(θ)|0⟩=cos(θ/2)|0⟩+sin(θ/2)|1⟩ and Rᵧ(θ)|1⟩=−sin(θ/2)|0⟩+cos(θ/2)|1⟩.

The source emits N elementary pairs into distinguishable memory slots. N is Poisson with mean a × `loading`. The common loading factor a is calibrated by measuring the probability of zero emissions at known `exposure` and lies in [0.75,1.15]. Photon-number heralds reveal no polarization or source-qubit state. The science experiment retains N≥2. An additional `parity` of 0 or 1 selects even or odd N; null accepts either. All probabilities below are in this ready, parity-selected ensemble.

The two entries of `bank` specify kets |B₀⟩ and |B₁⟩, each listed in port order (L,R). Conditional on N, the pure state immediately after loading is

    [ |0⟩ₘ |B₀⟩⊗ᴺ + |1⟩ₘ |B₁⟩⊗ᴺ ] / √2.

N is classical. The source qubit m is common to all slots. A source analyzer projects it onto (|0⟩+`sign` exp(i `phase`)|1⟩)/√2, with parameters from `marker`. Selected shots continue. The arms use bank slots 0 and 1. The other stored pairs remain unread and are discarded. The electronics do not observe the preparation label 0 or 1.

## Bell analyzers and feed-forward

Arm j has its own ordered outer pair `outer[j]` at (remote user sⱼ, relay qubit mⱼ). Its bank slot supplies a relay qubit nⱼ and a remote output tⱼ. Arm 0 uses n₀=L, t₀=R. Arm 1 uses n₁=R, t₁=L. The two slots and both outer pairs are distinguishable.

On arm j, rotate nⱼ by Rᵧ(`analyzers[j]`). The optical analyzer performs a Bell measurement on (mⱼ,nⱼ). Only the two orthogonal outcomes

    |Ψ_b⟩ = [|01⟩ + (−1)ᵇ |10⟩]/√2,      b=0,1

are conclusive. The other Bell outcomes are discarded. The local electronics learn b and apply X Zᵇ to the remote tⱼ, with the rightmost operator acting first. The remaining output pair is in order (sⱼ,tⱼ). If both arms are conclusive, the central recorder keeps only b₀ XOR b₁. An integer `report` selects that parity. Null accepts all doubly conclusive records. Different Bell outcomes are measured alternatives even when the central recorder retains the same label.

## Readings

- `observable="marker"` is the probability of the specified marker outcome among ready, parity-selected loadings.
- `observable="herald"` is the probability that both Bell analyzers are conclusive and the report matches, **conditioned on the specified marker outcome**.
- `observable="fringe"` is the mean product of four Pauli measurements after all those selections and both feed-forward corrections. For `spin=["ZX","XY"]`, for example, measure Z on s₀, X on t₀, X on s₁ and Y on t₁. All four measurements are on the same shot; Pauli matrices use the standard convention Y|0⟩=i|1⟩.

The graded ensemble has loading in [0.5,3], β in [0,π/2], all other angles in [−π,π], and positive probabilities for the conditioning events. The attenuated memories can produce nonmaximally entangled pairs.

## Worked optical checks

Set a=1, loading=ln 2, both outer pairs to [π/4,0,0], both analyzer rotations to zero, and marker sign +1 and phase zero.

First set the bank to [0,0,0] and [π/2,0,0], the two kets |00⟩ and |11⟩. Select the even Bell report and measure XX on both remote pairs. The marker probability is 1/2 and the conditional herald probability is 1/8. When N=2, the two preparations interfere on the four surviving qubits. An unused |00⟩ or |11⟩ slot distinguishes the two preparations perfectly. Consequently the fringe at even occupancy is Pr(N=2 | even N≥2)=2(ln 2)² ≈ 0.960906028. It is zero at odd occupancy.

For a feed-forward check use identical bank preparations [0,π/2,0], or |0+⟩. Select even bank occupancy and measure ZX on arm 0 and XZ on arm 1. The marker probability is 1. The even and odd Bell reports each have conditional probability 1/8. The joint fringe is +1 for the even report and −1 for the odd report.

## Interface and grading

Implement `Model` in `/app/model.py`. `fit(records)` receives calibration dictionaries with `input` and `value`, stores `activity`, and returns self. `predict(inputs)` returns a finite one-dimensional real NumPy array in order, including an empty array for empty input.

The starter fully propagates the two actual optical analyzers, including their outcome probabilities and corrections. It assumes the source qubit lost its off-diagonal coherence before the eraser measurement, and consequently mixes the two bank preparations. The public `apparatus.arm` computes an unnormalized amplitude for one specified Bell outcome of an outer pair and a bank ket.

Ten calibration readings require maximum absolute error below 0.00002. The public optical checks and each of the three held-out groups require maximum absolute error below 0.003. Each group contains nine readings at settings distinct from the worked examples.
