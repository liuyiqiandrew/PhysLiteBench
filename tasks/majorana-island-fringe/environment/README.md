# Ramsey readout after coherent charge counting

A superconducting island meets a sequence of fresh pairs of spin-polarized quantum dots. Each encounter writes a shared quantum counter, then the outgoing dots are erased. The same island and counter persist through all encounters. A Ramsey probe reads the island after the selected counter outcome. Set ℏ=k_B=1.

## Initial island

The island has 2n Majoranas γ with {γⱼ,γₖ}=2δⱼₖ. Define γ(u)=∑ uⱼγⱼ and Q(u,v)=−iγ(u)γ(v) for an ordered orthonormal pair. Write Qⱼ=−iγ₂ⱼγ₂ⱼ₊₁. Initially the island is in thermal equilibrium at inverse temperature `beta` for H=−∑ EⱼQⱼ/2 within the given eigenvalue `parity` of ∏Qⱼ. Here Eⱼ=1+`ripple` cos[2π(j+1/2)/n], j=0,…,n−1. The preparation Hamiltonian is switched off during the experiment. The integer n is in [3,32], beta in [0.4,6], ripple in [−0.45,0.45] and parity is ±1.

For `frame="site"` the contact vectors uₐ are the coordinate basis of ℝ²ⁿ. For `frame="wave"` they are rows of its orthogonal cosine matrix. Row zero is constant 1/√(2n), and row a>0 is uₐⱼ=cos[πa(j+1/2)/(2n)]/√n. At most six distinct contact indices occur in an input.

## Encounters

The counter has D=2,…,4 orthonormal states |k⟩, k=0,…,D−1. Its initial pure state is `counter.prepare`. Dot, counter and analyzer states are lists of complex amplitudes [real,imaginary]. Normalize each complete nonzero list to one. The initial counter and island are independent.

The top-level `encounters` lists zero to four encounters in time order. In each encounter perform the following operations.

1. Prepare a fresh dot pair with one electron in the normalized state (p₀ d₀†+p₁ d₁†)|vac⟩, with p=`prepare`. Dot operators obey the canonical fermion anticommutators and anticommute with the island Majoranas.
2. Apply the two or three switches in `pulses` in order. Each [j,a,b,τ,φ,χ] connects dot j=0 or 1 to the island channel c=(γ(uₐ)+iγ(uᵦ))/2, where a and b are distinct. For a counter component |k⟩ the integrated tunneling Hamiltonian is τ[exp(iφ′)dⱼ†c+exp(−iφ′)c†dⱼ] with φ′=φ+j k `feedback`. This operation is diagonal in the counter basis and preserves its coherence. Following the tunneling pulse, apply the charging phase exp(−iχ n₀n₁) where nⱼ=dⱼ†dⱼ. Intermediate empty and doubly occupied dots are allowed.
3. Tag the normalized single-electron dot state |t⟩ specified by `tag`. On the projector P=|t⟩⟨t|, send |k⟩ → |(k+1) mod D⟩ and create a marker |m₁⟩. On I−P leave the counter unchanged and create |m₀⟩. These alternatives act linearly on a superposition. Each encounter uses an independent fresh marker with normalized states and overlap ⟨m₀|m₁⟩=`marker`, a single [real,imaginary] of modulus at most one. No observer reads a marker.
4. Retain only the projection of the dots onto the pure single-electron state `erase` and discard the dots. During the counter dwell each resulting component |r⟩ acquires exp[−i `dwell` r(r−1)/2], with r the counter label after the tag. Continue to the next encounter without resetting the counter or island.

All switch angles, feedback and dwell lie in [−π,π]. Between named operations there is no additional evolution. After the last encounter project the counter onto the pure state `counter.analyze`. The joint probability of all retained dot and counter outcomes is greater than 0.001 in the supplied runs.

## Ramsey probe

Prepare an arm qubit in (|0⟩+|1⟩)/√2. The top-level `probe` contains the two arm histories. Each contains zero to five pulses [a,b,θ] in chronological order, with integrated Hamiltonian θQ(uₐ,uᵦ)/2 for |θ|≤2π. The qubit is measured along cos(λ)X+sin(λ)Y with λ=`phase` and ⟨0|Y|1⟩=−i. The reported reading is this conditional arm expectation times one positive visibility V in [0.65,0.95]. The island and every marker are unobserved.

For a check, take n=3, site frame, beta=1.5, ripple=0 and parity=+1. Use D=3, counter input |0⟩ and analyzer |1⟩. Let both encounters prepare and erase the equal real dot superposition, tag dot 0, use marker [1,0], feedback=0 and dwell=0. Both encounters use [1,0,1,π/2,0,χ] followed by [1,0,1,−π/2,0,0]. Set χ=0 in encounter one and χ=π in encounter two. The final probe is [[],[[0,1,π]]] and phase=−π/2. Two tag histories reach counter 1. For an initially filled island channel the charging dwell reverses the sign of exactly one history, so they cancel. Only the empty channel remains. The retained probability is 0.228341417760 and the reported fringe is V. Setting both markers to [0,0] changes them to 0.125 and 0.826731342082 V, respectively.

Implement Model.fit(records), returning self with `visibility`, and Model.predict(inputs), returning a finite real NumPy array of shape (len(inputs),), in input order. Calibration records in data/calibration.json have identical arm histories and no encounters. Absolute calibration error must be below 0.000002. Maximum absolute error in each new family must be below 0.003. The starter assumes an independent thermal island for the final probe. The supplied `dot_core.Island` represents every addressed contact exactly. Its `outgoing(encounter,k)` returns the two island amplitudes for the electron to exit through dot 0 or 1 at a given incoming counter label, before the tag and eraser.
