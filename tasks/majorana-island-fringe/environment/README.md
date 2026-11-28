# Ramsey readout with a finite Fermi lead

A superconducting island meets a sequence of fresh pairs of spin-polarized quantum dots. Each encounter writes a shared quantum counter, then the outgoing dots are erased. The same island and counter persist through all encounters. A finite, isolated lead can record the tagged dot histories. A Ramsey probe reads the island after the selected counter and lead outcomes. Set ℏ=k_B=1.

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

All switch angles, feedback and dwell lie in [−π,π]. Between named operations there is no additional evolution. After the last encounter project the counter onto the pure state `counter.analyze`. The joint probability of all retained outcomes, including any lead measurement, is greater than 0.001 in the supplied runs.

## The Fermi lead

When the optional top-level `lead` field is present, prepare one closed ring of M spin-polarized fermion sites before the first encounter. The ring and island start independently. The lead is in thermal equilibrium at its own `beta` in the sector of exactly `particles` electrons. There is no particle exchange with the dots or the island. The one-body Hamiltonian has diagonal `onsite[j]` and hopping matrix element h[(j+1) mod M,j]=−`hop` exp(i `flux`/M), with the Hermitian conjugate on the reverse bond. All remaining elements are zero. The many-body Hamiltonian is ∑ hᵢⱼ aᵢ†aⱼ.

There is one entry [j,v,t] in `lead.quenches` for each encounter, in the same order. During its tag operation the lead evolves for time t with Hamiltonian H+b v nⱼ, where b is 1 on the tagged dot state and 0 on its complement, and nⱼ=aⱼ†aⱼ. Each unitary is controlled coherently by the dot state. The same lead persists from one encounter to the next. Its terminal many-body state is unobserved except for the charge measurements specified below. The fresh markers described above are separate degrees of freedom.

If `lead.readout` is null, there is no charge meter. Otherwise it has distinct site indices in `sites` and an integer `delta`. Just before the first encounter, perform an ideal projective measurement of the **total** electron count on those sites. After the last encounter, repeat that measurement and retain a run if final count minus initial count equals `delta`. The apparatus stores only the two total counts. Given a count k, it applies the projector onto its full eigenspace and preserves coherence within that space. All retained outcomes with the requested difference contribute, and neither count itself is used for the Ramsey probe.

Here M is 4 to 8, the number of electrons is 1 to M−1, onsite potentials lie in [−2,2], hop in [0,1.5], flux in [−π,π] and lead beta in [0.4,3.5]. Quench potentials lie in [−5,5] and durations in [0,2]. A region contains between one and M−1 sites.

## Ramsey probe

Prepare an arm qubit in (|0⟩+|1⟩)/√2. The top-level `probe` contains the two arm histories. Each contains zero to five pulses [a,b,θ] in chronological order, with integrated Hamiltonian θQ(uₐ,uᵦ)/2 for |θ|≤2π. The qubit is measured along cos(λ)X+sin(λ)Y with λ=`phase` and ⟨0|Y|1⟩=−i. The reported reading is this conditional arm expectation times one positive visibility V in [0.65,0.95]. The island and every marker are unobserved.

For a check, take n=3, site frame, island beta=1.5, ripple=0 and parity=+1. Use D=3, counter input |0⟩ and analyzer |1⟩. Both encounters prepare and erase the equal real dot superposition, tag dot 0 and use marker [1,0], feedback=0 and dwell=0. Both use [1,0,1,π/2,0,χ] followed by [1,0,1,−π/2,0,0]. Set χ=0 in encounter one and χ=π in encounter two. The final probe is [[],[[0,1,π]]] and phase=−π/2.

Use a four-site lead with two electrons, all onsite energies and hopping zero, flux=0, lead beta=1, quenches [[0,π,1],[1,π,1]] and null readout. The two relevant tag histories apply the parity of site 0 or site 1. Among the six equally likely pairs of occupied sites, four have odd occupation in the first two sites and two have even occupation. The overlap between the histories is −1/3. The retained probability is 0.090552860747 and the reported fringe is 0.681091139309 V. Without the lead these become 0.228341417760 and V.

Implement Model.fit(records), returning self with `visibility`, and Model.predict(inputs), returning a finite real NumPy array of shape (len(inputs),), in input order. The calibration runs in data/calibration.json have identical arms, no encounters and no lead. Absolute calibration error must be below 0.000002. Maximum absolute error in each new family must be below 0.003. The calibrated starter ignores coupling to the lead.

The supplied `histories.history_table(q)` gives the tag strings and two complex matrices in that order. Their entries indexed by bra string a and ket string b are the contributions from the island, counter, dots and fresh markers to the retained probability and the complex arm coherence ⟨U₀†U₁⟩. The helper `histories.measurement(q, detector)` takes an optional callable `detector(lead, histories)` supplying additional lead overlaps with the same bra and ket indexing. It returns retained probability and normalized fringe before visibility. You can use these helpers or implement the full instrument yourself.
