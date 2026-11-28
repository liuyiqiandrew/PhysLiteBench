# Order and disorder echo on an Ising ring

A Ramsey interferometer uses the critical periodic spin chain

    H = −Σⱼ (Xⱼ Xⱼ₊₁ + Zⱼ),       P = ∏ⱼ Zⱼ,

where X and Z are Pauli matrices and site N is site 0. Use N→∞ through multiples of 16. The source prepares the lowest-energy spin state |s⟩ with P=+1 for sector `even` or P=−1 for `odd`. Let Eₛ be the energy of this state. Both sectors use the same periodic physical spin Hamiltonian.

The pulse list is in chronological order. For a pulse [kind,a,b], let l=Na and r=Nb. A `spin` pulse applies XₗXᵣ. A `string` pulse applies the product ZₗZₗ₊₁⋯Zᵣ₋₁. The apparatus applies each pulse coherently in the signal arm. After pulse j and before pulse j+1, the corresponding `segments` entry [e,t] applies the energy filter and evolution

    Kⱼ = exp[−(N/2)(e + i t)(H − Eₛ)].

The factor N/2 uses the critical propagation speed of this Hamiltonian. Negative t reverses real-time evolution. The filter attenuates excited-state amplitudes. Keep that attenuation when evaluating the interference signal.

For pulses O₁,…,Oₘ the complex return amplitude is ⟨s|Oₘ Kₘ₋₁ Oₘ₋₁ ⋯ K₁ O₁|s⟩. At each N the apparatus also measures each Oⱼ separately in a fresh **even-sector ground state**, obtaining dⱼ=⟨even|Oⱼ|even⟩. Write A for the limit of the return amplitude divided by ∏ⱼ dⱼ. The reported real signal is V Re[exp(−i `phase`) A]. This normalized signal can exceed one in magnitude. Blank calibration records contain only `phase` and give V cos(phase), rounded to nine decimal places. The common visibility V lies in [0.65,0.95].

Each query has two to four pulses with endpoints on the grid j/16, 0≤a<b≤15/16 and 3/16≤b−a≤9/16. Endpoints at different pulses may coincide. The positive filter settings are in [0.085,0.18], |t|≤0.70, and phases are in radians in [−π,π]. The even-ground-state normalization and scaling limit apply to every sequence.

For example, use `pulses=[["spin",0.125,0.375],["string",0.25,0.8125]]` and `segments=[[0.12,0.30]]`. The normalized amplitude in the even sector is 1.12733483+0.26515265i. In the odd sector it is −0.22673216−0.03834057i. At V=0.80 and phase π/2 the corresponding reported readings are 0.21212212 and −0.03067246.

Implement Model.fit(records) and Model.predict(inputs) in /app/model.py. Fit takes records with `input` and `value`, stores V in `self.visibility`, and returns self. Predict returns a finite real NumPy array of shape (len(inputs),) in input order, including an empty array for an empty list. The maximum absolute error must be below 0.000002 for calibration and below 0.004 in each of the even, odd and reversed-evolution sets. The starter computes the single-pulse scaling limits and factorizes them between pulses. NumPy and SciPy are available.
