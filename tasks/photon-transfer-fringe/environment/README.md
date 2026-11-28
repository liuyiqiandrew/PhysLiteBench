# A two-photon echo in a coherent delay line

Two photons pass a single Λ atom. A lossless optical delay returns the complete outgoing field through the same atom a second time. The delay mixes two temporal arms and can replay each one in reverse order. Predict the final polarization-transfer fringe after selecting two photons in one temporal mode. The only unknown apparatus parameter is the counter visibility V.

## Counter and waveguide

The counter measures q incident and q′ outgoing photons in polarization 0. The fringe is V times the mean of cos[λ(q′−q)−φ], with `lambda` and `phase` giving λ and φ. In the 12 calibration records `source=[[n,0,0],[0,0,0]]` denotes a thermal state of mean n in polarization 0 and vacuum in 1. `before` contains a single mixer between the counters. The provided `optics.py` evaluates these unfiltered calibration shots. V is in [0.65,0.95].

A polarization or atomic ground-state rotation `R([θ,p])` has matrix [[cos θ, −exp(−ip)sin θ], [exp(ip)sin θ, cos θ]]. Its columns give the outgoing kets of the incident basis states. Complex numbers in the inputs are `[real,imag]`.

In echo shots, one photon enters in each polarization, so q=1. The two rows of `input_modes` give their normalized temporal envelopes f₀ and f₁. They pass through the common `input_rotation` before the atom. The incident state there is A₀†A₁†|vac⟩ with Aᵢ†=∑c R(input_rotation)_{ci}∫fᵢ(t)b_c†(t)dt. Traveling fields satisfy [b_c(t), b_d†(u)]=δ_cd δ(t−u).

Each passage consists of two successive arms of equal length D=`duration`. Each arm contains n equal drive intervals, with n either 1 or 2. A temporal mode has shape (2,n) in chronological order. Its entries sum in squared magnitude to one. Divide each entry by √(D/n) to get its continuous envelope on that interval. The envelopes vanish outside their passage. The atomic detunings `detuning` and `return_detuning` also have shape (2,n) in chronological order, one array per passage.

The atom begins in R(`atom`)|g₀⟩, using the ground doublet (g₀,g₁). Both polarizations couple to the same excited state e. `rates=[Γ₀,Γ₁]` gives its two emission rates. For detuning δ set L_c=√Γ_c |g_c⟩⟨e| and H=δ|e⟩⟨e|. Each encounter with right-moving field increments obeys the lossless Itô equation

    dU = [∑c(L_c dB_c† − L_c† dB_c) − (iH + ½∑c L_c†L_c)dt] U.

Direct transmission is identity. The public `single_pass.transfer` implements one such passage between specified optical Fock modes. Its docstring gives the array and normalization conventions.

## Delay and readout

After the first passage, `control=[θ,p,s]` applies R(θ,p) on the atomic ground doublet and exp(is) on e. During `gap=[duration,detuning]` the atom evolves into vacuum with the same rates. Gap radiation leaves by a separate port. No atomic measurement occurs between passages.

For a local delay coordinate x in [0,D], define τ(r,x)=D−x when r is true and x otherwise. Call the fields leaving passage one A and those entering passage two B. The delay acts on **every** temporal component by the lossless creation-operator map

    A†_{j,c}(τ(input_reverse[j],x)) → ∑k,d T_{kj} W_j[d,c] B†_{k,d}(τ(output_reverse[k],x)).

The indices j and k denote first-pass and second-pass arms in their respective chronological order, each numbered 0 and 1. Both times inside parentheses are local to their arm. `temporal` is the unitary T of shape (2,2) in complex-pair format. For `arms[j]=[θ,p,α]`, use W_j=exp(iα)R(θ,p). A true reversal reorders waveform samples while preserving their complex amplitudes. The two arms interfere at equal delay coordinate x. The entire optical state, including any radiation emitted by the atom during the first passage, goes through this map. Only the separate gap port is discarded.

In the second outgoing field select two photons in the same normalized temporal envelope `output_mode`. Apply `output_rotation` to their polarizations and let q′ be the number detected in polarization 0. The annihilator for detected color r is ∑c R(output_rotation)_{rc}∫output_mode*(t)b_c(t)dt. If `analyzer=[θ,p]`, also select the atom in R(θ,p)|g₀⟩. A null analyzer sums over atomic states. Let I be the indicator of this joint outcome. `conditional=true` reports V E[I cos(λ(q′−1)−φ)]/E[I]; `conditional=false` reports V E[I cos(λ(q′−1)−φ)] per launched pair. D is in [0.5,1.7], rates in [0,1.8] with positive sum, |δ|≤1.1, gap duration in [0,0.5], and all angles have magnitude ≤π. Every graded echo has E[I]>0.005.

For a worked shot set D=1/2 and n=1. Both input modes and the output mode have coefficient 1/√2 on each arm. Set rates `[1,0]`, atom g₀, analyzer null, detunings and gap zero, all rotations identity, λ=φ=0 and `conditional=false`. Set `arms=[[0,0,π],[0,0,π]]`, `input_reverse=[false,false]`, and let T swap the arms with unit positive amplitudes. With `output_reverse=[true,true]`, the field returns in wholly reversed chronology. The second passage undoes the scattering, giving V. With T set to identity and `output_reverse=[false,false]`, the field returns in the original chronology and the reading is 0.8057582941 V.

## Interface and scoring

Implement `Model` in `/app/model.py`. `fit(records)` takes records with `input` and `value`, sets `visibility`, and returns self. `predict(inputs)` returns a finite one-dimensional real NumPy array in order, including an empty array for an empty list. The supplied calibrated starter retains one flat optical waveform per drive interval and polarization in the delay. It coherently returns those modes and the intermediate atomic state.

Calibration requires maximum absolute error below 0.00002. The two worked shots and each of three held-out groups require maximum absolute error below 0.001. The groups have eight readings each and cover direct temporal routing, interfering early and late arms, and spin-selected echoes. Both per-pair and conditional fringes are included.
