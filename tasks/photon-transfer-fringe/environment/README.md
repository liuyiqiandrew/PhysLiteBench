# A two-photon echo from a finite spin memory

Send two shaped photons through a single atom, write its outgoing light into a small spin memory, and return the retrieved light through the same atom. The counter measures photon transfer between polarizations. Its visibility V is the only unknown apparatus parameter. The supplied `single_pass.transfer` computes the complete continuous-field scattering matrix of the waveguide atom on either passage. The experiment requires the joint state of the returning field, memory and atom.

## Calibration and conventions

A number counter measures q incident and q′ outgoing photons in polarization 0. Its fringe is V times the mean of cos[λ(q′−q)−φ]. The fields `lambda` and `phase` are λ and φ. V lies in [0.65,0.95]. The twelve calibration records have `source=[[n,0,0],[0,0,0]]`, a thermal state of mean n in polarization 0 and vacuum in 1. `before` is one polarization mixer `[0,1,θ,p]` between the counters. The supplied `optics.py` implements these unfiltered calibration records.

Every polarization or spin rotation `R([θ,p])` has outgoing basis kets as its columns,

    R = [[cos θ, −exp(−ip)sin θ], [exp(ip)sin θ, cos θ]].

A complex entry is `[real,imag]`. A temporal envelope consists of complex entries fⱼ with ∑ⱼ|fⱼ|²=1. On interval j it equals fⱼ/√durationⱼ and it vanishes outside the listed intervals. Polarization labels c are 0 and 1. All phases and pulse areas are dimensionless.

## One pass through the waveguide

The atom has a common excited level e and ground states g₀,g₁. `rates=[Γ₀,Γ₁]` gives its emission rates into the two polarizations of a lossless right-moving field. The scattering matrices of the direct field are identity. In interval j, L_c=√Γ_c |g_c⟩⟨e| and Hⱼ=detuning[j]|e⟩⟨e|. The unitary obeys the vacuum Itô equation

    dU = [∑c(L_c dB_c† − L_c† dB_c) − (iHⱼ + ½∑c L_c†L_c)dt] U.

`single_pass.transfer(rates, durations, detunings, incoming, outgoing)` returns a 3×3 atomic matrix in the ordered basis (g₀,g₁,e). Each list contains zero, one or two arrays of shape (intervals,2) giving the continuous polarization amplitudes of the indicated optical modes. Input array f means ∑c∫f_c(t)b_c†(t)dt on the ket. Output array h means ∑c∫h_c(t)b_c(t)dt on the bra. Conjugate the temporal mode when building h. Waveguide modes outside these lists are projected on vacuum. Repeating an identical array applies the field operator twice, with the usual normalized Fock ket or bra requiring a factor 1/√2. The helper follows from the equation above and can also propagate an atom initially in e.

## Write, store and read

The emitter starts in R(`atom`)|g₀⟩. In the `first` passage one photon in polarization 0 occupies `input_modes[0]`, and one in polarization 1 occupies `input_modes[1]`. Hence q=1. Both pass through `first.input_rotation` before the emitter. The outgoing temporal mode `first.write_mode` is routed into the write port for both polarizations. The rest of the outgoing light is discarded. All subsequent optical inputs are vacuum apart from what the memory releases.

The memory contains `store.cells` identical cells at positions xⱼ=j−(cells−1)/2, numbered from zero. Each cell has three states, an empty state |v⟩ and occupied spin states |0⟩, |1⟩. It starts empty. Each cell holds at most one excitation. The write and read gates each have unitary exp(−iG). For either gate use the two `write` or `read` pulse areas θ_c and the corresponding `write_k` or `read_k` wavevectors k_c:

    G = ∑j,c θ_c [exp(ik_c xⱼ) a_c |cⱼ⟩⟨vⱼ| + exp(−ik_c xⱼ) a_c† |vⱼ⟩⟨cⱼ|].

Here |cⱼ⟩⟨vⱼ| acts only on cell j. For writing, a_c annihilates the selected first-pass wavepacket in polarization c. The field remaining at the write port after the gate is discarded. Before readout apply the map exp(i `store.phase`) R(`store.rotation`) on the occupied doublet of each cell, with |v⟩ unchanged. For reading, a_c is a fresh, initially empty optical mode with temporal envelope `second.read_mode`. After the read unitary, this light is sent through the waveguide atom. The cells do not couple to the waveguide atom except through these fields. Capture, rotation and retrieval are ideal coherent operations on the selected modes and cells.

The atom itself also evolves between passages. Immediately after the first passage, `handoff="ground"` retains the atom's ground doublet using a single projector |g₀⟩⟨g₀|+|g₁⟩⟨g₁|. `handoff="unmeasured"` applies no projection. Next `control=[θ,p,s]` applies R(θ,p) on its ground doublet and exp(is) on e. It then evolves for `gap=[duration,detuning]` into vacuum under the same rates. Radiation in the gap is discarded. The memory operations act on a separate system during this handoff. The stated gap accounts for all atomic evolution between the first and second windows.

## Final reading

The two gates and both passages conserve total excitation number. Keep the prescribed handoff outcome and two final photons in the same selected temporal envelope `second.output_mode`, in any polarization. Apply `second.output_rotation` before the final polarization counters. If `analyzer=[θ,p]`, also project the waveguide atom onto R(θ,p)|g₀⟩. If it is null, trace over the atom. Cells and discarded fields are unobserved. Let I indicate that the full shot was retained. For `conditional=true` report V E[I cos(λ(q′−1)−φ)]/E[I]. For `conditional=false` report V E[I cos(λ(q′−1)−φ)] per launched pair. Thus a joint reading at λ=φ=0 gives V times the selection probability. The phase `store.phase` is referenced to the atomic control.

Each passage has one to four intervals, with durations in [0.4,1.4] and |detuning|≤1.1. Rates are in [0,1.8] with positive sum. There are two or three cells. Write and read areas are in [0,1.7], wavevectors have magnitude ≤π, gap duration is in [0,0.7] and all rotation and readout phases have magnitude ≤π. Each graded geometry has a selection probability above 0.01.

For a check, use one unit-duration, resonant square interval in each passage, unit coefficient `[1,0]` for every temporal mode, rates `[1,0]`, atom g₀, null analyzer, unmeasured handoff, zero gap, and identity rotations. In two cells take both write and both read areas π/(2√2), all wavevectors zero, λ=φ=0 and `conditional=false`. With `store.phase=0` the reading is 0.7868098728 V; at `store.phase=π` it is 0.1235676703 V. The polarization-1 photon is a spectator at the waveguide atom and still participates in the memory gates.

A separate check on the memory alone uses two cells with identity spin rotation. For polarized input 0, set `write_k[0]−read_k[0]=π`. Opposite site phases cancel the one-excitation return. If two polarization-0 excitations are stored, both cells are occupied and their relative site phases sum to zero. The two-excitation return can therefore be bright.

## Model and grading

Implement `Model` in `/app/model.py`. `fit(records)` accepts dictionaries with `input` and `value`, sets `visibility` and returns self. `predict(inputs)` returns a finite one-dimensional real NumPy array in order, including an empty array for an empty list. The starter uses the calibrated waveguide transfer, projects the atom onto its ground doublet at handoff, and applies an independent single-excitation optical memory to each stored photon.

Calibration must have maximum absolute error below 0.00002. The public square-pulse readings and each of three held-out groups require maximum absolute error below 0.001. Each group has nine readings. They cover ground-selected finite spin storage, a phase-matched optical return with unmeasured handoff, and spatially phased retrieval with final ground-state analysis. Both joint and conditional readouts occur.
