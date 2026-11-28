# Spectroscopy of a random-singlet cut

Two copies of an infinite antiferromagnetic spin-½ XX chain have **identical fixed exchange couplings**. Prepare both in their ground state. In the strong-disorder scaling model below, each ground state consists of singlet pairs. A Ramsey probe acts on one copy. An optional herald detector acts on the other. Repeat with fresh disorder realizations and pool all retained shots.

## Spectrum across one cut

The chain is bipartite with alternating original site signs +1 and −1. At a running cutoff Ω, the strongest active bond forms a singlet between neighboring active spins. Remove both spins and replace the adjacent bonds by one bond J′=J_left J_right/Ω. A crossing is the formation of a singlet whose two spins lie on opposite sides of a fixed cut in the original chain. All readouts concern these crossing singlets. The ideal spectrometer addresses a pair by the energy scale at which it formed and keeps its original site and sublattice labels.

Use the asymptotic fixed-point scale Γ=ln(Ω₀/Ω) and scan coordinate μ=ln(Γ/Γ_start). The input bands are consecutive intervals of μ in increasing order, starting at μ=0. Their widths are positive. For quantitative predictions use the following local strong-disorder flow, which defines the scaling model at the marked cut.

- There is one active bond across the cut. Write x=ln(Ω/J)/Γ for its nonnegative scaled log coupling. While no nearby bond is decimated, dx/dμ=−(1+x).
- Neighboring decimations on the two sides occur independently at rate one per side per unit μ. Each adds a fresh independent exponential variable of mean one to x. They do not produce a crossing singlet.
- When x reaches zero, register a crossing and remove its two endpoint spins. The new marked bond has x equal to the sum of two independent mean-one exponentials. This restarts the same flow. Outside the marked cut a decimation always removes two neighboring active spins.

For `start="crossing"` the scan opens immediately **after** a crossing, which lies outside the observed bands. For `start="bulk"` the scan opens at a point sampled uniformly in μ along a record of asymptotically long duration. The optional `last` is the original sublattice sign of the left spin of the most recent crossing before the scan. If null, that sign is equally likely to be +1 or −1, independent of the magnitude disorder. The couplings and sublattice realized on every band are common to both copies. No scan boundary resets the chain.

## Pair operations and the herald

Each band specifies `width`, two `arms` and a `herald`. In each arm the pulses are listed in chronological order. A pulse [side,axis,u,v] applies exp[−i(u+s v)σ_axis/2] to every crossing singlet in the band, on its left or right spin according to side ("L" or "R"). Here s is that spin's **original site sign**, and axis is "x", "y" or "z". Use the usual Pauli matrices with ⟨↑|σ_y|↓⟩=−i. A singlet is (|↑↓⟩−|↓↑⟩)/√2 in left-right order. All other pairs remain untouched.

The Ramsey ancilla starts in (|0⟩+|1⟩)/√2 and coherently selects the corresponding arm. At the end measure cos(λ)X+sin(λ)Y with λ=`phase`. The signal copy is unobserved.

A band with `herald=[efficiency,h,g]` rotates the **left** spin of each crossing singlet in the herald copy about z by angle h+s g. Measure the pair in the singlet and triplet subspaces. A triplet produces one registered click with the stated efficiency, independently for each pair, and a singlet never clicks. If `count` is an integer, pool only runs with exactly that many clicks across all bands. If it is null, retain every run. The answer is the conditional mean Ramsey reading over this pooled ensemble times a common visibility V.

Each input contains zero to four bands, with widths in [0.2,4.0] and zero to three pulses per arm per band. Angles lie in [−π,π], efficiencies in [0,1] and integer target counts in [0,2]. The retained fraction exceeds 0.025 in the supplied runs.

## Worked scan and interface

Open just after a crossing, set phase to zero and use a single band of width 1. Arm 0 applies no rotation and arm 1 applies ["L","z",π,0]. A single crossing makes the two spin states orthogonal. With count=null the reported fringe is 0.786645599303 V. For the bulk opening the fringe is 0.695775953416 V. With herald=[1,π,0] and count=0 the retained fractions are those two coefficients respectively, and the conditional fringe is V in both cases.

Implement Model.fit(records), returning self and setting `visibility`, and Model.predict(inputs), returning a finite real NumPy array in input order with shape (len(inputs),). Calibration records in data/calibration.json use empty scans with V in [0.65,0.95]. Maximum absolute calibration error must be below 0.000002. The maximum absolute error per new family must be below 0.003. The starter fits the visibility and assumes that crossing singlets form a Poisson process with the correct bulk mean rate 1/3. The supplied `apparatus.pair_response(band,left_sign)` returns the single-pair arm overlap ⟨ψ₀|ψ₁⟩ and the herald click probability.
