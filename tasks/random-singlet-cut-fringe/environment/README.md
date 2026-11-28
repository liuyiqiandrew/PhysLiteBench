# Returning a carrier spin through a random-singlet chain

Two copies of an infinite antiferromagnetic spin-½ XX chain have identical fixed random exchange couplings. Prepare both in the strong-disorder singlet ground state. A single carrier spin interacts twice with each crossing singlet in the signal copy. The Ramsey ancilla controls those exchanges. A herald detector acts on the other copy. Pool repeated shots over fresh disorder realizations.

## Geometry and cutoff flow

The strongest active exchange Ω binds its two adjacent active spins into a singlet. Remove the spins and join their two outside neighbors by the exchange J′=J_left J_right/Ω. Original sites alternate sublattice signs +1 and −1. Effective neighbors remain on opposite sublattices, since a decimation removes two neighboring active spins.

At the beginning of the scan choose a block of `central` consecutive **active spins**, either one or two, sampled uniformly by active-spin index. Fix one spatial cut immediately to the left of the block and another immediately to its right. Every subsequent singlet whose sites lie on opposite sides of a cut is detectable. Label a pair `left` or `right` if it crosses just that cut, and `both` if it spans both. A pair inside the block crosses neither. The cuts stay in place after the original active spins leave. `left_sign` gives the original sublattice sign of the exterior active spin immediately to the left of the left cut at the beginning of the scan. If null, it is equally likely to be ±1, independently of the exchange magnitudes.

Use Γ=ln(Ω₀/Ω), μ=ln(Γ/Γ_start) and x=ln(Ω/J)/Γ. Numerical targets use the following strong-disorder fixed-point model. At μ=0 each internally tracked link, including the two boundary links, has an independent exponential x of mean one. With m active spins remaining between the cuts, track the m+1 consecutive links from the left exterior spin to the right exterior spin. When m=0 there is a single common link.

- All tracked x drift according to dx/dμ=−(1+x) between decimations.
- The untracked exterior link adjacent to either end decimates at rate one per side per unit μ. Such an event removes two exterior spins on that side. Add a fresh independent exponential of mean one to the adjoining tracked boundary link. When m=0 either side updates the same tracked link.
- When a tracked link reaches x=0, its two spins form a singlet. Remove both spins and their neighboring links. The single replacement link has x equal to the **sum** of the old left and right neighboring x at that instant. Each missing exterior neighbor in this sum is a fresh independent exponential of mean one. The untouched tracked links keep their current x. Continue the same evolution across all band boundaries.

If an interior singlet consumes the two central spins, its replacement spans both cuts. If separate singlets first consume the central spins at the boundaries, the resulting common link still follows the same exchange rule. No reset is made merely because the two cuts now share a link.

## Carrier passes

The carrier and the two chain copies start independently. The carrier begins in the normalized spin state `prepare`, given as two [real,imaginary] amplitudes in the ↑,↓ order. The signal-chain pair states are singlets (|↑↓⟩−|↓↑⟩)/√2, with the left member written first. The Ramsey ancilla starts in (|0⟩+|1⟩)/√2 and selects corresponding arm settings throughout both passes.

First, visit every crossing pair in increasing order of its formation coordinate μ. On the outbound pass the carrier couples to each pair's **left** member. After all outbound interactions apply the arm's specified `turnaround` to the carrier. Next visit the **same pairs in exactly reverse order**, coupling the carrier to each pair's right member. The carrier retains its state throughout. No member of a pair is measured between passes.

Each band contains `width` in μ and entries named `left`, `right`, and `both` for the three crossing geometries. Each entry's `arms` contains one [a,b,c,d] for each Ramsey arm. For a singlet with left sublattice s, the outbound unitary is exp[−i(a+s b) SWAP] on carrier and left member. The return unitary is exp[−i(c−s d) SWAP] on carrier and right member. SWAP interchanges the two spin states; the stated operator includes its complete phase. The two top-level `turnaround` entries [axis,θ] apply exp(−iθσ_axis/2) to the carrier on the corresponding arm, with axis x, y or z and ⟨↑|σ_y|↓⟩=−i.

If `analyze` is a normalized pure spin state specified in the same format as `prepare`, postselect that carrier state at the end. Normalize a nonzero input list before using it. If null, discard the carrier without measuring it. The entire signal chain is finally unobserved. Measure the Ramsey ancilla along cos(λ)X+sin(λ)Y with λ=`phase`.

In the herald copy of the same disorder realization, each geometry's `herald=[efficiency,h,g]` rotates the left member of that singlet about z by angle h+s g. Measure singlet versus triplet. A triplet independently registers a click with the stated efficiency, and a singlet never clicks. If `count` is an integer, retain exactly that many clicks across all bands. If null, retain every herald outcome. Report the pooled conditional Ramsey expectation after all required herald and carrier selections, times V. The joint retained probability exceeds 0.025 for supplied probes.

Inputs contain zero to three bands of width 0.2 to 1.8. Coupling parameters and rotation angles lie in [−π,π], efficiencies in [0,1] and integer click counts in [0,2].

## Worked exchange

Let one central spin be active, with left_sign null, phase 0, and a single width-one band. At every geometry, let the herald register every crossing: efficiency=1,h=π,g=0. Select exactly one click. Prepare an up carrier, use identity turnarounds and discard the carrier at the end. Arm 0 has no exchanges. Arm 1 performs a full −i SWAP on each pass. The conditioned fringe is 0.5 V and the retained fraction is 0.757402756814. Selecting the final up carrier as well gives a conditioned fringe of (2/3) V.

```json
{"central":1,"left_sign":null,"phase":0,"count":1,
 "prepare":[[1,0],[0,0]],"analyze":null,"turnaround":[["z",0],["z",0]],
 "bands":[{"width":1,
   "left":{"arms":[[0,0,0,0],[1.5707963267948966,0,1.5707963267948966,0]],"herald":[1,3.141592653589793,0]},
   "right":{"arms":[[0,0,0,0],[1.5707963267948966,0,1.5707963267948966,0]],"herald":[1,3.141592653589793,0]},
   "both":{"arms":[[0,0,0,0],[1.5707963267948966,0,1.5707963267948966,0]],"herald":[1,3.141592653589793,0]}}]}
```

Implement Model.fit(records), returning self and setting `visibility`, and Model.predict(inputs), returning a finite real NumPy array of shape (len(inputs),) in input order. Calibration records in data/calibration.json use empty scans with identical turnarounds and V in [0.65,0.95]. Maximum absolute calibration error must be below 0.000002. Maximum absolute error in each new family must be below 0.003.

The supplied `words.ordered_average(q,event,v)` integrates the specified two-cut continuum flow exactly to numerical precision. Your callable `event(band,geometry,s)` supplies a square matrix M and the herald click probability p for that singlet. Given formation order 1,…,N, the helper averages M₁M₂…Mₙ v, weighted by the probability of the selected herald outcome. Its result has not been divided by the retained probability. The optional `partner` callback in `apparatus.interferometer` receives a pair of outbound angles and a pair of return angles in bra, ket order. It transforms the nested carrier map, whose matrix is indexed by output ket/bra and input ket/bra in row-major order. The calibrated starter gives the two partners independent maximally mixed spin states.
