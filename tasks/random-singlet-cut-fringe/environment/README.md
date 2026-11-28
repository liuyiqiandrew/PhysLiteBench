# Passing a carrier spin along a random-singlet chain

Two copies of an infinite antiferromagnetic spin-½ XX chain have identical fixed random exchange couplings. Prepare both in the strong-disorder singlet ground state. A single carrier spin travels from left to right, interacting with the two members of each spectrally selected singlet as it reaches their physical sites. The Ramsey ancilla controls those exchanges. A herald detector acts on the other copy. Pool repeated shots over fresh disorder realizations.

## Geometry and cutoff flow

The strongest active exchange Ω binds its two adjacent active spins into a singlet. Remove the spins and join their two outside neighbors by the exchange J′=J_left J_right/Ω. Original sites alternate sublattice signs +1 and −1. Effective neighbors remain on opposite sublattices, since a decimation removes two neighboring active spins.

At the beginning of the scan choose a block of `central` consecutive **active spins**, either one or two, sampled uniformly by active-spin index. Fix one spatial cut immediately to the left of the block and another immediately to its right. Every subsequent singlet whose sites lie on opposite sides of a cut is detectable. Label a pair `left` or `right` if it crosses just that cut, and `both` if it spans both. A pair inside the block crosses neither. The cuts stay in place after the original active spins leave. `left_sign` gives the original sublattice sign of the exterior active spin immediately to the left of the left cut at the beginning of the scan. If null, it is equally likely to be ±1, independently of the exchange magnitudes.

Use Γ=ln(Ω₀/Ω), μ=ln(Γ/Γ_start) and x=ln(Ω/J)/Γ. Numerical targets use the following strong-disorder fixed-point model. At μ=0 each internally tracked link, including the two boundary links, has an independent exponential x of mean one. With m active spins remaining between the cuts, track the m+1 consecutive links from the left exterior spin to the right exterior spin. When m=0 there is a single common link.

- All tracked x drift according to dx/dμ=−(1+x) between decimations.
- The untracked exterior link adjacent to either end decimates at rate one per side per unit μ. Such an event removes two exterior spins on that side. Add a fresh independent exponential of mean one to the adjoining tracked boundary link. When m=0 either side updates the same tracked link.
- When a tracked link reaches x=0, its two spins form a singlet. Remove both spins and their neighboring links. The single replacement link has x equal to the **sum** of the old left and right neighboring x at that instant. Each missing exterior neighbor in this sum is a fresh independent exponential of mean one. The untouched tracked links keep their current x. Continue the same evolution across all band boundaries.

If an interior singlet consumes the two central spins, its replacement spans both cuts. If separate singlets first consume the central spins at the boundaries, the resulting common link still follows the same exchange rule. No reset is made merely because the two cuts now share a link.

## Spatial carrier circuit

The carrier begins in the normalized spin state `prepare` (two [real,imaginary] amplitudes in ↑,↓ order). It and the chains start independently. The signal-chain pair states are (|↑↓⟩−|↓↑⟩)/√2 with the left member first. Prepare the Ramsey ancilla in (|0⟩+|1⟩)/√2. On arm j rotate the carrier by `launch[j]=[axis,θ]`, using exp(−iθσ_axis/2).

First use the bands to identify all crossing pairs that formed in the scanned μ range. The carrier then arrives from the left and visits the **physical sites of all selected pair members in increasing original lattice-site order**, regardless of formation energy or crossing label. There is exactly one visit to each selected site. Other sites cause no interaction. The carrier stays coherent throughout. The spatial order follows from the original chain and the neighboring-active-spin decimation rule.

Each band contains `width` and entries `left`, `right` and `both` for singlets of the three crossing geometries. Each geometry has two `arms`, each [a,b,c,d]. If s is the left member's original sublattice, couple the carrier to that member by exp[−i(a+s b) SWAP] on the appropriate arm. At the right member use exp[−i(c−s d) SWAP]. SWAP interchanges the two spin states and the complete phase of each exchange matters. The same geometry has `field=[[axis_L,θ_L],[axis_R,θ_R]]`. Immediately **after** each exchange, apply its corresponding field rotation exp(−iθσ_axis/2) to the carrier. These lab-frame rotations are common to both Ramsey arms. Axes are "x", "y" or "z" with ⟨↑|σ_y|↓⟩=−i.

After the final site, select the carrier spin state `analyze`, in the same amplitude format as `prepare`. If null, discard the carrier without measuring it. Normalize each nonzero amplitude list before use. The chain is unobserved. Measure the Ramsey ancilla along cos(λ)X+sin(λ)Y with λ=`phase`.

In the matched herald copy, `herald=[efficiency,h,g]` rotates the left member of that singlet about z by angle h+s g. Measure singlet versus triplet. A triplet independently registers one click with the specified efficiency; a singlet never clicks. There are separate counters for left-only, right-only and joint crossings. The top-level `counts` array has entries in that order. Each integer selects exactly that many registered clicks in its channel across all bands. Each null entry leaves that channel unselected. Pool all jointly retained herald and carrier outcomes. Report the conditional Ramsey reading times visibility V.

Scans have zero to three bands of width 0.2 to 1.8. Angles lie within [−π,π], efficiencies in [0,1], and each integer click count in [0,2]. At most two channels are selected at once. The supplied probes have retained fraction above 0.025.

## Worked spatial scan

Choose two central spins, a single band of width 1, and perfect herald clicks on every one-sided pair. Select exactly one left-only and one right-only click. Give every spin zero field. Both angles on the left-only pair are zero in arm 0 and π/2 in arm 1. The right-only pair has zero left exchange and π/2 right exchange in both arms. The joint-pair exchanges are all zero. Use identity launch rotations, an up carrier and no final carrier selection, with phase zero.

The selected one-sided singlets are disjoint. The common rightmost swap cancels in the Ramsey overlap, leaving the two full swaps applied to the left singlet on arm 1. The retained probability is 0.2775250920 and the conditional fringe is 0.5 V.

```json
{"central":2,"left_sign":null,"phase":0,"counts":[1,1,null],
 "prepare":[[1,0],[0,0]],"analyze":null,"launch":[["z",0],["z",0]],
 "bands":[{"width":1,
   "left":{"arms":[[0,0,0,0],[1.5707963267948966,0,1.5707963267948966,0]],"field":[["x",0],["z",0]],"herald":[1,3.141592653589793,0]},
   "right":{"arms":[[0,0,1.5707963267948966,0],[0,0,1.5707963267948966,0]],"field":[["x",0],["z",0]],"herald":[1,3.141592653589793,0]},
   "both":{"arms":[[0,0,0,0],[0,0,0,0]],"field":[["x",0],["z",0]],"herald":[1,3.141592653589793,0]}}]}
```

Implement Model.fit(records), returning self and setting `visibility`, and Model.predict(inputs), returning a finite real NumPy array of shape (len(inputs),) in input order. The empty calibration scans use identical launch rotations and a visibility in [0.65,0.95]. Maximum absolute calibration error must be below 0.000002. Maximum absolute error in each new family must be below 0.003.

The supplied `apparatus.pair_map(spec,sign,bra,ket)` returns the exact single-singlet transform and its herald click probability. The 16×16 transform acts on the row-major flattening of an interior 4×4 carrier map E and inserts the given singlet's left-site operation before E and its right-site operation after E. E is indexed by output ket/bra and input ket/bra in row-major order. The free carrier is the identity map. The calibrated starter wraps all earlier interactions in every newly formed crossing singlet.

The supplied `walk.advance(q,start,step)` solves the entire two-cut disorder and herald process for an arbitrary linear record vector. It applies updates in **formation order**. Your callback `step(band,geometry,s,merges)` returns a square record-update matrix and the click probability for that newly formed singlet. If an interior pair merges the two cuts silently, geometry is None and merges is true. A left or right crossing can also complete the merger. The helper returns two unnormalized final record vectors: the contributions from samples with separate cuts and from samples with a common cut. The callback can be called for combinations that have zero weight. Their returned values must still have the specified shapes.
