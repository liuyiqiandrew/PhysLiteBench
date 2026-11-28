# First spectral lines of a random-singlet chain

Prepare two copies of the same infinite antiferromagnetic spin-½ XX chain in the strong-disorder singlet ground state. Their random exchanges are identical. A Ramsey probe acts on one copy. Two energy-resolving spin detectors act on the other copy. Record the formation coordinates of their first clicks, condition the Ramsey signal on both readings, and repeat over fresh disorder realizations.

## Geometry and cutoff flow

The strongest active exchange Ω binds its two adjacent active spins into a singlet. Remove the spins and join their two outside neighbors by the exchange J′=J_left J_right/Ω. Original sites alternate sublattice signs +1 and −1. Effective neighbors remain on opposite sublattices, since a decimation removes two neighboring active spins.

For `central` equal to one or two, at the beginning of the scan choose that many consecutive active spins, sampled uniformly by active-spin index. Fix one spatial cut immediately to the left of the block and another immediately to its right. Every subsequent singlet whose sites lie on opposite sides of a cut is detectable. Label a pair `left` or `right` if it crosses just that cut, and `both` if it spans both. A pair inside the block crosses neither. The cuts stay in place after the original active spins leave. `left_sign` gives the original sublattice sign of the exterior active spin immediately to the left of the left cut at the beginning of the scan. If null, it is equally likely to be ±1, independently of the exchange magnitudes.

Use Γ=ln(Ω₀/Ω), μ=ln(Γ/Γ_start) and x=ln(Ω/J)/Γ. Numerical targets use the following strong-disorder fixed-point model. For one or two central spins, every initially tracked link, including the boundary links, has an independent exponential x of mean one. A separate preparation `central=0` opens immediately after a known singlet has crossed both cuts, after the last central spin has vanished. The fresh common link then has x equal to the sum of two independent mean-one exponentials. The known preceding singlet is outside the scan. With m active spins remaining between the cuts, track the m+1 consecutive links from the left exterior spin to the right exterior spin. When m=0 there is a single common link.

- All tracked x drift according to dx/dμ=−(1+x) between decimations.
- The untracked exterior link adjacent to either end decimates at rate one per side per unit μ. Such an event removes two exterior spins on that side. Add a fresh independent exponential of mean one to the adjoining tracked boundary link. When m=0 either side updates the same tracked link.
- When a tracked link reaches x=0, its two spins form a singlet. Remove both spins and their neighboring links. The single replacement link has x equal to the **sum** of the old left and right neighboring x at that instant. Each missing exterior neighbor in this sum is a fresh independent exponential of mean one. The untouched tracked links keep their current x. Continue the same evolution across all band boundaries.

If an interior singlet consumes the two central spins, its replacement spans both cuts. If separate singlets first consume the central spins at the boundaries, the resulting common link still follows the same exchange rule. No reset is made merely because the two cuts now share a link.

## Signal and spectral detectors

The consecutive bands start at μ=0. Each gives its `width` in μ and three entries `left`, `right`, `both` for singlets formed across just the left cut, just the right cut, or both cuts. There is no optical interaction with a pair that crosses neither cut. Each entry has two `arms` and two `detectors`. The sublattice sign s always refers to the **left member of that singlet in physical site order**.

Prepare a Ramsey ancilla in (|0⟩+|1⟩)/√2. On the signal copy, each arm lists pulses [side,axis,u,v] in chronological order. A pulse applies exp[−i(u+s_local v)σ_axis/2] to the selected member "L" or "R" of every singlet of this geometry forming in the band. The local sign is s on the left member and −s on the right. Each pair starts in (|↑↓⟩−|↓↑⟩)/√2. No measurement is made on the signal copy. Read cos(λ)X+sin(λ)Y on the ancilla, with λ=`phase` and ⟨↑|σ_y|↓⟩=−i.

On the matched herald copy, detector L is active on left-only and common pairs and measures the **left** member. Detector R is active on right-only and common pairs and measures the **right** member. A detector entry [η,axis,h,g] first rotates its member by exp[−i(h+s_local g)σ_axis/2], then measures σ_z. If the result is up, the detector registers a click with efficiency η. A down result never clicks. Inefficiencies are independent between the devices and between different pairs. A common pair supplies the two members of **one singlet** to the two detectors. A one-sided pair cannot trigger the opposite detector. Optical operations on one pair leave every other pair unchanged. The two chain copies have independent spin states for a given realization of their common couplings.

The field `lines=[left,right]` gives each detector's **first registered click coordinate** in the scan, or null if it registered no click by the end. Later clicks on a detector that has already fired have no effect on its reported line. Null requires the detector to be silent for the entire scan.

A finite reading denotes a resolved spectral line. Pool runs whose specified finite first readings lie in intervals of width ε around their stated coordinates, keep any required silent channels, form the conditional Ramsey reading, and take ε to zero. At the terminal time use the inward interval ending there. Two equal finite readings use the **same** shrinking interval. Two different finite readings use intervals centered separately. Each supplied conditional event has positive probability or positive leading density in this limit. The measured value is the pooled reading times one visibility V.

Scans contain zero to three bands of widths 0.4 to 2.5. Finite line coordinates lie in a band interior or at the final endpoint. Angles lie in [−π,π], efficiencies in [0,1] and axes are "x", "y" or "z". All coordinates and cutoffs are dimensionless.

## Worked scan

Open immediately after a common crossing (`central=0`) with one band of width 1. In that band, arm 0 does nothing, and arm 1 makes two successive π rotations about z on the left spin of every pair. Each common singlet gives an arm factor of −1. Both detectors are fully efficient. Detector L measures along z and detector R first rotates its member by π about y. The singlet then makes the two detectors register together or stay silent together, each with probability 1/2. Require `lines=[1,1]` and phase=0. The joint line density in μ is 0.147073693528. The conditional fringe is −0.857006510892 V.

```json
{"central":0,"left_sign":null,"phase":0,"lines":[1,1],"bands":[{"width":1,
 "left":{"arms":[[],[]],"detectors":[[1,"y",0,0],[1,"y",3.141592653589793,0]]},
 "right":{"arms":[[],[]],"detectors":[[1,"y",0,0],[1,"y",3.141592653589793,0]]},
 "both":{"arms":[[],[["L","z",3.141592653589793,0],["L","z",3.141592653589793,0]]],
         "detectors":[[1,"y",0,0],[1,"y",3.141592653589793,0]]}}]}
```

Implement Model.fit(records), returning self and setting `visibility`, and Model.predict(inputs), returning a finite real NumPy array of shape (len(inputs),) in input order. The empty scans in data/calibration.json have V in [0.65,0.95]. Maximum absolute calibration error must be below 0.000002. Maximum absolute error in each new family must be below 0.003.

The starter models the three optical geometries by independent stationary Poisson streams, each at rate 1/6, with independent fair sublattices. It includes the common-pair spin correlation. The supplied `apparatus.optics(spec,geometry,s)` returns the single-pair overlap ⟨ψ₀|ψ₁⟩ and four click probabilities ordered none, left only, right only, both. You may use the helper or implement the complete readout yourself.
