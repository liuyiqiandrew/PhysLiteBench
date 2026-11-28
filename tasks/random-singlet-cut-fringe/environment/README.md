# Spectroscopy across two cuts in a random-singlet chain

Two copies of an infinite antiferromagnetic spin-½ XX chain have identical fixed random exchange couplings. Prepare both in the strong-disorder singlet ground state. An energy-resolving Ramsey spectrometer acts on one copy and a herald detector acts on the other. Pool repeated shots over fresh disorder realizations.

## Geometry and cutoff flow

The strongest active exchange Ω binds its two adjacent active spins into a singlet. Remove the spins and join their two outside neighbors by the exchange J′=J_left J_right/Ω. Original sites alternate sublattice signs +1 and −1. Effective neighbors remain on opposite sublattices, since a decimation removes two neighboring active spins.

At the beginning of the scan choose a block of `central` consecutive **active spins**, either one or two, sampled uniformly by active-spin index. Fix one spatial cut immediately to the left of the block and another immediately to its right. Every subsequent singlet whose sites lie on opposite sides of a cut is detectable. Label a pair `left` or `right` if it crosses just that cut, and `both` if it spans both. A pair inside the block crosses neither. The cuts stay in place after the original active spins leave. `left_sign` gives the original sublattice sign of the exterior active spin immediately to the left of the left cut at the beginning of the scan. If null, it is equally likely to be ±1, independently of the exchange magnitudes.

Use Γ=ln(Ω₀/Ω), μ=ln(Γ/Γ_start) and x=ln(Ω/J)/Γ. Numerical targets use the following strong-disorder fixed-point model. At μ=0 each internally tracked link, including the two boundary links, has an independent exponential x of mean one. With m active spins remaining between the cuts, track the m+1 consecutive links from the left exterior spin to the right exterior spin. When m=0 there is a single common link.

- All tracked x drift according to dx/dμ=−(1+x) between decimations.
- The untracked exterior link adjacent to either end decimates at rate one per side per unit μ. Such an event removes two exterior spins on that side. Add a fresh independent exponential of mean one to the adjoining tracked boundary link. When m=0 either side updates the same tracked link.
- When a tracked link reaches x=0, its two spins form a singlet. Remove both spins and their neighboring links. The single replacement link has x equal to the **sum** of the old left and right neighboring x at that instant. Each missing exterior neighbor in this sum is a fresh independent exponential of mean one. The untouched tracked links keep their current x. Continue the same evolution across all band boundaries.

If an interior singlet consumes the two central spins, its replacement spans both cuts. If separate singlets first consume the central spins at the boundaries, the resulting common link still follows the same exchange rule. No reset is made merely because the two cuts now share a link.

## Spectrometer

Each band contains a `width` in μ and three entries named `left`, `right` and `both`. Each entry contains its two `arms` and `herald` for singlets of that geometry forming within that band. The bands are consecutive in increasing μ starting at zero. There is no other operation.

For each crossing singlet the Ramsey ancilla coherently selects arm 0 or arm 1. An arm lists pulses [side,axis,u,v] in chronological order. A pulse applies exp[−i(u+s v)σ_axis/2] to the selected spin with original sublattice sign s. The side "L" or "R" means the left or right member of the singlet, and the axis is "x", "y" or "z". Use (|↑↓⟩−|↓↑⟩)/√2 in left-right order and ⟨↑|σ_y|↓⟩=−i. The signal copy is unobserved. Prepare the ancilla in (|0⟩+|1⟩)/√2 and read cos(λ)X+sin(λ)Y, with λ=`phase`.

In the separate herald copy of the **same disorder realization**, `herald=[efficiency,h,g]` rotates the **left member** about z by angle h+s g. Test whether this formerly singlet pair is now in the triplet subspace. Each triplet independently registers one click with the stated efficiency. Singlets never click. If `count` is an integer, retain exactly that many registered clicks across all three geometries and all bands. If null, retain every shot. The answer is the conditional mean Ramsey reading times one visibility V. In the supplied runs the retained fraction exceeds 0.025.

Each input contains zero to three bands of width 0.2 to 1.8. Each arm contains zero to three pulses. Angles lie in [−π,π], efficiencies in [0,1], and integer target counts in [0,2].

## Worked scan

Set phase=0, count=null and open a single band of width 1. For each one-sided crossing use arms [[],[["L","z",π,0]]]. Use [[],[]] for pairs spanning both cuts. The two arm states become orthogonal on the first one-sided crossing. With one central active spin the predicted fringe is 0.135335283237 V. With two central active spins it is 0.412860375226 V. Either result is independent of left_sign.

The two-spin input for the worked scan is

```json
{"central":2,"left_sign":null,"count":null,"phase":0,"bands":[
  {"width":1,
   "left":{"arms":[[],[["L","z",3.141592653589793,0]]],"herald":[0,0,0]},
   "right":{"arms":[[],[["L","z",3.141592653589793,0]]],"herald":[0,0,0]},
   "both":{"arms":[[],[]],"herald":[0,0,0]}}
]}
```

Implement Model.fit(records), which returns self and sets `visibility`, and Model.predict(inputs), which returns a finite real NumPy array in input order with shape (len(inputs),). The calibrations in data/calibration.json use empty scans with V in [0.65,0.95]. Maximum absolute calibration error must be below 0.000002. Maximum absolute error per new family must be below 0.003.

The supplied `apparatus.pair_response(geometry,left_sign)` returns ⟨ψ₀|ψ₁⟩ for one pair and its click probability. The calibrated starter treats left-only, right-only and joint crossings as three independent stationary Poisson streams at rate 1/6 each, with independent fair sublattice signs.
