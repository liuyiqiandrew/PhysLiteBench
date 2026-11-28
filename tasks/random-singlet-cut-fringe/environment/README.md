# Resolving pairs of spectral lines with a shared frequency reference

Prepare two copies of one infinite random-singlet XX chain with identical exchange couplings. One copy drives a Ramsey ancilla. In the other copy, two counters report the first energies at which they detect a singlet spanning the same spatial cut. A shared spectrometer reference shifts the two reported line coordinates oppositely and sets the Ramsey demodulation phase on that shot. Average over disorder, the two optical measurements and the reference state.

## Sharp source

Use the dimensionless formation coordinate μ=ln(Γ/Γ_start), with Γ=ln(Ω₀/Ω) the strong-disorder scale. In the specified fixed-point source, successive singlets form across the cut at renewal intervals equal in law to the sum of independent exponentials of rates (3−√5)/2 and (3+√5)/2 per unit μ. For `start="renewed"` the scan opens just after a crossing outside the observed spectrum. For `start="bulk"` it opens uniformly in μ along a record of very long duration. All pairs start in the singlet (|↑↓⟩−|↓↑⟩)/√2, in left-right order. A given pair has **one** formation coordinate for both copies and both members.

Scan the consecutive `bands` from μ=0 in increasing order. Each band gives a positive `width`, two `arms` and two `detectors`. The ancilla starts in (|0⟩+|1⟩)/√2 and selects the corresponding signal-copy arm. Each pulse [side,axis,angle] in an arm applies exp(−i angle σ_axis/2) to that member of every crossing singlet formed in the band. Pulses are listed in chronological order. Sides are "L" and "R", axes are "x","y","z", and ⟨↑|σ_y|↓⟩=−i. Never measure the signal-chain spins.

In the separate matched copy, detector L measures the left member and detector R the right member. Each [efficiency,axis,angle] rotates its member by the given angle about the stated axis and then measures σ_z. An up outcome independently registers one click with the stated efficiency. A down outcome never clicks. For every shot with at least one registered click on each channel, let T_L,T_R be the respective first click coordinates. If either counter had no click before the scan ended, that shot produces no pair report. Subsequent clicks do not alter the report.

## Spectrograph and phase reference

A single real detuning ξ is chosen for each shot, independently of the source and the spins. With probability p it is locked at k, where `lock=[p,k]`. Otherwise ξ is uniform on the interval `dither=[lo,hi]`. The same ξ adds to the first left reporter and subtracts from the first right reporter. Thus the measured center and signed half-splitting are

    C = (T_L+T_R)/2
    D = (T_L−T_R)/2 + ξ.

The electronics also measures the Ramsey ancilla along cos(λ+τξ)X+sin(λ+τξ)Y, where λ=`phase` and τ=`lag`. It is the reference value from this same shot. Gate on measured center `center`. For `kind="signed"`, the second electronics channel displays D. For `kind="beat"`, it displays |D| and loses the ordering of the reported lines. A slit `[a,b]` with a<b retains displayed values in that closed interval. A slit with a=b is a resolved reading at the specified display value.

To condition on continuous readings, pool all shots in a width-ε interval around the specified measured center. Apply the slit above, replacing a resolved slit reading by a width-ε interval centered at the reading (and restricted to nonnegative values on a beat display). Form the conditional Ramsey expectation and then take ε to zero. Both the locked and continuously dithered components enter this pooling with their stated probabilities. Multiply by one visibility V.

The center lies strictly inside the scan. Widths are in [0.6,2.5], with one to three bands. The controls use angles in [−π,π], efficiencies in [0,1], detunings and slit limits in [−0.8,0.8], and |τ|≤7. All supplied selections have positive leading weight. Resolved settings in the supplied scans avoid band edges and unlocked-dither endpoints. A resolved splitting may exactly equal a locked line.

## Worked spectrum

Use a renewed common cut and a single band of width 2. Arm 0 is idle and arm 1 makes two successive π rotations about z on the left spin. This gives a factor −1 per crossing singlet in the arm overlap. Both detectors are perfectly efficient. Detector L has no spin rotation and detector R rotates by π/2 about y, so the four joint pair outcomes all have probability 1/4. Take center=1, kind="signed", slit=[0.15,0.15], lock=[0,0], dither=[−0.3,0.3], phase=0 and lag=0.

The signal is −0.436157362603 V. Sources with one genuinely shared first line contribute 0.118020401638 to the density per measured center and signed half-splitting. Sources with distinct true first lines contribute 0.012974947916. In this measurement the reported pair is separated even when both detections originally came from the same singlet.

```json
{"start":"renewed","phase":0,"lag":0,"center":1,"kind":"signed","slit":[0.15,0.15],
 "lock":[0,0],"dither":[-0.3,0.3],"bands":[{"width":2,
 "arms":[[],[["L","z",3.141592653589793],["L","z",3.141592653589793]]],
 "detectors":[[1,"z",0],[1,"y",1.5707963267948966]]}]}
```

Implement Model.fit(records), returning self and setting `visibility`, and Model.predict(inputs), returning a finite real NumPy array in input order with shape (len(inputs),). The calibration records use `{"calibration":true,"phase":λ}` and have reading V cos λ with V in [0.65,0.95]. Maximum absolute calibration error must be below 0.000002. Each probe family has maximum absolute tolerance 0.003.

The supplied `source.Sharp(q)` calculates the true source before dither. Its `raw(tL,tR,diagonal=False)` returns [raw acceptance, raw complex arm coherence] per dtL dtR for distinct true times. For a common true line use `raw(t,t,diagonal=True)`, which returns both values per dt. At equality `raw(t,t,diagonal=False,order="LR")` or order="RL" instead gives the smooth limiting component for the indicated order of true clicks. All source values already include runs lost during preparation or counting. They have not been conditioned on the measured slit or multiplied by visibility. The starter replaces the reference by its ensemble mean and uses the slit midpoint.
