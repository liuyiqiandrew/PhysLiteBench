# Counting a surviving photon after pair upconversion

The instrument measures the photon number q in a nonempty proper bank of input rails and the number q′ in the same bank of output rails. It reports V times the mean of cos[λ(q′−q)−φ] over retained shots. The inputs `lambda` and `phase` give λ and φ in radians. The unknown visibility V lies in [0.65,0.95]. Ideal counters acquire no information about frequency, arrival time or individual occupations inside the bank.

The calibration inputs have `source=[[n,0,0],[0,0,0]]` with a thermal state of mean n in rail 0 and vacuum in rail 1. Here n is in [0,0.9]. The bank is [0], `before` is one mixer, `prep` and `after` are empty, and `gate` is null. For any mixer [i,j,t,p] the outgoing coherent amplitudes are cos(t)aᵢ−exp(−ip)sin(t)aⱼ and exp(ip)sin(t)aᵢ+cos(t)aⱼ. Mesh entries are applied in their listed order. Unlisted rails are unaffected. An empty mesh is the identity.

## Fixed-number frequency-entangled sources

A top-level `spectral` object replaces `source` on the fixed-number shots. There are N=2 or 3 input rails, with exactly one photon in each rail. The preparation mesh is the identity and is omitted. The entrance number record is the number of rails in `bank`. The `before` and `after` meshes are achromatic and act on every frequency as specified above. Detectors and the optional parity gate resolve only rail counts, with no frequency or arrival-time information.

Write xⱼ for the angular-frequency offset of the photon initially in rail j, divided by the common bandwidth unit. Its joint spectral amplitude is attached to the input rail labels. For one preparation `mean` is the vector μ of offsets and `cov` is the covariance matrix C of their intensity distribution. The real symmetric matrix `chirp` is K and `delay` is the vector t. The normalized amplitude is

    F(x) = [(2π)ᴺ det C]^(−1/4) exp[−(x−μ)ᵀC⁻¹(x−μ)/4 + i xᵀKx/2 − i tᵀx].

The emitted optical state is the integral of F(x) a₀†(x₀) ⋯ a_{N−1}†(x_{N−1}) |vac⟩ over all offsets in ℝᴺ. Emission is already heralded at the entrance. `spectral.preparations` lists one or two such preparations. With two alternatives, let |F₀⟩ and |F₁⟩ be their optical states. The joint source and marker state is proportional to |F₀⟩|d₀⟩ + c|F₁⟩|d₁⟩. `spectral.weight` gives c and `spectral.marker` gives ⟨d₀|d₁⟩, each as [real,imaginary]. The marker states are normalized and go unobserved. The squared norm before normalizing this superposition is at least 0.2. The two fields are absent for a single preparation.

The plate `spectral.plate=[r,T,B]` is placed on rail r immediately after `before` and before the parity gate. A photon of offset x propagating through that rail acquires the factor exp(−iTx+iBx²/2). The carrier phase is compensated. Other rails transmit every frequency with unit amplitude. When `gate` is null, all shots proceed. Otherwise `gate.rails` is a nonempty subset of rail indices, and `gate.parity` is +1 for even or −1 for odd. The gate retains the chosen total-count parity without measuring the occupied rails or frequencies. Each linear spectral shot has parity acceptance at least 0.05. Each mesh contains at most 15 mixers. The pulse then passes through `after`. Report the same conditional transfer fringe at `lambda` and `phase`, multiplied by V.

Each mean component has magnitude at most 0.75 and each input delay has magnitude at most 1.5. Eigenvalues of C lie in [0.5,1.55] and eigenvalues of K in [−0.9,0.9]. The source weight has magnitude at most 1.25 and the marker overlap at most one. The plate has |T|≤1.6 and |B|≤0.75. Delays and chirps use the reciprocal bandwidth and reciprocal squared bandwidth as their respective units.

For a check, use N=3, `bank=[0]`, `before=[[0,1,π/4,0]]`, `after=[]`, `gate=null`, `lambda=π`, `phase=0`, and `plate=[0,0,0]`. Use one preparation with μ=0, K=0, t=[1,0,0]. Let C have unit diagonal and C₀₁=C₁₀=ρ as its only nonzero off-diagonal entries. The third photon exits its own rail. With ρ=1/2 the fringe is −exp(−1/2)V. With ρ=−1/2 it is −exp(−3/2)V. Both preparations have the same single-rail frequency spectra.

## Pair conversion between the meshes

An optional top-level `conversion` object is available when N=3. Immediately after the plate and parity gate, a weak nonlinear interaction annihilates two fundamental photons and creates one photon in a separate sum-frequency band. All converter ports feed the same sum-frequency field c. Use [c(z),c†(z′)]=δ(z−z′) and the same continuum normalization on each fundamental rail. The sum-frequency input is vacuum. In units of ℏ, the interaction is exp[−iε(C+C†)] with

    C = ∑ₕ ∫du dw h(u,w) c†(u+w) aᵣ(u) aₛ(w).

Here u,w are the offsets from the common fundamental carrier and z=u+w is the offset from twice that carrier. Each entry of `conversion.paths` specifies one term [r,s,A_re,A_im,Δ,W,d,L] in the sum. For that term,

    h(u,w) = (A_re+i A_im) exp[−(u−w−Δ)²/(4W²) − i d(u−w) − i L(u+w)].

The first ordered rail r supplies the frequency u and the second s supplies w. The two indices are distinct. The list contains one or two distinct ordered rail pairs. W is in [0.6,1.4], |Δ|≤0.8, |d|≤1, |L|≤1.2 and |A_re+i A_im|≤1.2. The phases include propagation to the common trigger receiver. The apparatus retains no record of which converter or which two fundamental frequencies produced the trigger.

Retain one sum-frequency photon and take ε→0 after conditioning. All conversion inputs have joint probability of the parity and trigger selections divided by ε² above 0.01 in that limit. The surviving fundamental photon continues through `after` for the exit count. The trigger travels on its own line and is excluded from both bank counts.

`conversion.detector` chooses the trigger measurement.

* `null` accepts the full sum-frequency band and traces out the trigger. All frequency partitions u+w=z can interfere at a fixed z. Distinct sum-frequency outcomes are summed as probabilities.
* `{"band":[lo,hi]}` retains z∈[lo,hi] with a frequency-intensity detector. Its individual z outcome is discarded after selection.
* `{"mode":[μ,s,t,K]}` projects the trigger onto the normalized one-photon mode with spectral amplitude (2πs²)^(−1/4) exp[−(z−μ)²/(4s²)+iKz²/2−itz]. Here |μ|≤1, s∈[0.65,1.55], |t|≤1.3 and |K|≤0.65.
* `{"time":[lo,hi]}` detects the trigger in time with Fourier convention ⟨t|z⟩=exp(−izt)/√(2π). It retains t∈[lo,hi] and discards the individual arrival time. The time and frequency windows lie in [−2.5,2.5] and each has width at least 0.4. Clock time uses the reciprocal input bandwidth unit.

For a worked conversion, use three independent standard Gaussian spectra: one source preparation with μ=t=0, C=I, K=0. Set bank=[0], before=[], after=[[0,2,π/4,0]], gate=null, plate=[0,0,0], λ=π and φ=0. Use paths [[0,1,1,0,0,1,0,0],[1,2,1,0,0,1,0,1]]. With detector=null the result is −exp(−1)V. Selecting mode [0,√2,0,0] gives −2 exp(−1)/(1+exp(−2)) times V. Each path leaves a different fundamental rail occupied before the final mixer. The one-unit trigger delay changes the coherence between these two alternatives.

Implement `Model.fit(records)` returning self and storing V in `self.visibility`. The public calibration values are rounded to eight decimal places. `Model.predict(inputs)` returns a flat finite real NumPy array in input order, including empty inputs. Absolute errors must be below 0.00002 for the calibration and below 0.001 on each new shot. The supplied `spectral.spectral_signal` gives the unscaled fringe and acceptance for shots without conversion. The starter treats conversion paths and incoming photon labels as distinguishable, using the carrier offsets to estimate conversion weights.
