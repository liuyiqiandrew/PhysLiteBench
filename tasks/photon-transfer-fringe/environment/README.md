# Counting photons transferred between optical rails

A pulse occupies 2–6 ideal bosonic modes of a common frequency. All optics are lossless. Each rail starts in an independent thermal state with the specified mean occupation n. A degenerate parametric pump on that rail then applies S with

    S† a S = a cosh(r) − exp(i chi) a† sinh(r).

There is no displacement. The rails pass through a preparation mesh, then a nondestructive meter measures the total photon number Q in a selected bank of input rails. The meter writes the integer q on a classical register. Within each eigenspace the measurement is an ideal Lüders projection. It acquires no information about the individual occupations within the bank.

The light passes through a second mesh called `before`. Some experiments then insert a nondestructive parity meter on a selected set of rails. This meter makes a projective measurement of (−1) to the power of their combined photon number. Only shots with the specified parity are retained. It resolves no further information. When the meter is absent all shots are retained. The pulse then passes through the `after` mesh. A final detector records the total photon number q′ in the original bank of rail labels at the exit.

The reading is the average of cos(lambda (q′−q) − phase) on retained shots multiplied by a common visibility V in [0.65,0.95]. All angles are in radians. Each input has the following fields.

* `source` lists [n,r,chi] for each rail in index order, with n in [0,0.9], r in [0,0.85] and chi any phase.
* `prep`, `before` and `after` are lists of up to 15 two-rail mixers in the order encountered. An empty mesh is the identity. For [i,j,t,p], rail indices are distinct and zero based. The outgoing coherent amplitudes are cos(t) a_i − exp(−ip) sin(t) a_j and exp(ip) sin(t) a_i + cos(t) a_j, respectively. All other amplitudes are unchanged.
* `bank` is a nonempty proper subset of rail indices. `gate` is null or an object with `rails` (a nonempty subset) and `parity` (1 for even, −1 for odd). Settings with a parity gate have retention probability at least 0.05. `lambda` and `phase` are readout angles.

For example, use source=[[0,0.7,0],[0,0,0]], bank=[0], before=[[0,1,π/2,0]], empty prep and after, gate=null, lambda=π and phase=0. Every pulse contains an even number of photons. Swapping the rails therefore gives a reading equal to V.

Calibration in `data/calibration.json` consists of {input,value} records, rounded to eight decimal places. These two-rail measurements use bank=[0], empty prep and after, no gate, and one mixer in before that mixes unsqueezed thermal rail 0 with vacuum rail 1. Implement Model.fit(records), returning self with fitted attribute `visibility`, and Model.predict(inputs), returning a finite real NumPy array of shape (len(inputs),) in input order. The maximum absolute error must be below 0.00002 in calibration and below 0.001 on each new reading. The starter models the squeezed preparation, number measurements, passive meshes and coherent weak triggers.

## Coherently ordered single-photon operations

In a `switch` experiment, the entrance number meter, before mesh and exit detectors work as above. Between the before and after meshes insert two weak light–matter interactions, A and B, separated by the specified middle mesh. The parity meter is absent. An independent control qubit starts in (|0⟩+|1⟩)/√2. The operation order is A, middle mesh, B on arm 0 and B, middle mesh, A on arm 1. The middle mesh is the same on both arms.

Each interaction X couples the optical field briefly to its own vacuum trigger mode h_X. In units of ℏ the unitary is exp[−iε(h_X† L_X + L_X† h_X)]. The six numbers [i,j,t,p,beta,f] specify

    b = cos(t) a_i + exp(ip) sin(t) a_j
    L_X = cos(beta) b + exp(if) sin(beta) b†.

Here i and j are distinct rails, beta is in [0,π/2] and all phases refer to the same pump reference. Retain one photon in each trigger and project the control with bra (⟨0|+exp(i angle)⟨1|)/√2. Apart from the entrance number record these retained shots contain no record of which order occurred. The trigger temporal modes of the two arms overlap perfectly. Take ε→0 after conditioning on successful detection. All weak-switch settings have success probability divided by ε⁴ above 0.02 in this limit. The measured reading uses the same final-bank count difference and normalization over retained shots as before.

The optional `switch` object has `clicks` [A,B], `middle` in mesh format, and `angle` in radians. As an example take two rails, source=[[0.5,0,0],[0,0,0]], bank=[0], empty prep, before and middle, and after=[[0,1,π/2,0]]. Use A=[0,1,0,0,π/2,0], B=[0,1,0,0,0,0], angle=π, gate=null, lambda=π and phase=0. The reading is V/2.

## Finite interaction time

If a top-level `pulse=[ε_A,ε_B]` field appears alongside `switch`, the two quoted interaction strengths are finite, in [0.35,1.2]. Apply the same trigger unitaries as above at these strengths, using ε_X whenever interaction X occurs. Each trigger enters in vacuum. At the exit each trigger is measured with an ideal number-resolving detector. Retain exactly one photon in each trigger and the specified control outcome. All retained paths obey the same entrance and exit counting protocol as in the weak experiment. The two orders use the same physical trigger labels, and the outputs carry no timing information. Finite settings have joint acceptance above 0.0005. The supplied model implements these finite-trigger settings.

For a finite-strength example use the preceding two-rail commutator setting and set `pulse=[0.5,0.65]`. The joint acceptance is 0.0108482475291 and the calibrated reading is 0.892232430316 V. To see how the finite strength affects the weights, start with m photons in rail 0. Finite subtraction and addition have a rotation or squeezing survival amplitude. The two ordered amplitudes on this input sum to a common factor times (m+1)/cosh(0.5) − m/cos(0.65); the squared common factor is tanh²(0.5) sin²(0.65) [cos(0.65)/cosh(0.5)]^(2m)/4. Average their squared sum over the geometric input with mean 0.5. Each accepted input contributes (−1)^m to the fringe.

## A shared quantum pump

A weak switch may have a top-level `pump` object alongside `switch` and in place of `pulse`. The two interactions now draw the energy for optical creation from the *same* undriven oscillator c, with [c,c†]=1. In each quoted interaction replace L_X by

    cos(beta) b + exp(if) sin(beta) b† c.

The unitary, vacuum triggers and single-photon trigger requirements are as in the weak switch. Each absorption of one quantum from the pump creates one optical quantum along the creation path. The pump is independent of the original rails and of the control. Initially it is in the pure state given by `pump.state`. Alongside the middle optical mesh it evolves with the integrated Hamiltonian χ c†c(c†c−1)/2, where χ=`pump.kerr` is in [−π,π]. This evolution occurs exactly once, after the first trigger interaction and before the second on either arm. There is no pump measurement at that time.

If `pump.detect` is null, trace out the pump at the exit. Otherwise retain its projection onto the pure state given there, in addition to the optical trigger and control selections. Each state is a list of distinct [m,x,y] entries, meaning amplitude x+iy on occupation m; normalize the complete nonzero vector. Input and analyzer occupations are integers from 0 through 4. The reported signal, conditioning, weak-coupling limit and error tolerance are the same as for the weak switch. All these pump preparations have retained probability divided by ε⁴ above 0.02. The supplied model implements these shared-pump settings.

For a check, use the two-rail weak commutator example above, set χ=π, trace out the pump, and supply one pump quantum. The result is V/2. With two pump quanta it is −V/28. In the second preparation the intermediate evolution changes sign on the two-quantum component and leaves the one-quantum component unchanged.

## Frequency-entangled single-photon source

A top-level `spectral` object replaces `source` on a separate set of fixed-number shots. There are N=2 or 3 input rails, with exactly one photon in each rail. The preparation mesh is the identity and is omitted. There is no trigger switch. The entrance number record is the number of rails in `bank`. The `before` and `after` meshes are achromatic and act on every frequency as specified above. Detectors and the optional parity gate resolve only rail counts, with no frequency or arrival-time information.

Write xⱼ for the angular-frequency offset of the photon initially in rail j, divided by the common bandwidth unit. Its joint spectral amplitude is attached to the input rail labels. For one preparation `mean` is the vector μ of offsets and `cov` is the covariance matrix C of their intensity distribution. The real symmetric matrix `chirp` is K and `delay` is the vector t. The normalized amplitude is

    F(x) = [(2π)ᴺ det C]^(−1/4) exp[−(x−μ)ᵀC⁻¹(x−μ)/4 + i xᵀKx/2 − i tᵀx].

The emitted optical state is the integral of F(x) a₀†(x₀) ⋯ a_{N−1}†(x_{N−1}) |vac⟩ over all offsets in ℝᴺ. Emission is already heralded at the entrance. `spectral.preparations` lists one or two such preparations. With two alternatives, let |F₀⟩ and |F₁⟩ be their optical states. The joint source and marker state is proportional to |F₀⟩|d₀⟩ + c|F₁⟩|d₁⟩. `spectral.weight` gives c and `spectral.marker` gives ⟨d₀|d₁⟩, each as [real,imaginary]. The marker states are normalized and go unobserved. The squared norm before normalizing this superposition is at least 0.2. The two fields are absent for a single preparation.

The plate `spectral.plate=[r,T,B]` is placed on rail r immediately after `before` and before the parity gate. A photon of offset x propagating through that rail acquires the factor exp(−iTx+iBx²/2). The carrier phase is compensated. Other rails transmit every frequency with unit amplitude. The parity gate, when present, retains the chosen total-count parity without measuring the occupied rails or frequencies. The pulse then passes through `after`. Report the same conditional transfer fringe at `lambda` and `phase`, multiplied by V.

Each mean component has magnitude at most 0.75 and each input delay has magnitude at most 1.5. Eigenvalues of C lie in [0.5,1.55] and eigenvalues of K in [−0.9,0.9]. The source weight has magnitude at most 1.25 and the marker overlap at most one. The plate has |T|≤1.6 and |B|≤0.75. Delays and chirps use the reciprocal bandwidth and reciprocal squared bandwidth as their respective units. The maximum absolute signal error on each new measurement remains 0.001. The starter propagates distinguishable photons drawn from each emission's marginal frequency intensities.

For a check, use N=3, `bank=[0]`, `before=[[0,1,π/4,0]]`, `after=[]`, `gate=null`, `lambda=π`, `phase=0`, and `plate=[0,0,0]`. Use one preparation with μ=0, K=0, t=[1,0,0]. Let C have unit diagonal and C₀₁=C₁₀=ρ as its only nonzero off-diagonal entries. The third photon exits its own rail. With ρ=1/2 the fringe is −exp(−1/2)V. With ρ=−1/2 it is −exp(−3/2)V. Both preparations have the same single-rail frequency spectra.
