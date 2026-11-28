# A scattering camera for a freely diffusing rigid particle

A dilute suspension contains identical rigid particles in a quiescent Newtonian fluid. Translational and rotational inertia are negligible. Use dimensionless length and time units. The pose is (x,R), where x is a material origin in the laboratory and R maps body vectors to laboratory vectors. The initial R is Haar-uniform on SO(3), and x starts at zero. The fixed camera axis is the laboratory z unit vector e. A fluorescent tag is at body coordinate b, so its measured coordinate is z=e·(x+Rb).

The following infinitesimal rigid motions specify all six Brownian degrees of freedom. The body axes e₀,e₁,e₂ are right handed. Put r=(0.55,0.82,1.12) and d=(0.18,0.27,0.36). For rotational channel j, draw a signed angle δ with variance 2νrⱼ dt. During this rotation, R and x follow the body-fixed screw

    dR/dδ = R [eⱼ]×,       dx/dδ = R hⱼ.

Here [a]×v=a×v. The body translation per radian hⱼ is `hand` times the j-th vector in ((0.27,0.46,−0.18), (0.04,−0.23,0.33), (−0.38,0.17,0.29)). Separately translate along each instantaneous body axis Reⱼ by a signed distance of variance 2νdⱼ dt. The six increments are independent centered Gaussians, renewed at every time step. Compose their finite rigid motions in the stated order and take dt→0. No other forces, torques, flow, or drift act. The common mobility ν lies in [0.7,1.2]. The mirror species are selected by hand=±1.

There are two protocols.

* A `rotation` input has a body `axis` j (0,1,2) and a lag `t` in [0.1,1.5]. Measure 3 times the ensemble average of (e·R(0)eⱼ)(e·R(t)eⱼ). The rotational calibration values for this protocol are rounded to eight decimal places.
* A `fringe` input specifies `times` [t₁,t₂] giving two successive durations in [0.12,1.4], signed `wave` [k₁,k₂] in inverse length units with |k|≤2.4, the body vector `tag` with |b|≤1.3, `hand`, and a detector `phase` in radians. At time zero and again at time t₁, apply a weak optical gate. Gate j accepts independently with probability (1+e·R aⱼ)²/4 where `gates` [a₀,a₁] are two body vectors of lengths at most one. A gate observes the orientation instantaneously and leaves the pose and subsequent dynamics undisturbed.

On particles accepted by both gates, return the mean of cos[k₁(z₁−z₀)+k₂(z₂−z₁)−phase], where z₁ is measured at t₁ and z₂ at t₁+t₂. All three positions follow the same tag on the same trajectory. For example, when both wave numbers vanish the fringe is cos(phase), including for displaced tags.

With the calibrated solvent, hand=+1, times=[0.45,0.65], wave=[1.5,−1.6], tag=[0.65,−0.4,0.5], gates=[[0.93,0,0],[0,0.88,0]] and phase=π/2, the measured fringe is approximately −0.05087.

Implement `Model.fit(records)` returning self and storing the fitted ν in `self.mobility`; every calibration record contains an `input` and `value`. `Model.predict(inputs)` returns a finite real NumPy vector of shape (len(inputs),) in input order, including empty input. The maximum absolute error must be below 0.000002 on calibration readings and below 0.003 on each fresh reading. The starter fits rotational calibration and propagates the solvent motion.


## Interferometer driven by a stored gyroscope record

Some solvent shots include a gyroscope and a separate spin-½ interferometer. The complete list of signed rotational solvent increments is stored in its physical order, with the axis j recorded for each angle δ. Translations do not make entries. After the camera shot a controller uses that same record on both coherent arms of its spin interferometer. Playback leaves no additional which-arm record and does not affect the particle or its gates. For a kick from axis j the electronic control applies

    Vⱼ(δ) = exp[−i δ gⱼ·σ/2].

The real vector gⱼ specifies the rotation per radian in fixed electronic spin axes. The three Pauli matrices have σ_z=diag(1,−1) and (σ_y)₀₁=−i. They act on the auxiliary spin. For each time discretization use the same finite screw ordering as in the solvent experiment, then take the same continuous-time limit.

On the forward arm, play the record of interval one in temporal order, apply the programmed forward pulse, then play the record of interval two in temporal order. On the replay arm, play the record of interval two in **reverse temporal order**, apply the replay pulse, then play the record of interval one in reverse temporal order. In replay multiply every recorded angle by the stated `sense` (+1 or −1) before applying its kick. Both arms use the same component gⱼ for the same original entry. A programmed pulse with vector p denotes exp(−i p·σ/2); its sign and axis are taken as supplied on either arm.

Let F and R be the resulting spin unitaries, with the operation performed first acting on the input ket first. The path qubit is prepared in (|0⟩+|1⟩)/√2 and selected with bra (⟨0|+exp(iχ)⟨1|)/√2. Thus its retained spin amplitude for one record is T=(F+exp(iχ)R)/2. The initial electronic spin state is ρ=(I+s·σ)/2. After the path measurement a spin filter accepts with effect E=(I+d·σ)/2, or E=I when no analyzer is installed. The joint click probability on that record is tr(E TρT†).

Each shot passing the two optical gates contributes its usual cosine when both electronic outcomes are accepted and contributes zero otherwise. Return the mean of those contributions among the optically gated shots. Its denominator counts all optically gated shots. In particular, at zero wave and phase this reading is the gated joint click probability.

The top-level `replay` object has `coils` [g₀,g₁,g₂], each of length at most 1.8, the signed `sense`, and `forward_pulse` and `replay_pulse` of length at most π. `initial` is the input Bloch vector s and `analyzer` is the vector d or null. Both vector lengths are at most one. `control_phase` gives χ. All existing solvent, tag, gate and wave ranges apply. The starter smooths the stored rotations to their zero mean before applying the programmed pulses.

For a check set both gate vectors and wave numbers to zero, detector phase=0, times=[0.52,0.81], and use no analyzer. Set both pulse vectors to zero, χ=0, and gⱼ=eⱼ/√rⱼ in the electronic axes. With sense=−1, reverse playback is the inverse of the original word. The click probability is [1+3 exp(−2ν(0.52+0.81))]/4, about 0.31598 at the calibrated mobility. With collinear coils, identity pulses, no analyzer, χ=0 and sense=+1 the two words are identical for every path, and the reported fringe equals the ordinary solvent fringe.
