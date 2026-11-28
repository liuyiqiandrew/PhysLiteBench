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

## Rapidly cycling body motor

Some fringe preparations also apply a three-state motor. All six solvent motions still act. Let ε>0 be the motor's dimensionless microscopic time-scale parameter. In motor state j=0,1,2, put c=cos(2πj/3), s=sin(2πj/3). In addition to the solvent motion the motor imposes the following body-fixed velocities:

    ω = (1.5√2 ν/ε) [c (1,0,0) + s (0,1,0)]
    v = (1.5√2 ν/ε) hand [c (0.3,1.1,−0.6) + s (−0.8,0.2,1.2)].

These contribute dR/dt=R[ω]× and dx/dt=R v at the material origin. Independently of the solvent, the motor jumps from j to j+1 mod 3 at rate ν(1+B)/ε², and from j to j−1 mod 3 at rate ν(1−B)/ε². The real setting B=`motor.bias` is in [−0.8,0.8]. A jump changes the velocities instantaneously and leaves the pose continuous. Each shot starts with the same uniform orientation distribution as before.

* A motor with `flux=[f₁,f₂]` starts in a uniformly random state. In each time interval i, a counter adds one on every forward motor jump and subtracts one on every backward motor jump. Both f values lie in [−3,3]. At fixed ε keep shots only if the signed count in each interval equals the nearest integer to ν tᵢ fᵢ/ε² (round ties upwards). The two optical gates still select shots. Report the limit as ε→0 of the conditional fringe.
* In the synchronized protocol `pair=[p,r]` is a top-level field giving two distinct state indices. The motor object supplies only its bias. Start one ensemble in state p and another in state r. Count every motor jump history in each ensemble. Write Fⱼ(ε) for the usual fringe on shots passing both optical gates when the motor starts in j. Report the limit of [Fₚ(ε)−Fᵣ(ε)]/ε as ε→0. The two conditional averages are formed separately before taking their difference.

For a synchronized example use times=[0.27,0.56], wave=[0.9,1.4], tag=[0.7,−0.4,0.5], gates=[[0.34,−0.81,0.3],[0.38,−0.7,0.53]], hand=1, phase=π/2, motor.bias=0.75, and pair=[0,1]. At the calibrated mobility the reading is approximately −0.10982. The same 0.003 absolute error limit applies to both motor protocols.

## Return of the body frame

A further solvent-only experiment tracks the full orientation throughout the shot. Keep it if the angle of the final frame relative to the initial frame is less than a positive aperture η. Take η→0 after forming the conditional averages. The center of the particle may translate freely. Each interval in this protocol lasts between 1.0 and 1.4 time units. The two optical gates, the tag and the solvent increments work as before.

The frame tracker distinguishes two ways to complete a rotation. Start a unit quaternion at (1,0,0,0) in scalar-first convention. For every body rotation through δ about a body unit vector a, right-multiply it by (cos(δ/2), a sin(δ/2)). Body translations leave it unchanged. Equivalently, continuously lift the relative frame R(0)ᵀR(t) from the identity. At closure the resulting quaternion ends near +1 or −1. Let σ be the sign of its scalar part. This sign can also be read by comparing a spin-½ rotation with a fixed reference in an interferometer. A net 2π rotation has σ=−1, whereas a net 4π rotation has σ=+1.

The top-level field `loop` selects the readout. With `"plus"` or `"minus"`, keep only returns with σ=+1 or σ=−1 respectively and report the usual fringe conditional on that selection and both optical gates. With `"contrast"`, keep both signs and report the mean of σ times the usual cosine among returns accepted by both optical gates. The tag and the tracker follow the same sample path. These preparations have no motor.

With the calibrated solvent, times=[1.1,1.2], zero wave numbers, both gate vectors [0,0,0] and phase=0, the `"contrast"` reading is 0.85249. For hand=−1, times=[1.07,1.18], wave=[1.24,−1.35], tag=[0.61,0.44,−0.52], gates=[[0.68,0.27,−0.39],[0.53,−0.64,0.33]] and phase=0.41, the plus and minus fringes are 0.09298 and 0.01280.

## Two returns of a frame of lines

In a `director` preparation the tracking camera sees three labeled, mutually perpendicular lines attached to the body. It distinguishes the three axes, but each line is unmarked at its ends. A turn through π about a body axis therefore restores the image of all three lines. The optical tag and orientation gates remain attached to the ordinary marked body. This experiment has no motor. Each interval lasts 1.0 to 1.4 and uses |k|≤1.85 and |tag|≤1.15.

Keep a path when, at the end of **each** interval, all three lines lie within a small angle η of their respective directions at the beginning of that interval, allowing either sense. Take η→0 after conditioning. Integrate the relative body quaternion as for the full-frame tracker, restarting at the identity for interval two in its actual starting body frame. Write A and B for the resulting rotations in temporal order. At the two returns they approach elements of {±1, ±i, ±j, ±k}, with i, j and k the **positive** π rotations about body axes 0, 1 and 2, with ij=k and ji=−k. The full marked body orientation after the turns is represented by its initial quaternion multiplied on the right by AB.

A `director` object has a `read` setting. `"commute"` retains words satisfying AB=BA. `"crossed"` retains those with AB=−BA. Each returns the usual cosine among the selected paths accepted by both gates. A gate at the intermediate exposure sees the actual orientation reached after A.

The `"exchange"` setting includes `axes=[a,b]`, two distinct ordered body-axis indices. Keep the paths on which each of these axes supplied one half-turn, in either temporal order and either sense. In an exchange preparation the optional field `first_sign` is +1 or −1. When present, the signed tracker retains only that sense of the first half-turn, as read in its starting body frame. Let C be the positive half-turn about a multiplied on the right by the positive half-turn about b. A spin interferometer measures the marked rotation relative to this C. Let Q denote either spin lift of the initial lab frame, Aₗ=QAQ⁻¹, Bₗ=QABA⁻¹Q⁻¹ and Cₗ=QCQ⁻¹. An optional `pulse=[tilt,theta]` rotates a spectator spin through theta about the **laboratory** unit vector (sin(tilt),0,cos(tilt)); write E for its unit quaternion. Use the identity E if the pulse is absent. The path arm applies Aₗ, E, then Bₗ. The reference arm applies E, then Cₗ. Both spin operations leave the particle motion unchanged. For an unpolarized spin the marker is the scalar part of (CₗE)⁻¹ Bₗ E Aₗ. Return its product with the usual cosine, averaged over the selected paths and both gates. With no pulse the marker is +1 for AB=C and −1 for AB=−C. Angles are in radians.

The provided apparatus solves the earlier solvent, motor and single-return readouts. Its approximation for director preparations uses the open solvent fringe for commuting or crossed words and zero for exchange.

For example, use hand=1, times=[1.12,1.25], wave=[1.45,−1.33], tag=[0.72,0.30,−0.53], gates=[[0.71,−0.54,0.30],[0.15,0.77,0.53]] and phase=0.54. The commuting and crossed readings are about 0.10651 and −0.01527. For exchange with axes=[0,2] and first_sign=−1, the reading is 0.00765 with no pulse and −0.00335 with pulse=[0.4,2.7].
