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

## Two-band spin monitor

The particle carries a projective spin analyzer whose axes follow the body orientation. It observes an auxiliary spin with angular momentum 3/2. Between observations the spin is free in the laboratory. The analyzer makes only the projective measurements specified here. It does not push or rotate the solvent particle.

In the standard body z basis order the four spin levels as +3/2,+1/2,−1/2,−3/2. Write J for the usual dimensionless spin-3/2 angular momentum matrices, with real positive raising-matrix elements. A filter setting has a three-component body rotation vector `turn` with length at most π and a `band` equal to `central` or `upper`. Its body-frame basis vectors are the columns of exp(−i turn·J), acting on those four standard levels. The actual laboratory subspaces are obtained by rotating this complete body basis with the particle. The arbitrary sign of the spin representation of R has no effect on a projector or a density operator.

The filter records one binary label at each observation. For `central`, label 0 transmits the two levels (+1/2,−1/2), and label 1 transmits (+3/2,−3/2). For `upper`, label 0 transmits (+3/2,+1/2), and label 1 transmits (−1/2,−3/2). Within each doublet keep the indicated order for preparations and analyzers. Each observation projects onto the recorded two-dimensional subspace with its Born probability. It preserves coherence between the two levels in that subspace. Different binary records are distinguishable.

Initially the spin occupies the first filter's `start` doublet (0 or 1). Its two-level density matrix in that body basis is (I+s·σ)/2 for the specified Bloch vector `initial`=s. The Pauli matrices have σ_z=diag(1,−1) and (σ_y)₀₁=−i. The change counter begins at zero with the known start label. At the end of every microscopic rotational solvent kick, measure the spin with the filter at its new body orientation. Each time the reported label differs from the previous label, increment the counter. The independent body translations leave the analyzer stationary in orientation. Take the stated solvent limit dt→0 **after** performing these observations for the finite kicks.

Use the first setting throughout interval one. Immediately after the last observation in interval one, switch to the second setting and observe once at the same particle orientation. A label change at this switch counts once. Use the second setting for all kicks in interval two. At the end, keep records with exactly the requested total number of changes. In the last occupied doublet the final analyzer has effect E=(I+d·σ)/2 for `analyzer`=d, or E=I for null. The vectors s and d have length at most one.

These inputs carry a top-level `monitor` object with `filters` [first setting,second setting], `start`, `initial`, `changes` (0,1,2), `analyzer`, and `read`. For `read="weight"` measure the usual tagged cosine multiplied by the indicators of the requested count and a positive final analyzer result, averaged among all shots passing the two optical gates. For `read="selected"` average the cosine multiplied by the positive analyzer indicator among the optically accepted shots that already have the requested count. The analyzer never enters either denominator. For selected readings the requested count occurs in at least 1% of the optically accepted shots. All spatial, timing and solvent settings are as in the ordinary fringe.

For a worked check use both filters `{"band":"central","turn":[0,0,0]}`, null analyzer, zero wave, zero optical gate vectors, phase=0, times=[0.52,0.81], and `read="weight"`. The labels change with constant rate 3ν(r₀+r₁)/2. The probabilities for exactly zero, one and two changes are respectively 0.08229, 0.20551 and 0.25664 at the calibrated mobility, independent of the initial doublet and spin. In the same experiment a selected reading is 1 for each of these three counts.

The starter retains solvent scattering, the mean transition rates of unpolarized doublets and the instantaneous switch probabilities. It treats the spin as fully depolarized in each doublet during observation.
