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

## Gyroscope pickup in a field gradient

For a pickup shot, plot two normalized angular channels of this same solvent trajectory as a planar record. The ordered `axes` [a,b] are distinct. Its horizontal coordinate X accumulates each signed body rotation about a divided by √rₐ. Its vertical coordinate Y accumulates the rotations about b divided by √r_b. Start at the origin and draw a straight segment for each microscopic kick. Each free coordinate has variance 2νt. Translations do not enter the recorded polygon. They still move the tagged particle.

Each pickup shot also uses a calibrated rapid actuator. In each solvent tick of duration dt, first apply the free Brownian solvent kicks above. Then apply the following programmed body screws before the next exposure. The actuator displaces the particle by the same complete finite body screw per angle as a thermal rotation. Its programmed angles are independent of ν. All programmed angular increments appear in the gyroscope record. The ordinary fringes and the rotational calibration have the actuator off.

Write e=(|stroke| dt)^(1/3). A is a positive body rotation by e√rₐ about a, and B is a positive rotation by e√r_b about b. A superscript −1 reverses a motion. Define C to be the four motions A, B, A⁻¹, B⁻¹ in **chronological order**. A positive stroke performs A, C, A⁻¹, C⁻¹ chronologically, where C⁻¹ executes the four reversed motions in reverse order. A negative stroke executes the inverse of the entire positive word. At stroke=0 skip it. Use the net pose reached by all these motions. The returned response is the limit dt→0 of this driven protocol, with physical interval durations and the supplied stroke coefficient fixed. In the planar gyro drawing the word traverses two opposite squares separated horizontally by one edge. It returns to its initial point and the two signed areas cancel. The field gradient distinguishes their locations.

After the final exposure draw the straight closing chord from the last (X,Y) back to the origin. A phase pickup in a spatially linear field measures the first horizontal moment M of the polygon's **signed** area. Counterclockwise loops have positive signed area. For any oriented edge from (x₀,y₀) to (x₁,y₁), including the closing chord, its contribution to M is

    (y₁−y₀)(x₀²+x₀x₁+x₁²)/6.

Sum these contributions and take the same continuous limit as the solvent. Equivalently, M is the closed integral ½∮X² dY. Self-intersecting polygons are included with their signed contributions. The readout scale is S=(νT)^(3/2) where T=t₁+t₂.

Sort the optically accepted shots by requiring the **vertical** recorder coordinate Y(T) to equal `end`. An optional scalar `middle` additionally requires Y(t₁) to equal that value. Null `middle` leaves Y(t₁) unobserved. Each pin means accepting a symmetric interval around the specified ordinate and taking its width to zero. The horizontal coordinate and the body orientation are free at every exposure. The pin coordinates use the original origin at time zero.

The optical fringe at dimensionless pickup strength g is the mean of

    cos[K + g M/S − phase],        K = k₁(z₁−z₀) + k₂(z₂−z₁),

among shots satisfying both optical gates and the specified vertical pins. The gradient setting changes only the reported fringe phase. The instrument measures a weak-field derivative at g=0. For `read="slope"` return its first derivative, −E[(M/S) sin(K−phase)]. For `read="curvature"` return the negative second derivative, E[(M/S)² cos(K−phase)]. Both expectations use the same pin and gate selection.

The top-level `pickup` object has `axes`, `end`, `middle`, `read` and signed actuator coefficient `stroke` in [−1,1]. Both pickup durations lie in [0.38,1.1], |end|≤1.25 and |middle|≤1.25 when supplied, with |end−middle|≤1.5. Other camera ranges are as above. The starter substitutes the free ensemble average of the thermal X² at every time. It retains the vertical Brownian bridge, closing chord and the programmed squares’ pickup moment. It propagates the unconditioned optical solvent.

For a check set stroke=0 and both gate vectors and wave numbers to zero. With only the final Y=d pin, a π/2 detector phase gives the slope d/(6√(νT)). A zero detector phase gives the curvature 5/6+5d²/(36νT). For times [0.52,0.73] and end=0.81 these read about 0.12631 and 0.91311 at the calibrated mobility. With the additional middle pin −0.53 the corresponding readings are about 0.53191 and 1.28624. Keeping that middle pin and setting stroke=+0.65 gives about 1.19746 and 2.43722. Reversing the stroke to −0.65 gives about −0.13365 and 1.02118.
