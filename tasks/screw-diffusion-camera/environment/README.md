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

## Area swept by two gyroscope channels

A subset of camera shots also records two of the **same** rotational solvent channels defined above. Their ordered body-axis labels are `axes` [a,b], with a≠b. Plot the signed angle from channel a horizontally after dividing by √rₐ, and that from b vertically after dividing by √r_b. The plotted coordinate (X,Y) starts at (0,0), and each incoming increment adds to its coordinate. Thus each free component has variance 2νt at time t. The third rotational channel and the independent translations continue to move the body as usual.

Join successive recorded positions by straight segments throughout both intervals. At the end close this single polygon by a straight chord to its starting point. Let A be its signed area, positive for counterclockwise traversal in the (X,Y) plot. For vertices (Xⱼ,Yⱼ) it is half the sum of XⱼYⱼ₊₁−YⱼXⱼ₊₁. The continuous limit uses the same finite screw increments that drive the solvent. This is an area in the two-dimensional *integrated angular record*. The complete body orientation keeps evolving during the experiment.

Sort the shots by requiring (X,Y) at the final exposure to equal the two-vector `end`. An optional `middle` vector additionally pins (X,Y) at the intermediate exposure, in the same coordinates with the same original origin. Each pin means conditioning on a circular acceptance disk and taking its radius to zero. Null `middle` leaves the intermediate gyro coordinate unobserved. Both weak optical gates still operate. The denominator is the probability of the same pin selection and both optical gates, taken in the same small-disk limit as the numerator.

For these records return the mean, among the selected shots, of

    cos[k₁(z₁−z₀) + k₂(z₂−z₁) + field × A − phase].

The top-level `area` object contains `axes`, the signed real `field`, `end` and `middle`. Its two interval durations lie in [0.55,1.1] and 1.25≤|field|≤2.25. The lengths of both supplied position vectors are at most 1.25 and, when two are given, |end−middle|≤1.5. All passive camera parameters use the earlier ranges. The starter approximates each recorded path by the polygon joining only the supplied positions. It propagates the unconditioned solvent for the optical part.

For a check with no spatial phase, set both wave numbers and both gate vectors to zero. If C(t,d) denotes the conditional characteristic function of the signed area of one gyro arc with endpoint displacement d, then with z=ν|field|t,

    C(t,d) = (z/sinh z) exp[−|d|²(z coth z−1)/(4νt)].

For times [0.59,0.73], field=1.4, end=[0.43,0.70], middle=null and phase=0.87 the reading is about 0.37207 at the calibrated mobility. Pinning middle=[0.69,−0.41] in the same experiment gives about 0.60102. The total polygon then includes the signed triangle joining the origin, middle and end.
