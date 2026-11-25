# Entropy anomaly, revision 6: contacts on the inertial time scale

This candidate retains the charged, two-velocity particle, periodic square, anisotropic drag, temperature profiles and reservoir calorimeters of the entropy-anomaly family. It changes the contact time scale and defines an operational excess measurement through a separately measured matched background. The target is three unhinted `gpt-5.6-luna` high trials. Scientific controls do not establish a 0/3 model outcome.

Revision 5 is preserved in `archives/entropy-anomaly-r5` with the complete task, scripts, validation and peer reports, frozen evaluation tasks, all three final submissions and trajectories, Docker controls, native context provenance and causal repairs. Its reward count was 2/3, with zero exceptions. The failing model retained the position/cubic-anomaly shortcut; adding only the omitted contact kinetic term at its fitted friction repaired all hidden cases and 48 independent interior preparations without changing calibration. One passer derived the full finite-mass cubic/second-moment hierarchy and matched the oracle off-grid. The other recognized contact kinetic heat and passed the frozen tolerance while omitting a small positional contact-label entropy term. Its reward remains a pass; the detailed review records that residual physical omission separately.

## Apparatus and observable

The public SDE describes exactly two particle velocity components with positive mass m, drag G=diag(gamma,gamma*drag_ratio), Lorentz matrix through Gamma=[[gamma,-B],[B,gamma_y]], constant force F, and bath temperatures T_s(x)=(1+s*delta)T(x), where T(x)=T0(1+a cos(kx)). Position and velocity are continuous at valve changes. Neither the magnetic field nor an instantaneous contact change performs particle work. Bath entropy is recorded from the temperature-divided Stratonovich heat on every contact interval.

For each positive mass the actual valve flip rate is kappa/m in each direction. The public field `switch_rate` is the fixed control kappa, with units mass/time like the friction coefficients; it is not an inverse-time rate held fixed as m tends to zero. This differs from revision 5 and is explicit in the apparatus. Gamma, B, kappa, delta, F, T0, a and k stay fixed throughout the mass limit.

The first measurement is the long-time stationary mean bath entropy for all requested settings at that mass. A second measurement uses the same particle mass, drag, magnetic field, mean temperature, bath contrast and valve control, but both forces and spatial contrast are zero. Its two temperatures are spatially uniform T0*(1+s*delta). Each preparation is independently brought to joint stationarity. Subtract the matched background mean at that same finite mass, and only then let m tend to zero. This matched measurement cancels the bath-to-bath kinetic leak while retaining the finite force/gradient calorimeter signal. It does not subtract an oracle answer, a fitted correction, or a model-dependent numerical convention.

The mass is not an input or a fitted unknown. Every prediction setting uses its own matched background. The original public force, magnetic field, drag-ratio, T0, a and integer-k ranges remain unchanged; delta is allowed in [0,.65] and kappa in [.2,2]. Temperatures remain strictly positive everywhere. Only the common friction gamma in [.7,1.6] is unknown.

## Fast physical law and hierarchy

Write c=sqrt(m)*v. The stationary forward generator, multiplied by m, consists of a fast OU/contact operator L0 plus sqrt(m) times spatial transport and the force operator. The contact generator has the fixed coefficient kappa in L0. The fast velocity and contact processes must be solved jointly. Their stationary law is a mixture of thermal Gaussian histories rather than a local Maxwell law at the averaged temperature when kappa and friction are comparable.

Let M_pq,s(x)=integral c_x^p c_y^q p_s(x,c)dc. The fast operator on homogeneous degree n=p+q moments has diagonal -(p*gamma+q*gamma_y)-kappa, contact coupling kappa to the other label, magnetic couplings p*B M_(p-1,q+1) and -q*B M_(p+1,q-1), and diffusion sources p(p-1)*gamma*T_s M_(p-2,q) and q(q-1)*gamma_y*T_s M_(p,q-2). The force/spatial source at the next order is

    p F_x M_(p-1,q)+q F_y M_(p,q-1)-partial_x M_(p+1,q).

All sources and moment matrices follow the public SDE. No contact heat impulse is introduced.

The leading state probabilities are M_00,s^0=rho/2. Summing the leading second-moment equations gives M_20,+^0+M_20,-^0=rho*T, M_02,+^0+M_02,-^0=rho*T and the summed mixed moment zero. Consequently the leading spatial current still satisfies

    Gamma J = F rho-e_x partial_x(T rho),
    partial_x J_x=0,        integral rho dx=1.

The implementation uses one thermal phase period theta=kx, multiplying derivatives by k and normalizing its marginal density over that phase period. Because k is an integer and all coefficients are periodic with that shorter period, this is equivalent to repeating the solution across the full periodic square and keeping a uniform y distribution. It is not a changed circumference or an undisclosed boundary condition.

The oracle solves the hierarchy in this order:

1. M2^0 from M0^0, then M4^0 from M2^0.
2. M1^1 from F*M0^0 and partial_x M2^0.
3. M3^1 from diffusion*M1^1, F*M2^0 and partial_x M4^0.
4. The antisymmetric order-m density correction from the contact continuity equation.
5. M2^2 from diffusion*M0^2, F*M1^1 and partial_x M3^1.

The sign-antisymmetric density correction is M_00,+^2=-partial_x M_10,+^1/(2*kappa) and M_00,-^2=-M_00,+^2. The undetermined common correction can be set to zero for this observable: its integral is zero by normalization, and the leading local entropy coefficient divided by rho is spatially constant for the proportional T_± profiles. Hence its contribution to the integrated finite entropy vanishes. This gauge would not be justified for arbitrary unrelated temperature profiles; those are outside the public apparatus.

The limiting excess signal is

    integral sum_s [gamma*(M_20,s^2/T_s-M_00,s^2)
                   +gamma_y*(M_02,s^2/T_s-M_00,s^2)] dx.

The raw finite-mass entropy has a leading leak L/m. L is independent of the local temperature scale and position because the two profiles are proportional; integrating the normalized rho gives exactly the same L as the uniform/no-force background. That background has no finite term. For B=0 its leak is

    L=sum_i gamma_i*kappa*delta²/[(gamma_i+kappa)*(1-delta²)].

For general B it follows from the uniform coupled conditional velocity-covariance solve. The oracle's private `leak` diagnostic is used only for scientific consistency checks. The independent production reference does not subtract that diagnostic.

## Coherent shortcut and calibration

The completed shortcut treats the valve as fully mixed relative to velocity relaxation, kappa/gamma_i tending to infinity. The velocity sees the base temperature T(x); contact-resolved calorimetry still weights the bath heat with mean inverse temperature 1/[T(x)*(1-delta²)]. After subtraction of the same uniform kinetic background, its prediction is the complete single-bath charged, anisotropic entropy anomaly divided by (1-delta²). It retains the exact stationary base-temperature spatial distribution, both force components, the full magnetic cubic relaxation coefficient, and the stated calorimeter weighting. Its lack of finite-kappa dependence is the intended physical approximation, not an unfinished implementation or forgotten argument.

The starter contains this complete approximation with the fit left unimplemented. The named baseline implements the common-friction fit. At delta=0 both contacts have identical temperatures and the valve affects no particle observable. The matched uniform/no-force background is then zero. Thus every old calibration value, uncertainty and input remains valid. All 192 public/private calibration records are preserved byte for byte from revision 5. At B=delta=0 the response scales exactly as 1/gamma, so weighted inverse-friction regression uniquely identifies gamma. Clipping uses the existing [.7,1.6] apparatus interval; it does not create hidden information.

Three diagnostic groups vary bath contrast, finite contact control and magnetic/drag anisotropy in gradient-dominated preparations. The same .04 group RMS prediction gate is retained. An equal-contact anchor checks recovery of the existing complete single-bath physics; a spatially uniform force anchor checks the exact finite force signal. Both completed controls must pass both anchors. The group normalization is the true RMS within the group, without singular individual relative errors.

## Independent closed limits

A second derivation provides an exact scalar anchor for isotropic drag gamma, zero B, zero force and the cosine temperature profile. Put r=kappa/gamma. Then

    excess = 2*T0*k²*(1-sqrt(1-a²))/(3*gamma*(1-delta²))
             *[1+delta²*(2*r-1)*(r+3)/((3+2*r)*(1+r)²)].

The first factor is the completed fully mixed shortcut. The bracket is the finite-contact kinetic correction. At kappa=gamma/2 it is exactly one: there is an accidental calorimetric equality even though the fast velocity law is still nonGaussian. The validator checks this equality and the author evidence reports it; separation is not claimed for every allowed setting. The diagnostic low-contact controls are chosen in the regime where the finite law and fully mixed approximation differ substantially at the identified friction.

The leading unconditional velocity fourth moment in the same isotropic limit is

    <c_x^4>=3*T(x)²*[1+delta²*gamma/(gamma+kappa)],

while <c_x²>=T(x). This independently identifies the Gaussian-mixture kinetic channel suppressed by the shortcut. At delta=0 the full matrix hierarchy recovers the existing single-bath charged anomaly. At kappa tending to infinity the excess approaches the completed shortcut with inverse-kappa corrections. For spatially uniform temperature, at any mass and contact control, the excess is exactly

    mu^T G mu/[T0*(1-delta²)],       mu=Gamma^-1 F.

That identity checks magnetic signs, force weighting and the matched subtraction separately from gradient physics.

## Independent finite-mass reference and validation

The private reference solves the full finite-mass, contact-labelled stationary Kramers equation in Fourier space and Hermite velocity functions, retaining the finite magnetic coupling, force, spatial temperature multiplication and contact generator kappa. It computes raw bath entropy directly from velocity second moments and the Stratonovich noise correction. At each of three positive masses it solves a second full stationary Kramers problem for the matched uniform/no-force apparatus and subtracts those two raw measurements. A quadratic extrapolation gives the limiting excess. The reference imports no oracle hierarchy, closed correction, cubic coefficient or leak subtraction.

The author validator checks the complete 18 hidden preparations with half-mass refinement and degree-16/spatial-49 refinement against the oracle's degree-four moment hierarchy. The default reference uses degree12, spatial41 and mass epsilon*(gamma*min(1,drag_ratio))²/(T0*k²), epsilon=.006. The three extrapolation masses are h,h/2,h/4. It separately checks eight calibration preparations, harsh small-drag/high-contrast/high-field corners, independent uniform covariance and scalar leak equations, the isotropic closed anchors and the accidental equality, and the fast-contact source limit.

At 512 public-box corners it checks positive normalized position density, contact-density correction normalization, conserved spatial current, first-moment currents, summed leading second moments, the spatially constant leak, leading conditional velocity-covariance positivity, and Gaussian-mixture fourth-moment bounds. It checks label reversal and y/field reflection. Finite-mass samples independently check the global first law (bath energy equals constant-force work), the weighted energy/entropy balance including contact energy exchange at rate kappa/m, and conditional velocity covariance positivity.

The reported excess is a background-subtracted calorimeter observable. It is not asserted to be the full nonnegative thermodynamic entropy production. Positivity checks apply to actual densities/covariances and the raw physical bath accounting; an observed positive excess on selected controls is a numerical observation, not a general second-law constraint.

Noise controls use 256 independent calibration draws with seed9161066. Every draw fits both completed controls, checks the unchanged .03 parameter and 1.5 calibration gates, evaluates all three diagnostics and both anchors, and verifies robust oracle/shortcut separation at the unchanged .04 prediction gate. The source code may use the exact zero-contact/zero-field inverse-drag scaling for calibration speed, but general predictions and physics checks retain the full hierarchy.

Run `OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -B scripts/validate_entropy_anomaly.py` from the staged candidate root. It reads and verifies the preserved calibration, writes `results/entropy-anomaly-r6-validation.json`, saves separate public/private pytest controls under `results/entropy-anomaly-r6-controls`, and hashes all task files and the named baseline/validator. Local controls run in distinct app copies with unchanged tests. The oracle must pass nine checks; the completed shortcut must pass six and fail exactly the three physical prediction groups. Fresh Docker oracle/shortcut runs, a frozen task checksum and all three high-effort Luna trajectories remain the parent task's required next evaluation. Every new run must be reported; no model difficulty claim follows from the shortcut control alone.

The ordinary agent image copies only public README, starter, public tests and calibration. The independent reference, true friction, hidden inputs, hint, author formulas and solution stay outside it. Agent and verifier limits remain 600 and60 seconds. The source/public apparatus received independent physics review before packaging; final source review and immutable hashes precede model evaluation.
