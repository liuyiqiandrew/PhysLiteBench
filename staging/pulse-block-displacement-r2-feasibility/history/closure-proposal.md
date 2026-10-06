# Bounded pulse-family closure proposal

One explicit material model closes the internal-readout problem. It is a different apparatus from r1, not a force law silently inferred from its matched magnetodielectric. No numerical gap, viable grading domain, or Luna difficulty is established. This author-only proposal is not permission to package or evaluate a candidate.

Use a planar, electrically polarizable sheet carried by a rigid optical frame of mass M/2. A second, optically inert mass M/2 is connected to the frame by an internal spring k. Both move only along the incident x direction. All sheet dipole centers belong to the frame; their relative charge motion is transverse, along y. Include their masses in the frame mass. There are no magnetic dipoles, absorption, coatings, external supports during a pulse, or unnamed force reservoirs. The spring is outside the illuminated region. The ideal plane-wave sheet and rigid-frame constraints are part of the apparatus, not a finite-thickness homogenization correction.

Each sheet constituent is a neutral harmonic electric oscillator with positive reduced mass and fixed charges. The continuum electric-dipole model couples those charges to Maxwell fields through their ordinary electric and magnetic Lorentz forces. The moving-dipole interaction is p dot (E + V cross B) to first order in frame velocity; including this velocity term fixes mechanical versus canonical frame momentum. The sheet has one unknown positive areal oscillator strength f and known resonance frequency 1. Units set c = epsilon0 = mu0 = resonance frequency = illuminated area = 1. Thus P below is the sheet electric dipole moment per area. This f is not r1's bulk dimensionless susceptibility strength; old data cannot be reused.

In the weak-pulse limit, fields and transverse oscillators are evaluated at the unperturbed sheet x=0. Let e(t) be the real incident electric field there, with no incoming wave on the right. Maxwell continuity and the electric-current jump give

    e_r = -P_dot/2
    e_t = e - P_dot/2
    E_sheet = e - P_dot/2
    B_average = e
    P_ddot + P = f*E_sheet

or equivalently P_ddot + (f/2)*P_dot + P = f*e. The damping here is radiation into the two outgoing waves, not absorption or an added reservoir. Use the retarded solution, with P and P_dot tending to zero in the past. The real pulse can retain r1's compact positive-frequency amplitude proportional to (1-u^2)^4, together with its conjugate negative frequencies. Its energy is E = integral e(t)^2 dt, and the incident energy centroid defines t=0. No carrier averaging is needed in these equations.

The force on the complete optical frame is the transverse current's magnetic Lorentz force, including the symmetric self-field prescription fixed by the two Maxwell limits:

    F = P_dot*B_average = P_dot*e.

There is no missing interface force. The same answer follows independently from the vacuum Maxwell stress just outside the sheet:

    T_left - T_right = e^2 + e_r^2 - e_t^2 = P_dot*e.

The oscillator energy is U = (P_dot^2 + P^2)/(2*f), which is positive, and

    U_dot = P_dot*E_sheet = e^2 - e_r^2 - e_t^2.

This establishes optical energy and momentum closure for the stationary leading-order fields. For moving matter the stated charge/Lorentz coupling supplies the order-velocity work and momentum terms. A future validator must not present the stationary field identity alone as a full finite-amplitude moving-body simulation. Mechanical displacement is order E, mechanical energy is order E^2, and the task would extract the coefficient linear in E before any long-time limit.

The longitudinal mechanics is fully specified by

    (M/2)*X1_ddot = F - k*(X1-X2)
    (M/2)*X2_ddot =     k*(X1-X2),

with both coordinates and velocities zero in the distant past. Set Omega_m^2 = 4*k/M. Measure the inert component's finite-clock displacement z(t*) = M*X2(t*)/E, not a final displacement after an infinite time: reflection leaves a nonzero shared drift. A complete forward expression is

    z(t*) = integral_{-infinity}^{t*}
            [(t*-s) - sin(Omega_m*(t*-s))/Omega_m]*F(s)/E ds.

It retains translation and internal motion. It is not a relative-mode amplitude whose rigid-limit value is deliberately zero. Positive mechanical frequencies and an ordinary nonzero readout interval would still need validation. Exact clock choices, pulse and spring ranges, and thresholds are not frozen here.

The proposed completed source solves the same causal optical oscillator, radiation reaction, and two-mass equations. Its sole physical error is assigning the induced electric-gradient force to mechanical frame acceleration:

    F_source = P*(partial_x E)_average = -P*e_dot.

It neglects the total derivative connecting this canonical force to kinetic frame momentum:

    F - F_source = d(P*e)/dt.

This is not an incorrect derivative or a neglected spring mode. Both derivatives, oscillator storage, reflection and spring filtering are evaluated correctly. Both forces vanish at f=0, agree when cycle averaged in every stationary harmonic state, and have the same complete-pulse impulse. They can differ at a finite component readout. The exact difference in the above readout is integral [1-cos(Omega_m*(t*-s))]*P(s)*e(s)/E ds. Its existence is not a lower bound on a usable normalized gap.

Use separate stationary calibration preparations: hold the frame fixed, illuminate harmonically at known off-resonant frequency omega and intensity I, and measure mean supporting force. Remove that support before the pulse preparation. For exp(-i*omega*t),

    P/e = f/[1-omega^2-i*f*omega/2]
    r = i*f*omega/[2*(1-omega^2-i*f*omega/2)]
    R = f^2*omega^2/[4*(1-omega^2)^2+f^2*omega^2]
    mean force = 2*I*R.

Both source and physical model predict this identically for every f. At each nonzero, off-resonant omega, R is strictly increasing for f>0, so a single nonzero mean-force setting identifies f globally; multiple settings and independent repeats can supply ordinary fixed instrument noise. This proves noiseless identifiability and shared calibration, not a noisy optimization or prediction-margin result.

An independent reference would evolve the Maxwell field and charged-sheet oscillator with a jump-condition or symmetric thin-sheet discretization, obtain frame force from the two exterior stresses, and integrate Newton's equations. An oracle could instead solve the retarded scalar oscillator in frequency space, form P_dot*e, and use the mechanical Green function above. The reference must not insert either compact force formula as its driving force. Check temporal and sheet-limit convergence, optical energy balance, stress versus charge force, positive oscillator energy, total pulse impulse, stationary calibration equality, weak-drive scaling, and mechanical limits before any candidate packaging. Runtime and noise margins remain unknown.

The force decomposition follows the electric-dipole Lorentz law in Bradshaw et al., [Electromagnetic Momenta and Forces in Dispersive Dielectric Media](https://arxiv.org/pdf/0910.0770), Eq. 17. The sheet radiation reaction, stress identities, calibration map and mechanical response above are derived from the stated apparatus, not borrowed from that paper's bulk-medium or narrow-band approximations.

The nearest archive is decisive but does not already contain this exact transient repair. In the actual electric-dipole-force pass `KMXhb9w`, public step 15 and the final source replace a Poynting-vector term by the exact **time-averaged** electric-gradient expression for arbitrary complex coherent polarizations. That expression corresponds to the stationary limit retained by this proposal's source. The archived calculation has no pulse envelope, time-resolved material momentum or d(P*e)/dt term. Its success is strong evidence that Luna can rederive Lorentz forces, so this proposal has substantial ease risk; it is not evidence of an already performed repair of the omitted transient term.

The actual resonator-energy pass `ntqD6wz` explicitly restored internal oscillator kinetic and spring energy. This proposal's source already retains all transverse oscillator energy and radiation reaction, so repeating that repair does not fix it. The pulse r1 pass `z8sUiqU` used the final complete-body energy-centroid relation. That relation does not determine this finite component coordinate. Quantum-residence and dipole-recoil also concern observable/momentum attribution, but respectively use stationary entrance interference and coherent electric/magnetic outgoing recoil rather than this time derivative.

The spring itself is only a known physical filter. Adding it to an already correct force law would add mathematics, not new physics. Here the substantive distinction is the transient material-force assignment; an internal detector makes it observable but does not establish hardness. The construction is analytically closed within its explicit electric-sheet model. It merits independent closure/overlap review before deciding whether one bounded feasibility calculation is worthwhile. It is neither a ready r2 nor a claim that this family must be retained.
