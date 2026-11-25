# Odd-viscosity pressure: rejected simple candidate

The proposed channel is physically definite, but the simple benchmark is not worth packaging. No scientific program, task, calibration data or model evaluation was created.

Consider a planar periodic channel with walls at y=±h, no normal flow, fixed total mass and known linear barotropic equation of state p=c_s²(rho−rho0). Fix the mean density rho0. Apply constant force per current volume (Fx,Fy), with constant ordinary viscosity eta>0 and known odd-viscosity ratio beta=eta_o/eta. A momentum-free thermostat can remove viscous heating. Small forcing keeps density perturbations small. For a stationary parallel velocity v=(u(y),0), continuity holds even with the transverse density profile, and convective acceleration vanishes.

Use epsilon_xy=+1 and the symmetric odd stress convention in Ganeshan and Abanov, equation2. The velocity gradient gives tau_xy=eta*u', tau_xx^odd=eta_o*u' and tau_yy^odd=−eta_o*u'. Direct momentum balance therefore gives

    eta*u'' = −Fx,
    p' = Fy−eta_o*u'' = Fy+beta*Fx.

The vorticity is omega=−u'. The reduced Stokes pressure is p_tilde=p−eta_o*omega=p+eta_o*u', so p_tilde'=Fy. Fixed mass determines the additive thermodynamic pressure constant. In a centered channel, the optical density perturbation is (Fy+beta*Fx)*y/c_s². Reading the normalized reduced pressure as thermodynamic pressure instead gives Fy*y/c_s². These are pressure or density readouts, not mechanical normal wall traction, which also contains odd stress.

With longitudinal forcing alone, the proposed wrong density readout is identically zero. Adding a known transverse force produces an ordinary nonzero hydrostatic source signal, but does not resolve the benchmark weakness: beta remains absent from the complete wrong prediction if calibration measures only the velocity profile or ordinary Couette shear force. The latter does identify eta globally through eta*(U_top−U_bottom)/(2h), but supplies no nontrivial use of beta in that wrong model. The exercise would expose an obvious omitted coefficient rather than the desired substantial completed approximation.

Rejected branches are preserved here:

- No-slip closed-contour lift: not a valid odd-viscosity signal. The pressure change alone cannot be integrated as if it were the whole force.
- Parallel channel with Fy=0: physically valid optical distinction, but a zero-source toy.
- Parallel channel with Fy nonzero: a nonzero hydrostatic baseline, but beta still drops out of the proposed source and calibration.
- Spatially varying porous-wall traction calibration: could make local odd stress observable, but was not developed. Root requested no further apparatus engineering to rescue this simple candidate.

Primary source actually read: [Ganeshan and Abanov, Odd viscosity in two-dimensional incompressible fluids](https://arxiv.org/html/1703.04522v1), equations2–9. It establishes the reduced-pressure relation and vanishing odd contribution to net force on a closed velocity-controlled contour. Its incompressible analysis does not itself assert an optical equation of state; the density calculation above follows directly from the explicitly added barotropic channel model.

The archive scan found no odd-viscosity task by that name or terminology in AUTHOR/ASSESSMENT/input-outline files. Actual nearby author documents read were Hall-bar r3 (graded transverse kinetic stress and voltage) and acoustic-cavity-pressure r1 (fixed-mass second-order pressure). They share stress/readout or mass-normalization themes, but neither is this constant odd-pressure shift. Lack of an exact duplicate does not overcome the zero/unused-coefficient issue.

The adiabatic-capture final package review remains the priority. This feasibility record authorizes neither a task package nor model trials.
