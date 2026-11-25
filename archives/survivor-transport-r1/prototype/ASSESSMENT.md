# Absorption-conditioned transport prototype

This is a scientific prototype, not a task package. No agent evaluation has run. The proposal uses a circular tube with irreversible wall capture and measures the accumulated axial displacement of particles that survive the whole observation interval. It does not introduce disorder, a switching wall, a new surface, or a missing hydrodynamic coupling.

## Apparatus and candidate closure

Inject a dilute, noninteracting tracer pulse at axial position zero, uniformly across the cross-section of an axially unbounded tube. The prescribed steady axial velocity is `u(r)=plug+peak*(1-r²/R²)`. Molecular diffusion is isotropic with unknown D. The cylindrical wall irreversibly captures solute with known Robin rate kappa: `D*d_r c+kappa*c=0`. There is no reinjection, saturation, wall storage or bulk reaction. The initial transverse distribution has positive overlap with the principal mode. Proposed units are radius in micrometers, D in micrometers squared per second, kappa in micrometers per second, and velocity in micrometers per second. Uniform plug transport can be realized by translation of the tube and its contents; the superposed Poiseuille profile is defined in the laboratory frame.

Two physical readouts would be available: the asymptotic loss exponent `-lim log(P(alive at t))/t`, and `lim E[Z_t-Z_0 | no capture by t]/t`. The latter is the long-time slope of mean accumulated displacement among particles surviving the complete interval. It is not an instantaneous velocity measurement of the surviving endpoint population. Ordinary axial Brownian diffusion is included but has zero contribution to this mean.

Both controls solve the complete radial Robin eigenproblem and its loss rate. The completed approximation replaces the conditional trajectory average by the velocity average over the exact late-time endpoint density. This is a definite, positive distribution with the correct radial shape and loss exponent; every capture, diffusion, geometry and flow control enters. Its error is treating the endpoint ensemble as the sampling distribution throughout each surviving trajectory. The alternative is a coherent readout closure, not a broken Bessel equation or an omitted input.

For the principal mode `phi(r)=J0(q*r/R)`, `q*J1(q)=(kappa*R/D)*J0(q)` and the loss exponent is `D*q²/R²`. The endpoint density is proportional to phi with area measure `2*pi*r dr`. Conditioning an interior point of a long path on both its preceding and subsequent survival instead gives weight phi squared. Thus the two drift averages are

    source:   integral(u*phi*r dr)/integral(phi*r dr),
    physical: integral(u*phi²*r dr)/integral(phi²*r dr).

The complete measured survival law need not be approximated to make this distinction. No apparatus text would supply these weights or name the correction.

## Calibration and scientific evidence

Positive wall-capture loss measurements at several radii and reactivities identify D. At fixed physical kappa, the principal Rayleigh quotient is `min [D*integral(|grad phi|²)+kappa*surface_integral(phi²)]/integral(phi²)`. Its derivative with respect to D is positive for positive kappa. Uniform-flow drift preparations agree exactly at every capture strength. Reflecting-wall Poiseuille drift also agrees exactly. Those channels can validate both readouts without exposing survival-path selection.

The prototype checks D in [.7,1.4], radius .7 or1.2, capture .5,2,8,24, and signed peak velocity .6 or-1.3. It evaluates48 combinations. The independent reference is a conservative annular finite-volume Robin diffusion generator. A diagonal axial counting field and axial diffusion term form the tilted generator. Differencing its largest eigenvalue yields the drift; this calculation does not use the squared analytic mode.256/512 cells with Richardson extrapolation give maximum relative errors1.47e-8 in drift and1.42e-10 in the loss exponent.

Three ordinary Poiseuille cases have source drift errors12.37–12.85% and physical absolute velocities .536–.978. These errors are smaller than20%, but comfortably above a .04 prediction gate without subtracting a background or choosing nearly vanishing signals.512/1024 reference refinement, exact flow reversal and reflecting/plug-flow anchors also pass. A separate zeroth/first axial-moment evolution from uniform injection confirms convergence of successive finite-time slopes within7.2e-6; the remaining difference is consistent with its128-cell mesh.25 noiseless fits across the entire D interval recover the parameter within2.88e-8. The smallest sampled calibration derivative is .05465. Full noise validation and instrument choices remain for any later authorized package.

## Archive overlap and primary sources

The archive audit scanned task/assessment text for survival, conditioning, reactive walls, killed diffusion and Taylor transport, then read the closest five author records. Archived tube-dispersion ignores shear-history dispersion in a reflecting tube; its source replaces flow by its mean. Here the radial capture solution is exact and the observable is survival-selected drift. Gated-capture concerns a dynamically switching absorbing coating and a stationary uptake flux. Fixed-trap-memory concerns persistent random escape rates on return visits. Neither has this conditional path ensemble. Pressure-surface-waves concerns follower mechanical traction, and acoustic-cavity-pressure concerns second-order fixed-mass mean redistribution; the abandoned two-interface outline would have been closer to them and was not built. This is a new physical closure within the broad transport theme, not a claim that its mathematics or conditioning principle is novel.

[Biswas and Sen, Taylor dispersion with absorbing boundaries: A Stochastic Approach](https://arxiv.org/abs/cond-mat/0701392), derive the survival-conditioned moments and explicitly distinguish the endpoint mode from the squared-mode path occupation in their equation14 and its discussion. Their displayed example is a planar channel; this prototype independently derives the circular Robin implementation. [Basic characteristics of Taylor dispersion in a laminar tube flow with wall absorption](https://doi.org/10.1016/j.ijheatmasstransfer.2017.02.051), supplies primary background on tube loss, advection and dispersion moments. No result about empirical agent difficulty follows from either paper.

Run the preserved prototype with the pinned numerical dependencies. It writes report.json, containing all cases and its source hash. Root owns any decision to package or evaluate it.
