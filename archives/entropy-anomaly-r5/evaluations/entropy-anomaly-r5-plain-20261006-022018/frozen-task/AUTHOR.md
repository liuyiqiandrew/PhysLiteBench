# Entropy anomaly, revision 5

This same-family revision adds recurrent contact changes between two local thermal reservoirs. It preserves the charged Brownian particle, anisotropic drag, smooth spatial temperature profile, finite-mass calorimetry and stationary mass-to-zero limit. The starter already includes the complete switched position process and the correct static magnetic kinetic anomaly. It omits the kinetic rethermalization cost of changing contacts repeatedly.

Revision 3 remains canonical while this package is staged. Revision 4's entropy-variance task remains archived. The fresh neutral r3 and r4 records both have two passes out of three; those outcomes are preserved. Their successful methods include matrix-drag kinetic relaxation and fast/slow entropy fluctuations. This revision changes the physical preparation: velocities stay continuous while their connected thermal reservoir changes. Kinetic endpoint events now recur extensively instead of appearing only at the beginning and end of one measurement. No new Luna outcome is claimed here.

## Apparatus and unknown

The public underdamped equations are the prior two-dimensional anisotropic Langevin equations with signed Lorentz force, but the local temperature is selected by an independent symmetric telegraph valve. The known flip rate kappa stays fixed as mass tends to zero. Exactly one reservoir is connected at a time, with

    T_s(x)=T0*(1+s*delta)*(1+a*cos(k*x)),  s=+1,-1.

At a valve flip the particle's position and velocity stay continuous. There is no particle impulse, Hamiltonian change or instantaneous work. The subsequent relaxation transfers heat to the newly connected thermal bath. Calorimeters sum all particle heat, divided by each currently connected local temperature. External background costs of maintaining the bath profiles and empty valve are excluded. Every apparatus input, allowed range, noise law, convention and order of limits is stated publicly. The single fitted parameter is the common friction gamma.

## Conditional position currents

Normalize the two conditional position densities together on one thermal phase period, so each has integral one half. Equivalently they describe the repeated thermal cells of the full periodic square, with uniform y. Physical x derivatives include the integer wave number k. Write gx=gamma, gy=lambda*gamma and

    Gamma=[[gx,-B],[B,gy]].

The limiting physical currents and stationary contact balance are

    Gamma*J_s=F*rho_s-ex*(T_s*rho_s)',
    J_xs'=kappa*(rho_-s-rho_s).

The oracle and completed shortcut solve these coupled equations spectrally. All controls, including the overall switching rate, affect their full densities and currents. The completed shortcut computes

    sum_s integral J_s^T diag(gx,gy) J_s/(T_s*rho_s) dx
    + kappa*sum_s integral rho_s*log(rho_s/rho_-s) dx
    + C(B)*sum_s integral rho_s*(T_s')^2/T_s dx.

The first two terms are the entropy currents of the reduced position/contact process, including contact-state correlations. The last retains the full static thermal anomaly. In the H30,H21,H12,H03 basis, A has diagonal (3gx,2gx+gy,gx+2gy,3gy), neighboring entries A[i,i+1]=-B*sqrt((3-i)*(i+1)) and the opposite sign below the diagonal. With a=(sqrt(6),0,sqrt(2),0), C(B)=a^T A^-1 a/4. The source therefore already solves the magnetic physics that previous passing submissions restored.

This is a completed physical approximation: momentum is eliminated before computing the cost of changing its connected reservoir. It is not an omitted input, incorrect force law, failed fit or unstable numerical solver.

## Finite kinetic relaxation at repeated contact changes

A finite-mass conditional energy balance gives the limiting raw bath entropy

    sum_s integral [F.J_s/T_s - 2*J_xs*T_s'/T_s
                    + C(B)*rho_s*(T_s')^2/T_s] dx
    + kappa*sum_s integral rho_s*(T_s/T_-s-1) dx.

Here the leading local kinetic energy is T_s*rho_s because there are two velocity components. The kinetic energy flux is 2*T_s*J_xs minus the static thermal-gradient response. Integration by parts using the conditional contact balance rewrites this rate as the completed shortcut plus

    kappa*sum_s integral rho_s*
        [T_s/T_-s-1-log(T_s/T_-s)] dx.

This additional term is the relative entropy between the old and new local Maxwell distributions. Every integrand is nonnegative. It survives the small-mass limit because each contact change transfers finite kinetic heat, even though the relaxation duration vanishes. For the specified proportional profiles and symmetric valve, the two conditional masses are one half and the correction simplifies exactly to

    2*kappa*delta^2/(1-delta^2).

The oracle evaluates the unsimplified spatial integral. The verifier contains neither formula. There is no ambiguity about magnetic path-reversal conventions: the target is direct bath calorimetry from the stated finite-mass model.

## Calibration and identifiability

All calibration has delta=B=0. The two bath temperatures coincide, rho_s=rho/2, and both contact entropy terms vanish identically for every kappa. The forward response is the prior r3 coefficient divided by gamma. Weighted regression for inverse friction therefore identifies gamma globally over [.7,1.6]. The magnetic and two-reservoir hidden settings introduce no new fitted parameter.

The 192 existing calibration values and all their uncertainties are preserved exactly; only bath_contrast=0 and known switch_rate fields were appended to each input. Both public and private data copies match. The prior calibration seed remains 9332. No measurements were regenerated or new observational noise selected. Equality with the previous forward solution was checked at gamma=.7,1.1,1.6, to within 8.7e-14. Eight representative calibration settings also agree with independent finite-mass calorimetry within 3.4e-8.

The fitted friction is 1.0993577853, with relative error .00058383 and reduced calibration chi-square .955566. The oracle's three diagnostic group errors are .0000187,.0000412,.0000825; the completed shortcut's are .877866,.876982,.745560. Both pass the equal-contact magnetic anchor with error .00026025.

## Independent finite-mass reference

The private reference constructs the two-state stationary Kramers generator in x Fourier modes and Gaussian-weighted Hermites of both scaled velocities. It includes the full finite-mass spatial transport, forces, anisotropic collision operator, Lorentz rotation and inter-reservoir switching. Multiplication of the kinetic generator by m puts the contact coupling at m*kappa. The two reservoirs use their actual temperatures in thermal diffusion; the Gaussian basis temperature is only a numerical reference scale.

The default uses 41 Fourier modes and velocity Hermites through total degree12. For each of three positive masses h,h/2,h/4, the reference directly computes the mean Stratonovich bath heat divided by the connected temperature, including both noise-work subtractions. It then extrapolates quadratically to zero mass. The mass scale is h=.006*min(gx,gy)^2/(T0*k^2). This path never calls the positional oracle, static-anomaly coefficient or contact correction.

Every graded preparation was checked with half the mass scale and with degree16/spatial49 refinement. Additional independent tests compare uniform-bath heat with direct conditional velocity-covariance equations, finite-mass local energy transport including contact energy exchange, and positive conditional velocity covariance. High contrast, strong field, unequal drag and wave-number corners are included. The original 600-second agent and60-second verifier limits are retained.

## Validation and gates

The inherited gates remain3% parameter error, calibration reduced chi-square below1.5 and4% group RMS prediction error. Group normalization is the RMS reference entropy within each group. No threshold was tightened to obtain a failure. The new diagnostic signals are positive and ordinary in size; none uses a near-zero denominator.

All256 independent noise realizations, seed9161055, pass calibration and parameter checks for both controls. All256 oracle realizations pass all prediction groups. All256 shortcut realizations fail the three intended contact groups and pass the equal-contact anchor. Worst oracle diagnostic error is .00063135 and worst anchor error .00199403; the minimum shortcut diagnostic error is .745297. Maximum parameter error is .44784% and maximum reduced chi-square1.26876.

Independent finite-mass agreement over the graded inputs is within4.1e-9, half-mass refinement within3.6e-9 and basis refinement within5.3e-12. State normalization, current balance, contact correction, magnetic reflection, label reversal and exact uniform/zero-force limits were checked on512 domain corners. All position densities are positive. Direct finite-mass energy balance closes within9.2e-13 and the minimum scaled conditional velocity covariance eigenvalue is .33493.

The isolated local oracle passes8/8 tests in about7.5seconds. The completed shortcut passes5 tests and fails exactly the3 contact diagnostics in about7.4seconds. The public unfinished starter is not the completed shortcut control. Docker controls, independent final peer review and Luna evaluation are separate steps; these numerical results establish scientific separation, not model difficulty.

Reproduce from the repository root with pinned NumPy2.3.3, SciPy1.16.3 and pytest8.4.2:

    OPENBLAS_NUM_THREADS=1 uv run --no-project --python3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python staging/luna-high-hardening-20261005/scripts/validate_entropy_anomaly.py

The validator reads checked-in calibration and does not regenerate it. Reports and isolated control outputs are under staging/luna-high-hardening-20261005/results/entropy-anomaly-r5*. Files in environment/ are the only public Docker build context. AUTHOR, hint, solution, verifier and true friction stay outside the ordinary agent image. Fresh unhinted gpt-5.6-luna high evaluation with three trials is the target; its result is pending.
