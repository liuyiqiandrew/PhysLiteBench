# Entropy anomaly revision 4

This is an observable extension of the entropy-anomaly family, not a new family. Revisions 1 and 2 are preserved in `archives/entropy-anomaly-r1` and `archives/entropy-anomaly-r2`; canonical revision 3 remains untouched. Revision 3's fresh October 4 batch passed 2/3, with the remaining failure classified as mixed mathematical/physical reasoning. Those outcomes remain in `results/retained-three-brownian-trial-reviews.json` and are not evidence for revision 4.

The microscopic apparatus returns to one-dimensional, force-free motion. Calibration now measures the nonzero mean entropy rate in a spatially nonuniform bath. The supplied forward model includes the correct mean and the exact limiting positional process. Its physical approximation replaces the entropy record by its conditional mean given that position process. It then solves the positional Poisson equation and computes the resulting nonzero occupation-time variance exactly. Thus the prediction error is not a missing stochastic-calculus convention, an unfinished variance calculation, or an error in the positional dynamics.

For constant friction and temperature T(x), the limiting position generator and stationary density are

    L = (T/gamma) d²/dx²,       rho proportional to 1/T.

Write a(x)=T'(x)²/(2 gamma T(x)) and mu=integral rho*a. Velocity elimination jointly with calorimetric entropy gives an additional independent Wiener contribution:

    dA = a(X)dt + sqrt(2a(X))dW_hat.

Here W_hat is independent of the Wiener process driving X. Given the entire position path, this extra integral has mean zero and variance 2*integral a dt, so its covariance with the conditional mean functional vanishes. If -L u=a-mu, with integral rho*u=0, the correct long-time variance rate is

    2*integral rho*(a-mu)*u + 2*mu.

The completed shortcut contains the first term. The physical oracle adds the second. Both are positive and use every supplied control. Calibration and all other mean measurements agree for every friction. For the specified sinusoidal bath,

    mu = T0*k²*(1-sqrt(1-contrast²))/(2 gamma).

Every nonuniform calibration preparation therefore uniquely identifies reciprocal friction. The 288 records are 24 independent repeats of 12 settings. The standard uncertainty is fixed at .0001; calibration and noise-validation seeds are 118141 and 118147. Generation uses the physical limiting mean, checked against the independent finite-mass reference. Public and private data copies are identical.

The independent verifier does not insert a reduced entropy correction. At finite mass, integration by parts in the stated Stratonovich calorimetry gives

    S_env = A_m - Delta[m V²/(2T)] - (3/2)*Delta log T,
    A_m = integral (3T-mV²)*V*T'/(2T²) dt.

The stationary endpoint variables have finite moments. They do not affect the long-time first two cumulant rates at fixed mass. The reference solves the full Kramers equation in Fourier position and Gaussian-weighted Hermite velocity. If H multiplies by the cubic integrand, G=m*L_forward, and p is stationary, it solves

    G p1 = -m*(H-mu)*p,       integral p1=0,
    variance rate = 2*integral H*p1.

This is a derivative of the finite-mass counting eigenproblem, not a copy of the reduced variance formula. Three positive masses cancel the first two corrections in the small-mass extrapolation. The mean from this cubic representation also matches direct friction-minus-bath-work calorimetry.

The limiting anomalous functional obeys the counting generator L+(lambda+lambda²)*a, for E[exp(lambda*A)]. This gives the symmetry lambda to -1-lambda and eigenvalue zero at lambda=-1. These are checks on that limiting functional. No claim is made that bare finite-mass bath entropy has the same full exponential-moment domain: its unbounded kinetic endpoint can matter there. The benchmark asks only for long-time first and second cumulant rates.

Primary background is [Bo, Lim and Eichhorn, Functionals in stochastic thermodynamics (2019), Eq. 40b](https://arxiv.org/abs/1907.07361), which explicitly contains an independent anomalous Wiener process. [Celani et al., Anomalous thermodynamics at the micro-scale (2012)](https://arxiv.org/abs/1206.1742) explains the hidden kinetic contribution and its fluctuation relation. The task-specific one-dimensional reference follows directly from the stated Langevin and calorimetric equations.

The full report is `results/entropy-anomaly-r4-validation.json` inside the stage. Across 256 noise realizations, all calibration, parameter and physical predictions pass; all completed shortcuts fail the three variance groups while passing mean anchors. The worst physical prediction error is 3.75e-5 and the smallest shortcut error is .9863, at the unchanged .04 normalized RMS gate. The actual fit is gamma=1.100011519 and reduced chi-square 1.13145. Local controls give oracle 9/9 in 2.24 seconds and shortcut 6 passes/3 intended failures in 2.18 seconds.

Independent hidden disagreement is below 1.31e-6 relative; spatial/Hermite refinement is below 1.11e-13 and mass halving changes the extrapolated result by at most 1.15e-6 relative. Public corner checks are below 1.65e-7 relative. Stationary densities and both variance models remain positive in the checked cases. The exact zero-gradient limit vanishes. At small contrast, the physical variance tends to twice the mean, while the conditional-mean variance scales as contrast^4/64 in unit controls. Full-range noiseless calibration recovery is at roundoff.

The archive audit is `results/family-history-and-duplicate-audit.json`. The earlier finite-time thermal-calorimeter task concerned endpoint stored energy; the long-time statistic here removes that mechanism. Exclusion-current and boson-transfer-noise concerned different conservation and quantum counting correlations. This revision retains the existing entropy family designation and all previous evidence.

Reproduce from the repository root:

    uv run --no-project --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python staging/entropy-anomaly-r4/scripts/validate_entropy_fluctuation.py

Add `--generate` only for deliberate data regeneration. The stage has the exact neutral full-replacement instruction, an unfinished fit, a separate completed shortcut, independent grading, and the standard 600/60-second limits. No model evaluation has been run for this revision; empirical difficulty is unknown. Root owns frozen controls and three fresh unhinted trials.
