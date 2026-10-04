# Gated capture prototype

Scientific prototype only. No full task package, calibration data, or model-agent evaluations.

A fixed sphere irreversibly removes dilute independent Brownian particles from a maintained reservoir. Its entire surface switches between two finite intrinsic reactivities according to one stationary telegraph process independent of particle positions and counts. The readout is the stationary mean total absorption rate. Unknown common diffusivity D is identified by steady uptake with equal state reactivities at multiple radii and reaction strengths. Changing either rate still defines the gate process during calibration, although equal reactivities make its effect physically vanish.

The completed alternative is the exact stationary spherical diffusion solution with the time-averaged Robin coefficient kappa_bar=pi0*kappa0+pi1*kappa1. This is a coherent fast-switching homogenization. It has positive density, exact particle balance and correct surface geometry. It uses both rates through their stationary probabilities but loses the overall switching timescale. That absence is honestly the physical approximation; no unused dummy input or incomplete solver is inserted. This replaces the rejected idea of keeping the exact joint state and multiplying independent readout averages, which would have exposed an immediate balance inconsistency.

The actual state-resolved joint mean concentrations c_i=E[n(r)*1_(gate=i)] satisfy

    0 = D*(c_i''+2*c_i'/r) + sum_j Q_ij*c_j,
    D*c_i'(a) = kappa_i*c_i(a),
    c_i(infinity) = pi_i*c_infinity.

Here Q=[[-k01,k10],[k01,-k10]], pi=(k10,k01)/(k01+k10), and positive uptake is 4*pi*a^2*sum_i kappa_i*c_i(a). The exact decaying solution is

    c(r) = pi*c_infinity + A*pi*a/r
           + B*(1,-1)*a/r*exp[-sqrt((k01+k10)/D)*(r-a)].

A two-by-two boundary system determines A and B. The total uptake also equals -4*pi*a*D*A. Since gate transition rates do not depend on particle counts and absorption is linear, the joint mean equations close even for one common surface gate. Correlations between different particles would matter for count fluctuations, which are not this task's observable.

The independent reference solves the full radial first-order boundary-value problem in a finite outer shell for the two joint concentrations and two diffusive fluxes. It applies the reaction boundary directly and a deterministic outer reservoir. Three outer radii are extrapolated in1/R; no infinite-domain mode coefficients enter this reference. The analytic and BVP uptake agree within7.80e-8 relative across six controls. Doubling outer radii confirms the extrapolation, and direct inner reaction and outer diffusive flux agree.

The alternative overestimates diagnostic uptake by34.7–162.7%. These are healthy positive signals, not near-zero relative-error artifacts. Eighteen equal-reactivity calibration settings at31 diffusivities in[.7,1.3] agree to4.3e-14; noiseless scalar fits recover the parameter throughout the range. Identifiability is analytic: dJ/dD=4*pi*a*c_infinity*(a*kappa)^2/(D+a*kappa)^2>0. The smallest sampled derivative is.265. Fast switching converges to the alternative, slow switching converges to the stationary mixture of separately held-surface rates, state relabeling is exact, and jointly scaling all kinetic rates scales uptake. All sampled joint density profiles are nonnegative and bounded by their no-sink values.

The source-input outline is in INPUT-OUTLINE.md. It gives the actual surface process, standard reactive boundary meaning, reservoir, units and stationarity, without prescribing a conditional-density solver or an effective reactivity. A future package should use an unfinished fit, None diffusivity, a separate completed control, fixed measurement sigma, neutral whole-implementation permission and independent numeric grading. No claim is made that Luna will miss this mechanism.

Primary sources: Szabo, Shoup, Northrup and McCammon, [Stochastically gated diffusion-influenced reactions, J. Chem. Phys.77,4484–4493 (1982)](https://doi.org/10.1063/1.444397); Bressloff, [Stochastically switching diffusion with partially reactive surfaces, Phys. Rev. E106,034108 (2022)](https://arxiv.org/abs/2205.13985). The latter gives a local-time interpretation of switching Robin boundaries. Our specified target is nonsaturating and the requested mean closes linearly; no occupancy-blocking or receptor-recharge model is implied.
