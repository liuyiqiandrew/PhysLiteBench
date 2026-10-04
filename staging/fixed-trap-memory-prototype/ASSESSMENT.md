# Fixed-trap first-passage memory

Prototype only. No task package, calibration files or model-agent evaluations. The proposal tests a physical disorder approximation, not a changed first-passage boundary or an inaccurate generator.

A finite channel has transient sites 0 through N-1 and an absorbing detector at N. One particle starts at0. The sole departure from0 goes to1 at its site's escape rate. At every other transient site, the total escape rate is the site's rate and departure goes right with known probability p or left with probability1-p. Direction choices do not depend on site type. A fresh specimen is fabricated for each shot: independently, each transient site has residence-time mean1/gamma or c/gamma with known probabilities1-s and s. Those site properties remain fixed throughout the shot. Holding times are independent exponentials conditional on the site rates, including repeated visits. The measurement is either the mean or the variance of absorption time pooled over fresh specimens and thermal trajectories. It is not the mean of within-specimen variances.

The unknown is the common escape-rate scale gamma. The complete shortcut is an exact2N-state Markov model for position and current rate class, resampling the destination's class on every arrival. It retains the binary waiting-time distribution, jump-direction bias, finite geometry, reflecting endpoint and absorbing detector. Every supplied control participates. This is the annealed renewal approximation, an internally valid positive stochastic process. It loses the persistent identity of a physical trap on returning to a visited site.

The full mean first-passage calibration is identical in both processes for every channel, bias and disorder contrast. The embedded sequence of sites is independent of the rates. If M_i counts visits to site i, and M=sum_i M_i, both means are E[tau]*E[M]. Thus calibration can vary size, bias, contrast and type probability while gamma is uniquely identified through an exact positive coefficient divided by gamma. Zero-contrast variance records could provide additional exact calibration without removing the disorder discrimination. No control has to be absent during calibration.

For the prototype oracle, let P be the transient embedded jump matrix and G=(I-P)^(-1). From site0,

    E[M] = (G 1)_0,
    E[M^2] = ((2G-I) G 1)_0,
    E[sum_i M_i(M_i-1)] = 2 sum_i G_0i (G_ii-1).

Write mu=E[tau] and mu2=E[tau^2]. The renewal process has raw second passage-time moment

    mu^2 E[M^2] + (2mu2-mu^2) E[M].

The physical fixed-landscape process has the additional contribution

    (mu2-mu^2) E[sum_i M_i(M_i-1)].

Both means agree, so that difference also holds for the centered variances. Conditional on the embedded path, independent exponential waiting fluctuations give mu2*sum_i M_i, while the shared random residence mean contributes Var(tau)*sum_i M_i^2. Repeated encounters with a slow site therefore do not behave like independent draws from the same marginal wait distribution.

The independent reference enumerates every binary specimen, constructs its explicit continuous-time backward generator from the local escape rates, solves the first and second hitting-time equations, averages raw moments, then subtracts the square of the pooled mean. It contains no visit-count variance correction. The shortcut's2N-state generator is separately solved to verify the renewal expression, so neither control is an uncompleted mathematical model.

Twelve prototype diagnostic settings give variance deficits13.1–54.7%, with positive finite mean and variance. The exact specimen reference agrees with the visit formula to2.52e-15 relative; the independent renewal generator agrees with its moment expression to2.38e-15. Mean calibration equivalence is within1.37e-15. Nine mixed calibration settings at33 gamma values across[.6,1.4] recover gamma within4.00e-8. Directed no-return motion, zero contrast and pure-type populations give complete equality of both models. Rescaling gamma rescales the mean by gamma^-1 and the variance by gamma^-2. The reference separately checks the law of total variance, confirming the pooled specimen/trajectory convention.

The finite model has no heavy-tail or infinite-time convergence issue: all escape rates are positive and the absorbing boundary is reachable. The selected prototype size is at most9, so direct enumeration is small. The report records every checked point and all54 limit combinations; the whole check ran in .62seconds. No claim is made about Luna difficulty. Recognizing that the public site properties persist and replacing the source correctly would be a physical success; a correct fixed-landscape attempt with a matrix or variance subtraction error would be classified separately.

The overlap audit scans current task, stage and archive AUTHOR files. Closest comparisons are single-file memory (particle ordering and joint asymptotic covariance), exclusion current (joint conserved counters), terminal current noise (electrical measurement location), and finite-band reservoir (unitary preparation-dependent mode populations). None uses return visits to fixed random residence parameters. The broad preparation/correlation theme is shared and acknowledged.

Primary background: Akimoto and Saito, [Exact Results for First-Passage-Time Statistics in Biased Quenched Trap Models](https://arxiv.org/abs/1901.00624), Phys.Rev.E99,052127(2019), explicitly distinguishes fixed random energy traps from a renewal CTRW and derives distinct first-passage statistics. Luo and Tang, [Sample-dependent first-passage time distribution in a disordered medium](https://arxiv.org/abs/1507.07409), Phys.Rev.E92,042137(2015), studies sample dependence of trap-model passage statistics. Those papers use different geometries and disorder laws; the finite reflecting-chain binary-specimen formulas above are derived here and checked independently, rather than copied from their asymptotic results.
