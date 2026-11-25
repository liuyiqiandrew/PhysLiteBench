# Single-file two-time covariance: author-only prototype

This is a proposed new task, not a packaged or evaluated benchmark. No model run has been launched. The independent numerical prototype passes. Difficulty remains unknown.

## Proposed apparatus

Identical Brownian point particles move on an infinite straight line. Between encounters each has diffusion constant D; collisions are perfectly reflecting in the ordered configuration space, so particles cannot pass. There are no external forces, hydrodynamic interactions, attractions or other noise sources. Every experimental shot starts with particle j exactly at j/rho, for every integer j, and the particle initially at zero is marked. Independent thermal noise is renewed between shots; the initial positions are identical. D is the sole unknown and rho is a known control.

The detector follows that same marked particle without applying forces. For fixed positive dimensionless time multipliers t and s, the requested output is lim_(L->infinity) Cov[X(L t),X(L s)]/sqrt(L), with both times scaled together and all other controls held fixed. The covariance is across the independently repeated shots. No finite-time Gaussian law is assumed in the apparatus. Planned ranges are D in [.8,1.2], rho in [.7,1.5], and t,s in [.25,4]. Calibration would measure equal-time covariance at several t and rho using a fixed, control-independent instrument uncertainty. The standard instruction would permit replacing the whole implementation.

## Complete physical approximation

The supplied predictor would already contain the exact one-time long-time variance for this preparation. It models the effective trajectory as fractional Brownian motion with H=1/4 and that fitted amplitude. Its covariance is

    C_source(t,s) = sqrt(D/(2*pi))/rho * [sqrt(t)+sqrt(s)-sqrt(abs(t-s))].

This is a positive Gaussian covariance with stationary increments, not a numerical mistake or a missing fit. Every diagonal value is exactly sqrt(2*D*t/pi)/rho. There is no unused preparation input or output-specific branch. Equal-time calibration therefore identifies D without testing the stationary-increment assumption.

The physical limiting covariance is

    C(t,s) = sqrt(D/pi)/rho * [sqrt(t+s)-sqrt(abs(t-s))].

Both processes have the same self-similar exponent and one-time Gaussian scaling law. The difference is temporal memory of the fixed-density preparation in a system with a conserved particle order. Local equilibration does not make the entire trajectory stationary in its increments when both observation times grow at a fixed ratio from preparation. This interpretation is different from merely using an incorrect one-time annealed coefficient.

## Independent microscopic route

Allow independent Brownian walkers to cross, while assigning the tagged particle to its conserved rank in the sorted positions. At large joint time scale, its displacement is the net crossing count at the origin divided by rho, up to fluctuations lower than the leading L^(1/4) scale. For each deterministic initial position x, the two threshold indicators have covariance equal to the bivariate Gaussian CDF minus the product of marginal CDFs. Summing these independent contributions gives the current covariance; after rescaling, the lattice sum becomes rho times a spatial integral.

The reference evaluates this indicator covariance by integrating the bivariate normal density over its correlation parameter. Substitution correlation=sin(theta) removes the endpoint square-root singularity. A separate spatial Gauss quadrature performs the continuum integral. It does not call the closed tagged covariance. A finite-lattice sum checks the integral, and independent Brownian rank simulations use the same walker paths at both readout times before sorting. Those simulations are supplementary finite-time consistency checks, not the limiting oracle or a proposed grading method.

The primary results are Krapivsky, Mallick and Sadhu, [Dynamical properties of single-file diffusion](https://arxiv.org/abs/1505.01287), especially their deterministic and fluctuating preparation two-time covariances, and Sadhu and Derrida, [Large deviation function of a tracer position in single file diffusion](https://arxiv.org/abs/1505.04572), for the microscopic independent-particle mapping and multi-time extension. The numerical crossing construction and proposed calibration are independently implemented here.

## Evidence and limits

The 26 corner/off-grid reference comparisons agree within 8.76e-13; spatial quadrature refinement changes predictions by at most 4.81e-13. Equal-time calibration agrees within 2.23e-16, and noiseless fits at 17 D values recover D within 6.67e-16. The smallest calibration derivative is .1213. Both 20-time covariance matrices are positive in all 24 sampled sets; the smallest sampled eigenvalues are .0137 and .00969. Joint-time scaling holds within 4.45e-16.

At D=rho=1 and (t,s)=(4,1), the physical signal is .28436 and the source gives .50584, a 77.9% gap. Ratios t/s=2 and 4 give 36.6% and 77.9% gaps, respectively; these do not rely on nearly zero responses. The rank simulations at scales16,64,256 give .2890,.2783,.2762 with standard errors around .0064, consistent with the limiting .28436. Their statistical uncertainty does not establish a monotone convergence rate. Lattice-current agreement is already essentially roundoff at the tested scales; it is not evidence of exact equality between finite-time tracer and crossing-count covariances.

Before any full task, retain the microscopic preparation and joint limit, check independent quadrature throughout the chosen domain including near-equal times, add repeated fixed-sigma calibration and 256-noise controls, and peer-review the public text. No empirical failure probability can be inferred from this prototype.
