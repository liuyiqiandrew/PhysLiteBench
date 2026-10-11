# Driven mobile-trap survival, revision 2

Revision 1 measured the sqrt(time) survival coefficient of an undriven
Brownian target. All three Luna High solutions recovered its correct
Pascal-principle coefficient. That candidate, its reviews and all outcomes
are preserved under `archives/physics-expansion-20261011/`.

Revision 2 adds constant force and controlled target noise temperature and
measures the exponential rate instead. Calibration uses deterministic target
motion while traps retain their separately prescribed Brownian diffusion.
This is a new physical setup and observable, not a new draw of revision 1.
The old zero-force coefficient is subleading for the new measurement.

## Rate from matching path bounds

Write u=mu*abs(F) and D=mu*T. For a fixed target path x, marked-Poisson
thinning gives S[x]=exp(-rho*E_B Range(x-B)). Convexity and E[B]=0 imply
E_B Range(x-B)>=Range(x)>=abs(x(t)). Consequently

    S(t) <= E_X exp(-rho*abs(X_t)).

The endpoint is Gaussian with mean u*t and variance 2*D*t. Its elementary
Laplace principle bounds the asymptotic rate below by

    min_w [(w-u)^2/(4*D)+rho*abs(w)].

For a matching lower survival bound, choose a deterministic moving interval
[w*s-ell,w*s+ell]. Require all traps to avoid it and the target to remain
inside. The first event has probability
exp(-rho*(2*ell+E Range(w*s-B_s))). A drift change from u to w, bounded
endpoint in that interval, and positive Dirichlet ground-state martingale
bound the second event below by

    exp(-(u-w)^2*t/(4*D) - abs(u-w)*ell/(2*D) - pi^2*D*t/(4*ell^2)).

For any fixed w, the expected drifted Brownian range divided by t tends to
abs(w). Choosing ell proportional to t^(1/3) makes every other correction
sublinear. Thus the lower survival bound gives the same rate. Reflection
allows w>=0 for u>=0. The optimum w=max(u-2*rho*D,0) yields

    K = mu*F^2/(4*T),                  abs(F)<=2*rho*T,
    K = mu*(rho*abs(F)-rho^2*T),        abs(F)>2*rho*T.

At T=0 the target is deterministic and K=rho*mu*abs(F). At F=0, K=0;
trap encounters still give slower stretched-exponential decay. Every fixed
positive finite D_B disappears from the exponential rate. This does not
mean it disappears from finite-time survival. The low-force branch is
sustained by target paths with zero asymptotic drift; the high-force branch
by paths with a reduced nonzero drift. No ambiguous average order or
unreported boundary condition is needed.

## Independent reference and finite-time checks

The reference numerically minimizes the Gaussian velocity cost plus the
Poisson interval cost; it does not use the oracle's piecewise formula.
The zero-velocity and deterministic endpoints are explicitly included.
A second reference supplies bounds on -log(S(t))/t at finite times:

- The lower bound integrates exp(-rho*abs(X_t)) against the Gaussian
  endpoint density, using two truncated normal moments in log space.
- The upper bound uses the moving interval above with its optimizing width.
  Its trap factor uses the exact expected range of drifted Brownian motion.

These bound the finite-time normalized log survival, not the limiting rate
at each finite time. Both converge to the oracle. The validator checks
153 random, corner and crossover preparations at times 1, 10^3, 10^6, 10^9.
The largest endpoint/tube differences from the limiting rate at the last
time are 8.75e-5 and 9.47e-5. The slowest endpoint case is the stationary,
zero-temperature target, whose exact correction decays as t^(-1/2).
An initial 1e-6 check for that finite-time correction was too tight and was
corrected before trials; no physical source, data or grading changed.

The range is independently checked by quadrature of the drifted Brownian
first-passage tail for both signs of drift at 24 settings. The maximum
absolute discrepancy is below 8e-15. All survival calculations use log
probabilities, without exponentiating extremely small survival values.

## Calibration and physical shortcut

The completed shortcut replaces the target path by its deterministic
mean drift. It is the exact exponential rate of that auxiliary process,
K_short=rho*mu*abs(F), for any fixed trap diffusion. It already has the
correct zero-force exponential rate. Its mistake is evaluating trap
exposure on a typical target trajectory before averaging survival; it
omits the drift reduction selected among surviving target paths.

All 256 calibration readings use T=0 at 32 density/force/trap-diffusion
settings repeated eight times. Both controls agree there for every allowed
mobility and recover its unique linear amplitude. True mu is 1.08;
sigma=.001, data seed 611901, and the 256-draw noise seed is 611909.
Prediction groups use relative Euclidean error below 4%, parameter error
below 3%, and reduced calibration chi-square below 1.5. Group norms are
nonzero even though cold anchors include zero-force cases. The weakest
thermal shortcut error exceeds 14%, while the largest noisy oracle error
is below .014%.

Run `scripts/validate_mobile_trap_survival.py`. Normal validation reads
checked-in data; `--generate` intentionally replaces both calibration
copies and hidden truth. The report is
`results/expansion-mobile-trap-survival-r2-validation.json`.
After independent source review and fresh Docker controls, evaluate exactly
three frozen unhinted gpt-5.6-luna High trials. Review final code and all
trajectories. Keeping the mean target drift after fitting mu is a physical
failure. A correct rare-path contraction with a subsequent mathematical or
coding error must be classified separately. Preserve every outcome and
never resample an unchanged revision.
