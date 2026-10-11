# Rotational heat: independent concept review

Reviewed 2026-10-11. **Approved for packaging, subject to the ordinary final-source review.** The apparatus, central-region calibration, completed work approximation, and Gaussian-path reference form a viable minimal task. This is not a difficulty prediction. No task code, data, grading, or model trials were changed or run in this review.

## Apparatus and formula

Use the stated stationary overdamped diffusion itself, with positive k, T, and mobility mu, and nonzero Omega. Its stationary position law is the normalized Gaussian with covariance `(T/k)*I`, even though the rotational force maintains a probability current. State that preparation explicitly. Neither the initial nor final position or energy is fixed or conditioned on. Heat into the bath is the Stratonovich integral of the complete force, so `Q=W-[U(t)-U(0)]` with `W=integral Omega*R*x circ dx`.

Set `tau=mu*k*t`, `q=sqrt(k/T)*x`, `r=Omega/k`, and `a=T*lambda`, using the plus-sign exponential moment. The mean heat rate is `2*mu*Omega**2*T/k`. The interior dimensionless bulk generating rate is

```
g(a) = 1 - sqrt(1 - 4*r**2*a*(1+a)).
```

One direct derivation applies the backward work tilt to the terminal heat factor `exp(-a*|q|**2/2)`. A radial Gaussian ansatz `C(tau)*exp(-b(tau)*|q|**2/2)` gives

```
b' = -2*(b*b + b + r*r*a*(1+a)),  b(0)=a,
(log C)' = -2*b,
E exp(a*Q/T) = C/(1-a+b).
```

The attracting root is `b=(-1+D)/2`, where `D=sqrt(1-4*r*r*a*(1+a))`. Its basin and the stationary initial Gaussian overlap give the open heat interval `-1<a<a_plus`. Here `a_plus=(1-r*r)/(1+r*r)` for `r*r<1/3`, and `a_plus=(sqrt(1+1/r**2)-1)/2` otherwise. Thus the proposed signs, factor of two, physical time scale, and piecewise upper boundary are correct.

Writing `b=r*r` only in the following rate formula, the work approximation is

```
I_work(z)/(mu*k) = sqrt((1+b)*(1+b*z*z)) - 1 - b*z.
```

For heat, replace it by `-2*b*z` when `z<-1`. If `b<1/3`, also replace it for `z>(3-b)/(1-3*b)` by

```
2*b*(1-b)*z/(1+b) - 4*b/(1+b).
```

The rate and its first derivative match at each switch. The positive-tail switch is always at z greater than 3. Calibration restricted to `.4<=z<=1.6` therefore agrees exactly with work for every allowed drive. Include nonzero rate readings away from z=1; mobility is an identifiable common multiplicative scale. Temperature cancels from this normalized rate function, so fixing T is the minimal choice, or its invariance should be disclosed.

The rate is a finite observable at ordinary z values. Rare initial/final energies of order t contribute to its tails even though typical endpoint energy changes are order one. This is not justified merely by applying the differentiable version of the Gartner-Ellis theorem: the finite endpoint slopes are precisely the non-steep case. The known explicit Gaussian heat distribution supports the constrained transform and its linear tails. Noh and Park's [Fluctuation Relation for Heat](https://arxiv.org/html/1204.1004v2), especially equations 22–26, gives this same model and rate after changing the exponential-moment sign convention. The continuation is discussed in Noh's [Fluctuations and correlations in nonequilibrium systems](https://arxiv.org/abs/1402.0193).

Do not identify the actual generating rate at an exact boundary with its interior limit. For example, at a=-1 the Riccati solution stays at the repelling root b=-1 and the moment is `exp(2*tau)`, whereas `g(-1+)` is zero. The rate's linear branches use limits from inside the open domain. A task asking for the rate function avoids this endpoint ambiguity.

## Independent reference

The prototype supplies an adequate independent route. Exact OU transitions have mean matrix `exp(-h)*(cos(r*h)*I+sin(r*h)*R)` and variance `1-exp(-2*h)`. Midpoint path heat is the exact potential-energy difference plus the polygonal rotational line integral. Integrating its exponential over the stationary Gaussian path gives a block-tridiagonal precision matrix. Its scalar Schur pivots include distinct initial and final blocks; their positivity derives the admissible domain instead of importing the oracle's clipped counting field. The log determinant gives the discrete-time generating rate.

This transition law is exact, but the polygonal heat is a time-discretization of continuous Stratonovich heat. Retain step refinement and continuum extrapolation. Finite-chain pivot recursion is a useful additional check of finite-time moments and divergence outside the limiting domain. A separate continuum Gaussian calculation, rather than a second implementation of the clipped saddle, supplies the physics cross-check.

I independently checked 81 rate cases, including z=-1 on both sides, positive-tail switches, and drive ratios on both sides of `r*r=1/3`. Steps `.01,.005,.0025` with the prototype's quadratic extrapolation agreed with the oracle within `2.230e-7` absolute. In 48 finite-time cases, that extrapolation agreed with the exact continuum Gaussian moment within `6.983e-8`. These checks used NumPy 2.5.2 and SciPy 1.18.0, not a future pinned task image.

Before packaging, use the stable algebraic discriminant

```
1 - 8*a*r*decay*sin(r*h)/variance - 4*a*a*r*r
```

instead of subtracting `diagonal**2-4*cross`; those terms are individually order `h**-2`, but their difference is order one. The linear coefficient is 8 because squaring the off-diagonal term produces a factor 2. My initial note incorrectly wrote 4; the author's calibration check caught that error before any task data were generated, and this correction records it. The additional reviewer checks above used the original subtractive expression, so their results are unaffected. This preserves the independent path derivation. The present steps are already accurate enough for ordinary percentage grading; arbitrarily shrinking the step would otherwise lose accuracy through cancellation. The prototype assumes positive r in its oracle saddle formula, so keep that disclosed restriction or implement dependence on absolute r correctly.

## Shortcut, margins, and archive distinction

The work approximation is a valid, nonzero observable of the same stationary process. Its incorrect physical inference is discarding unbounded energy endpoints for a tail statistic merely because they disappear from the mean and low cumulant rates. It reproduces the entire proposed calibration, not only a zero-drive limit. Example dimensionless heat/work rates are `.29915648/.49632496` at `(r,z)=(.15,9)`, `1.08/1.68979069` at `(.3,-6)`, and `1.5/1.76556444` at `(.5,-3)`: relative separations about 66%, 56%, and 18%. Use nonzero group norms for grading and retain central-region anchors.

The nearest archive is `archives/screened/thermal-bodies-r12/tasks/thermal-bodies`, which asks for finite-window thermostat-heat variance and tests stored-energy/current covariance. The present effect survives the long-time limit and concerns rare endpoint energies and exponential-moment existence. `archives/entropy-anomaly-r4/tasks/entropy-anomaly/AUTHOR.md` explicitly distinguishes its first two long-time rates from endpoint-sensitive full moment domains. Its later `staging/entropy-anomaly-hardening-review/ASSESSMENT.md` leaves tail/domain experiments open because no finite operational readout, shared calibration, stable margin, or practical independent reference had then been validated. This proposal supplies those missing pieces. It is related stochastic thermodynamics, but not a relabeling of either archived observable or failure mechanism.

This review inspected prototype SHA-256 `a40965979be7633c8c8f673e8c3264c5556a9d63e263c25e1b37a0d988e4e4b9` at `staging/gpt6-expansion/rotational_heat_prototype.py`. The numerical-stability suggestion above was sent to the author; this report does not claim review of later modified bytes.
