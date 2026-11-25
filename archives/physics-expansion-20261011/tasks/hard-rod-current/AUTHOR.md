# Hard-rod current, revision 1

The calibration determines the common speed from interacting means and
variances and from point-particle generating functions. The completed
approximation predicts all these exactly, but freezes the collision-dressed
fluctuation statistics at their ordinary, unbiased values. Full current
fluctuations instead change the moving cut in compressed coordinates and
therefore change the crossing intensities themselves.

## Derivation

Write rho for physical center density, a for rod length, p for right-moving
fraction, v for speed, h=1-a*rho, and c=rho/h. Compress ordered centers by
y_n=x_n-a*n. Elastic equal-mass collisions exchange velocities, so the
compressed positions follow free marked trajectories and are then sorted.
The compressed ensemble is a marked Poisson gas of density c, apart from
the origin convention discussed below.

A physical current J(t) approximately j*t corresponds to a compressed cut
at a*j*t. At that cut the two independent Poisson crossing rates are

```
r_plus(j) = c*p*(v-a*j)
r_minus(j) = c*(1-p)*(v+a*j).
```

The current rate function is the minimum Poisson relative entropy of two
nonnegative flows whose difference is j. Its Legendre transform reduces to

```
lambda = s + a*c*[p*(exp(s)-1)-(1-p)*(exp(-s)-1)]
F(lambda) = c*v*[p*(exp(s)-1)+(1-p)*(exp(-s)-1)].
```

The first equation is strictly increasing in s and has a unique root for
every disclosed field. The saddle current remains strictly inside +/-v/a
when a>0. There is no phase choice or boundary saddle to guess. The oracle
uses this scalar root. Differentiation at zero gives

```
mean = rho*v*(2*p-1)
variance = rho*v*h*[1-a*rho*(2*p-1)^2].
```

At a=0 the SCGF becomes that of independent signed Poisson crossings.
Reflection maps (p,lambda) to (1-p,-lambda); speed scales every statistic
linearly. The SCGF can be negative opposite the mean current.

## Completed approximation

Set d=2*p-1 and freeze the ordinary crossing intensities

```
R_plus = c*p*v*(1-a*rho*d)
R_minus = c*(1-p)*v*(1+a*rho*d).
```

For independent Poisson counts N_plus/minus with those rates, approximate
the current by its exact mean plus centered dressed shot noise,

```
J_short(t) = rho*v*d*t + h*[(N_plus-R_plus*t)-(N_minus-R_minus*t)].
```

This is a valid real-valued effective current with fractional jumps and a
deterministic background, not an exact integer crossing process. Its SCGF
is implemented without truncation. It retains all inputs, both velocity
species, and finite excluded volume, and reproduces both exact cumulants
used for calibration. Its physical error is holding these event rates and
the dressing fixed under finite current conditioning.

## Independent reference and initial ensemble

The reference minimizes the sum of two Poisson relative entropies at each
current and then numerically maximizes lambda*j-I(j). It does not use the
oracle's implicit field equation. A further microscopic check constructs
the finite-time integer current distribution. Label the last initial free
point left of zero by n=0 and use x_n=y_n+a*n. Then

```
P(J(t)>=k) = P(Q_t(a*(k-1))>=k),
```

where Q_t(z) is a difference of independent Poissons with means

```
c*[p*max(v*t-z,0)+(1-p)*max(-v*t-z,0)]
c*[p*max(z-v*t,0)+(1-p)*max(z+v*t,0)].
```

Adjacent tails give integer-current probabilities. Stable Skellam sums and
log-sum-exp then give the finite-time log moment generating function. Time
and summation refinement check its limit for symmetric and flowing gases.
The fourth case has a negative finite-time estimate and a positive limit.
The observed 1/t correction is used only for
these checks, not asserted to be exact over the domain.

The public initial ensemble has a uniform physical origin. A stationary
Tonks gap is a+E with E exponentially distributed at rate c, and the gap
containing the origin is size biased. Splitting it into its length-a piece
and its remaining free piece gives probabilities a*rho and h. Compression
then gives a Palm Poisson gas with one distinguished point or an ordinary
stationary Poisson gas, plus a coordinate shift bounded by a. The extra
point changes counts by a bounded amount, and bounded-window Poisson counts
have superexponentially small probabilities of being order t. Thus these
origin conventions have the same speed-t current large-deviation limit.
An arbitrary random O(1) boundary correction would not by itself justify
that conclusion. The thermodynamic-before-time limit and redrawing both
positions and marks are explicit in the public specification.

## Validation and evaluation

There are 240 independent readings at 48 settings, sigma=.002, true speed
1.12, data seed 611501, and 256 noise draws with seed 611509. Every readout
is linear in speed, and the nonzero variance coefficients identify that
parameter uniquely. Prediction groups use error norm divided by truth norm
with a 4% limit; no division by an individual SCGF near a zero is used.
Parameter error must be below 3% and reduced chi-square below 1.5.

`scripts/validate_hard_rod_current.py` recomputes frozen reference data,
checks both completed controls and the noise ensemble, compares 152 domain
and boundary cases, and checks reflection, speed scaling, zero field,
positive variance, Jensen's bound, and first two field derivatives. It also
checks four microscopic finite-time distributions at t=16,32,64. Only
`--generate` replaces calibration and hidden truth. The report is
`results/expansion-hard-rod-current-r1-validation.json`.

An initial derivative check at step 1e-4 differed by 2.28e-7 because the
second difference amplified the root tolerance. Changing only the validation
step to 3e-4 reduces this to 2.50e-8. Task inputs, truth, oracle, and grading
were unchanged. This was before model evaluation.

Run both Docker controls, then exactly three frozen unhinted gpt-5.6-luna
High trials. Review final source and trajectories: retaining a fixed-state
independent-event closure is a physical failure; an otherwise correct rank
mapping with a sign, root, or implementation error must be classified
separately. Do not resample an unchanged completed revision.

The existing single-file-memory problem concerns Brownian tagged-particle
covariances, and ring-current-fluctuations concerns a diffusive nonuniform
conditioned profile. This task uses ballistic rods and a homogeneous
current-dependent crossing contraction. Primary context is
[Myers et al., Transport fluctuations in integrable models out of equilibrium](https://arxiv.org/abs/1812.02082).
The specialized two-velocity formulas above follow directly from the stated
mechanical mapping.
