# Brownian target among mobile traps: concept review

Reviewed 2026-10-11. **Approved for prototyping and packaging.** The leading survival coefficient, calibration loophole, and proposed physical shortcut are correct. A direct bounding reference is sufficient and avoids impractical simulation of extremely rare survival. No task source or model evaluation was changed or run.

## Physical law and ensemble

State that all particles are points on the infinite line, `dX=sqrt(2*D_A)*dW_A` and `dY_i=sqrt(2*D_B)*dW_i`, with independent standard Brownian drivers. Traps start from a homogeneous Poisson point process independent of X(0)=0. They neither react with one another nor obstruct one another. The target is killed at the first equality with any trap. The survival probability averages initial trap positions and all paths before taking its logarithm. The infinite system is defined before the long-time limit; no finite-box depletion or conditional initial void is implicit.

For fixed positive trap diffusivity and any fixed finite target diffusivity, the proposed coefficient is

```
C = lim[-log S(t)]/sqrt(t) = 4*rho*sqrt(D_B/pi).
```

Its units are inverse square-root time. Independence from D_A is a property of this leading asymptotic measurement, not an assertion that finite-time survival is unchanged when the target moves. D_A=0 is included exactly; D_B=0 is correctly excluded from the proposed domain. The expression agrees with the primary results of [Bray and Blythe](https://arxiv.org/abs/cond-mat/0205381) and [Blythe and Bray](https://arxiv.org/abs/cond-mat/0209353).

## Direct independent bounding argument

There is a short continuum proof of the needed Pascal bound in one dimension. Fix a continuous target path x starting at zero. For a Brownian trap increment path B, the initial trap locations that lead to contact are the range of `x(s)-B(s)` for `0<=s<=t`. This is an interval because both paths are continuous. Marked-Poisson thinning therefore gives

```
S[x] = exp(-rho * E_B Range(x-B)).
```

The range functional is convex and even. Brownian symmetry makes `f(x)=E_B Range(x-B)` both convex and even, so `f(x)>=f(0)`. Thus every fixed target path satisfies `S[x]<=S[0]`, and averaging target motion preserves that bound. The Brownian reflection principle gives `E Range(B)=4*sqrt(D_B*t/pi)`.

For the opposite bound, choose a half-width ell. Require no initial traps in `[-ell,ell]`, no exterior trap to enter it by time t, and the target to remain inside. These independent events imply survival. Their first two probabilities are `exp(-2*rho*ell)` and `exp(-4*rho*sqrt(D_B*t/pi))`.

Confinement of a target starting at zero has probability at least `exp(-pi**2*D_A*t/(4*ell**2))`. This bound follows from the positive Dirichlet ground-state martingale: the surviving expectation of `cos(pi*X/(2*ell))` equals that exponential, and the cosine is at most one. It avoids treating the first term of an alternating survival series as a lower bound.

Optimizing at `ell=(pi**2*D_A*t/(4*rho))**(1/3)` proves

```
0 <= -log(S(t))/sqrt(t) - C
   <= 3*(pi**2*rho**2*D_A/4)**(1/3)*t**(-1/6).
```

For D_A=0 the stationary-target solution is exact; use that branch instead of dividing by the zero optimizing width. This squeeze establishes the coefficient without presuming independent relative traps or assuming fast finite-time convergence.

The private reference can compute the stationary target's no-entry integral by independent quadrature,

```
2*rho * integral_0^infinity erfc(x/sqrt(4*D_B*t)) dx,
```

and combine it with the confinement/empty-interval cost. Rescale the quadrature coordinate by `sqrt(4*D_B*t)` and work with logarithms. Verify quadrature refinement, bound ordering, and the certified `t**(-1/6)` shrinking gap over the allowed parameter domain. This is an independent microscopic bounding calculation, not a second copy of the closed coefficient. An exact interval-survival solver or alternating series with an explicit remainder can supplement it, but is unnecessary for a minimal reference.

The convergence is slow: large finite times and very small raw survival probabilities do not by themselves validate the leading asymptotic coefficient. A simulation-based extrapolation would be weaker than the explicit bounds. Numerical demonstrations may use large times safely in log space; they must not claim to resolve underflowed probabilities.

Citation precision matters. The full text of cond-mat/0209353 explicitly states that its Pascal inequality was not proved there. Later [Moreau et al.](https://arxiv.org/abs/cond-mat/0303273) establish the trapping principle for lattice walks, with the all-time treatment in [their subsequent paper](https://arxiv.org/abs/cond-mat/0401190). The direct expected-range argument above applies to this one-dimensional continuum apparatus without requiring a lattice-limit argument.

## Calibration and physical shortcut

For one trap, the relative coordinate has diffusivity `D_A+D_B`. For two different traps, however, their relative increments share target noise and have covariance `2*D_A*dt`. Treating all relative trap paths as independent keeps the correct one-pair process but removes this common stochastic motion. It is a coherent auxiliary process: an immobile target among independent traps with diffusivity `D_A+D_B`. Its coefficient is positive and finite,

```
C_short = 4*rho*sqrt((D_A+D_B)/pi).
```

At D_A=0 it is exact for the entire calibration family. Density variation identifies the positive scale `sqrt(D_B)` uniquely and therefore D_B. All true signals are nonzero in the proposed domain. At D_A at least .5, the smallest fractional shortcut error over D_B in [.5,1.5] exceeds 15%; larger target mobilities give stronger margins. Hidden tests can thus avoid near-zero signals and subtle asymptotic corrections.

The missing physics is the common target trajectory and the survival selection of unusually confined target paths. It is not an incorrect relative diffusivity for one pair. The target-diffusivity input deliberately disappears only from the leading true readout, which should be explained in author notes without calling every public input asymptotically active.

## Archive distinction

The closest author records checked were fixed-trap-memory, survivor-transport, and gated-capture. Fixed-trap-memory concerns repeated visits to static random escape rates and pooled finite-passage variance. Survivor-transport concerns accumulated displacement conditioned on a wall-survival event in a finite cross-section. Gated-capture concerns a common switching reactive surface and mean capture. None has an infinite Poisson gas of mobile traps or a stretched-exponential survival coefficient determined by a target-path/void bound. The general themes of shared noise and conditioning are related, but the apparatus, asymptotic scaling, and required approximation correction are distinct.

Retaining independent relative traps after fitting D_B would be the intended physical error. Deriving the shared-path survival problem but mishandling a bound, factor of two, or fit would require a separate classification. The known simplicity of the final coefficient is not evidence that the target model will fail; empirical difficulty remains to be tested once the package is frozen.
