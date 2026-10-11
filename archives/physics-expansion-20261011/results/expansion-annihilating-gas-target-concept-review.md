# Annihilating gas target: independent concept review

Verdict: approve the physical concept for packaging on the proposed bounded domain. The finite-time mapping, two-speed calibration, and truncation certificate are correct. Interpretation of future failures needs care: the exact generating function uses specialized mathematics, so recognizing the missing collision physics but failing to derive or compute it is not by itself the intended physical failure.

## Exact observable and preparation

[Haslegrave and Tournier, Corollary 3](https://arxiv.org/html/2004.09119v1#S2.SS5) establishes the joint-law independence needed to combine the first-hit index with its renewal distance. The stated quartic agrees with that corollary and with [Theorem 2 of the phase-transition paper](https://arxiv.org/html/1811.08709v3). This is more information than eventual-hit probability alone and is essential for a finite-time observable.

For Poisson density `rho`, the position of particle `n` on one side has a gamma distribution. A first incoming particle keeps speed `v`, so its arrival time is its original position divided by `v`. Thus, with `z=rho*v*t`,

```
q(t) = sum_{n>=1} a_n*gammainc(n,z),
S(t) = (1-q(t))^2,
```

where `a_n` is the nonnegative coefficient of the analytic-at-zero branch of the proposed generating function. Until the tag is hit, the two half-lines do not communicate, so their no-hit probabilities multiply. There is no factor of two in `z`: the two-moving-particle closing speed is already accounted for by the collision probabilities.

Specify a separately inserted stationary tag, with an independent Poisson background on both sides. This is also the Poisson Palm preparation conditioned on the tagged velocity being zero. Continuous Poisson gaps make simultaneous three-body collisions probability zero, so they do not change this observable. A deterministic convention can still be supplied for implementation ties. All particles disappear permanently after annihilation, with no exchange of velocities or new particles.

At `p=0`, the one-sided no-hit probability is `exp(-z)*(I0(z)+I1(z))`; the proposed squared scaled-Bessel expression is therefore correct. It equals one at time zero and decreases strictly with positive `z`, so nonzero-time calibration identifies the common speed. At `p=1`, survival is one. The shortcut is exactly the gas obtained by deleting stationary background particles before evolution, retaining all moving-particle collisions at density `(1-p)*rho`.

## Numerical certification and independent validation

Retain coefficients through degree `N`. Nonnegativity and total mass at most one give

```
0 <= q-q_N <= gammainc(N+1,z),
0 <= (1-q_N)^2-S <= 2*gammainc(N+1,z).
```

For the proposed `rho<=1.2`, `t<=60`, and `v<=1.3`, `z<=93.6`; degree 256 has uniform absolute survival truncation bound `1.3207242e-43`. This certifies the omitted tail, not floating-point arithmetic. The coefficient recurrence contains a subtraction despite its coefficients being probabilities. Check nonnegativity and refinement rather than silently clipping a numerical failure.

The complex-root/Cauchy calculation is a useful independent numerical representation of the same quartic. Track the branch from zero, check polynomial residuals, and vary contour radius and sample count. At radius `r<1` with `M` samples, positive-coefficient aliasing is bounded by `r^M`; coefficient roundoff is amplified by `r^(-n)`. These competing errors should be checked over the retained degrees. This representation alone does not independently validate the microscopic mapping.

The direct event queue and small-system enumeration provide that additional physical check. At horizon `t`, finite propagation speed limits the relevant initial region to `[-v*t,v*t]`; one side can be simulated with an absorbing tag and a Poisson number of particles in `(0,v*t)`. Record statistical uncertainty for Monte Carlo rather than calling agreement exact. For fixed small particle counts, average over all gap permutations and all velocity assignments. Do not replace randomly ordered unequal gaps with an unqualified equally spaced system, which creates triple collisions.

I independently enumerated 58,320 five-particle configurations using two unequal gap sets. For six stationary probabilities, the first three nonzero coefficients agree within `5.56e-17`:

```
a1=(1-p)/2,
a3=(1+2*p)*(1-p)^2/8,
a5=(1-p)^3*(2+7*p+9*p^2)/32.
```

I also checked the prototype's calibration formula against its coefficient sum through `z=93.6` (maximum discrepancy `4.45e-16`) and repeated the complex calculation at 41 values of `p`, using 4096 samples and radius `.975` (maximum coefficient discrepancy `1.91e-15`). These supplement the author's reported microscopic Monte Carlo checks.

## Distinction and failure classification

The omitted effect is the change in collision histories caused by stationary background particles. An individual stationary particle can absorb an incoming threat or remove a moving particle that would otherwise have blocked it; protection is not a pathwise monotonic statement. Already the short-time difference from the deletion shortcut is `p*(1-p)*z^2/2+O(z^3)`, while both share the initial linear loss. This gives a direct physical diagnostic beyond a bare appeal to a quartic equation.

This family is distinct from elastic hard-rod current fluctuations and from noninteracting Brownian-trap survival: particles are irreversibly removed, and the background changes the arrival process through reactions among its own particles. No matching annihilation apparatus appeared in the reviewed task/archive descriptions. The proposed hidden separations are ample at ordinary nonzero survival probabilities.

The exact finite-time theorem is not a routine result every trained physicist should recall. A solver can instead build microscopic dynamics; the apparatus must leave that route open. Count a future trial as a physical failure only when its final model and emitted reasoning support deletion, independent exposure, or another incorrect collision assumption. A correct collision model with insufficient sampling, a failed recurrence, or missing specialized derivation requires a separate numerical or mathematical classification. Preserve every outcome under the fixed-three protocol.
