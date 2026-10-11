# Driven target among mobile traps: revision-2 concept review

Verdict: approve for prototyping and packaging. The proposed exponential decay rate is correct, including its independence of every fixed finite trap diffusivity. The drive, target fluctuation law, and observable change the problem substantially from the archived zero-drift revision. No model outcome is assumed here.

Write `u=mu*F`, `D=mu*T`, and let the target start at zero. For `T>0`, the result is

```
K = inf_w { (w-u)^2/(4D) + rho*abs(w) }
  = mu*F^2/(4T),                  F <= 2*rho*T,
  = mu*(rho*F-rho^2*T),           F >= 2*rho*T.
```

The optimal surviving drift is `w=max(0,mu*(F-2*rho*T))`. At `T=0`, the target is deterministic and `K=rho*mu*F`. At `F=0`, `K=0`, including when `T>0`: stretched-exponential survival remains possible but has zero exponential rate. The branches and first derivative match at the threshold.

## Matching path bounds

For a fixed continuous target path `x`, the marked Poisson preparation gives the exact conditional survival

```
S[x] = exp(-rho * E_B range(x-B)),
```

where `B` is one trap's displacement on `[0,t]`. For each trap path, the forbidden initial positions form the continuous image of `x-B`, whose length is its range. This identity requires averaging the initial Poisson cloud and trap paths before the logarithm, as proposed.

Range is convex. Since `E B=0`, Jensen's inequality gives `E_B range(x-B)>=range(x)>=abs(x_t)`. Thus

```
S(t) <= E exp(-rho*abs(X_t)).
```

The endpoint is Gaussian with mean `u*t` and variance `2D*t`. Its Laplace principle gives the displayed infimum as a lower bound on the survival decay rate. This argument allows all target trajectories; it does not assume that the target follows a straight line.

For the reverse bound, require the target to stay within distance `ell=t^alpha`, `0<alpha<1/2`, of a deterministic line `w*s`, and require all traps to avoid that entire moving tube. The latter probability is exactly

```
exp(-rho*(2*ell + E range(w*s-B_s))).
```

The two events depend on independent random objects. Changing target drift from `u` to `w` costs `(u-w)^2*t/(4D)`, with an endpoint correction bounded by `abs(u-w)*ell/(2D)`. Under drift `w`, confinement has probability at least `exp(-pi^2*D*t/(4*ell^2))`; this follows directly from the positive cosine eigenfunction of the absorbing interval. Finally,

```
abs(w)*t <= E range(w*s-B_s)
         <= abs(w)*t + 4*sqrt(D_B*t/pi).
```

All extra terms are sublinear in time. Optimizing over `w` proves the matching bound, including `w=0`. There is no uncontrolled assumption about target-path correlations or a typical displacement. For a deterministic target (`T=0`), the same expected-range bounds prove the calibration law directly.

## Independent numerical reference

A finite-time squeeze is a suitable independent route. Evaluate the Gaussian endpoint expectation by quadrature or a stable log-normal-CDF expression. Independently integrate the drifted-Brownian maximum hitting tails for drifts `w` and `-w` to obtain the moving-tube expected range. Numerical minimization over `w` and `ell` then gives the upper rate bound

```
rho*(2*ell + E range(w*s-B_s))/t
+ (u-w)^2/(4D) + abs(u-w)*ell/(2D*t)
+ pi^2*D/(4*ell^2).
```

The lower rate bound is `-log(E exp(-rho*abs(X_t)))/t`. These bounds can be refined without inserting the piecewise limiting rate into the reference. Include both phases, the threshold, several fixed trap diffusivities, and separate exact checks at `T=0` and `F=0`. Handle the deterministic branch explicitly rather than divide by zero in the finite-temperature bounds. Direct rare-survival simulation is unnecessary.

## Calibration, shortcut, and scope

Positive-force, zero-temperature calibration determines the common mobility uniquely. The completed shortcut `K_short=rho*mu*F` is the exact rate of the auxiliary deterministic target following its unconditioned drift, with the same mobile traps. It is finite and nonzero for positive force and exactly calibrated. It misses the exponentially relevant probability of slower target motion. Positive-temperature hidden cases with `F/(rho*T)=1,2,3,4` have respectively 300%, 100%, 50%, and 33.3% relative separation, so large robust errors do not require zero true rates or singular inputs.

Specify that `T` is the target's noise temperature with `k_B=1`, while the trap diffusivity is independently prescribed. A common bath should not be implied when traps still diffuse during `T=0` calibration. Keep all physical parameters fixed before taking the long-time limit, and retain the fresh Poisson/path annealed preparation.

The closest overlap is revision 1's conditional expected-range machinery. That revision asks for a square-root-time coefficient at zero drift and tests shared-target correlations. This revision asks for a time-extensive rate in a driven thermal model and requires optimizing the rare target drift, with a nontrivial threshold. Archive the 3/3-passing revision unchanged and record this as revision 2. The standard zero-drift results of [Bray and Blythe](https://arxiv.org/abs/cond-mat/0205381) and their [fuller bounds paper](https://arxiv.org/abs/cond-mat/0209353) are context; the driven formula here is justified by the direct bounds above, rather than attributed to those papers.
