# Hard-rod current: concept review

Independent review, 2026-10-11. **Approved for packaging, subject to final source review, numerical validation, and complete controls.** The proposed formulas are correct for the annealed stationary positional Tonks preparation and the stated thermodynamic-before-long-time limit. Difficulty remains unmeasured.

## Mapping and current statistics

Label ordered centers and compress coordinates by `y_n=x_n-a*n`. Equal-mass elastic collisions exchange velocities, so the compressed configuration evolves by free marked trajectories followed by sorting. The free-point density is `c=rho/(1-a*rho)`. A physical signed current `J_t≈j*t` moves the corresponding compressed cut to `a*j*t`, up to a nonextensive indexing correction.

For `abs(a*j)<v`, the free right/left crossing counts at that cut are independent Poisson variables with rates

```
r_plus(j) = c*p*(v-a*j),
r_minus(j) = c*(1-p)*(v+a*j).
```

This fixes both mapping signs. The required difference of the two counts is j. Their relative-entropy contraction therefore gives exactly the proposed rate function

```
I(j) = sup_s {s*j-r_plus(j)*(exp(s)-1)
                  -r_minus(j)*(exp(-s)-1)}.
```

Taking the current Legendre transform gives

```
lambda = s + a*c*[p*(exp(s)-1)-(1-p)*(exp(-s)-1)],
F(lambda) = c*v*[p*(exp(s)-1)+(1-p)*(exp(-s)-1)].
```

The lambda map has derivative `1+a*c*(p*exp(s)+(1-p)*exp(-s))>0`, so the solution is unique globally, including the full proposed bias range. Its saddle current is

```
j = c*v*[p*exp(s)-(1-p)*exp(-s)]
    /[1+a*c*(p*exp(s)+(1-p)*exp(-s))].
```

For positive a this always lies strictly between `-v/a` and `v/a`; no hidden branch switch or endpoint saddle is omitted. At a=0 the ordinary bidirectional Poisson SCGF is recovered. Differentiation at lambda=0 gives exactly `rho*v*(2*p-1)` and `rho*v*h*(1-a*rho*(2*p-1)**2)`, with `h=1-a*rho`.

## Initial ensemble and boundary issue

The averaging must redraw both initial positions and independent velocity marks between realizations. The positional law is equilibrium Tonks, while the bimodal velocities define a stationary, generally current-carrying preparation rather than a Maxwell thermal ensemble. Explicitly retain the thermodynamic limit before the long-time limit; a finite periodic ring measured indefinitely is a different ensemble problem.

A generic random quantity described only as O(1) is not automatically harmless for a speed-t large-deviation principle. Here the stationary Tonks boundary can be handled explicitly. A center spacing is `S=a+E`, with E exponential of rate c. The gap containing a uniform physical origin is size-biased. Split the origin's position in this gap into a length-a part and the remaining free part. Their probabilities are `a*rho` and h. After compression these become, respectively, a Palm Poisson configuration with one distinguished point and an ordinary stationary Poisson configuration, with only a bounded coordinate shift of at most a. The extra point changes counts by a bounded amount; bounded-window Poisson counts have superexponentially small probabilities of becoming order t. These boundary conventions therefore preserve the stated speed-t current LDP and SCGF. Rare long free gaps are already included in the Poisson spatial ensemble.

## Shortcut and calibration

The centered compound-Poisson shortcut is a defensible effective fluctuation closure. Its event intensities are the ordinary quasiparticle crossing intensities `rho*p*abs(v_eff_plus)` and `rho*(1-p)*abs(v_eff_minus)`; its dressed fluctuation charge is h. Holding these at the unbiased state gives the proposed centered shot noise and reproduces both mean and variance exactly. All rates remain positive on the disclosed domain.

Describe it as a **real-valued effective current with fractional dressed jumps and a deterministic background**, not as an exact integer center-crossing counter. Equivalently,

```
J_short(t) = j_mean*t + h*[(N_plus-r_plus*t)-(N_minus-r_minus*t)].
```

This is one valid process for every preparation, with the advertised SCGF. The physical mistake is extending the fixed-state independent-event closure to finite current bias; the compressed cut and crossing intensities actually depend on the conditioned current. At a=0 it becomes the correct uncentered integer bidirectional Poisson process. Calibration of interacting means/variances plus point-particle full SCGFs is therefore exact. Every observable scales linearly with the sole unknown speed, and nonzero variance readings identify it uniquely even when p=1/2 gives zero mean.

SCGFs can be negative for a counting field opposed to the mean, and they have zeroes. Use group norms with nonzero reference norm rather than individual relative errors near zero. Do not impose rate-like nonnegativity on F itself.

## Independent reference and prototype checks

The relative-entropy contraction is a sound independent numerical route: minimize `D(u_plus||r_plus(j))+D(u_minus||r_minus(j))` subject to `u_plus-u_minus=j`, then maximize `lambda*j-I(j)`. It avoids the oracle's implicit counting-field equation. Supplementing this equivalent asymptotic derivation with finite-time microscopic counting is valuable.

For the prototype convention `x_n=y_n+a*n`, with label zero the last initial free point left of zero, the exact tail identity is

```
P(J_t >= k) = P(Q_t(a*(k-1)) >= k).
```

At a test coordinate z, Q is a difference of independent Poissons with means

```
c*[p*max(v*t-z,0)+(1-p)*max(-v*t-z,0)],
c*[p*max(z-v*t,0)+(1-p)*max(z+v*t,0)].
```

The difference of adjacent current tails gives the full finite-time current law for this compressed-point origin convention. The boundary argument above relates its long-time SCGF to stationary Tonks sampling. This construction independently checks the physical rank mapping, rather than only rearranging the saddle equation.

I ran the current prototype at `(rho,a,p,lambda)=(.5,1,.5,2),(.6,1.2,.3,-4),(.3,.8,.7,4)`. The implicit and entropy-contraction values agreed within `9e-16`. Finite-time log-MGFs at t=16,32,64,128 showed the expected boundary contribution; `2*F_128-F_64` agreed with the asymptotic result within `5e-16` in these cases, without warnings. This does not establish exact 1/t corrections throughout the domain. Before freezing, refine the Skellam summation tails and time windows, check moment derivatives, zero bias, reflection `(p,lambda)->(1-p,-lambda)`, speed scaling, positivity of variance, and the point-particle limit. Preserve honest absolute tolerances near SCGF zeroes.

## Relation to prior tasks

Single-file-memory uses a fixed Brownian lattice and tagged two-time covariance. The new apparatus uses stationary random positions, deterministic ballistic dynamics, finite rod length, and higher cumulants of a fixed-location current. Ring-current-fluctuations tests a spatially nonuniform conditioned profile in a driven diffusive exclusion system; this hard-rod result remains a homogeneous ballistic contraction with no profile instability. The shared broad themes of rank conservation and conditioned fluctuations do not make the present physical omission a relabeling.

[Myers, Bhaseen, Harris and Doyon, Transport fluctuations in integrable models out of equilibrium](https://arxiv.org/abs/1812.02082) supplies primary context for ballistic current statistics and hard-rod tests. The elementary two-velocity mapping and the specialized formulas above are derived directly for this proposed apparatus.

Reviewed prototype SHA-256: `2656cd8f855443fee69cfbc753ff3408dcb1e1b02e72c93ce44354ddbbcb03fe`.
