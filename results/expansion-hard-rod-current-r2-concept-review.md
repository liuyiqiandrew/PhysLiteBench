# Polydisperse hard-rod current: revision-2 concept review

Reviewed 2026-10-11. **Approved as a substantive revision for prototyping and packaging.** The proposed field equation is correct for the stated ensemble average, and the completed shortcut repairs revision 1 before making a new approximation. No task code or model evaluation was changed or run. Final-source review and independent microscopic controls remain necessary.

## Mechanical and statistical specification

Equal masses exchange velocities on collision even when their lengths differ. Lengths remain attached to the ordered physical rods. If rod n has length a_n, write `x_n=y_n+A_n`, where `A_(n+1)-A_n=(a_n+a_(n+1))/2`. Contact then becomes equality of the neighboring y coordinates. The ordered y coordinates follow the sorted free-particle trajectories. The fixed length sequence is independent of that compressed free gas.

The equal-population ring preparation fixes total occupied length exactly. Taking its thermodynamic limit first makes every finite ordered label block independent fair binary draws and makes free gaps independent exponentials with rate `c=rho/(1-rho*abar)`. This remains an equilibrium mechanical preparation for each fixed order. The two masses must be stated equal; specifying lengths without masses would not establish velocity exchange.

The crucial ensemble condition is that the exponential moment averages over freshly drawn length orders as well as positions and velocity marks, before its logarithm and long-time limit. Lengths stay fixed within a trajectory. A moment conditioned on one typical indexed infinite order at a fixed origin has a different limiting law: macroscopic prefix lengths self-average and recover the mean-length result. Avoid calling the proposed statistic a quenched disorder average merely because the labels do not change during a run.

Do not import a variable-length hard-rod mapping that transports lengths with the free velocity labels. Some mathematical models exchange both lengths and velocities on collision; for example, [Hard Rods and Box-ball Systems](https://epubs.siam.org/doi/10.1137/25M1806867) explicitly adopts that different interaction. The fixed-order compression above supplies the appropriate proof here.

## Contraction and exact formula

For positive current j, let f be the long-rod fraction among the approximately j*t net crossed labels. Its mean length is `b=abar*(1-contrast+2*contrast*f)`. The compressed moving cut has speed b*j. The two free crossing intensities are

```
A = c*(v-b*j)/2,
B = c*(v+b*j)/2.
```

The net-flow rate at that cut is the Poisson contraction `sup_s[s*j-A*(exp(s)-1)-B*(exp(-s)-1)]`. The crossed composition adds `j*D(f||1/2)`, not an extensive cost proportional to all rods in a ballistic interval. Minimizing over f gives

```
min_f [D(f||1/2)+c*b*sinh(s)]
  = c*abar*sinh(s)-logcosh(c*abar*contrast*sinh(s)).
```

Consequently the proposed positive-field equation and generating rate are correct:

```
abs(lambda) = s+c*abar*sinh(s)
                -logcosh(c*abar*contrast*sinh(s)),
F(lambda) = c*v*(cosh(s)-1),  s>=0.
```

The optimal long-rod fraction is `(1-tanh(c*abar*contrast*sinh(s)))/2`, and its mean length is `b_star=abar*(1-contrast*tanh(c*abar*contrast*sinh(s)))`. The field equation has derivative `1+c*b_star*cosh(s)>0`. It therefore has exactly one nonnegative root. The current `j=c*v*sinh(s)/(1+c*b_star*cosh(s))` remains strictly below `v/b_star`, so the two Poisson rates stay positive at every finite-field saddle. Reflection supplies negative fields. The minimum rod length is positive throughout the proposed domain.

With `h=1-rho*abar`, the small-field expansion is

```
F(lambda) = rho*v*h*lambda**2/2
          +rho**3*v*abar**2*contrast**2*h*abs(lambda)**3/2
          +O(lambda**4).
```

Thus the zero mean and variance growth `rho*v*h` are contrast independent. The even SCGF generally is not three-times differentiable at zero. This is an allowed thermodynamic/long-time effect of the random crossed-block size; do not impose higher-cumulant analyticity or infer a failed numerical derivative there. The proposed variance calibration and monodisperse full-SCGF calibration agree exactly with the shortcut and identify the common speed through nonzero responses.

## Independent reference and origin convention

The nested composition/current contraction is a suitable independent analytical implementation. For fixed j and f, its explicit flow cost is

```
j*log((j+sqrt(j*j+4*A*B))/(2*A))
  -sqrt(j*j+4*A*B)+A+B+j*D(f||1/2).
```

Enforce `b*j<v` during composition minimization, then maximize `abs(lambda)*j-I(j)` over `0<=j<v/a_minus`. This does not use the field root. In six reviewer checks spanning packing .04–.72, contrast .25–.75, and field .1–4, that contraction agreed with the root within `3.942e-15` absolute at v=1. Example shortcut errors are 69.1% at `(rho,abar,contrast,lambda)=(.6,1.2,.75,4)`, 29.7% at `(.5,1,.6,3)`, and 10.3% at `(.3,.8,.75,2)`. Weak-density/small-contrast settings need not give a useful margin.

A finite-time microscopic control is particularly useful here. With compressed index zero the last initial free point left of zero and `A_0=0`, the exact rank event is

```
P(J_t>=k | lengths) = P(Q_t(-A_(1-k))>=k),
```

where Q is the independent free Skellam crossing count. For n absolute index steps, the magnitude of A is a sum of n-1 full rod lengths plus half of each of two endpoint lengths. Therefore the finite mixture uses a Binomial(n-1,1/2) interior and a Binomial(2,1/2) endpoint half-weight contribution; n=0 has zero offset. Replacing all n lengths by equal full weights is asymptotically harmless but is not the exact center-current finite-time law. Keep the `k-1` index convention from revision 1.

The anchored construction is not literally the stationary physical-origin preparation. As in revision 1, handle the size-biased origin interval using the Poisson free-gap/Palm construction. The additional endpoint labels only receive bounded local reweighting because both lengths are positive and bounded; their half-length shifts are bounded. This transfers the limiting current rate without assuming that any random order-one change is harmless for exponential moments. The exact half-population constraint vanishes only because the thermodynamic limit is taken before time.

## Revision distinction

Revision 1's shortcut froze current-conditioned propagation rates and dressed charge. Revision 2's proposed shortcut solves the full monodisperse implicit law, so it already repairs that error. Its remaining approximation replaces the random length of the net crossed block by the global mean before taking the exponential moment. Hidden fields favor atypically short crossed rods. This is a new conditional-composition cost, not a change of coefficients or a resample of the old test.

The archived quenched-spin activity candidate also distinguishes typical and conditioned behavior, but its collective field responds dynamically to activity bias in a mean-field spin system. Here the rods cannot reorder and the relevant fluctuation is the initial composition of a contiguous crossed block. Retain the same hard-rod family/revision history and the previous 1/3 outcome; do not present this as an unrelated new family or as already qualified.
