# Collisionless ballistic echo: concept review

Verdict: the apparatus and rephasing effect are sound, but reject this as a distinct new expansion candidate. It is the zero-friction limit of the archived pulsed kinetic gas, which already tested the same two spatial velocity impulses, finite-amplitude mode mixing, and hidden velocity memory. That candidate passed 3/3 reviewed Luna High trials. Changing the fitted scalar to thermal width and weakening the starter to isolated-pulse superposition does not add a new physical omission. No task or prototype was built for this review.

## Exact finite-kick law

Let the ring length be `2*pi`, initial velocity have standard deviation `s`, and `u=T-tau`. For the zero-phase sine kicks in the proposal, the exact trajectory is

```
x_tau = x0 + v0*tau + A1*tau*sin(k1*x0)
x_T = x0 + v0*T + A1*T*sin(k1*x0)
      + A2*u*sin(k2*x_tau).
```

Positions can be unwrapped in this calculation because every wave number is an integer. Expand the second kick, then average over uniform `x0` and Gaussian `v0`. For `N_m=E exp(i*m*x_T)`, retain integer `n` for which `ell=-(m+n*k2)/k1` is an integer, and sum

```
J_n(m*A2*u) * J_ell(A1*(m*T+n*k2*tau))
* exp[-s^2*(m*T+n*k2*tau)^2/2].
```

All terms are real for the proposed zero-phase preparation. The measured cosine coefficient is this sum; sine coefficients vanish by inversion symmetry. A direct integration of the trajectory over initial position and Gaussian velocity is an independent representation. Its angular and velocity resolutions would need separate refinement because phase-space structure becomes fine at long delays.

With `A2=0`, the answer vanishes unless `m/k1` is an integer, in which case it is

```
J_(-m/k1)(m*A1*T) * exp[-s^2*m^2*T^2/2].
```

Thus nonzero positive-time calibration away from Bessel zeros identifies positive `s`. The kick amplitude has velocity units. State whether the impulse at `tau=0` is applied immediately after the first, and preserve incoming velocity by addition. At `T=tau`, the second kick has no effect on a position readout.

For `m=k2-k1>0`, the principal echo channel is `n=-1, ell=1`. Set `d=m*T-k2*tau`. Its contribution is

```
J_-1(m*A2*(T-tau))*J_1(A1*d)*exp(-s^2*d^2/2).
```

It has a node at the nominal echo time `T=k2*tau/m`, since `J_1(0)=0`. The density echo appears in lobes around that time; in the weak-kick limit this contribution is proportional to `d*exp(-s^2*d^2/2)`. Other channels can leave a small background at the node. Exact-center readouts would therefore be poor generic discriminators. With no self-consistent field, “ballistic kinetic echo” is more precise than “plasma wave echo”; the latter's collective setting is described in [O'Neil and Gould](https://authors.library.caltech.edu/records/qddf6-xjm06).

## Starter and overlap

The sum of isolated finite-kick density responses omits all mixed-pulse terms. It can be used as a formal superposition approximation, but it is not generally the density of a valid auxiliary process: adding two normalized densities and subtracting the uniform background can make a density negative. It also gives identically zero difference-mode predictions whenever neither isolated kick has that harmonic, for example `k1=2,k2=3,m=1`. This is a weaker and potentially less diagnostic starting approximation than the archived control.

The archived [pulsed kinetic gas derivation](../archives/physics-expansion-20261011/tasks/pulsed-kinetic-gas/AUTHOR.md) has response kernel `A(t)=(1-exp(-gamma*t))/gamma`, thermal displacement variance `V(t)`, and adjacent-interval covariance `C`. Taking `gamma` to zero gives `A(t)=t`, `V(t)=s^2*t^2`, and `C=s^2*tau*u`, reducing its exact trajectory and Bessel formula to the expressions above. Its completed shortcut already retained the full nonlinear impulses and exact displacement marginals; it omitted only that covariance. The [recorded outcomes](../archives/physics-expansion-20261011/results/expansion-pulsed-kinetic-gas-r1-results.json) report three physical passes, each explicitly recovering the missing velocity memory.

Single-file memory is less direct overlap: it concerns a tagged Brownian particle, preserved spatial order, and preparation-dependent two-time covariance. That distinction justified the original pulsed-gas proposal, but it cannot make this collisionless simplification a new family. Preserve the existing outcome and pursue a different mechanism instead of packaging this variant.
