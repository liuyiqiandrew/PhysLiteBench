# Finite-ion logarithmic cloud: backup concept review

**The Gibbs apparatus and finite-N pressure law are sound. Approve the refined Hartree control as a concept; reject the original continuous integer-sector relaxation as the benchmark control.** No package, numerical reference or model trial was reviewed. This backup should not displace an ongoing promotion review.

The proposed Hamiltonian fixes a genuine two-dimensional logarithmic Coulomb gas, with central charge `+Nq`, `N` negative point charges, hard inner radius `a>0` and uncharged hard outer radius `R`. State that it is a full canonical Gibbs ensemble, with finite `N` held fixed before `R/a -> infinity`; it is not the three-dimensional cylindrical PB apparatus. The hard core removes central collapse, and repulsion between the same-sign point ions makes coincidences integrable. Specify the complete allowed multiplier range and define pressure by the area derivative at fixed `N,a,Gamma`.

With `k` ions at outer scale R, the entropy contributes `2k*log(R/a)`. Central, bound/free and free/free interactions together contribute `Gamma*k*(k+1)*log(R/a)/2`. Thus

```text
A_k = 2*k-Gamma*k*(k+1)/2,
phi = max_(k=0,...,N) A_k/(2*N).
```

The maximizing free count is `min(N,floor(2/Gamma))`, with adjacent-sector ties at `Gamma=2/j`. Evaluate the finite maximum directly to avoid convention-dependent bound-count labels at a tie. Pressure is continuous, although its slope changes; `phi=0` for `Gamma>=2`. This agrees with the exact outer-wall contact result in [Mallarino, Tellez and Trizac, *The contact theorem for charged fluids*, equations 37–40](https://arxiv.org/pdf/1410.6499). [Burak and Orland](https://arxiv.org/pdf/cond-mat/0509126) and [Varghese, Vemparala and Rajesh](https://www.imsc.res.in/~rrajesh/Publications/pe_inequivalence.pdf) give the related finite-ion condensation structure. Coupling conventions matter: their Manning parameter corresponds to `N*Gamma/2` here.

The original proposal maximized the exact integer-sector exponent at arbitrary real k. That is a mathematically well-defined relaxation but a weak physical control: it can fail already at N=1, with no inter-ion correlations to neglect. An observed repair could be only restoration of an integer optimization constraint. Preserve this rejected version as concept history, without treating its failures as clean evidence of the intended physical mistake.

The refined control follows instead from a normalized independent-particle trial distribution. Let each ion independently occupy the outer cloud with probability `p=n/N`. Then `K` is binomial and

```text
E[K(K+1)] = (1-1/N)*n²+2*n,
A_H(n) = (2-Gamma)*n-Gamma*(1-1/N)*n²/2,
phi_H = max_(0<=n<=N) A_H(n)/(2*N).
```

For N>1, `n*=clip((2-Gamma)/(Gamma*(1-1/N)),0,N)`; N=1 uses the exact one-particle answer. I independently obtained this functional by averaging the actual pair Hamiltonian in a product measure. It retains all finite-N pair counting, while missing correlations in which ions occupy the bound and free populations. In the exact dominant sector away from ties, permutation symmetry gives each ion a fractional free probability but their indicators are anticorrelated so that the total free count is fixed. The Hartree state incorrectly makes that total binomial. This is a clear physical factorization error, rather than integer bookkeeping alone.

Both models give `1-Gamma*(N+1)/4` while all ions are free and share the first threshold `Gamma=2/N`. Calibration at `u<=.3`, `g<=1.4`, `N<=4` has `Gamma<=.42<.5`, so it is exactly shared and linearly identifies g. Both also give the exact N=1 curve and vanish when `Gamma>=2`. In the one-free-ion regime, the exact positive pressure is `(2-Gamma)/(2*N)`, while the Hartree value is `(2-Gamma)²/[4*Gamma*(N-1)]`. At Gamma=1.8 the relative errors are 88.89%, 91.67% and 92.59% for N=2,3,4. Use positive-pressure cases bounded away from Gamma=2, since zero truth and strong parameter sensitivity complicate relative grading unnecessarily.

For validation, a sector maximum implemented twice is insufficient. The ordered-radius partition integral supplies another route: angular pair-distance factors are bounded above, and bounded below on separated angular sectors, by constants times the larger radius. Constants independent of R leave the leading partition exponent unchanged. A finite ordered-radial convolution or direct correlated Gibbs integration can check this route, supplemented by the exact contact identity and the elementary N=1 integral. Verify finite-size refinement, ties and the hard-wall pressure normalization before packaging. Do not call ordinary PB an exact reference for this finite-N Gibbs task.

This shares counterion-condensation context with counterion-cloud r1/r2 but changes the governing ensemble and omission: correlated finite particle counts replace mean-field species selection. It is a defensible new scientific setup if the refined control is used and later failures are confirmed to retain product-state localization. No claim of agent difficulty is established by this concept review.
