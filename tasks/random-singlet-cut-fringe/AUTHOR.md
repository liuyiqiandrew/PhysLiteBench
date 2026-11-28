# Random-singlet cut fringe

A one-dimensional antiferromagnet with broad random exchange flows to the random-singlet regime. The spectrum across a cut has correlated singlet formation scales. This task treats the local strong-disorder flow as its specified scaling model and adds an idealized energy-selective Ramsey probe and matched-disorder herald. For the spin-chain background and the waiting-time statistics, see [Fagotti, Calabrese and Moore, Phys. Rev. B 83, 045110 (2011)](https://doi.org/10.1103/PhysRevB.83.045110).

## Marked-bond flow

Let Q(x,μ) denote a weighted population of still-active marked bonds. Drift gives ∂ₓ[(1+x)Q]. Adjacent decimations at total rate 2 give 2(Q∗exp(−x)−Q), where the convolution is on x≥0. Crossings occur with boundary flux Q(0,μ) and inject the new density x exp(−x). The coefficient may carry an optical and counting weight z. The density has the form (a+b x)exp(−x). Its evolution within a band for a scalar crossing weight is

    da/dμ = −2a+b
    db/dμ = a−b+za.

In the full calculation the last term arrives from the opposite sublattice and from the appropriate previous click count. A boundary prepared after a crossing has a=0,b=1. Stationarity at z=1 gives a=1/3,b=2/3. This also shows the bulk mean formation rate is 1/3. For the waiting law set z=0 after a fresh crossing. The loss density is a=(exp(−r₋μ)−exp(−r₊μ))/√5, with r₋=(3−√5)/2 and r₊=(3+√5)/2. Since r₋r₊=1, it is the waiting time for two sequential exponential stages. A random log-scale opening weights those stages by their respective mean residence times.

Consecutive crossing singlets nest. Between two consecutive left endpoints every intermediate original site belongs to a pair on the left, so the difference in site index is odd. The next crossing therefore reverses its left sublattice sign. For an individual crossing let m=⟨ψ₀|ψ₁⟩ and let p be the herald click probability. In a given disorder realization the two copies have the same crossing scales and sublattices. The coefficient of yᵏ in the average product of m(1−p+py) is the unnormalized arm coherence for k clicks. Setting every m to one gives the retained fraction. Dividing the two sums before taking the Ramsey quadrature gives the retained visibility.

## Independent checks

The oracle propagates the two sequential waiting stages, sublattice and click count using a matrix exponential per band. The private reference integrates the continuum bond-density coefficients directly and uses an independent spinor representation of the singlet. It keeps separate normalization and coherence channels. The public worked scan checks the integrated first-crossing law and the forward recurrence time at a stationary opening. Splitting a band at an arbitrary interior point, spin rotations that leave the singlet invariant and the exactly counted herald provide additional physical checks.

There are 12 calibration shots and 24 held-out scans across unconditional, zero-click and positive-click families. A Poisson crossing control preserves the mean rate and the shared herald but loses the boundary regeneration physics. An independent-copy control keeps the true crossing law and sublattice alternation while giving the herald different disorder. The latter passes the unconditional family and fails both conditioned families. Run `OPENBLAS_NUM_THREADS=1 python scripts/validate_random_singlet_cut_fringe.py` from the repository root and add `--generate` to write the stored data. The native solve and verifier allowances are 1,800 and 180 seconds.
