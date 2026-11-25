# Mobile-trap survival, revision 1

The target and traps are independent Brownian particles before killing.
The public experiment averages the initial Poisson gas and every path before
taking the logarithm of survival. Trap diffusivity is strictly positive;
target diffusivity is fixed and finite as time tends to infinity.

## Physical derivation and independent reference

For a fixed target path x, a trap increment path B hits it precisely when
the trap's initial position lies in the range of x-B. Continuity makes that
range an interval. Marked-Poisson thinning therefore gives

    S[x] = exp(-rho * E_B Range(x-B)).

Range is a convex, even functional. Brownian symmetry makes its expectation
convex and even as a function of x. Its minimum is at x=0, proving
S[x]<=S[0]. Averaging over the target preserves this inequality. The
reflection principle gives the stationary-target log survival by integrating
traps' first-passage probabilities on either side:

    -log S[0] = 2*rho*integral_0^infinity erfc(y/sqrt(4*D_B*t)) dy
              = 4*rho*sqrt(D_B*t/pi).

For a lower survival bound, require an initially empty interval [-ell,ell],
no exterior trap entering it, and confinement of the target inside. These
independent events have product probability at least

    exp(-2*rho*ell - 4*rho*sqrt(D_B*t/pi) - pi^2*D_A*t/(4*ell^2)).

The last factor is rigorous: the positive Dirichlet ground-state martingale
has surviving expectation exp(-pi^2*D_A*t/(4*ell^2)), and its cosine is at
most one. The first term of the alternating survival series is not used as
a lower bound. Optimizing ell=(pi^2*D_A*t/(4*rho))^(1/3) gives

    0 <= -log S(t)/sqrt(t) - C
      <= 3*(pi^2*rho^2*D_A/4)^(1/3)*t^(-1/6),
    C = 4*rho*sqrt(D_B/pi).

For D_A=0 the stationary-target expression is exact. The squeeze proves
independence of the leading coefficient from D_A, not independence of
finite-time survival. The slow convergence makes direct simulation a poor
way to measure this asymptotic coefficient.

The private reference integrates the erfc first-passage law by quadrature,
then minimizes the empty-interval/confinement cost in log width. It does
not copy the oracle's closed coefficient. The validator checks the squeeze
at five times from 1 to 10^36 in log probability, never treating underflowed
survival probabilities as data. It also checks 80 independent spectral
confinement probabilities against the ground-state bound.

The leading result appears in [Bray and Blythe](https://arxiv.org/abs/cond-mat/0205381)
and [Blythe and Bray](https://arxiv.org/abs/cond-mat/0209353). The latter did
not prove the Pascal inequality it used. The expected-range argument above
supplies a direct proof for this continuum one-dimensional apparatus. The
independent concept review records later related trapping-principle papers.

## Calibration loophole

One target-trap relative coordinate has diffusivity D_A+D_B. Two such
relative coordinates share target noise, with increment covariance
2*D_A*dt. The shortcut treats all relative paths as independent and applies
Poisson thinning after this replacement. Its coefficient is
4*rho*sqrt((D_A+D_B)/pi), exact for independent traps with that diffusivity
and an immobile target. It is exact for every calibration experiment, where
D_A=0. It fails for mobile targets because it removes the shared path and
the selection of unusually confined target paths among survivors.

Calibration has 256 independent readings at 32 densities, repeated eight
times. The true D_B is 1.1 mm^2/s; sigma is .001 s^-1/2, data seed 611801.
A positive linear amplitude sqrt(D_B) is uniquely identified. Both completed
controls use the same weighted least-squares fit. The 256 noise draws use
seed 611809. The group relative-norm prediction tolerance is 4%, parameter
tolerance 3%, and reduced chi-square tolerance 1.5. All readouts are
nonzero. The weakest mobile shortcut group is wrong by over 26%, while
noise changes the oracle prediction by under .013%.

Run `scripts/validate_mobile_trap_survival.py`. It reads frozen data;
`--generate` intentionally replaces both calibration copies and hidden
truth. The report is `results/expansion-mobile-trap-survival-r1-validation.json`.
It verifies quadrature refinement, 104 domain/corner cases, confinement
bounds, density and Brownian scaling, and 256 noise realizations.

After independent source review and fresh Docker controls, run exactly
three frozen unhinted gpt-5.6-luna High trials. Inspect final source and
trajectories. Keeping independent relative traps after a successful fit is
a physical failure. A correct collective argument followed by a wrong
factor, asymptotic limit, or implementation requires separate classification.
Preserve every outcome; never resample an unchanged candidate.
