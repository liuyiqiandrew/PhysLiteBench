# Expansion concept review

Author-only review, 2026-10-11. This is a design review, not evidence of model difficulty. Target evaluation: three fresh, frozen, unhinted `gpt-6-luna` trials at high effort per accepted revision; all three failures must be reviewed as physical mistakes. I inspected the current roster, adding-task rules, candidate archive/stage names, and the closest geometric-rotor and collisionless-screening apparatuses and author notes.

## 1. Prioritize a two-kick kinetic echo

Use independent classical particles on a ring of circumference `2*pi`, with

    dx = v dt
    dv = -gamma*v dt + sqrt(2*gamma)*s dW.

Position is initially uniform, velocity is independently Gaussian with variance `s**2`, and the bath remains attached throughout. At times zero and `tau`, instantaneous impulses change velocity by `a1*sin(k1*x+phi1)` and `a2*sin(k2*x+phi2)`. Measure the complex density coefficient `N_m = E[exp(-i*m*x)]` after a further delay `u`. Supply both real quadratures, or specify a fixed real detector phase unambiguously. All wavenumbers are integers; kick amplitudes have velocity units, and `gamma` has inverse-time units. The common unknown is positive `gamma`; the known thermal speed is `s`. No self-consistent electric field is present, so call this a kinetic echo rather than a full plasma echo.

For `A(t)=(1-exp(-gamma*t))/gamma`, the exact trajectory can be written

    x_tau = x0 + a1*A(tau)*sin(k1*x0+phi1) + Y
    x_final = x0 + a1*A(tau+u)*sin(k1*x0+phi1)
              + Y + Z + a2*A(u)*sin(k2*x_tau+phi2).

The thermal displacement increments are jointly Gaussian, with

    Var(Y) = V(tau), Var(Z) = V(u)
    V(t) = 2*s**2*(gamma*t-1+exp(-gamma*t))/gamma**2
    Cov(Y,Z) = D = s**2*A(tau)*A(u).

A completed shortcut keeps this full nonlinear trajectory and both correct displacement marginals but renews the thermal velocity at the second impulse, setting `D=0`. It preserves deterministic relaxation from the first kick. This is a definite physical closure, and its predictions are generally nonzero. The actual apparatus specifies that the velocity and bath trajectory continue through the impulse.

**Exact calibration loophole.** Use second-pulse-only calibration (`a1=0`), with nonzero `a2` and `m=k2`. The initial ring position is uniform, so the pre-pulse thermal displacement drops out. Both models predict

    N_k2 = -J1(k2*a2*A(u))*exp(-k2**2*V(u)/2)*exp(i*phi2).

Vary moderate readout delays and amplitudes to identify `gamma`, avoiding Bessel zeros. Its two effects on kick displacement and thermal dephasing compete; establish uniqueness on the allowed interval numerically. Do not assume monotonicity at arbitrary delays. First-pulse-only calibration with a positive intermediate delay would already distinguish the proposed shortcut and cannot serve as its exact calibration subset.

**Oracle.** Expanding only the two sinusoidal kicks gives a rapidly convergent Bessel sum. For each integer `l` for which `r=(m-l*k2)/k1` is an integer, add

    J_l(-m*a2*A(u))
    * J_r(a1*(l*k2*A(tau)-m*A(tau+u)))
    * exp(i*(l*phi2+r*phi1))
    * exp(-0.5*((l*k2-m)**2*V(tau) + m**2*V(u)
                - 2*m*(l*k2-m)*D)).

**Independent reference.** Integrate the displayed trajectory over uniform `x0` and Gaussian `Y`, analytically averaging the independent residual in `Z|Y`. This replaces `Y+Z` by `(1+D/V(tau))*Y` and multiplies the result by `exp[-m**2*(V(u)-D**2/V(tau))/2]`. Handle `tau=0` directly. This method uses no Bessel expansion. Check convergence in ring and Gaussian quadrature separately, normalization at `m=0`, conjugation under `m -> -m`, one-pulse limits, and the stationary OU covariance.

An independent numerical probe at `gamma=.25`, `s=.8`, `tau=u=2`, `a1=.7`, `a2=.9`, `k1=1`, `k2=2`, `m=1`, `phi1=.3`, `phi2=.8` gave

    exact:    0.0485118900020 + 0.0281734699004 i
    shortcut: -0.0235151122093 - 0.0045668380961 i.

Direct two-dimensional Gaussian trajectory quadrature agreed within `3.57e-9` at order 240; orders 60 and 120 were inadequate for tight validation (`9.31e-4` and `7.66e-6` errors). These are concept checks, not final domain validation.

The distinction from archived collisionless-screening is nonlinear mixing of retained velocity information after a finite preparation, rather than the causal prescription for linear susceptibility. The overlap with single-file-memory is retained two-time information; the new mechanism is velocity-dependent phase mixing converted back to density by a spatial impulse. The echo motivation is supported by [Gould, O'Neil and Malmberg's original plasma-echo paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.19.219); the OU apparatus and formulas above are derived here.

An agent that explicitly renews the unresolved velocity has made the intended physical mistake. An agent that recognizes the continuous velocity process but miscomputes a covariance, Bessel index, or phase has made a mathematical or implementation mistake. Inspect this distinction before counting failures.

## 2. Reserve a periodically driven jump ring

A minimal alternate is one particle on three states with a stated clockwise order. Each phase of a repeated protocol supplies site energies `E_i`, bond barriers `B_i`, duration, and a common affinity `F`, all in units with thermal energy one. Specify continuous-time rates exactly:

    i -> i+1: kappa*exp(E_i-B_i+F/6)
    i+1 -> i: kappa*exp(E_(i+1)-B_i-F/6).

Let each clockwise hop add `1/3` and each counterclockwise hop subtract `1/3` to a passive counter. Prepare the unique periodic stationary state and observe uninterrupted dynamics for many complete cycles. Request `psi(chi)=lim_N log E[exp(chi*Q_N)]/N`, with a real, nonzero counting field. The common unknown `kappa` is an attempt frequency. State that phase changes and cycle boundaries do not reset the state or counter.

For a column generator `L_chi`, multiply only off-diagonal jump rates by `exp(+/-chi/3)`; keep the physical escape diagonals. The cycle propagator is the time-ordered product `M_chi` of phase matrix exponentials. The correct result is `log rho(M_chi)`. A complete independent-cycle shortcut computes the exact single-cycle law starting from periodic stationary `p*`, then reports `log(1^T M_chi p*)`. This omits correlations between cycles while retaining every phase, rate, count sign, and single-cycle fluctuation. Its predictions are nonzero and its average current is correct even in hidden protocols.

**Exact calibration loophole.** Use spatially uniform energies and barriers within each phase. Clockwise and counterclockwise rates then do not depend on the occupied state. Both models give the same Skellam cumulant-generating function,

    psi(chi) = sum_phases duration *
        [a*(exp(chi/3)-1)+b*(exp(-chi/3)-1)],

which identifies `kappa`. Hidden heterogeneous protocols expose cycle correlations. Keep substantial signal and inspect absolute as well as relative gaps; symmetric rotating protocols can produce disappointingly small separation. A preliminary random three-phase probe found examples with about 25% separation and `psi` around `.03`, but no final domain is selected.

Use uniformization and count-resolved probability propagation as the independent reference, extrapolating many-cycle growth, rather than duplicating the Perron-eigenvalue oracle. Also check the exact homogeneous formula, cycle-origin invariance of the asymptotic answer, and invariance to shifting the counted bond by a bounded endpoint displacement.

This is distinct from the qualified interacting exclusion ring's fluctuating density profile and archived stationary terminal-current noise. It is related to periodically driven counting statistics, whose geometric term is established in [Sinitsyn and Nemenman](https://arxiv.org/abs/q-bio/0612018). The proposed failure is cycle independence; do not claim it isolates the adiabatic geometric term. The matrix size is tiny, so difficulty remains uncertain and should be established only by fresh trials.

## 3. Deprioritize another non-Abelian transport task

Geometric-rotor revision 2 already used a four-state Hamiltonian with a degenerate low-energy doublet, noncommuting transport, and a local geometric-energy shortcut. Its single failed trial was excluded: the agent derived the right projected model, then lost a factor of two during implementation. Revision 1 passed 3/3. See its [archive diagnosis](../archives/geometric-rotor-r2/README.md) and [derivation](../archives/geometric-rotor-r2/tasks/geometric-rotor/AUTHOR.md).

A different time-dependent apparatus could be scientifically valid, but changing the geometry alone would not supply evidence of a fresh physical failure mechanism. Prefer the echo first, then the jump-ring concept. No third candidate is approved by this review yet.
