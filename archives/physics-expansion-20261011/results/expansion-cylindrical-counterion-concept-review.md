# Cylindrical counterion cell: concept review

**Verdict: approve the physics concept.** It is a distinct equilibrium condensation problem with an identifiable parameter and a complete pressure closure that fits calibration. No task package or model trial was reviewed. The Manning limit is well known, so physical soundness does not establish difficulty.

Put `lambda = u*tau0` (counterions per unit rod length) and `xi = lB*lambda`. Fix a positive rod radius `a`, homogeneous dielectric, temperature and line charge; exclude dielectric images, salt and specific adsorption. The uncharged outer wall confines ions. Specify the translationally invariant, cylindrically symmetric canonical PB equilibrium at each finite `R`, with no exchange of counterions with a reservoir. Take infinite rod length before `R/a -> infinity`, holding `a`, `lambda` and `lB` fixed. The measured quantity is the limit of the finite-cell density ratio, not a normalized equilibrium distribution imposed directly on an infinite transverse domain. State the length units of `tau0` and `lB`.

With dimensionless potential `psi=e*Phi/(kB*T)` and gauge `psi(R)=0`, the apparatus gives

```text
n(r) = n(R) exp(psi(r)),
psi'' + psi'/r = 4*pi*lB*n(r),
a*psi'(a) = -2*xi,  R*psi'(R) = 0,
2*pi*integral_a^R r*n(r) dr = lambda.
```

Thus the proposed accessible-volume mean `lambda/[pi*(R²-a²)]` is correct. The outer-wall osmotic pressure is `kB*T*n(R)`. The positive-rod convention, finite-cell PB solution, supercritical contact density and dilution law agree with [Deserno and Holm, *Cell model and Poisson-Boltzmann theory: A brief introduction*, Sections 2 and 4](https://arxiv.org/pdf/cond-mat/0112096).

An independent derivation also gives a useful interpretation of the shortcut. Let `x=log(r/a)`, `Delta=log(R/a)` and `z=log(2*pi*lB*r²*n(r))`. Then `z''=2 exp(z)`, with endpoint slopes `2(1-xi)` and `2`. Its first integral gives the exact finite-cell identity

```text
phi_R = (1-exp(-2*Delta)) * [1-xi/2 + pi*a²*n(a)/lambda].
```

Below and at `xi=1`, the inner contact contribution tends to zero. Above it, `a²*n(a) -> (xi-1)²/(2*pi*lB)`. Therefore

```text
phi = 1-xi/2             (xi <= 1),
phi = 1/(2*xi)           (xi >= 1).
```

The completed shortcut can consistently implement the **all-counterions-dilute-away pressure closure**, which drops the inner contact term. It stays positive over the proposed domain. Do not describe it as an exact, normalizable supercritical PB solution; that branch does not exist. The omitted physical contribution is the nonvanishing condensed layer. The true pressure lies above the continued linear law, so a generic claim that condensation lowers pressure relative to that continuation would be wrong.

Calibration with `u<=.4` has `xi<=.7` for every allowed parameter, hence exact agreement and the globally identifiable linear response `phi=1-lB*u*tau0/2`. For supercritical hidden cases, relative shortcut error is exactly `(xi-1)²`: 6.25% at 1.25, 12.25% at 1.35, and 36% at 1.6. Prefer groups above about 1.35; 1.25 leaves limited margin above a 4% gate. At true `tau0=1.3`, the public maximum is `xi=1.625`, giving 39.06% separation. The full public maximum 1.75 leaves shortcut pressure positive at .125.

For a finite-cell reference, write `C=z'²/4-exp(z)`. It is determined by

```text
Delta = integral_(1-xi)^1 du/(u²-C),
phi_R = (1-exp(-2*Delta))*(1-C)/(2*xi).
```

Choose `C<(1-xi)²` below threshold, and `C<0` when the integration interval crosses zero. This supplies a finite-cell route without using the limiting answer. A separate collocation BVP for `z''=2 exp(z)` checks its physical boundary conditions. I ran ten such checks at `xi=.3,.8,1,1.4,1.75` and `Delta=4,12`: BVP/first-integral pressure differed by at most `6.45e-13`; the contact identity residual was below `3.86e-13`.

Extrapolation requires care. For `xi>1`, set `C=-gamma²`; then `gamma*Delta=atan(1/gamma)+atan((xi-1)/gamma)`, so the relative finite-cell error is bounded above by `pi²/Delta²`. At `xi=1`, the leading coefficient changes to `pi²/4`. Subcritical convergence is exponential at fixed distance from threshold but is nonuniform as `xi -> 1`. Use log-radius variables, bracket the physical branch, and verify refinement on both sides and exactly at 1. Do not treat an arbitrary large finite `R/a` or a generic linear extrapolation as the infinite-dilution reference.

Archive search found no existing Manning/cylindrical-condensation task. The nearest PB task, `archives/electroviscous-flow-r2`, concerns diffuse-layer conductivity in a salted slit and already retains the correct equilibrium PB profile. This candidate instead tests a change in the dilute equilibrium ion population; it is not that task relabeled. Public apparatus should state PB and the limit, while the condensation diagnosis and formulas remain author-only.
