# Compensated waveguide, revision 1

The apparatus is a Cartesian Dirichlet channel with a straight, fixed center
and nonuniform width epsilon*a(x). The applied potential cancels the local
transverse ground energy exactly before the thin-width limit. This does not
cancel the kinetic energy associated with the changing normalized transverse
wavefunction. All wall locations, end caps, potentials, units, and the order
of the limiting spectral measurement are public.

## Physical reduction and approximation

The normalized transverse ground state is

```
chi(x,y) = sqrt(2/(epsilon*a(x)))*cos(pi*y/(epsilon*a(x))).
```

Its Berry connection vanishes: the integral of chi*d_x chi is zero. Its
derivative norm is `(pi²/12+1/4)*(a'/a)²`. After compensation and epsilon->0,
the longitudinal Hamiltonian is therefore

```
-d²/dx² + well*cos(2*pi*x) + (pi²/12+1/4)*(a'/a)²
```

with Dirichlet end conditions. This positive Born-Huang energy is finite as
the transverse gap diverges. Removing excited transverse populations does
not justify dropping derivatives of the retained transverse state. The
completed shortcut instead keeps only each cross section's local transverse
energy. Its effective longitudinal Hamiltonian is free of the derivative
term. It is a complete, nonzero spectral approximation with exact local
energy compensation and correct longitudinal confinement.

The compensation is stable. At each x, transverse Poincare's inequality
bounds the transverse kinetic form below by pi²/(epsilon*a)² times density.
The negative term cancels this lower bound, leaving a form bounded below
by the minimum longitudinal applied potential. There is no collapse or
unspecified energy renormalization. The width profiles remain fixed and
smooth in the limit.

## Independent reference

The oracle diagonalizes the effective one-dimensional operator in 80 sine
modes with 320-point quadrature. The reference uses the full two-dimensional
Rayleigh-Ritz quadratic form in normalized moving transverse modes. With
u=y/(epsilon*a), define D=u*d_u+1/2. Its Cartesian x derivative is

```
d_x[phi(x)*chi_n(u)/sqrt(epsilon*a)]
 = [phi'(x)*chi_n(u) - (a'/a)*phi(x)*D chi_n(u)]/sqrt(epsilon*a).
```

The reference integrates every cross term and all retained transverse
couplings directly; it does not insert the Born-Huang coefficient. Exact
transverse kinetic energies and the applied compensation cancel analytically
for the ground mode. This avoids subtracting large grid-discretization errors.
Gaps at epsilon=.03,.015,.0075 are extrapolated quadratically in epsilon².
The base uses 48 longitudinal and 7 transverse modes. Refinement uses
64 and 11 modes with all widths halved. Additional fixed-width checks
separately increase longitudinal, transverse, and quadrature resolution.

Hidden reference refinement changes a normalized group by at most 1.69e-7.
The refined reference and oracle agree within 7.92e-9 in group relative norm.
The validator checks 36 random and boundary preparations, with maximum
absolute difference 3.63e-6, and x-reflection symmetry. Oracle refinement
to 120 modes changes hidden gaps by less than 8.2e-10. An initial 48-mode
oracle missed the internal 1e-8 absolute refinement gate by 3.1e-8; this
developmental result is preserved, and the basis was increased before
evaluation. Calibration, hidden cases, and grading were unchanged.

## Calibration and evaluation

For constant width, separation is exact. Independent Mathieu characteristic
values give longitudinal energies pi²*b_n(well/(2*pi²)). All calibration
gaps are positive. The common gain enters linearly, so scalar weighted least
squares identifies it uniquely. There are 240 independent readings at 12
settings, with fixed sigma .02, true gain 1.07, and seed 611301. Both controls
fit 1.070014269, with reduced chi-square .989356.

The 256-noise study uses seed 611307 and exact scalar fits after checking
both control implementations. All calibration and parameter gates pass;
gain error stays below .00437%. The largest oracle prediction error is
.00438%; the weakest shortcut diagnostic error is 13.3%. Each hidden group
uses error norm divided by its true norm, with a 4% gate. The minimum graded
gap exceeds 25 energy units. Parameter error must be below 3%, and calibration
reduced chi-square below 1.5. These gates are well above numerical and noise
uncertainty.

Run `scripts/validate_compensated_waveguide.py` in the pinned numerical
environment. Only `--generate` replaces both calibration copies and frozen
hidden values. The report is
`results/expansion-compensated-waveguide-r1-validation.json`.

After independent source review and Docker controls, run exactly three
frozen, unhinted gpt-5.6-luna High trials with the standard 600/60-second
limits. Review source and trajectories: omitting the kinetic contribution
of the changing confined state is physical, whereas deriving the effective
operator correctly and coding its coefficient or eigensolver incorrectly
is mathematical or numerical. No difficulty claim precedes these trials.

This task differs from the archived geometric-rotor tasks, whose controls
already retained the local Born-Huang term but omitted global transport.
There is no degeneracy, twisting, bending, or holonomy here. Primary context:
[Haag, Lampart and Teufel, Generalised Quantum Waveguides](https://arxiv.org/abs/1402.1067),
and [Stockhofe and Schmelcher, Nonadiabatic couplings and gauge-theoretical structure of curved quantum waveguides](https://arxiv.org/abs/1311.6925).
The finite Cartesian apparatus independently fixes the coefficient and sign.
