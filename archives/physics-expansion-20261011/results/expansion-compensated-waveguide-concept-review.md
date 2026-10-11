# Compensated waveguide concept review

Independent review, 2026-10-11. **Approved for packaging, subject to full source validation and controls.** Difficulty remains unmeasured. This review covers `staging/gpt6-expansion/waveguide_prototype.py`, SHA-256 `82b18031be52ed7c2656dde960f414afd42c1122b4bc2a8febc0a95f9ea1aa92`.

The refined apparatus is unambiguous: a centered strip `0<x<1`, `-epsilon*a(x)/2<y<epsilon*a(x)/2`, Dirichlet on every wall and end cap, known kinetic coefficient `hbar**2/(2m)=1`, and the exact external potential

    U(x) = well*cos(2*pi*x) - pi**2/(epsilon*a(x))**2,
    a(x) = exp(A*cos(2*pi*x)+B*cos(4*pi*x+phase)).

Define the reported signal as a common dimensionless detector gain times the strict `epsilon -> 0` limit of `E_level-E_0`, where the first four full two-dimensional eigenvalues are ordered increasingly and `level` is 1, 2, or 3. Hold all shape controls and `well` fixed during this limit. This avoids an unknown-mass compensation rule and states exactly which spectrum is measured.

The singular negative compensation is stable. The transverse Dirichlet Poincare inequality bounds the transverse kinetic energy below by the locally subtracted term. The remaining quadratic form is bounded below by `min_x well*cos(2*pi*x)`. There is no unbounded collapse as the strip narrows. The potential is a physical scalar potential acting on every transverse state, not a projection restricted to the ground mode.

The normalized local ground mode is `sqrt(2/(epsilon*a))*cos(pi*y/(epsilon*a))`. Its real gauge has zero diagonal derivative connection. Its derivative norm is

    (pi**2/12+1/4)*(a'/a)**2.

Thus the finite limiting longitudinal Hamiltonian is `-d_xx + well*cos(2*pi*x)` plus this positive scalar potential, with Dirichlet endpoints. This coefficient requires the centered strip; shifting only one wall would change the physical geometry and its coefficient.

The completed local-mode shortcut is defensible: it keeps the transverse threshold and the exactly opposing applied potential, but omits the kinetic cost of the changing normalized transverse state. The resulting one-dimensional Hamiltonian is `-d_xx+well*cos(2*pi*x)`. This gives nonzero spectral gaps, not a zero-signal shortcut. At constant width, the derivative term vanishes exactly, so the two models share calibration at every known well and level. The unknown gain is uniquely identified by linear weighted least squares using the positive known calibration gaps.

The prototype's finite-width reference correctly includes all retained derivative couplings in a physical two-dimensional Rayleigh-Ritz form. For `u=y/(epsilon*a)` and `D=u*d_u+1/2`,

    partial_x[phi(x)*chi(u)/sqrt(epsilon*a)]
      = [phi'(x)*chi(u)-(a'/a)*phi(x)*Dchi(u)]/sqrt(epsilon*a).

The displayed `B=<chi,Dchi>` and `G=<Dchi,Dchi>` matrices therefore generate precisely its two cross blocks and quadratic dilation block. The transverse block `pi**2*(n**2-1)/(epsilon*a)**2` implements the same scalar compensation for every mode. This reference does not insert the oracle's scalar coefficient by hand.

My extra probe covered 324 shape/phase/well/level combinations on a coarse domain grid. The smallest sampled first gap was `8.49596`; the smallest second and third gaps were `52.46254` and `129.73978`. These are sampled values, not rigorous domain minima. Three strong-shape finite-width sequences at epsilon `.03,.015,.0075` approached the effective result; the simple epsilon-squared extrapolation differed by at most `.00116` energy units, on a gap of about `179`. Increasing the effective basis from 48 to 72 changed these examples by at most `1.09e-7`.

Before freezing, refine longitudinal and transverse bases and spatial quadrature independently of epsilon. Then compare two epsilon windows; do not silently ascribe a fixed basis error to the thin-strip extrapolation. Include zero shape, first/second harmonics, nonzero phase, both well signs, and all three levels. A naive Cartesian mesh is a poor reference because the large transverse kinetic and compensation energies cancel. Freeze well-converged reference gaps for fast grading, and verify that measurement noise leaves a wide oracle/shortcut separation. A spatially constant scalar correction would cancel from every gap, so check the actual nonconstant shapes and all scored gap groups rather than assuming separation.

This is related to the geometric-rotor family but tests a different omission. Both archived rotor controls retained the local scalar geometric energy and missed transport around a closed guide. Here the centerline is straight, the retained transverse mode is nondegenerate and real, and the missing effect is the local scalar kinetic cost. The rigid-linkage archive concerns a classical equilibrium measure. [Generalised Quantum Waveguides](https://arxiv.org/abs/1402.1067) and [Nonadiabatic couplings and gauge-theoretical structure of curved quantum waveguides](https://arxiv.org/abs/1311.6925) provide primary context for these adiabatic derivative terms; the coefficient and reference blocks above are derived for the stated strip.

The eventual failure review must distinguish dropping derivative coupling as a physical approximation from deriving it correctly and then making a coefficient, coordinate, basis, or extrapolation error. Fresh trials are still needed to determine whether this familiar adiabatic correction is difficult for the target model.
