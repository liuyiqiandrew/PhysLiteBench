# Finite-bath gravitating gas: concept review

Reviewed 2026-10-11. No task implementation or model trials were performed. Numerical values below come from an independent in-memory Lane–Emden calculation.

**Verdict: approve the refined physical concept.** The finite-bath first-instability temperature is a continuous, nontrivial prediction. The fitted gravitational strength now enters that prediction, and the bath capacity changes it. This addresses the constant-label and ancillary-fit weaknesses of the preceding stability-flag proposal. The proposed energy turning point and Hessian denominator are correct. Keep the local smooth mean-field scope explicit and leave sufficient separation from the prediction gate.

## Preparation and observable

The gas and a finite bath form an isolated composite during each stability test. Specify a non-gravitating, mechanically inactive bath with negligible coupling energy, positive constant heat capacity \(bNk_B\), and energy exchange sufficient for thermal equilibration. It supplies no extra gravitational potential, mass exchange, volume change, or external thermostat. Its energy can be written \(E_b=bM\theta+\mathrm{constant}\), where \(\theta=k_BT/m\), and its entropy follows from \(dS_b/dE_b=1/T\). The additive energy constant has no effect.

For \(b=0\), the bath is absent and the gas energy is fixed. For the separately declared infinite-bath case, temperature is externally fixed during perturbations. That case is the canonical limit; it does not have a finite composite energy that can be lowered.

Define the critical output by following the regular branch from dilute states to increasing density contrast until its **first local entropy instability**. Decreasing total energy refers to a sequence of independently prepared isolated composites, not an external energy leak during a stability test. In the infinite-bath protocol, lower the imposed temperature along the dilute stable branch. A finite bath allows the gas to pass its temperature minimum while remaining stable, so the first instability cannot be defined as the lowest temperature reached.

Retain the explicit Boltzmann mean-field, collisional, infinitesimal smooth spherical-perturbation criterion. It excludes Vlasov Casimir constraints, finite-N activated escape, binary formation, and global point-particle collapse. The canonical versus microcanonical local-stability distinction is supported by [Chavanis's isothermal-sphere analysis](https://arxiv.org/html/astro-ph/0103159); finite-N onset fluctuations are a different problem, as discussed by [Katz and Okamoto](https://arxiv.org/abs/astro-ph/0004179). No kinetic growth time is being predicted here.

A supplied equilibrium density contrast is a label of the initial stationary profile, not a quantity held fixed during a perturbation. Otherwise the intended instability mode could be inadvertently excluded. Ordinary-temperature and critical-temperature readouts should have clearly distinct input requirements.

## Turning-point calculation

Using the regular isothermal Lane–Emden solution, let

\[
u=\frac{\alpha e^{-\psi(\alpha)}}{\psi'(\alpha)},\qquad
v=\alpha\psi'(\alpha)=\eta,\qquad
\theta=\frac{GM}{Rv}.
\]

The gas energy includes the rigid-wall pressure contribution:

\[
E_g=-\frac{GM^2}{R}\frac{3/2-u}{v},\qquad
E_{\mathrm{tot}}=E_g+bM\theta.
\]

In terms of physical heat capacity,

\[
\frac{C_g}{Nk_B}=
\frac{4u^2+2uv-11u+3}{2(u-1)}.
\]

For every finite \(b\geq0\), the first loss is the first energy turning point after the canonical point \(u=1\). Equivalently, it is the root

\[
\boxed{4u^2+2uv+(2b-11)u+3-2b=0}
\]

on the first negative-gas-heat-capacity segment. For \(b>0\), the stable thermal-exchange condition on that segment is \(C_g<-bNk_B\), reaching equality at the loss. At \(b=0\), the endpoint is the isolated gas's first energy minimum. As \(b\to\infty\), it approaches the canonical temperature minimum. Do not select later roots of the spiraling equilibrium branch.

I integrated the regular Lane–Emden initial-value problem with relative tolerance 2e-12 and absolute tolerance 2e-14, then located the first roots. The canonical values are \(\alpha_c=8.993114931\), contrast \(32.12491256\), and \(\eta_c=2.517551324\). Independent finite-bath results are:

| \(b\) | Critical contrast | \(\eta_{\mathrm{crit}}\) | \(\theta_{\mathrm{crit}}/(GM/R)\) | Canonical shortcut error relative to truth |
|---:|---:|---:|---:|---:|
| 0 | 708.6083 | 2.0308545 | 0.4924036 | 19.3322% |
| 0.25 | 524.7827 | 2.0871883 | 0.4791135 | 17.0945% |
| 0.5 | 379.7137 | 2.1520044 | 0.4646831 | 14.5199% |
| 1 | 205.6454 | 2.2783857 | 0.4389073 | 9.4999% |
| 1.5 | 131.2622 | 2.3655516 | 0.4227344 | 6.0376% |
| 2 | 97.7496 | 2.4163202 | 0.4138524 | 4.0210% |
| infinite | 32.1249 | 2.5175513 | 0.3972114 | 0 |

The error here is \(1-\eta_{\mathrm{crit}}(b)/\eta_c\), evaluated at the same gravitational strength. In particular, the 4.1895% increase of the true temperature over the canonical value at \(b=2\) is **not** its relative-to-truth error. With a 4% grading gate, \(b=2\) leaves only 0.021 percentage points of margin before calibration noise. Keep that endpoint in domain validation if desired, but use lower capacities for robust discriminating groups. Capacities up to 1 or 1.5 give more useful gaps; verify the final group margins over the actual calibration-noise distribution.

## Independent entropy-Hessian reference

The bath extension can be derived directly, without importing the turning-point criterion. At fixed density, maximize the sum of gas kinetic entropy and bath entropy over their thermal energy split. Both thermal subsystems then have common \(\theta\). With mass-scaled entropy \(\mathcal S=mS/k_B\), their reduced entropy at fixed composite energy is, up to constants,

\[
\mathcal S_b[\rho]=-
\int\rho\log(\rho/\rho_*)\,dV
+(3/2+b)M\log[(E_{\mathrm{tot}}-W[\rho])/K_*].
\]

The quadratic coefficient for mass-preserving perturbations is

\[
\boxed{Q_b=-\frac12\int\frac{(\delta\rho)^2}{\rho}\,dV
-\frac{1}{2\theta}\int\delta\rho\,\delta\Phi\,dV
-\frac{1}{(3+2b)M\theta^2}
\left(\int\Phi\,\delta\rho\,dV\right)^2.}
\]

Thus the stated denominator is correct. The isolated expression is recovered at \(b=0\); the stabilizing rank-one term vanishes for an infinite bath. Eliminating the positive-capacity thermal split removes only stable thermal directions, so testing this reduced form covers the remaining density instability.

For radial mass perturbation \(q(r)=4\pi\int_0^r x^2\delta\rho(x)dx\), use \(q(R)=0\) and regular \(q=O(r^3)\). Its form is

\[
Q_b=-\frac12\int_0^R\frac{q'^2}{4\pi\rho r^2}\,dr
+\frac{G}{2\theta}\int_0^R\frac{q^2}{r^2}\,dr
-\frac{1}{(3+2b)M\theta^2}
\left(\int_0^R\frac{GM(r)q(r)}{r^2}\,dr\right)^2.
\]

A converged radial Hessian whose largest eigenvalue first crosses zero is a suitable independent stability reference. It should reproduce the complete finite-bath curve, the isolated and canonical limits, and the mode's sign on either side of the first root. It can share the equilibrium density profile with the oracle while independently testing stability. A second evaluation of the same turning-point polynomial would not supply that independence.

## Control, calibration, and overlap

The completed control is the infinite-bath approximation applied to all capacities. It retains the exact equilibrium profiles and ordinary temperatures. Critical-temperature calibration at infinite capacity is exact for every \(G\), while finite-bath ordinary-temperature calibration tests preparation without revealing the stability shift. Positive ordinary temperature coefficients identify \(G\) globally; hidden critical temperatures scale with that fitted value. Both the correct and approximate outputs are finite and nonzero.

There is a broad archive precedent: [screened finite-heat-reservoir](../archives/screened/tasks/finite-heat-reservoir/AUTHOR.md) distinguishes isolated oscillator phase-space weights from a canonical bath through equilibrium moments. The present mechanism is different: long-range gravity, a negative-heat-capacity branch, and a changing entropy eigenmode set the first instability. Disclose that shared ensemble theme, but it is not a relabeled version of the oscillator task.

This is a stronger candidate than the preceding binary-flag proposal. Its validity does not establish model difficulty; the finite-bath approximation must still be tested with complete controls and reviewed trials. Avoid additional complications introduced only to hide a standard thermodynamic argument.
