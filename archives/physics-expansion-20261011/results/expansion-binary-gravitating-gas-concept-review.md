# Binary gravitating gas: concept review

Reviewed 2026-10-11. **Approve the concept**, subject to the planned first-mode and numerical convergence checks before packaging. No task implementation, task edits, model trials, or delegation were performed. The calculations below were independent in-memory science checks.

## Apparatus and calibration

Use two conserved species with masses \(m_\ell\) and \(m_h=\chi m_\ell\), in the same rigid spherical box, at a common imposed kinetic temperature. Specify the same smooth classical Boltzmann mean-field, local spherical free-energy stability criterion as before. The reservoir fixes temperature but supplies no gravitational potential or mass exchange. Both species can redistribute spatially; their **global** particle numbers remain fixed. Do not impose a fixed local composition or fixed central composition.

The minimal composition inputs are total mass \(M\), heavy mass fraction \(q=M_h/M\), and mass ratio \(\chi\). With known light-particle mass, report \(\theta=k_BT/m_\ell\). The proposed ranges \(1\leq\chi\leq5\), \(0\leq q\leq0.5\) are reasonable. Define the readout as the first local instability encountered from the dilute branch as imposed temperature is lowered. No dynamical collapse time or finite-N equipartition failure is needed.

The completed control can be a genuine monodisperse auxiliary gas of the same total mass and total particle number, with

\[
\frac{\bar m}{m_\ell}=\frac1{1-q+q/\chi},\qquad
\theta_{c,\mathrm{mean}}=\frac{GM}{R}\frac{\bar m/m_\ell}{2.517551324\ldots}.
\]

It is exact whenever \(\chi=1\) or \(q=0\). Calibration on these complete monodisperse families, with varied \(M/R\), identifies the common positive \(G\) globally. Describe the control as replacing the species by their mean particle mass; calling the actual binary gas locally well mixed would violate its separate hydrostatic balance equations.

The physical omission is equilibrium mass segregation. The correct species profiles satisfy \(\rho_\ell\propto e^{-\Phi/\theta}\) and \(\rho_h\propto e^{-\chi\Phi/\theta}\). Their common temperature does not imply a common velocity dispersion or scale height. This multicomponent equilibrium and canonical stability framework is established in [Sopik, Sire, and Chavanis, *Self-gravitating Brownian systems and bacterial populations with two or more types of particles*](https://arxiv.org/pdf/cond-mat/0503251), sections 2–3. The task concerns their equilibrium thermodynamic mechanism, not the later nonequilibrium collapse discussed in that paper.

## Minimal equilibrium parameterization

Let \(h\) be the central heavy **mass** fraction, \(\rho_0\) total central density, and \(\alpha^2=4\pi G\rho_0R^2/\theta\). With \(\psi=(\Phi-\Phi_0)/\theta\),

\[
(\xi^2\psi')'=\xi^2[(1-h)e^{-\psi}+he^{-\chi\psi}],\qquad
\psi(0)=\psi'(0)=0.
\]

For each \(\alpha\), choose \(h\) to enforce

\[
q=\frac{h\int_0^\alpha \xi^2e^{-\chi\psi}\,d\xi}
{\alpha^2\psi'(\alpha)}.
\]

Then \(\eta=\alpha\psi'(\alpha)=GM/(R\theta)\). The first maximum of \(\eta\) along the fixed-global-composition dilute branch gives the canonical threshold. The total-central-density scale avoids the light-density parameterization's potential folds. It does not remove the need to scan from the dilute branch rather than assume a global optimizer selects the first peak.

## Independent stability reference

For separate enclosed-mass perturbations \(q_\ell(r),q_h(r)\), each regular as \(O(r^3)\) and zero at \(R\), the canonical free-energy quadratic coefficient is

\[
Q_F=\frac\theta2\int_0^R\left[
\frac{q_\ell'^2}{4\pi\rho_\ell r^2}
+\frac{q_h'^2}{4\pi\chi\rho_h r^2}\right]dr
-\frac G2\int_0^R\frac{(q_\ell+q_h)^2}{r^2}dr.
\]

Positive curvature means stability. This gives two entropy-gradient blocks and the common gravitational coupling; **both species constraints must be enforced separately**. A converged smallest-eigenvalue crossing provides an independent stability check against the fixed-composition temperature turning point. Drop an absent species at \(q=0\); do not divide by its zero density. Check the equal-mass limit, including its stable relative-composition directions.

I checked this two-species Hessian at the independently computed first peaks for \((\chi,q)=(3,.3)\) and \((5,.3)\). With 64 modes per species and 256 quadrature points, the lowest normalized curvatures were respectively 1.05e-7 and −3.51e-9, consistent with the preliminary peak-location accuracy. Increasing to 96 modes changed them by less than 2e-16. At 2% lower/higher \(\alpha\), curvatures were positive/negative, with magnitudes about 0.013–0.014. These support the proposed reference route; full-domain and first-crossing validation remain necessary.

## Separation and novelty

Independent fixed-composition continuation gives the following preliminary coefficients and mean-mass errors, at the same \(G\):

| \(\chi\) | \(q\) | True \(\theta_c/(GM/R)\) | Mean-mass coefficient | Relative shortcut error |
|---:|---:|---:|---:|---:|
| 2 | 0.3 | 0.5328982 | 0.4673075 | 12.31% |
| 3 | 0.1 | 0.4947399 | 0.4255836 | 13.98% |
| 3 | 0.3 | 0.6904106 | 0.4965142 | 28.08% |
| 3 | 0.5 | 0.8520748 | 0.5958170 | 30.07% |
| 5 | 0.1 | 0.6303189 | 0.4317515 | 31.50% |
| 5 | 0.3 | 1.0327693 | 0.5226465 | 49.39% |
| 5 | 0.5 | 1.3432994 | 0.6620189 | 50.72% |

Thus hidden mixtures around \(\chi=2\)–5 and \(q=.1\)–.5 offer substantial gaps without extreme mass ratios or trace-species limits. Equal-mass and absent-heavy limits belong in shared checks. Avoid claiming a nonzero minimum gap over the whole public domain, since the correct and control laws coincide continuously at calibration limits.

No direct mass-segregation/gravitational-mixture task was found in task/archive author or public specifications. The finite-bath predecessor tests energy exchange constraints; this candidate keeps the canonical bath correct and tests species-dependent spatial redistribution. It is a substantive new apparatus and omission, with the gravity-family lineage disclosed. The two-species calculation remains modest: one spherical field, one composition constraint, and a two-block radial Hessian. Preserve the completed finite-bath batch unchanged and judge any new trial failures from their actual physical assumptions rather than from numerical difficulty alone.
