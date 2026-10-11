# Isolated gravitating gas: concept critique

Reviewed 2026-10-11. Concept only; no implementation, task changes, or trials.

**Verdict: the physical distinction is correct and appears distinct from the archive, but do not package the unchanged proposed tests.** All proposed hidden contrasts 50–500 have the same true stability flag as the low-contrast calibration. An always-stable classifier would pass them. Shared unstable cases are necessary. Even after that repair, this is a comparatively recognizable ensemble-selection problem with a universal binary threshold; I would keep it as a lower-priority fallback rather than expect a 0/3 result. There is no scientific reason to reject it solely because the textbook result is accessible.

## Physical interpretation

Canonical and isolated mean-field isothermal spheres share the same stationary profiles but differ in local thermodynamic stability. The first canonical loss occurs at contrast approximately 32.1; isolated entropy stability persists to approximately 709, including the intervening negative-heat-capacity branch. These are established local stability results, not evidence of different hydrostatic equilibrium equations. [Chavanis, *Gravitational instability of finite isothermal spheres*](https://arxiv.org/html/astro-ph/0103159).

The task should define stability as the sign of the second entropy variation for infinitesimal smooth spherical perturbations of the classical mean-field distribution, at fixed total mass and energy. Do not define it as survival for arbitrarily long time of a finite collection of Newtonian point particles. Point-particle collapse prevents a global entropy maximum; a local smooth state can nevertheless be metastable. [Lynden-Bell, *Negative Specific Heat in Astronomy, Physics and Chemistry*](https://arxiv.org/html/cond-mat/9812172).

The mean-field, local criterion must be explicit. Finite-particle fluctuations can destabilize states before the nominal contrast 709, so an unspecified empirical finite-N collapse time is a different observable. [Katz and Okamoto, *Fluctuations in Isothermal Spheres*](https://arxiv.org/abs/astro-ph/0004179).

Keep the reflecting wall rigid and thermally insulating. Let collisions permit ordinary Boltzmann equilibration and energy redistribution; do not preserve collisionless Vlasov Casimirs or impose a fixed temperature during perturbations. No collision rate or observation time is needed for the defined thermodynamic criterion. If a dynamical growth rate were requested instead, the kinetic evolution and timescale would need specification. A short-distance regularization is unnecessary for this explicitly local smooth criterion, but would be necessary to discuss the collapsed phase or global equilibrium.

## Equilibrium and calibration

With known particle mass \(m\) and \(k_B=1\), solve

\[
(\xi^2\psi')'=\xi^2e^{-\psi},\qquad \psi(0)=\psi'(0)=0.
\]

For supplied contrast \(C>1\), \(\psi(\alpha)=\log C\) uniquely determines \(\alpha\), and

\[
\eta=\alpha\psi'(\alpha),\qquad T=\frac{GMm}{R\eta}.
\]

Thus nonzero temperature measurements identify the single positive \(G\) globally. The canonical control can solve this equilibrium exactly and still use its own fixed-temperature second variation; it is a complete physical comparison. The auxiliary bath changes the allowed perturbations, not the stationary profile.

The preparation should select the supplied profile, rather than merely an energy that may admit several stationary profiles. Density contrast labels that branch unambiguously. Its consistent energy, including the wall-pressure contribution, is

\[
E=-\frac{GM^2}{R}\Lambda,\qquad
\Lambda=\frac{3}{2\eta}-\frac{\alpha^2e^{-\psi(\alpha)}}{\eta^2}.
\]

Do not apply the unconfined virial relation to remove the wall term. Exclude exactly \(C=1\), whose regular finite-mass fixed-radius isothermal limit has infinite temperature. At unstable contrasts, call the supplied state a prepared stationary mean-field profile, not a state obtained by unrestricted equilibration.

## Independent stability reference

An entropy Hessian is a useful independent route from the equilibrium turning-point oracle. The following follows directly by eliminating the Maxwellian velocity distribution at fixed energy. Define mass-scaled entropy \(\mathcal S=mS/k_B\), kinetic energy \(K=E-W=3MT/(2m)\), and \(\beta=m/T\). Up to irrelevant constants,

\[
\mathcal S_E[\rho]=-\int\rho\log(\rho/\rho_*)\,dV
+\frac{3M}{2}\log[(E-W[\rho])/K_*].
\]

For mass-preserving perturbations, the quadratic coefficient is

\[
Q_E=-\frac12\int\frac{(\delta\rho)^2}{\rho}\,dV
-\frac\beta2\int\delta\rho\,\delta\Phi\,dV
-\frac{\beta^2}{3M}\left(\int\Phi\,\delta\rho\,dV\right)^2.
\]

The canonical quadratic form consists of the first two terms. The final negative rank-one term is the energy-conservation correction. It immediately explains why canonical stability implies isolated stability while the converse can fail.

For spherical perturbations define \(q(r)=4\pi\int_0^r u^2\delta\rho(u)du\), with \(q(R)=0\) and regular \(q(r)=O(r^3)\) at the center. Then

\[
Q_E=-\frac12\int_0^R\frac{q'^2}{4\pi\rho r^2}\,dr
+\frac{\beta G}{2}\int_0^R\frac{q^2}{r^2}\,dr
-\frac{\beta^2}{3M}\left(\int_0^R\frac{G M(r)q(r)}{r^2}\,dr\right)^2.
\]

A refined radial discretization can test its largest eigenvalue independently of the turning-point condition. Respect the origin regularity and resolve the central core. Compare both canonical and isolated signs on both sides of their first thresholds; do not infer stability merely from the sign of the local heat capacity beyond later turns. The oracle may instead locate the first temperature or energy turning point along the regular branch and track the loss of modes from the dilute stable limit.

## Benchmark limitations and minimum changes

- Add shared unstable high-contrast anchors above the isolated threshold and shared stable low-contrast anchors. Keep cases away from the thresholds and specify the exact-critical convention publicly. If the allowed domain stops at 500, the true stability output is constant throughout that domain and is not a useful target.
- Grade temperature and stability separately. A mixed continuous/binary norm could hide wrong flags behind correctly predicted temperatures. Define flags and their treatment as exact labels, rather than noisy temperatures.
- A stability error would be independent of the fitted \(G\): the dimensionless threshold depends only on contrast. This is legitimate but makes the inverse fit ancillary to the diagnostic physics.
- Searching task and archive author/public specifications for gravothermal, self-gravity, microcanonical, negative heat capacity, Lane–Emden, Antonov, and ensemble inequivalence found no direct predecessor. The broad distinction between conserving and clamping a thermodynamic variable is familiar, but this apparatus is not a relabeled current candidate.

I would not add arbitrary complications solely to obscure the standard answer. With the stated local mean-field scope and shared unstable anchors, the concept is fair and minimal. Whether the target model makes this physical mistake remains an empirical question; the scientific review offers no difficulty guarantee.
