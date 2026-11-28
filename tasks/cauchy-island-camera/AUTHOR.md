# Brownian excursion camera

The floor-contact trace of normally reflected planar Brownian motion, parametrized by boundary local time, is Cauchy. Each jump of that trace hides a physical Brownian excursion. The cameras measure the excursion marks and the previous path on the same shot. Brownian generators and hitting times use DΔ in bulk physical units. [Boundary traces of shift-invariant diffusions](https://arxiv.org/abs/1912.00072)

## Solution

For horizontal displacement d, the unlimited floor-to-floor excursion kernel is J∞(d) = 1/(πd²). With an absorbing ceiling at h the corresponding kernel is Jₕ(d) = π csch²(πd/2h)/(4h²). Both follow by taking the inward normal derivative of the bulk floor Poisson kernel at departure. Their ratio gives the marginal height CDF of a prescribed excursion. [Brownian excursion Poisson kernels in a strip](https://math.uchicago.edu/~lawler/annulus.pdf)

The earlier path occupies its reflecting floor with the inverse of the strip Dirichlet-to-Neumann operator, restricted to the live interval. The strip symbol is |k| coth(h |k|). Relative to the unlimited operator, the additional integral kernel is the smooth function J∞−Jₕ. The production solution expands the boundary trace in weighted Chebyshev modes and applies this inverse to the measured final-arch source. An independent ceiling on the preceding path and a height band on the final arch enter different factors.

To resolve the final duration, the horizontal heat density multiplies the two normal derivatives of the vertical absorbing heat kernel on (0,h). In units s = Dt, the latter is (2/h) times the sum over n ≥ 1 of (πn/h)² exp[−(πn/h)²s]. The production solution analytically integrates each mode over the requested physical-time window and subtracts the two ceilings to select the height band.

The independent reference maps the earlier strip by ζ = exp(πw/h) into a half-plane. The full reflecting segment maps to a single reflecting interval. The single-interval logarithmic boundary Green kernel in that coordinate gives the previous occupation measure. The normal derivative and line-element Jacobians cancel inside the boundary occupation integral. The reference computes final-arch currents by direct quadrature over log time, using the method of images at short times and a sine heat expansion at longer times. It uses neither the restricted boundary generator nor the analytic time integrals of the production solution.

## Validation

In the first worked shot the product of the two height-camera fractions equals the ratio of the terminal harmonic measures in the single-ceiling strip and the unbounded half-plane. The transformed interval Poisson kernel gives 0.2470258285 directly. The validator also checks that identity on every hidden source and landing pair. In both worked shots the duration test is independent of that harmonic identity.

Labels for 27 distinct held-out settings use the conformal reference. Solver and reference disagree by less than 0.00000002 on these readings. The same two methods agreed within 0.00000005 over 128 settings sampling the extrema of every public parameter. A 34-to-65-mode refinement changed the worst of these readings by less than 0.000000003. The public tolerance is 0.004.

The public single-arch approximation misses all three detector families. The more selective control recovers both height probabilities with the correct floor occupation. It draws the final duration from the single-arch time marginal for the actual horizontal displacement, ignoring the selected height. It misses the conditional duration fraction by up to 0.437. Run `python scripts/validate_cauchy_island_camera.py` to compare the implementations and controls. Use `--generate` to write the labels.
