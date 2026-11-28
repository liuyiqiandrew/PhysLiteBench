# Cauchy island camera

The camera measures the joint law of the last occupied site, the exterior landing, the time spent on each live island and the number of transfers between islands. A single time step across the gap can transfer the carrier between live islands.

## Law and implementations

For a chosen exterior landing y let H(x) be the monitored accepted landing density at launch x. Write V(x) for the gate rate on the occupied island and p for its transfer survival. The Lévy density from x to y is D/[π(y−x)²]. Multiplying this by the instantaneous donor response A(x) gives the source term. The occupation resolvent integrated against that source yields H. The killed Cauchy generator on one island uses zero continuation outside that island. Add the off-diagonal operator −p D/π times the integral of H(w)/(w−x)² over the other live island. This yields the positive resolvent operator to invert after adding V.

The solution expands H on each interval in the functions sin(nθ), with x = midpoint + half-width cos(θ). The killed diagonal Cauchy operator takes sin(nθ) to n sin(nθ)/(half-width sin(θ)). The remaining mass and transfer matrix elements are smooth angular integrals. The unit-ball weighted polynomial identity is given by Dyda, Kuznetsov and Kwaśnicki. [Fractional Laplace operator and Meijer G-function](https://arxiv.org/abs/1509.08529)

The independent reference starts at the first exit from the currently occupied interval. It uses the one-interval logarithmic Green kernel G for physical occupation and Poisson kernel P for the exit. On each island it solves H = G[source − V H] + p P[H on the other island]. Piecewise linear interpolation and split product integration in θ resolve the logarithmic singularity. Repeating with p = 1, V = 0 and the relevant donor profile supplies the normalizing densities. The analytic one-island Cauchy kernels are given by Kulczycki et al. [Spectral properties of the Cauchy process on half-line and interval](https://doi.org/10.1112/plms/pdq010)

## Checks

The validator regenerates the 27 held-out readings with 165 interior Green nodes per island and compares them to 72 spectral modes. Maximum group discrepancies were 0.000006 for landing, 0.000012 for donor and 0.000013 for clock. Use `python scripts/validate_cauchy_island_camera.py` from the repository root. Add `--generate` to rebuild the frozen labels. Both worked examples have separate settings.

A pre-freeze convergence scan compared 40 and 80 spectral modes on 169,722 scalar readings. It included the extrema of the geometry, offsets, launch positions, effective rates, relay and Bernstein profile entries. Maximum changes for the three observables were 0.00000042, 0.00000013 and 0.000011. An additional 126 comparisons, including single islands, compared 84-node Green integration with 72 spectral modes at parameter extrema. The maximum discrepancy was 0.000136. On a separate gap-camera bin, three million first-exit Monte Carlo paths matched the integrated terminal hit probability to 0.57 standard errors and the conditional relay survival to 0.10 standard errors.

The starter approximates the departing site and transfer launch by the island midpoint and replaces the residence time by its mean. It fits free transport exactly up to rounding but misses all three held-out observables. The stronger physical control solves the terminal and donor cameras exactly and uses an independent trajectory to supply the gate. It reproduces the first two families within 0.000012 and misses the selected clock fraction by up to 0.6364.
