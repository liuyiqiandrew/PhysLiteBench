# Spectral hole feedback

This original thermal spectroscopy model is inspired by equilibrium ensembles and external-field response in the [Princeton physics general exam archive](https://phy.princeton.edu/academics/graduate-program/degree-requirements/preliminary-exams).

For any fixed record, use the unedited Gibbs ensemble as the reference sampling measure. Its record probability factors as a root Poisson-Dirichlet law and conditionally tilted site laws. Given the labelled reference record, the distinguished masses and remainder at each level have a Dirichlet law. The response numerator for each step is multiplied by the applied activity of that labelled site. Its updated denominator is a function of every addressed unedited mass and their exact nesting. Averaging the product of these factors under the reference record posterior includes the correlation of all normalizations.

The oracle integrates the posterior with Gauss-Jacobi quadrature and marginalizes untouched labels into the remainder. The reference uses an independently constructed conditioning and finer quadrature, and checks a nonzero feedback case by direct simulation of the size-biased masses. The shortcut keeps the correct unedited posterior and averages each addressed mass before constructing the updated probability.
