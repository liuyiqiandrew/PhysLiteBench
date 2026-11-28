# Weak-field condensate coincidences, revision 1

Inspired by the fixed-particle Bose partition function in Princeton's May 2003 preliminary exam, Part II, section B, problem 3. The new readout uses a weak symmetry-breaking shift that scales inversely with volume and a passive internal rotation before detecting the condensate. [Princeton exam archive](https://phy.princeton.edu/academics/graduate-program/degree-requirements/preliminary-exams)

The completed shortcut uses the exact normal and condensed densities, so it fits the condensate-fraction calibration. It represents the condensed internal population by one set of deterministic fractions and predicts independent detector outcomes. The exactly fixed-N condensate instead has a distribution of macroscopic occupation fractions. The weak splitting tilts this distribution by a finite factor. Passive pulse transport retains exchange interference, represented by independent uniform phases conditional on those fractions.

The oracle sums phase-matched amplitude monomials and integrates their moments over the exponentially tilted simplex using Jacobi quadrature. The independent reference integrates a tilted complex Haar vector with scrambled low-discrepancy sampling. It checks the unsplit Dirichlet moments, mode conservation, and convergence. The 4% group RMS tolerance uses a 0.02 floor.
