# Reaction diffusion R10

R8 and R9 passed all three fresh model trials. R10 keeps the same chemical
apparatus and adds a fully specified equilibrium fluctuation detector.
Independent Poisson total constituents are not the thermal susceptibility:
charge neutrality anticorrelates free cations, while a neutral bound cluster
contributes correlated A and B constituents. The sign can change with binding.

The oracle starts from ideal species population covariance, conditions on
charge, and projects onto total A/B. The zero-current Onsager mobility follows
from the independent Brownian fluxes. The mode covariance is exp(-k^2 M C^-1 t)C.
The completed shortcut retains the correct transport relaxation but assumes
diagonal Poisson total-constituent susceptibility. It fits the unchanged mean
calibration and predicts every new observable finitely.

The independent reference retains all four species, a finite electrostatic
charge penalty and finite chemical reaction noise. It forms the complete
linear chemical-Langevin drift/noise and equilibrium correlation at finite
separation, then extrapolates the two fast processes away. It does not use
the reduced two-constituent susceptibility or mobility. Validation requires
thermodynamic Hessian agreement, Onsager symmetry, positive covariance,
finite-rate convergence, deterministic regression, and 256 noisy fits.
Exactly three fresh GPT-5.6 Luna High trials follow controls and validation.
