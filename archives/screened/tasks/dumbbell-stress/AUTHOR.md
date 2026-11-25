# Dumbbell stress, revision 1

This task tests the physical transport law of an elastic configuration tensor. Both controls are objective constitutive models: they transform consistently under a rotating frame, use every velocity-gradient component, and produce nonzero normal stresses. Calibration observes anisotropic stress relaxing under rigid solvent rotation, where the two models coincide exactly. Straining flows reveal whether molecular configurations stretch with the affine solvent motion.

With connector R=r2-r1, bead friction zeta=4*tau, and H=k_B*T=1, subtracting the two overdamped bead equations gives

    dR = [gradient - I/(2*tau)] R dt + sqrt(1/tau) dW.

The factor in the relative thermal noise comes from the two independent bead forces. The covariance C=<R R^T> therefore obeys

    Cdot = gradient*C + C*gradient.T - (C-I)/tau.

The polymer stress is S=C-I. Writing gradient=D+Omega separates the stress dynamics into

    Sdot = Omega*S - S*Omega + D*S + S*D + 2*D - S/tau.

The completed shortcut is the corotational Maxwell stress law, retaining the rotation commutator, strain source, and relaxation but omitting D*S+S*D. It is a valid phenomenological constitutive approximation, rather than an unfinished calculation. It is not the microscopic Hookean model specified in the apparatus. The exact connection between Hookean dumbbells and the upper-convected Maxwell law is also used in [Multiscale simulation of polymer melt spinning by using the dumbbell model](https://arxiv.org/abs/1609.00793).

For calibration D=0. Both models give

    S(t) = exp(-t/tau) R_rotation(t) S(0) R_rotation(t).T.

The known anisotropic Gaussian preparation and rotation determine the prefactor of the measured normal stress, so the decay identifies tau. The initial covariance is prepared by a known temporary anisotropic spring; the spring is switched to its isotropic value at the start. This is a specified nonequilibrium positional preparation, not an assumption about an unobserved stress history.

For all allowed tau and D, the shortcut covariance equation has source I/tau+2D with minimum eigenvalue at least 1/1.2-0.7=0.13333. Its homogeneous transport is damped rotation, so positive initial covariance remains positive. The true covariance has isotropic positive thermal source. The bound ||D||<=0.35 also keeps affine drift stable because 0.35<1/(2*1.2). Thus neither negative covariance nor a stretching instability explains the shortcut failures. Hidden cases measure shear normal stress, subcritical planar extension, and shear after a change of strain axis.

The oracle integrates the affine covariance equation with a 5 by 5 augmented matrix exponential. The independent reference solves the linear stochastic connector equation: its covariance is the transported initial covariance plus an integral of transported independent Gaussian noise increments. It evaluates that integral with 32-point Gauss-Legendre quadrature using 2 by 2 fundamental matrices. Doubling quadrature changes predictions by less than 3.6e-15. Closed-form simple shear, planar extension, rigid rotation, equilibrium, frame rotation, and interval-splitting checks also pass. Both constitutive models pass the frame-rotation and splitting checks; those are not the missing physics.

Calibration contains 200 measurements, true tau 0.8, seed 9329, and independent Gaussian errors with standard deviation 0.6% of the largest noiseless normal stress. Both controls fit tau=0.80136982, relative error 0.1712%, and reduced chi-square 0.84780. Exact calibration agreement is within 4.2e-15. Hidden errors are RMS-normalized within each group with limit 0.04. Oracle errors are 0.003018, 0.002292, and 0.001993; shortcut errors are 0.17139, 0.22663, and 0.20548.

All 256 additional calibration-noise realizations pass calibration and parameter checks, with maximum tau error 0.6960%. At fitted-tau extrema, oracle hidden error stays below 0.01224 and shortcut error above 0.16557. This is an extrema sensitivity check, not an exhaustive hidden Monte Carlo calculation. The independent transported-noise reference agrees below 1.0e-14. The tested covariance minimum eigenvalue is 0.7275, consistent with the global positive-source argument. Local isolated-copy tests give oracle 7/7 and shortcut 4 passes with 3 hidden failures.

Reproduce from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 scripts/validate_dumbbell_stress.py dumbbell-stress
```

Use `--generate` only to replace both calibration copies intentionally. The report is `jobs/dumbbell-stress-validation/summary.json`. The parent coordinates Harbor controls and paired Luna/high trials. Agent difficulty remains unmeasured until those trials are reviewed.
