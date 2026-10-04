# Capillary-mixture revision 1

The task tests whether an isotropic osmotic pressure captures the full mechanical effect of composition-gradient free energy. The completed shortcut solves the chemical diffusion equation correctly, computes its isotropic osmotic pressure, projects the resulting gradient force through incompressible Stokes balance, and obtains zero flow. It uses a consistent diffusion-only closure: global composition is conserved and the same free energy decreases. The missing anisotropic stress is a physical constitutive assumption, not an algebra defect.

For a volume-preserving displacement xi, material advection gives delta c=-xi dot grad c. The work rule in the apparatus therefore gives force mu grad c, or equivalently -c grad mu after absorbing a gradient into pressure. With mu=c-0.2 lap(c), the solenoidal force is pressure-equivalent to -0.2 lap(c) grad(c). The oracle projects that force, solves the periodic Stokes equation at viscosity 0.004 with zero mean velocity, and evolves

    c_t = -div(c*u) + mobility*lap(mu).

A single stripe has pressure-balanced reversible force at every time. Its amplitude is exactly exp[-mobility*k^2*(1+0.2*k^2)*t] times the initial amplitude in both models. The calibration therefore uniquely identifies the one unknown mobility without testing the stress closure. Superpositions at the same wave-number magnitude also have pressure-balanced force. Hidden patterns combine unequal wave-number magnitudes and read both redistribution of original modes and generated harmonics.

The physical energy identity is

    dF/dt = -mobility*integral|grad(mu)|^2 - viscosity*integral|grad(u)|^2.

The shortcut omits the fluid dissipation term because its velocity vanishes, but still decreases F. This is why conservation and monotone energy alone cannot diagnose the missing physics.

The oracle uses dealiased Fourier composition and Stokes projection on a 48-by-48 grid with fourth-order exponential time stepping. The independent reference uses cell chemical potentials, face composition averages, conservative face fluxes, and a staggered Stokes projection on a 128-by-128 grid. The same face average in mechanical forcing and composition transport ensures a discrete work identity. Its time integrator is exponential midpoint, independent of the oracle method.

Validation over 256 independent noisy calibrations gives a maximum relative mobility error of 0.425%, and all fits pass the calibration threshold. The nominal fitted coefficient is 0.0399769142, with reduced chi-squared 0.93619. Oracle hidden absolute RMSE is at most 1.75e-5; completed-shortcut errors are 0.02655, 0.01827, and 0.005860. At the fitted-parameter extrema the oracle stays below 9.60e-5 and the shortcut stays above 0.00586. The common hidden limit is 0.003 absolute composition amplitude; it does not divide by near-zero generated modes.

The independent grid error is below 3.97e-5. Refining the reference to 192 points per direction and halving its time step changes outputs by less than 2.21e-5. Refining the oracle to 64 points and halving its time step changes hidden outputs by less than 1.08e-9; high-wave-number corner changes are below 2.24e-7. The instantaneous work identity holds to 3.5e-18, composition conservation to 4.1e-17, and velocity divergence to 7.3e-15. Both controls lower free energy. Tested composition stays between -0.34 and 0.34, within the binary-mixture range.

Local isolated controls: oracle 7 tests passed in 4.54 seconds; shortcut 4 tests passed and all 3 intended hidden groups failed in 4.44 seconds. The frozen gpt-5.6-luna high batch scored 0/3 unhinted and 3/3 hinted. Source, public trajectories, and verifier metrics were reviewed for all six trials. All unhinted submissions accurately fitted mobility but retained the pressure-only reversible force; all hinted submissions added the variational capillary force or its anisotropic stress tensor.

For each unhinted submission, an in-memory diagnostic changed only the two force-component lines while keeping the submitted mobility, Stokes solver, and time stepping fixed. Every hidden group then passed, with maximum error 1.75e-5 and exactly unchanged calibration predictions. Retained artifacts were not edited. The six audit entries and per-trial fixed-mobility repairs are in [the trial review report](../../results/hardening-materials-trial-reviews.json), under `capillary-mixture__S73x8eA`, `capillary-mixture__cJNup34`, `capillary-mixture__wFi5MMs`, `capillary-mixture__26CYdzm`, `capillary-mixture__HrxcFrp`, and `capillary-mixture__xr8WA8Q`. This paired result establishes the observed failure mode for this batch; it is not a guarantee of failure on future trials.

Reproduce scientific validation from the repository root:

```bash
uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_capillary_mixture.py
```

The calibration seed is 29021, independent-noise seed 39021, and measurement standard deviation 0.0007. The private true mobility is 0.04. Use `--generate` only for intentional regeneration of both public and private calibration copies.

Evidence: [scientific report](../../results/capillary-mixture-validation.json) and [independent source review](../../results/brownian-cross-review-capillary-mixture.json).

Primary context: [Gurtin, Polignone and Viñals, continuum derivation of coupled binary-fluid mechanics](https://arxiv.org/abs/patt-sol/9506001) develops the mechanical/thermodynamic framework for composition stresses. [The incompressible Cahn–Hilliard/Navier–Stokes energy identity](https://link.springer.com/article/10.1007/s00033-024-02312-w) gives the capillary tensor and conserved composition transport. This task uses its explicit convex quadratic free energy and creeping-flow limit; it does not ask the agent to assume phase separation or an unspecified double-well potential.
