# Capillary-mixture revision 2

This revision tests the viscous constitutive law in an incompressible liquid with composition-dependent viscosity. Both controls include the correct capillary force, pressure projection, spatial viscosity field, and zero mean velocity. The completed shortcut uses componentwise momentum diffusion, div(eta grad u). The physical Newtonian stress is eta*(grad u+grad u transpose). For uniform viscosity, their difference is eta*grad(div u)=0. When viscosity varies, the missing transpose contribution generally survives the pressure projection.

This is a valid dissipative shortcut: its work loss is integral eta*|grad u|², and its Stokes operator is positive on mean-zero divergence-free velocities. The physical loss is integral 2*eta*D:D, with D the symmetric velocity gradient. The shortcut fails the local rigid-rotation constitutive check and need not conserve angular momentum through a symmetric local stress. Its hidden failure is a physical approximation, not an incorrect linear solver.

The apparatus uses a fresh prepared composition and an instantaneous velocity readout. The controlled zero-inertia limit allows momentum to relax before appreciable composition evolution; no external mechanism holds the pattern. The free energy gives chemical potential mu=c-0.2*lap(c) and reversible force mu*grad(c), pressure-equivalent to -0.2*lap(c)*grad(c). Unequal wave-number magnitudes drive nonzero solenoidal flow. Calibration has uniform viscosity, so both constitutive laws coincide exactly while the velocity scale identifies eta0.

The oracle solves the projected symmetric-stress equation on a 49-by-49 Fourier grid. The independent verifier minimizes Newtonian dissipation over a real streamfunction basis, so incompressibility is built into the variables. Its force uses mu*grad(c), and its stiffness uses symmetric strain components directly. It does not copy the oracle vector pressure projection or stress divergence.

The private eta0 is 0.004. Calibration has 144 measurements with a fixed, parameter-independent standard deviation of 0.001, calibration seed 31021, and independent-noise seed 41021. The fitted eta0 is 0.0039969210 and reduced chi-squared is 0.932143. All 256 noise realizations pass calibration and parameter checks; maximum parameter error is 0.0888%. The fixed standard deviation carries no noiseless-response side channel.

Nominal hidden absolute velocity RMSE is 0.000121–0.000205 for the oracle and 0.04524–0.10373 for the completed shortcut, against a 0.02 threshold. Across noise-fit extrema, oracle RMSE stays below 0.000237 and shortcut RMSE stays above 0.04511. This threshold is an absolute signed velocity-coefficient error; it does not divide by generated modes close to zero.

The independent reference differs from the oracle by at most 8.50e-8. Increasing the streamfunction cutoff from 12 to 16 changes outputs by at most 8.50e-8. Oracle grid refinement from 49 to 65 changes hidden outputs by 3.32e-14; a public high-wave-number corner refined to 81 changes by 1.79e-9. Both controls satisfy their own Stokes equations and positive work balance, with relative residual below 9.86e-12 and work error below 1.44e-19. Velocity divergence is below 4.35e-16. Translation covariance and composition scaling pass. The missing projected stress term is at least 13% of the applied force in the checked hidden preparations.

Local isolated verification gives oracle 7/7 passing tests in 0.93 seconds; the completed shortcut passes the four interface/calibration/parameter checks and fails all three intended hidden groups in 0.98 seconds. These controls establish scientific separation, not model difficulty. No revision-2 agent result is claimed here.

Revision 1, including its full task, controls, data, science report, and trial reviews, is preserved in [the neutral-v1 archive](../../archives/capillary-mixture-neutral-v1/README.md). It previously scored 0/3 plain and 3/3 hinted under the original instruction, then 3/3 plain under the neutral instruction. Those runs remain separate. Revision 2 preserves the neutral instruction byte for byte and changes the apparatus, fitted parameter, and readout to test a different constitutive approximation.

Reproduce the scientific checks with:

```bash
uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_capillary_mixture.py
```

Use `--generate` only to intentionally regenerate both calibration copies. See [the revision-2 science report](../../results/capillary-mixture-r2-validation.json) and [independent source review](../../results/brownian-cross-review-capillary-r2.json). The [Gurtin–Polignone–Viñals continuum theory](https://arxiv.org/abs/patt-sol/9506001) supplies background on thermodynamically consistent composition stresses; the explicit free energy and local Newtonian law in this task determine its particular equations.
