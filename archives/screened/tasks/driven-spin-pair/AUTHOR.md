# Driven spin pair, revision 1

This is a replacement candidate for the shared-noise qubit mechanism, not an additional independent mechanism to count alongside qubit-control. The earlier qubit revision and its run artifacts remain intact. Two probes undergo noncommuting continuous drive segments in a common quasistatic detuning field. The starter correctly composes each probe's rotations and drive segments and performs Gaussian quadrature. Its only modeling error is multiplying marginal averages instead of averaging the conditional product.

For a shot with detuning xi, local unitary evolution preserves a product state because the Hamiltonian has no interaction term. The measured joint probability is nevertheless

    P++ = integral p0(xi) p1(xi) Normal(xi; 0, width^2) dxi,

not the product of the two separately integrated probabilities. Classical shared fluctuations create correlations without requiring entanglement. The same detuning remains fixed through both segments. The relevant distinction between spatially correlated classical noise and quantum interactions is discussed by [Zou, Bosco, and Loss](https://arxiv.org/abs/2308.03054); this task uses only the explicitly defined classical quasistatic model.

Calibration leaves probe 1 in +z, undriven and read along +z, hence p1(xi)=1. Probe 0 is an undriven Ramsey probe, yielding P++ = [1+exp(-width^2 t^2/2)]/2. Both controls therefore match every noiseless calibration input exactly, and the envelope identifies the one unknown width. Hidden preparations rotate and drive both probes, including different axes in the two segments. The public apparatus states the field's uniformity, Gaussian law, long correlation time, local control conventions, shot averaging, and joint measurement.

The oracle uses 96-point Gauss-Hermite integration and Bloch rotations. The independent reference evolves complex two-component state vectors with explicit Pauli-matrix unitaries and uses adaptive Gaussian quadrature. Checks compare 96 and 144 quadrature nodes, zero-noise factorization, the exact shared-noise Ramsey formula, nonnegative covariance for identical probe controls, and both allowed noise-width endpoints.

Validation uses seed 9322 and 200 calibration records with Gaussian errors at 0.6% of the largest probability. The private parameter tolerance is 3%; reduced chi-square must be below 1.5. Hidden groups use absolute probability RMSE with a limit of 0.025. This is much larger than calibration uncertainty and quadrature error.

Checked-in calibration: width 0.94929914 radians/second (true 0.95), relative error 0.0738%, reduced chi-square 1.00578. Oracle hidden RMSE is at most 0.000141; shortcut RMSE ranges from 0.06497 to 0.08560. All 256 additional noise realizations pass calibration and parameter checks; maximum width error is 0.362%. At fitted-width extrema oracle hidden error remains below 0.000691 while shortcut error stays above 0.06440. Independent reference and analytic checks agree below 7e-16 on the tested sequences, including width endpoints.

Reproduce from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 scripts/validate_driven_spin_pair.py driven-spin-pair
```

Add `--generate` only to intentionally regenerate both calibration copies. The report is `jobs/driven-spin-pair-validation/summary.json`. Local isolated-copy pytest: physical oracle 7 passed; completed shortcut 4 passed and 3 hidden prediction tests failed. Harbor controls and paired gpt-5.6-luna/high 3-trial evaluation are coordinated separately; the requested failure rate remains an empirical question.
