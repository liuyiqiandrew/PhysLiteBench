# Active-trap revision 1

This candidate tests whether the agent distinguishes fixed-magnitude propulsion with rotational Brownian motion from Gaussian colored propulsion having the same two-time covariance. Both give exactly the same supplied free-displacement and stationary trap-variance calibrations. The hidden detector measures signed stationary density Fourier components, which depend on higher orientation correlations.

The microscopic dynamics are

    dr = (-k r + v u(theta)) dt + sqrt(2 Dt) dW,
    dtheta = sqrt(2 Dr) dWtheta.

The public starter already has a complete weighted calibration fit and the complete Gaussian predictor. Unlike several earlier tasks with an unfinished fit method, this revision can pass the public checks without edits. This deliberate recorded starter difference is frozen for the paired evaluation; the instruction, calibration, and hidden physics remain the same in plain and hinted runs.

The sole unknown is Dr. The calibration uses several speeds, thermal diffusivities, free observation times, and trap rates. Its nonzero-speed trap variance decreases strictly with Dr, so no second fitted parameter or degeneracy hides the model error. The completed shortcut is the exact prediction for stationary Gaussian propulsion with covariance delta_ij v^2 exp(-Dr |t-s|)/2. It conserves probability and matches every calibration preparation, rather than using an incorrect implementation of the covariance.

For the oracle, stationary position is an exponentially weighted propulsion history plus independent Gaussian thermal displacement. Stationary angular Brownian motion is reversible, so its past can be replaced by forward angular diffusion. The angular Feynman–Kac generator is

    L(t) = Dr * d_theta^2 + i*q*v*exp(-k*t)*cos(theta).

Starting in the uniform angular distribution, its zeroth Fourier component gives the active characteristic function. Thermal motion contributes exp(-Dt*q^2/(2*k)). The oracle uses angular modes -20 through 20 and truncates the weighted history at 25/k. The maximum omitted displacement phase is 7 exp(-25), below 1e-10.

The verifier uses an independent stationary polynomial hierarchy. For dimensionless complex active position z=k(x+iy)/v, rotational symmetry fixes the orientation exponent in <z^p zbar^q exp(i*l*theta)> to l=q-p. Its exact moment recursion is

    [k*(p+q) + Dr*(q-p)^2] M[p,q]
       = k*p*M[p-1,q] + k*q*M[p,q-1].

The radial moments M[n,n] enter the isotropic characteristic function through the J0 power series. This calculation uses neither time integration nor the oracle angular generator. Translational Brownian motion is independent in this linear trap, so its Gaussian factor is exact in both constructions.

The readout limit is an absolute RMSE of 0.025 for a bounded dimensionless mean. The hidden values include sign changes, so relative normalization would exaggerate errors near zeros. The Gaussian shortcut errors are 0.233–0.299; the fitted oracle errors are below 0.000125. Across 256 independent calibration-noise realizations, all fits meet the calibration and parameter thresholds, the physical-reference hidden error stays below 0.000889, and the shortcut error stays above 0.2326. Direct angular predictions at the fitted-parameter extrema give the same margin.

Scientific validation checks the whole allowed Dr/trap range at detector arguments 0, 3.1, 7. Oracle/reference differences are below 1.94e-9; increasing angular resolution and integration accuracy changes predictions by less than 1.45e-9. Increasing the verifier moment order 36 to 48 produces no resolved difference. Probability normalization, characteristic-function bounds, Fourier Gram positivity, exact second moments, the passive Gaussian limit, and the zero-rotation ring/Bessel limit all pass.

Local isolated controls: oracle 7 tests passed in 1.95 s; completed shortcut 4 calibration/interface/parameter tests passed and all 3 intended hidden groups failed in 0.42 s. Docker controls and Luna trials are scheduled separately by the root controller. Scientific margins alone do not establish agent difficulty.

Reproduce the scientific check from the repository root:

```bash
uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_active_trap.py
```

Use `--generate` only when intentionally regenerating both calibration copies. The fixed seeds are 28021 for the checked-in sample and 38021 for the 256-noise study. The calibration standard deviation is 0.008 times max(abs(noiseless value),0.05). The grader uses its own calibration copy. The private true coefficient is 0.65/s.

The shareable report is [active-trap-validation.json](../../results/active-trap-validation.json). Independent source review and hashes are in [brownian-cross-review-active-trap.json](../../results/brownian-cross-review-active-trap.json).

Primary context: [Chaudhuri and Dhar, exact moments of an active Brownian particle in a harmonic trap](https://arxiv.org/abs/2005.14234) derive position-moment dynamics and distinguish their non-Gaussian stationary statistics. [Malakar et al., exact stationary series](https://arxiv.org/abs/1902.04171) treat the same constant-speed, rotational-diffusion particle with independent translational diffusion in an isotropic trap. The present independent constructions and validation are supplied with this task.
