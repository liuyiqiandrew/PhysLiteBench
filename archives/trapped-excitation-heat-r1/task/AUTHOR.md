# Trapped excitation heat, revision 1

The calibration uses thermostatted concentration decay to identify the shared mobile-state diffusivity. The exact and shortcut models agree on every calibration input. In isolated measurements, a particle hop carries excitation energy even though it exchanges no heat with the lattice locally. The supplied predictor keeps the full local equilibrium excitation heat capacity and conserves total energy globally, but uses only the Fourier energy flux. The resulting temperature response to a concentration modulation has the opposite sign.

Let f=p(T)c be the mobile concentration, with p=1/(1+exp(Delta/T)). Number balance is c_t=D f_xx. Energy density is u=C T+Delta f and its flux is -kappa T_x+Delta j, with j=-D f_x. Eliminating f gives (C+Delta c p') T_t=kappa T_xx+Delta(1-p)c_t. The shortcut omits Delta j and instead has -Delta p c_t. Neither model violates the integrated mass or energy balances on a periodic domain; the difference is a local physical flux.

The oracle evolves concentration and temperature using a Fourier spatial method with 64 points. The independent reference evolves concentration and total energy using conservative finite-volume face fluxes on 256 cells, then inverts the local energy relation to recover temperature. A 512-cell refinement agrees with the oracle within 3.2e-5 of the signal scale. Mass and energy drifts are below 1e-15 and the ideal dilute two-state entropy increases. The entropy is C log T - c_bound log c_bound - c_free log c_free + c, up to conserved constants.

The prediction tolerance is relative RMS 0.04, normalized by each group signal RMS (floor 1e-6). It exceeds calibration uncertainty and numerical error while resolving the change in thermal pattern. Seed 15111 generates the checked-in calibration once; 256 additional noise realizations use seed 25111. All fits meet calibration chi-square and parameter thresholds. At observed fitted-parameter extrema, the maximum oracle error is 0.00222 and the minimum shortcut error is 0.5829. The actual local tests give oracle 7/7 and shortcut 4 passed, 3 intended hidden failures. Evidence: results/trapped-excitation-heat-science.json.

Reproduce scientific checks from the repository root:

```
uv run --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_trapped_excitation_heat.py
```

The planned evaluation is three unhinted and three hinted gpt-5.6-luna high trials, under the repository's pinned native runner and 600-second agent / 60-second verifier limits. Harbor difficulty evidence is pending. The hint states the particle energy flux; it does not reveal the fitted parameter or hidden cases.
