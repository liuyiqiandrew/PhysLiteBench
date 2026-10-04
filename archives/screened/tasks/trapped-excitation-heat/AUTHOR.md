# Trapped excitation heat, revision 2

Revision 1 scored 1/3 plain and 3/3 hinted. Two plain trials kept a Fourier-only energy flux; correcting only that flux at the fitted diffusivity repaired all predictions without changing calibration. The complete r1 task, scripts, six trial reviews and source hashes are preserved in archives/trapped-excitation-heat-r1.

Revision 2 has two mobile internal levels with energies (1,3), degeneracies (1,9), and hopping coefficients (D,0.4D), plus the immobile nondegenerate ground state. It retains the correctly transported mean excitation energy from r1 and tests the independent heat flux caused by relative transport between mobile levels.

Write p_i=g_i exp(-E_i/T)/Z, R=sum r_i p_i, A=sum E_i r_i p_i, and U=sum E_i p_i, with r=(1,0.4). Mass flux is j=-D grad(R c), and total energy density is C T+U c. The shortcut uses particle energy flux (A/R) j. This correctly carries the mean energy per particle current and conserves total energy globally, but omits relative transport. The exact energy flux is -D grad(A c). Their difference is -D c R Var_transport(E)/T^2 grad T, a positive thermal conduction channel even when the total particle current vanishes. The shortcut is a positive rank-one particle/energy transport closure, not a sign mistake or unstable solver.

Uniform-temperature clamped calibration gives the exact concentration decay rate D R(T) k^2 for both controls, identifying D independently of the isolated thermal response. Hidden thermal waves, approximately balanced particle gradients, and concentration-induced heat test the missing conduction. All initial states and observables remain in the public domain.

The oracle evolves (c,T) on 64 Fourier points with the energy flux above. The independent reference evolves (c,u) by conservative finite-volume currents j1+j2 and j1+3j2 on 256 cells. Refinement to 512 cells agrees with the oracle to 0.00016 of the signal scale. Integrated particle number and energy drift by less than 1.2e-15. The ideal internal-state entropy, including the level-2 degeneracy, increases.

The unchanged prediction tolerance is relative RMS 0.04, floor 1e-6. Seed 17133 generates calibration and seed 27133 supplies 256 independent noise draws. Every calibration and parameter check passes. At fitted-parameter extrema, the oracle error is at most 0.00458 and the shortcut error is at least 1.348. Nominal maximum oracle error is 0.000129. See results/trapped-excitation-heat-r2-science.json for actual local-control outcomes and the complete numerical report.

Reproduce with:

```
uv run --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_trapped_excitation_heat.py
```

Evaluate three plain and three hinted gpt-5.6-luna high trials under the unchanged native runner, 600-second agent limit and 60-second verifier limit. Difficulty results for r2 are pending. The hint supplies the energy-flux closure and no fitted parameter or hidden cases.
