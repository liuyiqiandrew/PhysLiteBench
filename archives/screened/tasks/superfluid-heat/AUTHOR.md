# Superfluid heat, revision 1

The normal component transports entropy while the superfluid can carry an opposite mass flux. Zero total mass current therefore permits a convective heat current. The supplied predictor includes the correct steady counterflow enhancement of thermal conductivity and uses every hydrodynamic parameter, but eliminates the counterflow velocity instantaneously. Uniform-temperature calibration cannot test that elimination; finite-time temperature waves depend on counterflow inertia.

With C=rho*c_p, c2^2=(rho_s/rho_n)*s^2*T0/c_p and nu=lambda*rho/(rho_n*rho_s), the temperature cosine coefficient theta and convective heat-current sine coefficient Q obey

theta_dot=-(G+kappa*k^2)*theta/C-k*Q/C,
Q_dot=C*c2^2*k*theta-nu*Q.

Both component velocities, hence Q, start at zero. The shortcut sets Q_dot=0 and obtains kappa_eff=kappa+C*c2^2/nu. This is a passive zero-frequency approximation, not an incorrect formula. It loses transient counterflow buildup and damped second-sound oscillations. Calibration mode k=0 gives theta=theta0 exp(-G*t/C) exactly for both models and identifies the common G.

The oracle uses a two-state matrix exponential. The independent reference integrates temperature, normal velocity and superfluid velocity directly using superfluid chemical-potential acceleration, mutual drag and total momentum conservation. Their off-grid differences are below 8e-14. The total mass current stays zero to 1.1e-17. The quadratic temperature/kinetic perturbation energy has exactly the expected decay from conductivity, bath exchange and mutual friction. Frictional heating is quadratic and is correctly omitted from the linear temperature equation.

The tolerance is relative RMS 0.04 of each group signal, floor 1e-6. Seed 18255 generates calibration; 256 additional draws use seed 28255. Every parameter and calibration check passes. The maximum oracle error at fitted-parameter extrema is 0.000808; the minimum shortcut error is 0.491. Actual local controls give oracle 7/7, shortcut 4 passed and 3 intended hidden failures. Evidence: results/superfluid-heat-science.json. Final source review: results/constraint-hardening-physics-review.json, superfluid-heat-r1.

Reproduce scientific checks with:

```
uv run --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_superfluid_heat.py
```

The planned evaluation uses three plain and three hinted gpt-5.6-luna high trials with the pinned native setup and unchanged 600-second agent / 60-second verifier limits. Difficulty results are pending. The hint explains the missing counterflow state without revealing the fitted heat leak or hidden cases.
