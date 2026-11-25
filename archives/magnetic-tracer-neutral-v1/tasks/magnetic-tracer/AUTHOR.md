# Magnetic tracer, revision 1

The completed shortcut retains the exact locally suppressed scalar diffusivity and conservative Fick transport. For every constant field, the divergence of the antisymmetric mobility flux vanishes; both models give exactly the same position-density evolution and calibration. Zero-field measurements remove the nonzero-field friction ambiguity.

The specified small-mass Kramers limit has flux −M grad(rho), where M is the inverse local drag-plus-Lorentz matrix. In a field gradient its antisymmetric part changes density transport. Both models conserve probability, preserve uniform equilibrium and dissipate density entropy. No local trajectory current or velocity-dependent observable is tested because those require care in the overdamped limit.

The oracle uses periodic x collocation and one exact y Fourier mode. The independent reference derives the coupled x Fourier-moment matrix directly: L_pq=−(p*q+ky²)D_(p−q)−ky*(p−q)A_(p−q), where D=gamma/(gamma²+B²) and A=B/(gamma²+B²). It uses independently sampled Fourier coefficients and a separate truncation. Hidden mixed Fourier projections expose field-gradient transport.

Primary derivation: [Abdoli et al., Nondiffusive fluxes in a Brownian system with Lorentz force, PRE 101, 012120 (2020)](https://arxiv.org/abs/1908.03101). The public apparatus specifies underlying forces and the controlled small-mass limit; it does not supply the reduced transport equation.

Validation uses calibration seed 871023 and 256 noise realizations with seed 905472. Measurement sigma is 0.0006; hidden relative RMS tolerance is 0.025. The fitted gamma is 0.8001185, reduced chi-square 0.9883. Across noise draws the correct hidden error is at most 0.000693 while the shortcut error is at least 0.1274. Independent Fourier reference agreement is 1.1e-14; both discretization refinements are below 3e-14. Field reversal with y reflection, uniform equilibrium, positive density and decreasing L2 norm pass. Actual local tests give seven passes for the oracle and four passes plus three hidden failures for the completed shortcut.

Run `uv run --no-project --python 3.13 --with numpy==2.3.3 --with scipy==1.16.3 --with pytest==8.4.2 python scripts/validate_magnetic_tracer.py`. Data regeneration requires `--generate` explicitly. The review is in results/brownian-cross-review-magnetic-tracer.json.

## Reviewed evaluation

The frozen revision scores 0/3 unhinted and 3/3 hinted on GPT-5.6 Luna high, Codex CLI 0.154.0 and Harbor 0.21.0, with 600/60-second agent/verifier limits. All three ordinary failures pass calibration and parameter recovery and are reviewed physical-model failures. Fixed-parameter repairs recover the correct hidden predictions. See `results/candidates.json` and `results/hardening-status.json` for job IDs, per-trial reviews and matched-source checks. These are development-batch outcomes, not a future failure-rate guarantee.
